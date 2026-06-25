from dataclasses import dataclass, field
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


APPLIANCES = {
    "dishwasher": 0,
    "fridge": 1,
    "microwave": 2,
    "washer_dryer": 3,
}

CSV_APPLIANCES = {
    "kettle": {"power": "kettle_power", "on": "kettle_on"},
    "fridge": {"power": "fridge_power", "on": "fridge_on"},
    "microwave": {"power": "microwave_power", "on": "microwave_on"},
    "dishwasher": {"power": "dishwasher_power", "on": "dishwasher_on"},
    "washingmachine": {"power": "washingmachine_power", "on": "washingmachine_on"},
}

ALL_APPLIANCES = sorted(set(APPLIANCES) | set(CSV_APPLIANCES))


APPLIANCE_DISPLAY_NAMES = {
    "dishwasher": "Dishwasher",
    "fridge": "Fridge",
    "microwave": "Microwave",
    "washer_dryer": "Washer dryer",
}


@dataclass
class SGNConfig:
    input_length: int = 864
    output_length: int = 64
    input_channels: int = 1
    target_appliances: list[str] = field(default_factory=list)
    num_appliances: int = 1
    scale: float = 1.0
    scale_mode: str = "aggregate_std"
    feature_columns: list[str] = field(default_factory=lambda: ["aggregate"])
    feature_mean: list[float] = field(default_factory=list)
    feature_scale: list[float] = field(default_factory=list)
    on_threshold_watts: float = 15.0
    batch_size: int = 16
    learning_rate: float = 1.0e-4
    epochs: int = 200
    patience: int = 30
    num_workers: int = 0
    hidden_fc: int = 1024
    dropout: float = 0.0
    train_stride: int = 1
    eval_stride: int = 32
    sae_period: int = 600
    seed: int = 1234
    gate_mode: str = "soft"
    standby_power: bool = False
    weight_decay: float = 0.0
    early_stop_metric: str = "total_loss"
    label_smoothing: float = 0.0
    reg_on_weight: float = 0.0
    gated_on_weight: float = 0.0
    on_confidence_weight: float = 0.0
    on_smooth_weight: float = 0.0
    bce_pos_weight: float = 1.0
    oversample_on: bool = False
    oversample_max_weight: float = 15.0
    grad_clip_norm: float = 1.0
    lr_scheduler_patience: int = 8
    lr_scheduler_factor: float = 0.5
    lr_min: float = 1e-6
    min_epochs: int = 5
    val_split_label: str = "validation"
    test_split_label: str = "test"


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "MATNILM" / "data" / "redd"


def default_csv_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "training_data_ukdale_cross_house.json"


def default_model_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "sgn_paper.json"


def describe_csv_split_label(csv_cfg: dict, split: str) -> str:
    """Human-readable split description for logs and waveform titles."""
    custom = csv_cfg.get("split_description")
    if custom and split == "val":
        return f"validating CSV ({custom})"
    if custom and split == "train":
        return f"training CSV ({custom})"

    split_mode = csv_cfg.get("split_mode", "temporal")
    val_mode = csv_cfg.get("val_mode")
    if split_mode == "holdout":
        if split == "test":
            return "house 2 (testing CSV)"
        if split == "val":
            if val_mode == "separate_files":
                val_houses = csv_cfg.get("val_house_ids")
                if val_houses:
                    house_text = ",".join(str(h) for h in val_houses)
                    return f"validating CSV (house {house_text})"
                return "validating CSV (separate file)"
            if val_mode == "by_house_tail":
                houses = csv_cfg.get("val_house_ids", [5])
                days = csv_cfg.get("val_last_days", 7)
                house_text = ",".join(str(h) for h in houses)
                return f"house {house_text} last {days:g} days (val)"
            if val_mode == "by_house":
                houses = csv_cfg.get("val_house_ids", [5])
                return f"house {','.join(str(h) for h in houses)} (val)"
        if split == "train":
            if val_mode == "separate_files":
                return "training CSV (houses 1+5 early timeline)"
            if val_mode == "by_house_tail":
                return "houses 1+5 train portion (excl. val tail)"
            return "train CSV (holdout)"
    return split


def load_model_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_std_scale(data_dir: str | Path) -> float:
    """SGN paper normalization: divide by std of aggregate training power."""
    path = Path(data_dir) / "train_small.pkl"
    with path.open("rb") as handle:
        frames = pickle.load(handle)
    mains = [np.asarray(frame.values[:, 0], dtype=np.float32) for frame in frames]
    scale = float(np.std(np.concatenate(mains)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid aggregate std scale computed from {path}: {scale}")
    return scale


def _resolve_csv_path(path_str: str, config_dir: Path) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return str(path)


def load_csv_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if "feature_columns" not in cfg or not cfg["feature_columns"]:
        raise ValueError(f"CSV config {path} must define non-empty feature_columns")
    cfg.setdefault("time_column", "readable_time")
    cfg.setdefault("aggregate_column", "aggregate")
    cfg.setdefault("appliances", CSV_APPLIANCES)
    cfg.setdefault("split_mode", "temporal")
    cfg.setdefault("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})

    split_mode = cfg["split_mode"]
    if split_mode == "holdout":
        if not cfg.get("train_csv_file") or not cfg.get("test_csv_file"):
            raise ValueError(
                f"CSV config {path} with split_mode='holdout' must define "
                "train_csv_file and test_csv_file"
            )
        cfg["train_csv_file"] = _resolve_csv_path(cfg["train_csv_file"], path.parent)
        cfg["test_csv_file"] = _resolve_csv_path(cfg["test_csv_file"], path.parent)
        if cfg.get("val_csv_file"):
            cfg["val_csv_file"] = _resolve_csv_path(cfg["val_csv_file"], path.parent)
        if cfg.get("val_mode") == "separate_files" and not cfg.get("val_csv_file"):
            raise ValueError(
                f"CSV config {path} with val_mode='separate_files' must define val_csv_file"
            )
        for key in ("train_csv_file", "val_csv_file", "test_csv_file"):
            file_path = cfg.get(key)
            if file_path and not Path(file_path).exists():
                raise FileNotFoundError(f"CSV config {path} references missing file: {file_path}")
    else:
        if "csv_file" not in cfg:
            raise ValueError(f"CSV config {path} must define csv_file")
        cfg["csv_file"] = _resolve_csv_path(cfg["csv_file"], path.parent)
    return cfg


def csv_path_for_split(csv_cfg: dict, split: str) -> Path:
    if csv_cfg.get("split_mode") == "holdout":
        if split == "test":
            return Path(csv_cfg["test_csv_file"])
        if split == "val" and csv_cfg.get("val_mode") == "separate_files":
            return Path(csv_cfg["val_csv_file"])
        return Path(csv_cfg["train_csv_file"])
    return Path(csv_cfg["csv_file"])


def describe_csv_sources(csv_cfg: dict) -> str:
    if csv_cfg.get("split_mode") == "holdout":
        parts = [
            f"train={csv_cfg['train_csv_file']}",
            f"test={csv_cfg['test_csv_file']}",
        ]
        if csv_cfg.get("val_csv_file"):
            parts.insert(1, f"val={csv_cfg['val_csv_file']}")
        return ", ".join(parts)
    return str(csv_cfg["csv_file"])


def csv_split_bounds(
    n_rows: int,
    split_ratios: dict[str, float],
    split: str,
    *,
    split_mode: str = "temporal",
) -> tuple[int, int]:
    if split_mode == "holdout" and split == "test":
        return 0, n_rows
    train_ratio = float(split_ratios.get("train", 0.7))
    val_ratio = float(split_ratios.get("val", 0.15))
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    if split == "train":
        return 0, train_end
    if split == "val":
        return train_end, val_end
    if split == "test":
        return val_end, n_rows
    raise ValueError("split must be one of: train, val, test")


def select_csv_split_df(df: pd.DataFrame, csv_cfg: dict, split: str) -> pd.DataFrame:
    """Select train/val/test rows according to csv config split rules."""
    split_mode = csv_cfg.get("split_mode", "temporal")
    if split_mode == "holdout" and split == "test":
        return df.copy()

    val_mode = csv_cfg.get("val_mode")
    if split_mode == "holdout" and val_mode == "separate_files":
        if split in {"train", "val", "test"}:
            return df.copy()
        raise ValueError(f"Unsupported split for separate_files mode: {split}")
    house_col = csv_cfg.get("house_column", "house")
    time_col = csv_cfg.get("time_column", "readable_time")

    if split_mode == "holdout" and val_mode == "by_house":
        train_houses = set(csv_cfg.get("train_house_ids", [1]))
        val_houses = set(csv_cfg.get("val_house_ids", [5]))
        ordered = df.sort_values([house_col, time_col]).reset_index(drop=True)
        if split == "train":
            return ordered[ordered[house_col].isin(train_houses)].reset_index(drop=True)
        if split == "val":
            return ordered[ordered[house_col].isin(val_houses)].reset_index(drop=True)
        raise ValueError(f"Unsupported split for by_house mode: {split}")

    if split_mode == "holdout" and val_mode == "by_house_tail":
        val_houses = csv_cfg.get("val_house_ids", [5])
        val_last_days = float(csv_cfg.get("val_last_days", 7))
        work = df.copy()
        work[time_col] = pd.to_datetime(work[time_col])
        val_mask = pd.Series(False, index=work.index)
        for house_id in val_houses:
            house_rows = work[work[house_col] == house_id]
            if house_rows.empty:
                continue
            end_time = house_rows[time_col].max()
            start_time = end_time - pd.Timedelta(days=val_last_days)
            val_mask |= (work[house_col] == house_id) & (work[time_col] >= start_time)
        ordered = work.sort_values([house_col, time_col]).reset_index(drop=True)
        val_mask = val_mask.reindex(ordered.index, fill_value=False)
        if split == "val":
            return ordered[val_mask.to_numpy()].reset_index(drop=True)
        if split == "train":
            return ordered[~val_mask.to_numpy()].reset_index(drop=True)
        raise ValueError(f"Unsupported split for by_house_tail mode: {split}")

    start, end = csv_split_bounds(
        len(df),
        csv_cfg["split_ratios"],
        split,
        split_mode=split_mode,
    )
    return df.iloc[start:end].copy()


def csv_training_stats(csv_cfg: dict, scale_mode: str) -> tuple[float, list[float], list[float]]:
    csv_path = csv_path_for_split(csv_cfg, "train")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV training file: {csv_path}")
    feature_columns = list(csv_cfg["feature_columns"])
    aggregate_column = csv_cfg.get("aggregate_column", "aggregate")
    house_col = csv_cfg.get("house_column", "house")
    time_col = csv_cfg.get("time_column", "readable_time")
    usecols = sorted(set(feature_columns + [aggregate_column, house_col, time_col]))
    df = pd.read_csv(csv_path, usecols=usecols)
    train_df = select_csv_split_df(df, csv_cfg, "train")

    feature_mean = train_df[feature_columns].mean().astype(float).tolist()
    feature_scale = train_df[feature_columns].std(ddof=0).replace(0, 1.0).astype(float).tolist()
    if scale_mode == "aggregate_std":
        scale = float(train_df[aggregate_column].std(ddof=0))
    else:
        scale = 612.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid CSV target scale computed from {csv_path}: {scale}")
    return scale, feature_mean, feature_scale


def csv_appliance_on_stats(csv_cfg: dict, appliance: str) -> dict[str, float]:
    """Summarize true ON power in the train CSV (helps interpret normalized targets)."""
    power_col = csv_cfg.get("appliances", CSV_APPLIANCES).get(appliance, {}).get("power")
    on_col = csv_cfg.get("appliances", CSV_APPLIANCES).get(appliance, {}).get("on")
    if not power_col or not on_col:
        return {}
    csv_path = csv_path_for_split(csv_cfg, "train")
    df = pd.read_csv(csv_path, usecols=[power_col, on_col])
    train_df = select_csv_split_df(df, csv_cfg, "train")
    on_power = train_df.loc[train_df[on_col] >= 0.5, power_col].astype(float)
    if on_power.empty:
        return {}
    return {
        "mean_on_watts": float(on_power.mean()),
        "p95_on_watts": float(on_power.quantile(0.95)),
        "max_on_watts": float(on_power.max()),
    }

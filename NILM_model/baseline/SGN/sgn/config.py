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
    eval_stride: int = 64
    sae_period: int = 1200
    seed: int = 1234
    gate_mode: str = "soft"
    standby_power: bool = False


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "MATNILM" / "data" / "redd"


def default_csv_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "training_data_ukdale_paper.json"


def default_model_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "sgn_paper.json"


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
    else:
        if "csv_file" not in cfg:
            raise ValueError(f"CSV config {path} must define csv_file")
        cfg["csv_file"] = _resolve_csv_path(cfg["csv_file"], path.parent)
    return cfg


def csv_path_for_split(csv_cfg: dict, split: str) -> Path:
    if csv_cfg.get("split_mode") == "holdout":
        if split == "test":
            return Path(csv_cfg["test_csv_file"])
        return Path(csv_cfg["train_csv_file"])
    return Path(csv_cfg["csv_file"])


def describe_csv_sources(csv_cfg: dict) -> str:
    if csv_cfg.get("split_mode") == "holdout":
        return (
            f"train={csv_cfg['train_csv_file']}, "
            f"test={csv_cfg['test_csv_file']}"
        )
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

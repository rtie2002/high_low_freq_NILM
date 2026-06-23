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
    return Path(__file__).resolve().parents[1] / "configs" / "training_data_house2.json"


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


def load_csv_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if "csv_file" not in cfg:
        raise ValueError(f"CSV config {path} must define csv_file")
    if "feature_columns" not in cfg or not cfg["feature_columns"]:
        raise ValueError(f"CSV config {path} must define non-empty feature_columns")
    cfg.setdefault("time_column", "readable_time")
    cfg.setdefault("aggregate_column", "aggregate")
    cfg.setdefault("appliances", CSV_APPLIANCES)
    cfg.setdefault("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    csv_file = Path(cfg["csv_file"])
    if not csv_file.is_absolute():
        csv_file = (path.parent / csv_file).resolve()
    cfg["csv_file"] = str(csv_file)
    return cfg


def csv_split_bounds(n_rows: int, split_ratios: dict[str, float], split: str) -> tuple[int, int]:
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


def csv_training_stats(csv_cfg: dict, scale_mode: str) -> tuple[float, list[float], list[float]]:
    csv_path = Path(csv_cfg["csv_file"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV training file: {csv_path}")
    feature_columns = list(csv_cfg["feature_columns"])
    aggregate_column = csv_cfg.get("aggregate_column", "aggregate")
    usecols = sorted(set(feature_columns + [aggregate_column]))
    df = pd.read_csv(csv_path, usecols=usecols)
    train_start, train_end = csv_split_bounds(len(df), csv_cfg["split_ratios"], "train")
    train_df = df.iloc[train_start:train_end]

    feature_mean = train_df[feature_columns].mean().astype(float).tolist()
    feature_scale = train_df[feature_columns].std(ddof=0).replace(0, 1.0).astype(float).tolist()
    if scale_mode == "aggregate_std":
        scale = float(train_df[aggregate_column].std(ddof=0))
    else:
        scale = 612.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid CSV target scale computed from {csv_path}: {scale}")
    return scale, feature_mean, feature_scale

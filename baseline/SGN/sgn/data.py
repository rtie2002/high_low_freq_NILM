from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import APPLIANCES, CSV_APPLIANCES, SGNConfig, csv_split_bounds


class REDDSGNWindowDataset(Dataset):
    """Windowed single-appliance dataset built from MATNILM REDD pickle files."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        appliance: str,
        config: SGNConfig,
        stride: int,
    ) -> None:
        if appliance not in APPLIANCES:
            raise ValueError(f"Unknown appliance '{appliance}'. Choices: {sorted(APPLIANCES)}")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.data_dir = Path(data_dir)
        self.split = split
        self.appliance = appliance
        self.appliance_index = APPLIANCES[appliance]
        self.config = config
        self.stride = stride

        path = self.data_dir / f"{split}_small.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed REDD file: {path}")

        with path.open("rb") as handle:
            frames = pickle.load(handle)

        self.sequences: list[np.ndarray] = []
        self.index: list[tuple[int, int]] = []
        for seq_id, frame in enumerate(frames):
            values = np.asarray(frame.values, dtype=np.float32)
            if values.shape[1] < 5:
                raise ValueError(
                    f"Expected main + 4 appliance columns in {path}, got shape {values.shape}"
                )
            if len(values) < config.input_length:
                continue
            self.sequences.append(values)
            for start in range(0, len(values) - config.input_length + 1, stride):
                self.index.append((seq_id, start))

        if not self.index:
            raise ValueError(f"No windows produced for split={split}, appliance={appliance}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_id, start = self.index[idx]
        values = self.sequences[seq_id]
        cfg = self.config

        in_start = start
        in_end = in_start + cfg.input_length
        out_start = in_start + (cfg.input_length - cfg.output_length) // 2
        out_end = out_start + cfg.output_length

        main = values[in_start:in_end, 0]
        appliance_watts = values[out_start:out_end, self.appliance_index + 1]
        appliance_scaled = appliance_watts / cfg.scale
        on_label = (appliance_watts > cfg.on_threshold_watts).astype(np.float32)

        return {
            "x": torch.from_numpy((main / cfg.scale).astype(np.float32)).unsqueeze(0),
            "y": torch.from_numpy(appliance_scaled.astype(np.float32)),
            "y_watts": torch.from_numpy(appliance_watts.astype(np.float32)),
            "on": torch.from_numpy(on_label),
        }


class CSVSGNWindowDataset(Dataset):
    """Windowed SGN dataset built from a merged feature CSV."""

    def __init__(
        self,
        csv_config: dict,
        split: str,
        appliance: str,
        config: SGNConfig,
        stride: int,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        appliances = csv_config.get("appliances", CSV_APPLIANCES)
        if appliance not in appliances:
            raise ValueError(f"Unknown CSV appliance '{appliance}'. Choices: {sorted(appliances)}")

        self.csv_config = csv_config
        self.split = split
        self.appliance = appliance
        self.config = config
        self.stride = stride

        app_cfg = appliances[appliance]
        power_column = app_cfg["power"]
        on_column = app_cfg.get("on")
        feature_columns = list(csv_config["feature_columns"])
        usecols = list(dict.fromkeys(feature_columns + [power_column] + ([on_column] if on_column else [])))

        csv_path = Path(csv_config["csv_file"])
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV training file: {csv_path}")

        df = pd.read_csv(csv_path, usecols=usecols)
        start, end = csv_split_bounds(len(df), csv_config["split_ratios"], split)
        df = df.iloc[start:end].copy()
        df = df.dropna(subset=usecols)
        if len(df) < config.input_length:
            raise ValueError(
                f"Split {split} is too short for input_length={config.input_length}: {len(df)} rows"
            )

        features = df[feature_columns].to_numpy(dtype=np.float32)
        mean = np.asarray(config.feature_mean, dtype=np.float32)
        scale = np.asarray(config.feature_scale, dtype=np.float32)
        if mean.shape[0] != len(feature_columns) or scale.shape[0] != len(feature_columns):
            raise ValueError("Feature normalization stats do not match feature_columns")
        self.features = (features - mean) / scale
        self.power = df[power_column].to_numpy(dtype=np.float32)
        if on_column and on_column in df:
            self.on = df[on_column].to_numpy(dtype=np.float32)
        else:
            self.on = (self.power > config.on_threshold_watts).astype(np.float32)

        self.index = list(range(0, len(df) - config.input_length + 1, stride))
        if not self.index:
            raise ValueError(f"No windows produced for CSV split={split}, appliance={appliance}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.index[idx]
        cfg = self.config
        in_start = start
        in_end = in_start + cfg.input_length
        out_start = in_start + (cfg.input_length - cfg.output_length) // 2
        out_end = out_start + cfg.output_length

        x = self.features[in_start:in_end].T
        appliance_watts = self.power[out_start:out_end]
        appliance_scaled = appliance_watts / cfg.scale
        on_label = self.on[out_start:out_end]

        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y": torch.from_numpy(appliance_scaled.astype(np.float32)),
            "y_watts": torch.from_numpy(appliance_watts.astype(np.float32)),
            "on": torch.from_numpy(on_label.astype(np.float32)),
        }

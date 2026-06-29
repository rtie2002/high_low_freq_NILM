from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import APPLIANCES, CSV_APPLIANCES, SGNConfig, csv_path_for_split, select_csv_split_df


def _as_appliance_list(appliance: str | list[str], choices) -> list[str]:
    if appliance == "all":
        return list(choices)
    if isinstance(appliance, str):
        return [appliance]
    return list(appliance)


class REDDSGNWindowDataset(Dataset):
    """Windowed multi-appliance dataset built from MATNILM REDD pickle files."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        appliance: str | list[str],
        config: SGNConfig,
        stride: int,
    ) -> None:
        appliances = _as_appliance_list(appliance, APPLIANCES)
        unknown = sorted(set(appliances) - set(APPLIANCES))
        if unknown:
            raise ValueError(f"Unknown appliance(s) {unknown}. Choices: {sorted(APPLIANCES)}")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.data_dir = Path(data_dir)
        self.split = split
        self.appliances = appliances
        self.appliance_indices = [APPLIANCES[name] for name in appliances]
        self.config = config
        self.stride = stride

        path = self.data_dir / f"{split}_small.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed REDD file: {path}")

        with path.open("rb") as handle:
            frames = pickle.load(handle)

        self.sequences: list[np.ndarray] = []
        self.times: list[np.ndarray] = []
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
            self.times.append(np.asarray(frame.index.astype(str)))
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
        aggregate_watts = values[out_start:out_end, 0]
        appliance_watts = values[out_start:out_end, [idx + 1 for idx in self.appliance_indices]].T
        appliance_scaled = appliance_watts / cfg.scale
        on_label = (appliance_watts > cfg.on_threshold_watts).astype(np.float32)

        return {
            "x": torch.from_numpy((main / cfg.scale).astype(np.float32)).unsqueeze(0),
            "y": torch.from_numpy(appliance_scaled.astype(np.float32)),
            "y_watts": torch.from_numpy(appliance_watts.astype(np.float32)),
            "on": torch.from_numpy(on_label),
            "aggregate_watts": torch.from_numpy(aggregate_watts.astype(np.float32)),
        }


class CSVSGNWindowDataset(Dataset):
    """Windowed multi-appliance SGN dataset built from a merged feature CSV."""

    def __init__(
        self,
        csv_config: dict,
        split: str,
        appliance: str | list[str],
        config: SGNConfig,
        stride: int,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        appliances = csv_config.get("appliances", CSV_APPLIANCES)
        target_appliances = _as_appliance_list(appliance, appliances)
        unknown = sorted(set(target_appliances) - set(appliances))
        if unknown:
            raise ValueError(f"Unknown CSV appliance(s) {unknown}. Choices: {sorted(appliances)}")

        self.csv_config = csv_config
        self.split = split
        self.appliances = target_appliances
        self.config = config
        self.stride = stride

        power_columns = [appliances[name]["power"] for name in target_appliances]
        feature_columns = list(csv_config["feature_columns"])
        aggregate_column = csv_config.get("aggregate_column", "aggregate")
        time_column = csv_config.get("time_column")
        house_column = csv_config.get("house_column", "house")
        on_columns = [appliances[name].get("on") for name in target_appliances]
        on_label_source = getattr(config, "on_label_source", "csv")
        target_columns = list(power_columns)
        if on_label_source == "csv":
            target_columns.extend(col for col in on_columns if col)
        usecols = list(
            dict.fromkeys(
                feature_columns
                + [aggregate_column]
                + target_columns
                + ([time_column] if time_column else [])
                + ([house_column] if csv_config.get("val_mode") else [])
            )
        )

        csv_path = csv_path_for_split(csv_config, split)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV file for split={split}: {csv_path}")

        df = pd.read_csv(csv_path, usecols=usecols)
        df = select_csv_split_df(df, csv_config, split)
        df = df.dropna(subset=usecols)
        if len(df) < config.input_length:
            raise ValueError(
                f"Split {split} is too short for input_length={config.input_length}: {len(df)} rows"
            )

        features = df[feature_columns].to_numpy(dtype=np.float32)
        if config.scale <= 0:
            raise ValueError(f"Invalid aggregate scale: {config.scale}")
        self.features = features / np.float32(config.scale)
        self.aggregate = df[aggregate_column].to_numpy(dtype=np.float32)
        self.power = df[power_columns].to_numpy(dtype=np.float32)
        self.time = df[time_column].astype(str).to_numpy() if time_column and time_column in df else None
        self.on_label_source = on_label_source
        if on_label_source == "csv":
            if not all(col and col in df for col in on_columns):
                missing = [col for col in on_columns if not col or col not in df]
                raise ValueError(f"Missing provided on/off label column(s): {missing}")
            self.on = df[on_columns].to_numpy(dtype=np.float32)
        else:
            self.on = None

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
        aggregate_watts = self.aggregate[out_start:out_end]
        appliance_watts = self.power[out_start:out_end].T
        appliance_scaled = appliance_watts / cfg.scale
        if self.on is not None:
            on_label = self.on[out_start:out_end].T
        else:
            on_label = (appliance_watts > cfg.on_threshold_watts).astype(np.float32)

        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y": torch.from_numpy(appliance_scaled.astype(np.float32)),
            "y_watts": torch.from_numpy(appliance_watts.astype(np.float32)),
            "on": torch.from_numpy(on_label.astype(np.float32)),
            "aggregate_watts": torch.from_numpy(aggregate_watts.astype(np.float32)),
        }

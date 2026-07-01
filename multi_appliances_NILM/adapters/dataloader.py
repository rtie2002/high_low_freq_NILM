"""
Shared NILM dataloader — load pre-split CSVs, apply model-specific windows.

Experiment yaml: data paths and column mapping (you split the data yourself).
Model yaml: windowing (input/output length, alignment, stride).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SplitName = Literal["train", "validation", "test"]
OutputAlignment = Literal["end", "center"]

_SPLIT_FILE_KEYS = {
    "train": "train_file",
    "validation": "validation_file",
    "test": "test_file",
}


def _resolve_input_length(windowing: dict[str, Any]) -> int:
    seq_len = int(windowing["input_window_length"])
    if windowing.get("force_even_input_length", False) and seq_len % 2 != 0:
        seq_len += 1
    return seq_len


def _output_slice(start: int, seq_len: int, windowing: dict[str, Any]) -> slice:
    out_len = int(windowing.get("output_window_length", 1))
    alignment: OutputAlignment = windowing.get("output_alignment", "end")
    if alignment == "end":
        out_end = start + seq_len
        out_start = out_end - out_len
    elif alignment == "center":
        out_start = start + (seq_len - out_len) // 2
        out_end = out_start + out_len
    else:
        raise ValueError(f"Unsupported output_alignment: {alignment}")
    return slice(out_start, out_end)


TargetMode = Literal["output_window", "full_input"]


class WindowDataset(Dataset):
    """Sliding-window dataset; windowing rules come from model config."""

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        states: np.ndarray,
        windowing: dict[str, Any],
        *,
        stride: int,
        target_mode: TargetMode = "output_window",
    ):
        self.inputs = inputs
        self.targets = targets
        self.states = states
        self.windowing = windowing
        self.target_mode = target_mode
        self.seq_len = _resolve_input_length(windowing)
        self.stride = max(1, stride)
        self.indices = np.arange(0, len(inputs) - self.seq_len, self.stride)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        start = int(self.indices[index])
        end = start + self.seq_len

        x = torch.tensor(self.inputs[start:end], dtype=torch.float32).unsqueeze(-1)
        if self.target_mode == "full_input":
            y = torch.tensor(self.targets[start:end], dtype=torch.float32)
            z = torch.tensor(self.states[start:end], dtype=torch.long)
            return x, y, z

        out = _output_slice(start, self.seq_len, self.windowing)
        y = torch.tensor(self.targets[out], dtype=torch.float32)
        z = torch.tensor(self.states[out], dtype=torch.long)

        out_len = int(self.windowing.get("output_window_length", 1))
        if out_len == 1:
            y = y.squeeze(0)
            z = z.squeeze(0)
        return x, y, z


from adapters.config import appliance_list
from adapters.unet_preprocess import preprocess_unet_arrays


def _appliance_order(experiment_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> list[str]:
    return appliance_list(experiment_cfg, model_cfg)


def _target_mode(windowing: dict[str, Any], split: str) -> TargetMode:
    if split == "train" and windowing.get("training_targets") == "full_input":
        return "full_input"
    return "output_window"


def _csv_column_map(csv_cfg: dict[str, Any], appliances: list[str]) -> tuple[list[str], list[str]]:
    app_cfg = csv_cfg.get("appliances", {})
    power_cols, state_cols = [], []
    for app in appliances:
        if app not in app_cfg:
            raise ValueError(f"Missing csv.appliances.{app} in experiment config")
        power_cols.append(app_cfg[app]["power"])
        state_cols.append(app_cfg[app]["state"])
    return power_cols, state_cols


def resolve_mains_column(experiment_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> str:
    """Experiment default; model yaml can override (e.g. UNet denoised vs noise mains)."""
    data_cfg = model_cfg.get("data", {})
    if mains_col := data_cfg.get("mains_column"):
        return str(mains_col)

    mains_map = data_cfg.get("mains_columns", {})
    if mains_map:
        use_denoised = data_cfg.get("use_denoised_mains", True)
        key = "denoised" if use_denoised else "noise"
        if key not in mains_map:
            raise ValueError(f"data.mains_columns.{key} missing in model config")
        return str(mains_map[key])

    csv_cfg = experiment_cfg.get("csv", {})
    if mains_col := csv_cfg.get("mains_column"):
        return str(mains_col)
    raise ValueError("Set csv.mains_column in experiment.yaml or data.mains_column in model yaml")


def load_csv_arrays(
    csv_path: Path,
    csv_cfg: dict[str, Any],
    appliances: list[str],
    *,
    mains_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    power_cols, state_cols = _csv_column_map(csv_cfg, appliances)
    usecols = list(dict.fromkeys([mains_column, *power_cols, *state_cols]))
    df = pd.read_csv(csv_path, usecols=usecols).dropna(subset=usecols)

    x = df[mains_column].to_numpy(dtype=np.float32)
    y = df[power_cols].to_numpy(dtype=np.float32)
    z = df[state_cols].to_numpy(dtype=np.float32)
    return x, y, z.astype(np.int64)


class NILMDataLoader:
    """Load your pre-split CSVs once, build per-model windowed datasets."""

    def __init__(
        self,
        experiment_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        data_root: str | Path,
    ):
        self.experiment = experiment_cfg
        self.model_cfg = model_cfg
        self.data_root = Path(data_root)
        self.csv_cfg = experiment_cfg.get("csv", {})
        self.appliances = _appliance_order(experiment_cfg, model_cfg)
        self._splits: dict[SplitName, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None

    def _resolve_csv_path(self, split: SplitName) -> Path:
        key = _SPLIT_FILE_KEYS[split]
        name = self.csv_cfg.get(key)
        if not name:
            raise ValueError(f"csv.{key} required — point to your pre-split CSV")
        path = Path(name)
        return path if path.is_absolute() else self.data_root / path

    def _load_split_csv(self, split: SplitName) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mains_col = resolve_mains_column(self.experiment, self.model_cfg)
        x, y, z = load_csv_arrays(
            self._resolve_csv_path(split),
            self.csv_cfg,
            self.appliances,
            mains_column=mains_col,
        )
        data_cfg = self.model_cfg.get("data", {})
        if data_cfg.get("preprocess") == "unet_nilm" and "seq2quantile" in self.model_cfg:
            x, y, z = preprocess_unet_arrays(x, y, self.appliances, self.model_cfg)
        return x, y, z

    def get_splits(self) -> dict[SplitName, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self._splits is not None:
            return self._splits

        self._splits = {
            "train": self._load_split_csv("train"),
            "validation": self._load_split_csv("validation"),
            "test": self._load_split_csv("test"),
        }
        return self._splits

    def build_dataset(self, split: str) -> Dataset:
        key: SplitName = "validation" if split in ("val", "validation") else split  # type: ignore[assignment]
        x, y, z = self.get_splits()[key]
        w = self.model_cfg["windowing"]
        stride = int(w["input_stride"]) if split == "train" else int(w.get("eval_stride", w["input_stride"]))
        return WindowDataset(
            x,
            y,
            z,
            w,
            stride=stride,
            target_mode=_target_mode(w, split),
        )

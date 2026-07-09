"""Shared NILM data pipeline.

Flow: experiment config -> load CSV arrays -> normalize -> sliding windows -> tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from adapters.config import appliance_list

SplitName = Literal["train", "validation", "test"]
OutputAlignment = Literal["end", "center"]
TargetMode = Literal["output_window", "full_input"]

_SPLIT_FILE_KEYS = {
    "train": "train_file",
    "validation": "validation_file",
    "test": "test_file",
}


def get_state_threshold(model_cfg: dict[str, Any]) -> float | None:
    val = model_cfg.get("data", {}).get("state_threshold_watts")
    return float(val) if val is not None else None


def get_state_label_source(model_cfg: dict[str, Any]) -> str:
    """Choose where ON/OFF supervision comes from: csv, threshold, or auto."""
    source = str(model_cfg.get("data", {}).get("state_label_source", "auto")).lower()
    if source not in {"auto", "csv", "threshold"}:
        raise ValueError("data.state_label_source must be one of: auto, csv, threshold")
    return source


def get_power_scale(model_cfg: dict[str, Any]) -> float:
    """Legacy divide-by-scale fallback when experiment has no normalization block."""
    return float(model_cfg.get("data", {}).get("power_scale", 1.0))


def get_normalization_cfg(experiment_cfg: dict[str, Any]) -> dict[str, Any] | None:
    norm = experiment_cfg.get("normalization")
    return norm if isinstance(norm, dict) else None


@dataclass
class NormalizationStats:
    """Single place for z-score (or legacy scale) normalize / denormalize."""

    input_mean: float | None = None
    input_std: float | None = None
    target_mean: np.ndarray | None = None
    target_std: np.ndarray | None = None
    legacy_scale: float = 1.0

    @classmethod
    def from_config(
        cls,
        experiment_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        appliances: list[str],
    ) -> NormalizationStats:
        legacy_scale = get_power_scale(model_cfg)
        norm = get_normalization_cfg(experiment_cfg)
        if not norm:
            return cls(legacy_scale=legacy_scale)

        aggregate = norm.get("aggregate", {})
        agg_mean, agg_std = aggregate.get("mean"), aggregate.get("std")
        input_mean = float(agg_mean) if agg_mean is not None else None
        input_std = float(agg_std) if agg_std is not None else None

        app_cfg = norm.get("appliances", {})
        means, stds = [], []
        for app in appliances:
            cfg = app_cfg.get(app)
            if not cfg or "mean" not in cfg or "std" not in cfg:
                raise ValueError(f"Normalization stats missing for appliance '{app}'")
            means.append(float(cfg["mean"]))
            stds.append(float(cfg["std"]))

        return cls(
            input_mean=input_mean,
            input_std=input_std,
            target_mean=np.asarray(means, dtype=np.float32),
            target_std=np.asarray(stds, dtype=np.float32),
            legacy_scale=legacy_scale,
        )

    @property
    def loss_scale(self) -> float | np.ndarray:
        """Scale for converting normalized MAE back toward watts."""
        if self.target_std is not None:
            return self.target_std
        return self.legacy_scale

    def normalize_inputs(self, x: np.ndarray) -> np.ndarray:
        if self.input_mean is not None and self.input_std is not None:
            return (x - self.input_mean) / self.input_std
        if self.legacy_scale != 1.0:
            return x / self.legacy_scale
        return x

    def normalize_targets(self, y: np.ndarray) -> np.ndarray:
        if self.target_mean is not None and self.target_std is not None:
            return (y - self.target_mean) / self.target_std
        if self.legacy_scale != 1.0:
            return y / self.legacy_scale
        return y

    def denorm(self, y: np.ndarray) -> np.ndarray:
        if self.target_mean is not None and self.target_std is not None:
            return np.maximum((y * self.target_std) + self.target_mean, 0.0)
        if self.legacy_scale != 1.0:
            return np.maximum(y * self.legacy_scale, 0.0)
        return y


def _split_key(split: str) -> SplitName:
    if split in ("val", "validation"):
        return "validation"
    return split  # type: ignore[return-value]


def _resolve_input_length(windowing: dict[str, Any]) -> int:
    seq_len = int(windowing["input_window_length"])
    if windowing.get("force_even_input_length", False) and seq_len % 2 != 0:
        seq_len += 1
    return seq_len


def _output_row_offset(windowing: dict[str, Any], seq_len: int) -> int:
    out_len = int(windowing.get("output_window_length", 1))
    alignment: OutputAlignment = windowing.get("output_alignment", "end")
    if alignment == "center":
        return (seq_len - out_len) // 2
    if alignment == "end":
        return seq_len - out_len
    raise ValueError(f"Unsupported output_alignment: {alignment}")


def _output_slice(start: int, seq_len: int, windowing: dict[str, Any]) -> slice:
    out_len = int(windowing.get("output_window_length", 1))
    offset = _output_row_offset(windowing, seq_len)
    return slice(start + offset, start + offset + out_len)


def _count_windows(n_timesteps: int, windowing: dict[str, Any], stride: int) -> int:
    seq_len = _resolve_input_length(windowing)
    if n_timesteps < seq_len:
        return 0
    return len(np.arange(0, n_timesteps - seq_len, max(1, stride)))


def _target_mode(windowing: dict[str, Any], split: str) -> TargetMode:
    if split == "train" and windowing.get("training_targets") == "full_input":
        return "full_input"
    return "output_window"


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
        normalization: NormalizationStats | None = None,
        state_threshold_watts: float | None = None,
        state_label_source: str = "auto",
    ):
        norm = normalization or NormalizationStats()
        self.inputs = np.ascontiguousarray(norm.normalize_inputs(inputs), dtype=np.float32)
        self.targets = np.ascontiguousarray(targets, dtype=np.float32)
        self.states = np.ascontiguousarray(states, dtype=np.int64)

        use_threshold_labels = state_label_source == "threshold" or (
            state_label_source == "auto" and state_threshold_watts is not None
        )
        if use_threshold_labels:
            if state_threshold_watts is None:
                raise ValueError("state_label_source='threshold' requires data.state_threshold_watts")
            self.states = (self.targets > float(state_threshold_watts)).astype(np.int64)

        self.targets = np.ascontiguousarray(norm.normalize_targets(self.targets), dtype=np.float32)

        self.inputs_t = torch.from_numpy(self.inputs)
        self.targets_t = torch.from_numpy(self.targets)
        self.states_t = torch.from_numpy(self.states)
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
        x = self.inputs_t[start:end].unsqueeze(-1)

        if self.target_mode == "full_input":
            return x, self.targets_t[start:end], self.states_t[start:end]

        out = _output_slice(start, self.seq_len, self.windowing)
        y = self.targets_t[out]
        z = self.states_t[out]
        if int(self.windowing.get("output_window_length", 1)) == 1:
            y = y.squeeze(0)
            z = z.squeeze(0)
        return x, y, z


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
    """Model yaml can override experiment csv.mains_column."""
    if mains_col := model_cfg.get("data", {}).get("mains_column"):
        return str(mains_col)
    if mains_col := experiment_cfg.get("csv", {}).get("mains_column"):
        return str(mains_col)
    raise ValueError("Set csv.mains_column in experiment yaml or data.mains_column in model yaml")


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
    z = df[state_cols].to_numpy(dtype=np.int64)
    return x, y, z


class NILMDataLoader:
    """Load pre-split CSVs once, build per-model windowed datasets."""

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
        self.appliances = appliance_list(experiment_cfg, model_cfg)
        self.state_threshold_watts = get_state_threshold(model_cfg)
        self.state_label_source = get_state_label_source(model_cfg)
        self.norm = NormalizationStats.from_config(experiment_cfg, model_cfg, self.appliances)
        self.loss_scale = self.norm.loss_scale
        self._splits: dict[SplitName, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None

    def _resolve_csv_path(self, split: SplitName) -> Path:
        key = _SPLIT_FILE_KEYS[split]
        name = self.csv_cfg.get(key)
        if not name:
            raise ValueError(f"csv.{key} required — point to your pre-split CSV")
        path = Path(name)
        return path if path.is_absolute() else self.data_root / path

    def _load_split_csv(self, split: SplitName) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return load_csv_arrays(
            self._resolve_csv_path(split),
            self.csv_cfg,
            self.appliances,
            mains_column=resolve_mains_column(self.experiment, self.model_cfg),
        )

    def _stride_for_split(self, split: str) -> int:
        w = self.model_cfg["windowing"]
        if split == "train":
            return int(w["input_stride"])
        return int(w.get("eval_stride", w["input_stride"]))

    def _make_window_dataset(self, split: str) -> WindowDataset:
        key = _split_key(split)
        x, y, z = self.get_splits()[key]
        w = self.model_cfg["windowing"]
        return WindowDataset(
            x,
            y,
            z,
            w,
            stride=self._stride_for_split(split),
            target_mode=_target_mode(w, split),
            normalization=self.norm,
            state_threshold_watts=self.state_threshold_watts,
            state_label_source=self.state_label_source,
        )

    def window_output_timesteps(self, split: str, n_windows: int) -> np.ndarray:
        """CSV row index for each sliding-window model output."""
        ds = self._make_window_dataset(split)
        n = min(int(n_windows), len(ds))
        w = self.model_cfg["windowing"]
        out_len = int(w.get("output_window_length", 1))
        offset = _output_row_offset(w, ds.seq_len)
        if out_len == 1:
            return ds.indices[:n] + offset
        return (ds.indices[:n, None] + offset + np.arange(out_len)).reshape(-1)

    def get_raw_csv_arrays(self, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return raw CSV mains/power/state arrays without threshold relabelling."""
        return self.get_splits()[_split_key(split)]

    def window_flattened_csv_states(self, split: str, n_points: int) -> np.ndarray:
        """Align dataset CSV *_on columns with flattened model output timesteps.

        Waveform plotting uses these labels only. Training loss and F1 metrics
        may still follow data.state_label_source in the model yaml.
        """
        key = _split_key(split)
        x, y, z_csv = self.get_splits()[key]
        w = self.model_cfg["windowing"]
        ds = WindowDataset(
            x,
            y,
            z_csv,
            w,
            stride=self._stride_for_split(split),
            target_mode=_target_mode(w, split),
            normalization=self.norm,
            state_label_source="csv",
        )
        if len(ds) == 0:
            return np.zeros((0, len(self.appliances)), dtype=np.int32)

        rows: list[np.ndarray] = []
        for i in range(len(ds)):
            _, _, z = ds[i]
            z_np = z.numpy() if hasattr(z, "numpy") else np.asarray(z)
            rows.append(z_np.reshape(-1, len(self.appliances)))
        flat = np.concatenate(rows, axis=0)
        return flat[: int(n_points)].astype(np.int32)

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
        return self._make_window_dataset(split)

    def denorm_to_watts(self, y: np.ndarray) -> np.ndarray:
        """Convert normalized targets or predictions back to watts."""
        return self.norm.denorm(y)

    def describe_split(self, split: str, *, batch_size: int) -> dict[str, Any]:
        key = _split_key(split)
        csv_path = self._resolve_csv_path(key)
        x, _, _ = self.get_splits()[key]
        w = self.model_cfg["windowing"]
        stride = self._stride_for_split(split)
        target_mode = _target_mode(w, split)
        n_windows = _count_windows(len(x), w, stride)
        n_batches = (n_windows + batch_size - 1) // batch_size if n_windows else 0
        return {
            "split": split,
            "csv_path": str(csv_path),
            "csv_name": csv_path.name,
            "timesteps": len(x),
            "n_appliances": len(self.appliances),
            "input_length": _resolve_input_length(w),
            "output_length": int(w.get("output_window_length", 1)),
            "output_alignment": w.get("output_alignment", "end"),
            "stride": stride,
            "target_mode": target_mode,
            "windows": n_windows,
            "batch_size": batch_size,
            "batches": n_batches,
        }

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
        self.inputs = np.ascontiguousarray(inputs, dtype=np.float32)
        self.targets = np.ascontiguousarray(targets, dtype=np.float32)
        self.states = np.ascontiguousarray(states, dtype=np.int64)
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
            y = self.targets_t[start:end]
            z = self.states_t[start:end]
            return x, y, z

        out = _output_slice(start, self.seq_len, self.windowing)
        y = self.targets_t[out]
        z = self.states_t[out]

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


def resolve_mains_columns(
    experiment_cfg: dict[str, Any], model_cfg: dict[str, Any]
) -> tuple[str, str | None]:
    """Return (mains, sub_mains) column names for UNet-style preprocessing."""
    data_cfg = model_cfg.get("data", {})
    mains_map = data_cfg.get("mains_columns", {})
    if mains_map:
        mains_col = str(mains_map.get("mains") or mains_map.get("noise") or "aggregate")
        sub_col = mains_map.get("sub_mains") or mains_map.get("denoised")
        return mains_col, str(sub_col) if sub_col else None

    mains_col = resolve_mains_column(experiment_cfg, model_cfg)
    return mains_col, data_cfg.get("sub_mains_column")


def load_csv_arrays(
    csv_path: Path,
    csv_cfg: dict[str, Any],
    appliances: list[str],
    *,
    mains_column: str,
    sub_mains_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    power_cols, state_cols = _csv_column_map(csv_cfg, appliances)
    usecols = list(dict.fromkeys([mains_column, *power_cols, *state_cols]))
    available = set(pd.read_csv(csv_path, nrows=0).columns)
    sub_col = sub_mains_column if sub_mains_column and sub_mains_column in available else None
    if sub_col:
        usecols = list(dict.fromkeys([*usecols, sub_col]))
    df = pd.read_csv(csv_path, usecols=usecols).dropna(subset=usecols)

    x = df[mains_column].to_numpy(dtype=np.float32)
    sub = df[sub_col].to_numpy(dtype=np.float32) if sub_col else None
    y = df[power_cols].to_numpy(dtype=np.float32)
    z = df[state_cols].to_numpy(dtype=np.float32)
    return x, y, z.astype(np.int64), sub


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
        data_cfg = self.model_cfg.get("data", {})
        if data_cfg.get("preprocess") == "unet_nilm":
            mains_col, sub_col = resolve_mains_columns(self.experiment, self.model_cfg)
        else:
            mains_col = resolve_mains_column(self.experiment, self.model_cfg)
            sub_col = None

        x, y, z, sub = load_csv_arrays(
            self._resolve_csv_path(split),
            self.csv_cfg,
            self.appliances,
            mains_column=mains_col,
            sub_mains_column=sub_col,
        )
        if data_cfg.get("preprocess") == "unet_nilm" and "seq2quantile" in self.model_cfg:
            x, y, z = preprocess_unet_arrays(
                x,
                y,
                self.appliances,
                self.model_cfg,
                sub_mains_watts=sub,
            )
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
        stride = self._stride_for_split(split)
        return WindowDataset(
            x,
            y,
            z,
            w,
            stride=stride,
            target_mode=_target_mode(w, split),
        )

    def _stride_for_split(self, split: str) -> int:
        w = self.model_cfg["windowing"]
        if split == "train":
            return int(w["input_stride"])
        return int(w.get("eval_stride", w["input_stride"]))

    def describe_split(self, split: str, *, batch_size: int) -> dict[str, Any]:
        key: SplitName = "validation" if split in ("val", "validation") else split  # type: ignore[assignment]
        csv_path = self._resolve_csv_path(key)
        x, y, z = self.get_splits()[key]
        w = self.model_cfg["windowing"]
        stride = self._stride_for_split(split)
        target_mode = _target_mode(w, split)
        n_windows = len(WindowDataset(x, y, z, w, stride=stride, target_mode=target_mode))
        n_batches = (n_windows + batch_size - 1) // batch_size if n_windows else 0
        return {
            "split": split,
            "csv_path": str(csv_path),
            "csv_name": csv_path.name,
            "timesteps": len(x),
            "n_appliances": y.shape[1] if y.ndim > 1 else 1,
            "input_length": _resolve_input_length(w),
            "output_length": int(w.get("output_window_length", 1)),
            "output_alignment": w.get("output_alignment", "end"),
            "stride": stride,
            "target_mode": target_mode,
            "windows": n_windows,
            "batch_size": batch_size,
            "batches": n_batches,
        }


def _data_preprocess_note(
    model_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    data_loader: NILMDataLoader | None = None,
) -> list[str]:
    data_cfg = model_cfg.get("data", {})
    lines: list[str] = []
    if data_cfg.get("preprocess") == "unet_nilm":
        mains_col, sub_col = resolve_mains_columns(experiment_cfg, model_cfg)
        lines.append("preprocess: unet_nilm (median filter + z-score, online from CSV)")
        lines.append(f"mains column: {mains_col}")
        if sub_col and data_loader is not None:
            header = set(pd.read_csv(data_loader._resolve_csv_path("train"), nrows=0).columns)
            if sub_col not in header:
                lines.append(f"sub_mains column: {sub_col} (not in CSV — falls back to mains)")
            else:
                lines.append(f"sub_mains column: {sub_col}")
        elif sub_col:
            lines.append(f"sub_mains column: {sub_col}")
        lines.append(f"mains path: {'denoise' if data_cfg.get('use_denoised_mains') else 'noise'} (paper default: noise)")
        lines.append(f"eval denorm: {data_cfg.get('denorm_style', 'standard')}")
    elif scale := data_cfg.get("power_scale"):
        lines.append(f"preprocess: divide power/mains by {scale}")
        if thr := data_cfg.get("state_threshold_watts"):
            lines.append(f"state labels: power > {thr} W")
    else:
        lines.append("preprocess: none (use CSV values as loaded)")
    return lines


def print_training_data_summary(
    *,
    experiment_id: str,
    model_name: str,
    appliances: list[str],
    model_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    data_loader: NILMDataLoader,
    batch_size: int,
    epochs: int,
    device: str,
) -> None:
    w = model_cfg["windowing"]
    train_cfg = model_cfg.get("training", {})
    width = 78
    bar = "=" * width
    thin = "-" * width

    print(bar, flush=True)
    print(f"EXPERIMENT: {experiment_id}  |  MODEL: {model_name}  |  DEVICE: {device}", flush=True)
    print(bar, flush=True)
    print(f"Appliances ({len(appliances)}): {', '.join(appliances)}", flush=True)
    print(flush=True)
    print("Windowing", flush=True)
    print(f"  input length (effective): {_resolve_input_length(w)}", flush=True)
    print(f"  output length:            {int(w.get('output_window_length', 1))}", flush=True)
    print(f"  output alignment:         {w.get('output_alignment', 'end')}", flush=True)
    print(f"  train stride:             {int(w['input_stride'])}", flush=True)
    print(f"  eval stride:              {int(w.get('eval_stride', w['input_stride']))}", flush=True)
    print(f"  train target mode:        {_target_mode(w, 'train')}", flush=True)
    print(f"  eval target mode:         {_target_mode(w, 'validation')}", flush=True)
    print(f"  batch size:               {batch_size}", flush=True)
    print(f"  epochs:                   {epochs}", flush=True)
    if train_cfg.get("use_amp"):
        print(f"  mixed precision:          {train_cfg.get('amp_dtype', 'bf16')}", flush=True)
    print(f"  dataloader workers:       {int(train_cfg.get('num_workers', 0))}", flush=True)
    if train_cfg.get("checkpoint_monitor"):
        print(f"  checkpoint monitor:         {train_cfg['checkpoint_monitor']}", flush=True)
    print(flush=True)
    print("Data", flush=True)
    for line in _data_preprocess_note(model_cfg, experiment_cfg, data_loader):
        print(f"  {line}", flush=True)

    for split in ("train", "validation", "test"):
        info = data_loader.describe_split(split, batch_size=batch_size)
        print(flush=True)
        print(thin, flush=True)
        print(f"SPLIT: {split.upper()}", flush=True)
        print(f"  csv file:      {info['csv_path']}", flush=True)
        print(f"  timesteps:     {info['timesteps']:,}  (rows after dropna)", flush=True)
        print(f"  input length:  {info['input_length']}", flush=True)
        print(f"  output length: {info['output_length']}", flush=True)
        print(f"  stride:        {info['stride']}", flush=True)
        print(f"  target mode:   {info['target_mode']}", flush=True)
        print(f"  windows:       {info['windows']:,}", flush=True)
        print(f"  batches:       {info['batches']:,}  (@ batch_size={batch_size})", flush=True)
        if split == "train":
            print(f"  used in:       training ({info['batches']:,} batches/epoch)", flush=True)
        elif split == "validation":
            print("  used in:       validation + checkpoint selection", flush=True)
        else:
            print("  used in:       final test evaluation (after training)", flush=True)

    print(flush=True)
    print(bar, flush=True)
    print(flush=True)

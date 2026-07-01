"""Shared adapter helpers — reused by every model plug-in."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from adapters.dataloader import NILMDataLoader, _resolve_input_length
from adapters.types import PredictionBundle


def get_state_threshold(model_cfg: dict[str, Any]) -> float | None:
    """Return watts threshold for ON labels, or None to use CSV state columns."""
    val = model_cfg.get("data", {}).get("state_threshold_watts")
    return float(val) if val is not None else None


def states_from_power(y: torch.Tensor, threshold_watts: float) -> torch.Tensor:
    return (y > threshold_watts).long()


def get_power_scale(model_cfg: dict[str, Any]) -> float:
    """Return 1.0 when CSV values are already scaled (e.g. UNet)."""
    return float(model_cfg.get("data", {}).get("power_scale", 1.0))


def scale_inputs(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return x
    return x / scale


def scale_targets(y: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return y
    return y / scale


def denorm_power_array(y: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return y
    return np.maximum(y * scale, 0.0)


def center_output_slice(windowing: dict[str, Any]) -> slice:
    """Center output window inside the input window (MATNILM eval)."""
    seq_len = _resolve_input_length(windowing)
    out_len = int(windowing.get("output_window_length", 1))
    start = (seq_len - out_len) // 2
    return slice(start, start + out_len)


def build_dataloader(
    dataset: Dataset,
    train_cfg: dict[str, Any],
    *,
    shuffle: bool,
) -> DataLoader:
    """PyTorch DataLoader with sensible defaults for GPU training."""
    num_workers = int(train_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_size": int(train_cfg["batch_size"]),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
    return DataLoader(dataset, **kwargs)


class AdapterDataMixin:
    """Lazy NILMDataLoader + standard PyTorch DataLoader construction."""

    cfg: dict[str, Any]
    experiment: dict[str, Any]
    model_cfg: dict[str, Any]
    data_root: str | None
    _data: NILMDataLoader | None = None

    def _data_loader(self) -> NILMDataLoader:
        if self._data is None:
            self._data = NILMDataLoader(self.experiment, self.model_cfg, self.data_root)
        return self._data

    def build_dataset(self, split: str) -> Dataset:
        return self._data_loader().build_dataset(split)

    def build_standard_dataloader(self, split: str) -> DataLoader:
        return build_dataloader(
            self.build_dataset(split),
            self.model_cfg["training"],
            shuffle=(split == "train"),
        )


def build_prediction_bundle(
    *,
    experiment_id: str,
    model_name: str,
    split: str,
    appliances: list[str],
    sample_index: np.ndarray,
    y_true_watts: np.ndarray,
    y_pred_watts: np.ndarray,
    y_true_on: np.ndarray,
    y_pred_on: np.ndarray,
) -> PredictionBundle:
    """Standard PredictionBundle constructor used by all adapters."""
    return PredictionBundle(
        experiment_id=experiment_id,
        model_name=model_name,
        split=split,
        appliances=appliances,
        sample_index=sample_index,
        y_true_watts=y_true_watts,
        y_pred_watts=y_pred_watts,
        y_true_on=y_true_on.astype(np.int32),
        y_pred_on=y_pred_on.astype(np.int32),
    )

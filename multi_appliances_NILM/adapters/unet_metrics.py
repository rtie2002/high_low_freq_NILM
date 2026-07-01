"""UNet-NILM validation metrics (author repo metrics.py)."""

from __future__ import annotations

import numpy as np


def _tp_fp_fn(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.sum(y_true * y_pred, axis=0).astype(np.float64)
    fp = np.sum((1 - y_true) * y_pred, axis=0).astype(np.float64)
    fn = np.sum(y_true * (1 - y_pred), axis=0).astype(np.float64)
    return tp, fp, fn


def _f1_from_stats(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, *, average: str) -> float:
    if average == "micro":
        return float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = (2 * tp) / (2 * tp + fp + fn)
    scores = scores[np.isfinite(scores)]
    return float(scores.mean()) if scores.size else 0.0


def compute_unet_state_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Macro / micro F1 over appliances (author val_F1 = maF1)."""
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp, fp, fn = _tp_fp_fn(y_true, y_pred)
    return {
        "val_f1": _f1_from_stats(tp, fp, fn, average="macro"),
        "val_maf1": _f1_from_stats(tp, fp, fn, average="macro"),
        "val_mif1": _f1_from_stats(tp, fp, fn, average="micro"),
    }

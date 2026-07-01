"""Shared NILM metrics for cross-model comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from adapters.types import PredictionBundle


def _tp_fp_fn(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.sum(y_true * y_pred, axis=0).astype(np.float64)
    fp = np.sum((1 - y_true) * y_pred, axis=0).astype(np.float64)
    fn = np.sum(y_true * (1 - y_pred), axis=0).astype(np.float64)
    return tp, fp, fn


def _micro_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    return float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1e-12))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred), axis=0)


def sae(y_true: np.ndarray, y_pred: np.ndarray, period: int = 1200) -> np.ndarray:
    n = len(y_true)
    n_periods = n // period
    if n_periods == 0:
        return np.full(y_true.shape[1], np.nan)
    out = np.zeros(y_true.shape[1], dtype=np.float64)
    for j in range(y_true.shape[1]):
        errors = []
        for k in range(n_periods):
            s, e = k * period, (k + 1) * period
            errors.append(abs(y_true[s:e, j].sum() - y_pred[s:e, j].sum()))
        out[j] = np.mean(errors) / period
    return out


def per_appliance_f1(y_true_on: np.ndarray, y_pred_on: np.ndarray) -> np.ndarray:
    """Binary F1 per appliance (MATNILM / sklearn style)."""
    scores = np.zeros(y_true_on.shape[1], dtype=np.float64)
    for j in range(y_true_on.shape[1]):
        yt = y_true_on[:, j].astype(bool)
        yp = y_pred_on[:, j].astype(bool)
        tp = np.logical_and(yt, yp).sum()
        fp = np.logical_and(~yt, yp).sum()
        fn = np.logical_and(yt, ~yp).sum()
        scores[j] = 2 * tp / max(2 * tp + fp + fn, 1)
    return scores


def _on_off_labels(
    bundle: PredictionBundle,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    on_threshold_watts: float,
) -> tuple[np.ndarray, np.ndarray]:
    if bundle.y_true_on is not None:
        z_true = bundle.y_true_on.astype(np.int32)
    else:
        z_true = (y_true > on_threshold_watts).astype(np.int32)

    if bundle.y_pred_on is not None:
        z_pred = bundle.y_pred_on.astype(np.int32)
    else:
        z_pred = (y_pred > on_threshold_watts).astype(np.int32)
    return z_true, z_pred


def evaluate_bundle(
    bundle: PredictionBundle,
    *,
    sae_period: int = 1200,
    on_threshold_watts: float = 15.0,
) -> pd.DataFrame:
    """Per-appliance MAE/SAE/F1 plus one overall summary row."""
    y_true = bundle.y_true_watts
    y_pred = np.maximum(bundle.y_pred_watts, 0.0)
    z_true, z_pred = _on_off_labels(bundle, y_true, y_pred, on_threshold_watts)

    mae_vals = mae(y_true, y_pred)
    sae_vals = sae(y_true, y_pred, sae_period)
    f1_vals = per_appliance_f1(z_true, z_pred)
    tp, fp, fn = _tp_fp_fn(z_true, z_pred)

    base = {
        "experiment_id": bundle.experiment_id,
        "model": bundle.model_name,
        "split": bundle.split,
    }
    rows = []
    for i, app in enumerate(bundle.appliances):
        rows.append({
            **base,
            "appliance": app,
            "mae": float(mae_vals[i]),
            "sae": float(sae_vals[i]),
            "f1": float(f1_vals[i]),
            "micro_f1": np.nan,
        })

    rows.append({
        **base,
        "appliance": "overall",
        "mae": float(np.mean(mae_vals)),
        "sae": float(np.mean(sae_vals)),
        "f1": float(np.mean(f1_vals)),
        "micro_f1": _micro_f1(tp, fp, fn),
    })
    return pd.DataFrame(rows)


def split_per_appliance_and_overall(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_app = metrics[metrics["appliance"] != "overall"].copy()
    overall = metrics[metrics["appliance"] == "overall"].copy()
    return per_app, overall

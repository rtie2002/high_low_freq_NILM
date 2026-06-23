from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-12


def _as_1d_float(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def mae(y_true, y_pred) -> float:
    y_true = _as_1d_float(y_true)
    y_pred = _as_1d_float(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def sae(y_true, y_pred, period: int = 1200) -> float:
    y_true = _as_1d_float(y_true)
    y_pred = _as_1d_float(y_pred)
    n_periods = len(y_true) // period
    if n_periods == 0:
        return float("nan")
    total_error = 0.0
    for idx in range(n_periods):
        start = idx * period
        end = start + period
        total_error += abs(float(np.sum(y_true[start:end]) - np.sum(y_pred[start:end])))
    return float(total_error / (period * n_periods))


def binary_scores(y_true_on, y_pred_on) -> dict[str, float]:
    y_true_raw = np.asarray(y_true_on).reshape(-1)
    y_pred_raw = np.asarray(y_pred_on).reshape(-1)
    y_true_on = y_true_raw >= 0.5 if np.issubdtype(y_true_raw.dtype, np.number) else y_true_raw.astype(bool)
    y_pred_on = y_pred_raw >= 0.5 if np.issubdtype(y_pred_raw.dtype, np.number) else y_pred_raw.astype(bool)

    tp = float(np.logical_and(y_true_on, y_pred_on).sum())
    fp = float(np.logical_and(~y_true_on, y_pred_on).sum())
    fn = float(np.logical_and(y_true_on, ~y_pred_on).sum())

    precision = tp / max(tp + fp, EPSILON)
    recall = tp / max(tp + fn, EPSILON)
    f1 = (2.0 * precision * recall) / max(precision + recall, EPSILON)
    return {
        "f1": float(f1),
    }


def compute_regression_metrics(y_true, y_pred, sae_period: int = 1200) -> dict[str, float]:
    y_true = _as_1d_float(y_true)
    y_pred = np.maximum(_as_1d_float(y_pred), 0.0)
    return {
        "mae": mae(y_true, y_pred),
        "sae": sae(y_true, y_pred, sae_period),
    }


def compute_nilm_metrics(
    y_true_watts,
    y_pred_watts,
    y_true_on=None,
    y_pred_on=None,
    *,
    on_threshold_watts: float = 15.0,
    sae_period: int = 1200,
) -> dict[str, float]:
    y_true_watts = _as_1d_float(y_true_watts)
    y_pred_watts = np.maximum(_as_1d_float(y_pred_watts), 0.0)

    metrics = compute_regression_metrics(y_true_watts, y_pred_watts, sae_period)
    if y_true_on is None:
        y_true_on = y_true_watts > on_threshold_watts
    if y_pred_on is None:
        y_pred_on = y_pred_watts > on_threshold_watts
    metrics.update(binary_scores(y_true_on, y_pred_on))
    return metrics


def compute_metrics_table(
    frame: pd.DataFrame,
    true_pred_pairs: dict[str, tuple[str, str]],
    *,
    true_on_cols: dict[str, str] | None = None,
    pred_on_cols: dict[str, str] | None = None,
    on_threshold_watts: float = 15.0,
    sae_period: int = 1200,
) -> pd.DataFrame:
    rows = []
    true_on_cols = true_on_cols or {}
    pred_on_cols = pred_on_cols or {}
    for appliance, (true_col, pred_col) in true_pred_pairs.items():
        true_on = frame[true_on_cols[appliance]] if appliance in true_on_cols else None
        pred_on = frame[pred_on_cols[appliance]] if appliance in pred_on_cols else None
        metrics = compute_nilm_metrics(
            frame[true_col],
            frame[pred_col],
            true_on,
            pred_on,
            on_threshold_watts=on_threshold_watts,
            sae_period=sae_period,
        )
        rows.append({"appliance": appliance, **metrics})
    return pd.DataFrame(rows)

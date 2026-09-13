"""Shared NILM metrics for cross-model comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from data.common import PredictionBundle


def _macro_f1_from_states(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro mean F1 over appliances (one batch or full epoch)."""
    f1_vals = per_appliance_f1(y_true.astype(np.int32), y_pred.astype(np.int32))
    return float(np.mean(f1_vals)) if len(f1_vals) else 0.0


def _macro_mae_norm(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro mean normalized MAE over appliances."""
    per_app = [
        float(np.mean(np.abs(y_pred[:, app_i] - y_true[:, app_i])))
        for app_i in range(y_true.shape[1])
    ]
    return float(np.mean(per_app)) if per_app else float("inf")


def _resolve_state_source(
    state_label_source: str,
    on_threshold_watts: float | np.ndarray | None,
) -> tuple[str, np.ndarray | None]:
    source = str(state_label_source).lower()
    if source not in {"auto", "csv", "threshold"}:
        raise ValueError("state_label_source must be one of: auto, csv, threshold")
    threshold = None if on_threshold_watts is None else np.asarray(on_threshold_watts, dtype=np.float32)
    if source == "threshold" and threshold is None:
        raise ValueError(
            "threshold state_label_source requires experiment evaluation.on_thresholds_watts"
        )
    return source, threshold


def _tp_fp_fn(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.sum(y_true * y_pred, axis=0).astype(np.float64)
    fp = np.sum((1 - y_true) * y_pred, axis=0).astype(np.float64)
    fn = np.sum(y_true * (1 - y_pred), axis=0).astype(np.float64)
    return tp, fp, fn


def _micro_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    return float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1e-12))


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise ratio; undefined zero-denominator entries remain NaN."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    return np.divide(numerator, denominator, out=out, where=denominator > 1e-12)


def _masked_mae(error: np.ndarray, mask: np.ndarray) -> np.ndarray:
    counts = mask.sum(axis=0, dtype=np.float64)
    return _safe_ratio((error * mask).sum(axis=0, dtype=np.float64), counts)


def _mean_finite(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _event_intervals(mask: np.ndarray, csv_timesteps: np.ndarray | None) -> list[tuple[int, int]]:
    """Return inclusive ON intervals, breaking events at discontinuous CSV rows."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return []
    if csv_timesteps is None:
        break_before = np.r_[True, np.zeros(mask.size - 1, dtype=bool)]
    else:
        timeline = np.asarray(csv_timesteps).reshape(-1)
        if timeline.size != mask.size:
            raise ValueError("csv_timesteps must have the same length as ON/OFF states")
        break_before = np.r_[True, np.diff(timeline) != 1]

    starts = np.flatnonzero(mask & (break_before | np.r_[True, ~mask[:-1]]))
    break_after = np.r_[break_before[1:], True]
    ends = np.flatnonzero(mask & (break_after | np.r_[~mask[1:], True]))
    return list(zip(starts.tolist(), ends.tolist()))


def _match_events(
    true_events: list[tuple[int, int]],
    pred_events: list[tuple[int, int]],
) -> int:
    """One-to-one event matches using any temporal overlap."""
    true_i = pred_i = matched = 0
    while true_i < len(true_events) and pred_i < len(pred_events):
        true_start, true_end = true_events[true_i]
        pred_start, pred_end = pred_events[pred_i]
        if true_end < pred_start:
            true_i += 1
        elif pred_end < true_start:
            pred_i += 1
        else:
            matched += 1
            true_i += 1
            pred_i += 1
    return matched


def event_detection_metrics(
    y_true_on: np.ndarray,
    y_pred_on: np.ndarray,
    csv_timesteps: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-appliance event metrics with one-to-one temporal-overlap matching."""
    y_true_on = np.asarray(y_true_on)
    y_pred_on = np.asarray(y_pred_on)
    if y_true_on.shape != y_pred_on.shape or y_true_on.ndim != 2:
        raise ValueError("event states must have matching (samples, appliances) shapes")

    true_counts = np.zeros(y_true_on.shape[1], dtype=np.int64)
    pred_counts = np.zeros_like(true_counts)
    matched_counts = np.zeros_like(true_counts)
    for app_i in range(y_true_on.shape[1]):
        true_events = _event_intervals(y_true_on[:, app_i], csv_timesteps)
        pred_events = _event_intervals(y_pred_on[:, app_i], csv_timesteps)
        true_counts[app_i] = len(true_events)
        pred_counts[app_i] = len(pred_events)
        matched_counts[app_i] = _match_events(true_events, pred_events)

    precision = _safe_ratio(matched_counts, pred_counts)
    recall = _safe_ratio(matched_counts, true_counts)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "missed_event_rate": 1.0 - recall,
        "true_events": true_counts,
        "pred_events": pred_counts,
        "matched_events": matched_counts,
    }


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


def per_appliance_average_precision(
    y_true_on: np.ndarray,
    y_score: np.ndarray,
) -> np.ndarray:
    """Threshold-free area under each appliance precision-recall curve.

    Average precision is computed at every distinct score threshold. Grouping
    equal scores makes the result independent of their input order and matches
    the standard non-interpolated AP definition:

        AP = sum_n (recall_n - recall_{n-1}) * precision_n

    A column with no positive labels receives AP=0 because it provides no
    evidence that the model can rank ON samples for that appliance.
    """
    y_true_on = np.asarray(y_true_on)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true_on.shape != y_score.shape or y_true_on.ndim != 2:
        raise ValueError("AP inputs must have matching (samples, appliances) shapes")

    scores = np.zeros(y_true_on.shape[1], dtype=np.float64)
    for app_i in range(y_true_on.shape[1]):
        target = y_true_on[:, app_i].astype(bool)
        positives = int(target.sum())
        if positives == 0:
            continue

        probability = y_score[:, app_i]
        if not np.isfinite(probability).all():
            raise ValueError("AP state scores must be finite")

        order = np.argsort(-probability, kind="mergesort")
        target = target[order]
        probability = probability[order]
        true_positive = np.cumsum(target, dtype=np.float64)
        false_positive = np.cumsum(~target, dtype=np.float64)

        # Evaluate precision/recall after the last sample at each tied score.
        threshold_ends = np.r_[
            np.flatnonzero(np.diff(probability) != 0),
            len(probability) - 1,
        ]
        precision = true_positive[threshold_ends] / (
            true_positive[threshold_ends] + false_positive[threshold_ends]
        )
        recall = true_positive[threshold_ends] / positives
        scores[app_i] = np.sum(np.diff(np.r_[0.0, recall]) * precision)
    return scores


def _on_off_labels(
    bundle: PredictionBundle,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    on_threshold_watts: float | np.ndarray | None,
    state_label_source: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    source, threshold = _resolve_state_source(state_label_source, on_threshold_watts)
    if source == "threshold":
        # True ON/OFF from per-appliance experiment thresholds.
        # Predicted ON/OFF from the model state head.
        z_true = (y_true > threshold).astype(np.int32)
        if bundle.y_pred_on is None:
            raise ValueError("threshold evaluation requires predicted ON/OFF states in the bundle")
        z_pred = bundle.y_pred_on.astype(np.int32)
        return z_true, z_pred

    if bundle.y_true_on is None:
        raise ValueError("CSV evaluation requires true ON/OFF labels in the bundle")
    z_true = bundle.y_true_on.astype(np.int32)

    if bundle.y_pred_on is None:
        raise ValueError("CSV evaluation requires predicted ON/OFF labels in the bundle")
    z_pred = bundle.y_pred_on.astype(np.int32)
    return z_true, z_pred


def evaluate_bundle(
    bundle: PredictionBundle,
    *,
    sae_period: int = 1200,
    on_threshold_watts: float | np.ndarray | None = None,
    state_label_source: str = "auto",
    power_postprocess: PowerPostprocessConfig | None = None,
) -> pd.DataFrame:
    """Per-appliance power, state, energy, and event diagnostics."""
    y_true, y_pred = apply_power_postprocess_pair(
        bundle.y_true_watts,
        bundle.y_pred_watts,
        power_postprocess,
    )
    z_true, z_pred = _on_off_labels(bundle, y_true, y_pred, on_threshold_watts, state_label_source)

    mae_vals = mae(y_true, y_pred)
    sae_vals = sae(y_true, y_pred, sae_period)
    f1_vals = per_appliance_f1(z_true, z_pred)
    if bundle.y_pred_state_prob is None:
        average_precision_vals = np.full(y_true.shape[1], np.nan)
    else:
        average_precision_vals = per_appliance_average_precision(
            z_true,
            bundle.y_pred_state_prob,
        )
    tp, fp, fn = _tp_fp_fn(z_true, z_pred)
    precision_vals = _safe_ratio(tp, tp + fp)
    recall_vals = _safe_ratio(tp, tp + fn)
    tn = np.sum((1 - z_true) * (1 - z_pred), axis=0).astype(np.float64)
    specificity_vals = _safe_ratio(tn, tn + fp)
    balanced_accuracy_vals = 0.5 * (recall_vals + specificity_vals)

    abs_error = np.abs(y_pred - y_true)
    true_on = z_true.astype(bool)
    on_mae_vals = _masked_mae(abs_error, true_on)
    off_mae_vals = _masked_mae(abs_error, ~true_on)
    energy_ratio_vals = _safe_ratio(y_pred.sum(axis=0), y_true.sum(axis=0))
    on_energy_ratio_vals = _safe_ratio(
        (y_pred * true_on).sum(axis=0),
        (y_true * true_on).sum(axis=0),
    )

    if len(y_true) > 1:
        contiguous = np.ones(len(y_true) - 1, dtype=bool)
        if bundle.csv_timesteps is not None:
            timeline = np.asarray(bundle.csv_timesteps).reshape(-1)
            if len(timeline) != len(y_true):
                raise ValueError("bundle.csv_timesteps length must match predictions")
            contiguous = np.diff(timeline) == 1
        delta_error = np.abs(np.diff(y_pred, axis=0) - np.diff(y_true, axis=0))
        delta_on = np.maximum(z_true[1:], z_true[:-1]).astype(bool)
        delta_mae_vals = _masked_mae(delta_error, delta_on & contiguous[:, None])
    else:
        delta_mae_vals = np.full(y_true.shape[1], np.nan)

    event = event_detection_metrics(z_true, z_pred, bundle.csv_timesteps)

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
            # Per-appliance binary F1 (ON class).
            "f1": float(f1_vals[i]),
            "macro_f1": float(f1_vals[i]),
            "micro_f1": np.nan,
            "precision": float(precision_vals[i]),
            "recall": float(recall_vals[i]),
            "balanced_accuracy": float(balanced_accuracy_vals[i]),
            "average_precision": float(average_precision_vals[i]),
            "on_mae": float(on_mae_vals[i]),
            "off_mae": float(off_mae_vals[i]),
            "delta_mae": float(delta_mae_vals[i]),
            "energy_ratio": float(energy_ratio_vals[i]),
            "on_energy_ratio": float(on_energy_ratio_vals[i]),
            "event_precision": float(event["event_precision"][i]),
            "event_recall": float(event["event_recall"][i]),
            "event_f1": float(event["event_f1"][i]),
            "missed_event_rate": float(event["missed_event_rate"][i]),
            "true_events": int(event["true_events"][i]),
            "pred_events": int(event["pred_events"][i]),
        })

    macro = float(np.mean(f1_vals)) if len(f1_vals) else 0.0
    micro = _micro_f1(tp, fp, fn)
    true_events = int(event["true_events"].sum())
    pred_events = int(event["pred_events"].sum())
    rows.append({
        **base,
        "appliance": "overall",
        "mae": float(np.mean(mae_vals)),
        "sae": float(np.mean(sae_vals)),
        # overall.f1 kept as macro for backward compatibility with older tables/plots.
        "f1": macro,
        "macro_f1": macro,
        "micro_f1": micro,
        "precision": _mean_finite(precision_vals),
        "recall": _mean_finite(recall_vals),
        "balanced_accuracy": _mean_finite(balanced_accuracy_vals),
        "average_precision": _mean_finite(average_precision_vals),
        "on_mae": _mean_finite(on_mae_vals),
        "off_mae": _mean_finite(off_mae_vals),
        "delta_mae": _mean_finite(delta_mae_vals),
        "energy_ratio": _mean_finite(energy_ratio_vals),
        "on_energy_ratio": _mean_finite(on_energy_ratio_vals),
        "event_precision": _mean_finite(event["event_precision"]),
        "event_recall": _mean_finite(event["event_recall"]),
        "event_f1": _mean_finite(event["event_f1"]),
        "missed_event_rate": _mean_finite(event["missed_event_rate"]),
        "true_events": true_events,
        "pred_events": pred_events,
    })
    return pd.DataFrame(rows)


def split_per_appliance_and_overall(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_app = metrics[metrics["appliance"] != "overall"].copy()
    overall = metrics[metrics["appliance"] == "overall"].copy()
    return per_app, overall


@dataclass(frozen=True)
class PowerPostprocessConfig:
    enabled: bool
    min_power_watts: float
    max_on_power_watts: np.ndarray

    def apply(self, power_watts: np.ndarray) -> np.ndarray:
        out = np.asarray(power_watts, dtype=np.float64).copy()
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        if out.shape[1] != len(self.max_on_power_watts):
            raise ValueError(
                f"Expected {len(self.max_on_power_watts)} appliance columns; got {out.shape[1]}"
            )
        out[out < self.min_power_watts] = 0.0
        for app_i, cap in enumerate(self.max_on_power_watts):
            out[:, app_i] = np.clip(out[:, app_i], 0.0, float(cap))
        return out


def resolve_power_postprocess(experiment_cfg, appliances, model_cfg=None) -> PowerPostprocessConfig | None:
    if model_cfg:
        if model_cfg.get("evaluation", {}).get("power_postprocess") is False:
            return None
    eval_cfg = experiment_cfg.get("evaluation", {})
    pp_cfg = eval_cfg.get("power_postprocess", {})
    if not bool(pp_cfg.get("enabled", False)):
        return None
    max_map = eval_cfg.get("max_on_power_watts", {})
    missing = [app for app in appliances if app not in max_map]
    if missing:
        raise ValueError(
            "evaluation.power_postprocess.enabled requires "
            f"evaluation.max_on_power_watts for: {missing}"
        )
    return PowerPostprocessConfig(
        enabled=True,
        min_power_watts=float(pp_cfg.get("min_power_watts", 5)),
        max_on_power_watts=np.asarray([float(max_map[app]) for app in appliances], dtype=np.float64),
    )


def apply_power_postprocess_pair(y_true_watts, y_pred_watts, config: PowerPostprocessConfig | None):
    y_true = np.maximum(np.asarray(y_true_watts, dtype=np.float64), 0.0)
    y_pred = np.maximum(np.asarray(y_pred_watts, dtype=np.float64), 0.0)
    if config is None or not config.enabled:
        return y_true, y_pred
    return config.apply(y_true), config.apply(y_pred)

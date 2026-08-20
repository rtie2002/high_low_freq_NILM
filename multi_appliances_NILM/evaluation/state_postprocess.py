"""State-probability calibration and temporal cleanup for NILM outputs."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from adapters.common import PredictionBundle


CALIBRATION_FILENAME = "state_calibration.json"


def _events(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    diff = np.diff(np.r_[False, m, False].astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _cleanup_mask(mask: np.ndarray, *, min_on: int = 1, merge_gap: int = 0) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()

    gap = int(merge_gap)
    if gap > 0:
        ev = _events(out)
        for (_, end), (next_start, _) in zip(ev, ev[1:]):
            if next_start - end <= gap:
                out[end:next_start] = True

    min_len = int(min_on)
    if min_len > 1:
        for start, end in _events(out):
            if end - start < min_len:
                out[start:end] = False

    return out.astype(np.int32)


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    tp = np.logical_and(yt, yp).sum()
    fp = np.logical_and(~yt, yp).sum()
    fn = np.logical_and(yt, ~yp).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def _as_app_values(
    raw: Any,
    appliances: list[str],
    default: int | float,
    cast,
) -> list[Any]:
    if raw is None:
        return [cast(default) for _ in appliances]
    if isinstance(raw, dict):
        return [cast(raw.get(app, default)) for app in appliances]
    if isinstance(raw, (list, tuple)):
        if len(raw) != len(appliances):
            raise ValueError(
                f"Expected {len(appliances)} values for appliances {appliances}, got {len(raw)}"
            )
        return [cast(v) for v in raw]
    return [cast(raw) for _ in appliances]


def _threshold_grid(cfg: dict[str, Any]) -> np.ndarray:
    raw = cfg.get("threshold_grid", [0.05, 0.98, 0.01])
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        start, stop, step = [float(v) for v in raw]
        if step <= 0:
            raise ValueError("state_calibration.threshold_grid step must be > 0")
        return np.arange(start, stop + 0.5 * step, step, dtype=np.float64)
    vals = np.asarray(raw, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        raise ValueError("state_calibration.threshold_grid must not be empty")
    return vals


def state_calibration_enabled(model_cfg: dict[str, Any]) -> bool:
    cfg = model_cfg.get("evaluation", {}).get("state_calibration", {})
    return bool(cfg.get("enabled", False))


def calibration_split(model_cfg: dict[str, Any]) -> str:
    cfg = model_cfg.get("evaluation", {}).get("state_calibration", {})
    return str(cfg.get("split", "validation")).lower()


def calibration_path(run_dir: Path) -> Path:
    return Path(run_dir) / CALIBRATION_FILENAME


def calibrate_state_postprocess(
    bundle: PredictionBundle,
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Choose per-appliance probability thresholds using the calibration split."""
    cfg = model_cfg.get("evaluation", {}).get("state_calibration", {})
    if bundle.y_true_on is None:
        raise ValueError("state calibration requires y_true_on")
    if bundle.y_pred_state_prob is None:
        raise ValueError("state calibration requires y_pred_state_prob")

    appliances = list(bundle.appliances)
    pp_cfg = cfg.get("postprocess", {}) or {}
    post_enabled = bool(pp_cfg.get("enabled", True))
    min_on = _as_app_values(pp_cfg.get("min_on_samples"), appliances, 1, int)
    merge_gap = _as_app_values(pp_cfg.get("merge_gap_samples"), appliances, 0, int)

    configured_thresholds = cfg.get("thresholds", "auto")
    if isinstance(configured_thresholds, dict):
        thresholds = [float(configured_thresholds[app]) for app in appliances]
        threshold_source = "configured"
    elif isinstance(configured_thresholds, (list, tuple)):
        thresholds = [float(v) for v in configured_thresholds]
        if len(thresholds) != len(appliances):
            raise ValueError("state_calibration.thresholds length must match appliances")
        threshold_source = "configured"
    else:
        thresholds = []
        grid = _threshold_grid(cfg)
        for app_i in range(len(appliances)):
            y_true = bundle.y_true_on[:, app_i].astype(np.int32)
            prob = bundle.y_pred_state_prob[:, app_i].astype(np.float64)
            best_threshold = 0.5
            best_f1 = -1.0
            for threshold in grid:
                pred = prob >= float(threshold)
                if post_enabled:
                    pred = _cleanup_mask(
                        pred,
                        min_on=min_on[app_i],
                        merge_gap=merge_gap[app_i],
                    )
                score = _f1(y_true, pred)
                if score > best_f1:
                    best_f1 = score
                    best_threshold = float(threshold)
            thresholds.append(best_threshold)
        threshold_source = "validation_sweep"

    calibrated = apply_state_postprocess_arrays(
        bundle.y_pred_state_prob,
        thresholds=thresholds,
        min_on_samples=min_on if post_enabled else [1] * len(appliances),
        merge_gap_samples=merge_gap if post_enabled else [0] * len(appliances),
    )
    scores = [
        _f1(bundle.y_true_on[:, app_i], calibrated[:, app_i])
        for app_i in range(len(appliances))
    ]

    return {
        "source_split": bundle.split,
        "threshold_source": threshold_source,
        "appliances": appliances,
        "thresholds": {app: float(th) for app, th in zip(appliances, thresholds)},
        "postprocess": {
            "enabled": post_enabled,
            "min_on_samples": {app: int(v) for app, v in zip(appliances, min_on)},
            "merge_gap_samples": {app: int(v) for app, v in zip(appliances, merge_gap)},
        },
        "validation_f1": {app: float(v) for app, v in zip(appliances, scores)},
        "validation_macro_f1": float(np.mean(scores)) if scores else 0.0,
    }


def save_calibration(path: Path, calibration: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")


def load_calibration(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def apply_state_postprocess_arrays(
    state_prob: np.ndarray,
    *,
    thresholds: list[float],
    min_on_samples: list[int],
    merge_gap_samples: list[int],
) -> np.ndarray:
    prob = np.asarray(state_prob, dtype=np.float64)
    if prob.ndim != 2:
        raise ValueError(f"state probabilities must be (N, A), got {prob.shape}")
    if prob.shape[1] != len(thresholds):
        raise ValueError("threshold count must match probability appliance columns")

    out = np.zeros(prob.shape, dtype=np.int32)
    for app_i, threshold in enumerate(thresholds):
        raw = prob[:, app_i] >= float(threshold)
        out[:, app_i] = _cleanup_mask(
            raw,
            min_on=int(min_on_samples[app_i]),
            merge_gap=int(merge_gap_samples[app_i]),
        )
    return out


def apply_state_calibration(
    bundle: PredictionBundle,
    calibration: dict[str, Any],
    *,
    apply_to_power: bool = True,
) -> PredictionBundle:
    if bundle.y_pred_state_prob is None:
        return bundle

    appliances = list(bundle.appliances)
    calib_apps = list(calibration.get("appliances", []))
    if calib_apps and calib_apps != appliances:
        raise ValueError(
            f"Calibration appliance order {calib_apps} does not match bundle {appliances}"
        )

    thresholds_map = calibration.get("thresholds", {})
    pp = calibration.get("postprocess", {}) or {}
    min_on_map = pp.get("min_on_samples", {})
    merge_gap_map = pp.get("merge_gap_samples", {})
    thresholds = [float(thresholds_map.get(app, 0.5)) for app in appliances]
    min_on = [int(min_on_map.get(app, 1)) for app in appliances]
    merge_gap = [int(merge_gap_map.get(app, 0)) for app in appliances]

    y_pred_on = apply_state_postprocess_arrays(
        bundle.y_pred_state_prob,
        thresholds=thresholds,
        min_on_samples=min_on,
        merge_gap_samples=merge_gap,
    )
    y_pred_watts = np.asarray(bundle.y_pred_watts, dtype=np.float64)
    if apply_to_power:
        y_pred_watts = y_pred_watts * y_pred_on.astype(np.float64)

    return replace(bundle, y_pred_on=y_pred_on, y_pred_watts=y_pred_watts)


def _apply_existing_state_gate(bundle: PredictionBundle) -> PredictionBundle:
    if bundle.y_pred_on is None:
        return bundle
    y_pred_watts = np.asarray(bundle.y_pred_watts, dtype=np.float64)
    y_pred_watts = y_pred_watts * bundle.y_pred_on.astype(np.float64)
    return replace(bundle, y_pred_watts=y_pred_watts)


def maybe_calibrate_and_apply(
    bundle: PredictionBundle,
    model_cfg: dict[str, Any],
    run_dir: Path,
    split: str,
) -> tuple[PredictionBundle, dict[str, Any] | None]:
    """Calibrate on validation, apply saved calibration on all enabled splits."""
    if not state_calibration_enabled(model_cfg):
        return bundle, None
    if bundle.y_pred_state_prob is None:
        return _apply_existing_state_gate(bundle), None

    cfg = model_cfg.get("evaluation", {}).get("state_calibration", {})
    apply_to_power = bool(cfg.get("apply_to_power", True))
    split_key = str(split).lower()
    cal_split = calibration_split(model_cfg)
    path = calibration_path(run_dir)

    calibration = None
    if split_key == cal_split:
        calibration = calibrate_state_postprocess(bundle, model_cfg)
        save_calibration(path, calibration)
    else:
        calibration = load_calibration(path)

    if calibration is None:
        if apply_to_power:
            return _apply_existing_state_gate(bundle), None
        return bundle, None
    return apply_state_calibration(bundle, calibration, apply_to_power=apply_to_power), calibration

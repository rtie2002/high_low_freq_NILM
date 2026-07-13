"""Watt-space power post-processing for evaluation (transfer-learning baseline parity).

Baseline (transfer_learning_multi-appliance/test/test_model.py + utils.tensor_cutoff_energy):

    1. Values below min_power_watts (default 5 W) are set to 0.
    2. Values are clipped to [0, max_on_power_watts] per appliance.

Applied to both ground truth and predictions before MAE/SAE reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PowerPostprocessConfig:
    enabled: bool
    min_power_watts: float
    max_on_power_watts: np.ndarray

    def apply(self, power_watts: np.ndarray) -> np.ndarray:
        """Return a copy with baseline-style cutoff applied."""
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


def resolve_power_postprocess(
    experiment_cfg: dict[str, Any],
    appliances: list[str],
    model_cfg: dict[str, Any] | None = None,
) -> PowerPostprocessConfig | None:
    """Read post-process settings from experiment yaml (optional model override).

    Model yaml may disable post-processing even when the experiment enables it:

        evaluation:
          power_postprocess: false
    """
    if model_cfg:
        model_eval = model_cfg.get("evaluation", {})
        if model_eval.get("power_postprocess") is False:
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


def apply_power_postprocess_pair(
    y_true_watts: np.ndarray,
    y_pred_watts: np.ndarray,
    config: PowerPostprocessConfig | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same watt-space cleanup to labels and predictions."""
    y_true = np.maximum(np.asarray(y_true_watts, dtype=np.float64), 0.0)
    y_pred = np.maximum(np.asarray(y_pred_watts, dtype=np.float64), 0.0)
    if config is None or not config.enabled:
        return y_true, y_pred
    return config.apply(y_true), config.apply(y_pred)

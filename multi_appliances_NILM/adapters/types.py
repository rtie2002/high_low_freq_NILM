"""Shared data shapes for train/eval and cross-model comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PredictionBundle:
    """Standard test output — every model saves predictions in this format."""

    experiment_id: str
    model_name: str
    split: str
    appliances: list[str]
    sample_index: np.ndarray
    y_true_watts: np.ndarray
    y_pred_watts: np.ndarray
    y_true_on: np.ndarray | None = None
    y_pred_on: np.ndarray | None = None
    csv_timesteps: np.ndarray | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            experiment_id=self.experiment_id,
            model_name=self.model_name,
            split=self.split,
            appliances=np.array(self.appliances),
            sample_index=self.sample_index,
            y_true_watts=self.y_true_watts,
            y_pred_watts=self.y_pred_watts,
            y_true_on=self.y_true_on if self.y_true_on is not None else np.array([]),
            y_pred_on=self.y_pred_on if self.y_pred_on is not None else np.array([]),
            csv_timesteps=self.csv_timesteps if self.csv_timesteps is not None else np.array([]),
        )

    @classmethod
    def load(cls, path: Path) -> "PredictionBundle":
        data = np.load(path, allow_pickle=True)
        appliances = data["appliances"].tolist()
        if isinstance(appliances, np.ndarray):
            appliances = appliances.tolist()
        y_true_on = data["y_true_on"]
        y_pred_on = data["y_pred_on"]
        csv_ts = data["csv_timesteps"] if "csv_timesteps" in data else np.array([])
        return cls(
            experiment_id=str(data["experiment_id"]),
            model_name=str(data["model_name"]),
            split=str(data["split"]),
            appliances=list(appliances),
            sample_index=data["sample_index"],
            y_true_watts=data["y_true_watts"],
            y_pred_watts=data["y_pred_watts"],
            y_true_on=None if y_true_on.size == 0 else y_true_on,
            y_pred_on=None if y_pred_on.size == 0 else y_pred_on,
            csv_timesteps=None if csv_ts.size == 0 else csv_ts,
        )


@dataclass
class StepOutput:
    loss: object
    logs: dict[str, float] = field(default_factory=dict)
    aux: dict[str, object] = field(default_factory=dict)

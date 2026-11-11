"""The one prediction format accepted by this project's evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PredictionBundle:
    """A reconstructed appliance-power timeline and optional ON/OFF estimates.

    Shapes use ``N`` evaluation time points and ``A`` appliances:

    - ``y_true_watts`` and ``y_pred_watts``: ``(N, A)``;
    - ``y_true_on``, ``y_pred_on``, and ``y_pred_state_prob``: ``(N, A)``;
    - ``csv_timesteps`` and ``segment_ids``: ``(N,)``.

    The class is deliberately model-agnostic.  A direct CNN and inverse solver
    produce the same object, so the metric and waveform rules are identical.
    """

    experiment_id: str
    model_name: str
    split: str
    appliances: list[str]
    sample_index: np.ndarray
    y_true_watts: np.ndarray
    y_pred_watts: np.ndarray
    y_true_on: np.ndarray | None = None
    y_pred_on: np.ndarray | None = None
    y_pred_state_prob: np.ndarray | None = None
    csv_timesteps: np.ndarray | None = None
    segment_ids: np.ndarray | None = None

    def save(self, path: Path) -> None:
        """Save raw float32 predictions without rounding evaluation values."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            experiment_id=self.experiment_id,
            model_name=self.model_name,
            split=self.split,
            appliances=np.asarray(self.appliances),
            sample_index=self.sample_index,
            y_true_watts=self.y_true_watts,
            y_pred_watts=self.y_pred_watts,
            y_true_on=self.y_true_on if self.y_true_on is not None else np.asarray([]),
            y_pred_on=self.y_pred_on if self.y_pred_on is not None else np.asarray([]),
            y_pred_state_prob=(
                self.y_pred_state_prob if self.y_pred_state_prob is not None else np.asarray([])
            ),
            csv_timesteps=self.csv_timesteps if self.csv_timesteps is not None else np.asarray([]),
            segment_ids=self.segment_ids if self.segment_ids is not None else np.asarray([]),
        )

    @classmethod
    def load(cls, path: Path) -> "PredictionBundle":
        """Load a bundle produced by :meth:`save`."""
        data = np.load(path, allow_pickle=True)

        def optional(name: str) -> np.ndarray | None:
            values = data[name] if name in data else np.asarray([])
            return None if values.size == 0 else values

        return cls(
            experiment_id=str(data["experiment_id"]),
            model_name=str(data["model_name"]),
            split=str(data["split"]),
            appliances=list(data["appliances"].tolist()),
            sample_index=data["sample_index"],
            y_true_watts=data["y_true_watts"],
            y_pred_watts=data["y_pred_watts"],
            y_true_on=optional("y_true_on"),
            y_pred_on=optional("y_pred_on"),
            y_pred_state_prob=optional("y_pred_state_prob"),
            csv_timesteps=optional("csv_timesteps"),
            segment_ids=optional("segment_ids"),
        )

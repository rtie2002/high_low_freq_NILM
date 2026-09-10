from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "multi_appliances_NILM"))

from adapters.common import PredictionBundle
from evaluation.state_postprocess import calibrate_state_postprocess


class StatePostprocessTests(unittest.TestCase):
    def test_temporal_postprocess_seconds_follow_sample_rate(self) -> None:
        bundle = PredictionBundle(
            experiment_id="unit",
            model_name="unit",
            split="validation",
            appliances=["washingmachine"],
            sample_index=np.arange(4),
            y_true_watts=np.zeros((4, 1), dtype=np.float32),
            y_pred_watts=np.zeros((4, 1), dtype=np.float32),
            y_true_on=np.asarray([[0], [1], [1], [0]], dtype=np.int32),
            y_pred_on=np.zeros((4, 1), dtype=np.int32),
            y_pred_state_prob=np.asarray([[0.1], [0.9], [0.9], [0.1]], dtype=np.float32),
        )
        model_cfg = {
            "evaluation": {
                "state_calibration": {
                    "threshold_grid": [0.5],
                    "postprocess": {
                        "enabled": True,
                        "min_on_seconds": {"washingmachine": 960},
                        "merge_gap_seconds": {"washingmachine": 16},
                    },
                }
            },
        }

        calibration = calibrate_state_postprocess(
            bundle,
            model_cfg,
            sample_seconds=8,
        )

        self.assertEqual(calibration["postprocess"]["sample_seconds"], 8.0)
        self.assertEqual(calibration["postprocess"]["min_on_samples"]["washingmachine"], 120)
        self.assertEqual(calibration["postprocess"]["merge_gap_samples"]["washingmachine"], 2)


if __name__ == "__main__":
    unittest.main()

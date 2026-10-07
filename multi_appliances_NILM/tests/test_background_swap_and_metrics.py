from __future__ import annotations

import unittest

import numpy as np
import torch

from data.common import PredictionBundle
from data.dataloader import WindowDataset, get_random_mix_mode
from evaluation.metrics import background_fpr_table, evaluate_bundle
from evaluation.state_postprocess import apply_state_calibration


def _bundle(
    y_true_watts: np.ndarray,
    y_pred_watts: np.ndarray,
    y_true_on: np.ndarray,
    y_pred_on: np.ndarray,
    *,
    segment_ids: np.ndarray | None = None,
) -> PredictionBundle:
    n_samples, n_apps = y_true_watts.shape
    return PredictionBundle(
        experiment_id="test",
        model_name="test_model",
        split="test",
        appliances=[f"app_{i}" for i in range(n_apps)],
        sample_index=np.arange(n_samples),
        y_true_watts=y_true_watts,
        y_pred_watts=y_pred_watts,
        y_true_on=y_true_on,
        y_pred_on=y_pred_on,
        y_pred_state_prob=y_pred_on.astype(np.float64),
        csv_timesteps=np.arange(n_samples),
        segment_ids=(
            np.zeros(n_samples, dtype=np.int64)
            if segment_ids is None
            else segment_ids
        ),
    )


class BackgroundSwapTests(unittest.TestCase):
    def test_default_mode_preserves_original_full_mix(self) -> None:
        self.assertEqual(get_random_mix_mode({"training": {}}), "full")

    def test_background_swap_keeps_anchor_targets_and_states(self) -> None:
        targets = np.asarray(
            [
                [1, 10], [2, 20], [3, 30], [4, 40],
                [5, 50], [6, 60], [7, 70], [8, 80],
            ],
            dtype=np.float32,
        )
        states = (targets > 15).astype(np.int64)
        background = np.asarray([100] * 4 + [200] * 4, dtype=np.float32)
        inputs = targets.sum(axis=1) + background
        dataset = WindowDataset(
            inputs,
            targets,
            states,
            {
                "input_window_length": 4,
                "output_window_length": 4,
                "output_alignment": "end",
            },
            stride=4,
            target_mode="full_input",
            random_mix_prob=1.0,
            random_mix_mode="background_swap",
        )

        mixed_input, mixed_targets, mixed_states = dataset[0]

        np.testing.assert_allclose(mixed_targets.numpy(), targets[:4])
        np.testing.assert_array_equal(mixed_states.numpy(), states[:4])
        np.testing.assert_allclose(
            mixed_input.squeeze(-1).numpy(),
            targets[:4].sum(axis=1) + 200,
        )


class DiagnosticMetricTests(unittest.TestCase):
    def test_state_calibration_can_update_detection_without_erasing_power(self) -> None:
        bundle = _bundle(
            np.asarray([[100.0], [100.0], [100.0]], dtype=np.float64),
            np.asarray([[80.0], [70.0], [60.0]], dtype=np.float64),
            np.asarray([[1], [1], [1]], dtype=np.int32),
            np.asarray([[1], [1], [1]], dtype=np.int32),
        )
        bundle.y_pred_state_prob = np.asarray([[0.9], [0.4], [0.2]], dtype=np.float64)
        calibration = {
            "appliances": ["app_0"],
            "thresholds": {"app_0": 0.5},
            "postprocess": {
                "min_on_samples": {"app_0": 1},
                "merge_gap_samples": {"app_0": 0},
            },
        }

        soft_power = apply_state_calibration(
            bundle, calibration, apply_to_power=False
        )
        hard_power = apply_state_calibration(
            bundle, calibration, apply_to_power=True
        )
        ramp_power = apply_state_calibration(
            bundle,
            calibration,
            power_gate_mode="ramp",
            ramp_width=0.2,
        )

        np.testing.assert_array_equal(soft_power.y_pred_on[:, 0], [1, 0, 0])
        np.testing.assert_allclose(soft_power.y_pred_watts[:, 0], [80.0, 70.0, 60.0])
        np.testing.assert_allclose(hard_power.y_pred_watts[:, 0], [80.0, 0.0, 0.0])
        np.testing.assert_allclose(ramp_power.y_pred_watts[:, 0], [80.0, 35.0, 0.0])

    def test_sample_fpr_fnr_energy_and_false_events(self) -> None:
        y_true_on = np.asarray([[0], [0], [1], [1], [0], [0]], dtype=np.int32)
        y_pred_on = np.asarray([[0], [1], [1], [0], [1], [0]], dtype=np.int32)
        bundle = _bundle(
            np.asarray([[0], [0], [20], [20], [0], [0]], dtype=np.float64),
            np.asarray([[0], [10], [20], [0], [30], [0]], dtype=np.float64),
            y_true_on,
            y_pred_on,
        )

        row = evaluate_bundle(bundle, state_label_source="csv", sample_seconds=8).iloc[0]

        self.assertAlmostEqual(row["false_positive_rate"], 0.5)
        self.assertAlmostEqual(row["false_negative_rate"], 0.5)
        self.assertAlmostEqual(row["false_positive_energy_wh"], 40 * 8 / 3600)
        self.assertEqual(row["false_event_count"], 1)

    def test_false_events_do_not_cross_segment_boundaries(self) -> None:
        bundle = _bundle(
            np.asarray([[10], [0]], dtype=np.float64),
            np.asarray([[10], [10]], dtype=np.float64),
            np.asarray([[1], [0]], dtype=np.int32),
            np.asarray([[1], [1]], dtype=np.int32),
            segment_ids=np.asarray([0, 1]),
        )

        row = evaluate_bundle(bundle, state_label_source="csv", sample_seconds=8).iloc[0]
        self.assertEqual(row["false_event_count"], 1)

    def test_background_bins_are_left_inclusive(self) -> None:
        aggregate = np.asarray(
            [0, 99.9, 100, 199.9, 200, 399.9, 400, 799.9, 800],
            dtype=np.float64,
        )
        n_samples = len(aggregate)
        bundle = _bundle(
            np.zeros((n_samples, 1), dtype=np.float64),
            np.ones((n_samples, 1), dtype=np.float64),
            np.zeros((n_samples, 1), dtype=np.int32),
            np.ones((n_samples, 1), dtype=np.int32),
        )

        table = background_fpr_table(
            bundle,
            aggregate,
            sample_seconds=8,
            state_label_source="csv",
        )

        self.assertEqual(table["off_samples"].tolist(), [2, 2, 2, 2, 1])
        self.assertEqual(table["false_positive_samples"].tolist(), [2, 2, 2, 2, 1])
        np.testing.assert_allclose(table["false_positive_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

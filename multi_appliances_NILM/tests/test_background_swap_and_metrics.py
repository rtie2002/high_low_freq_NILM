from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from data.common import PredictionBundle
from data.dataloader import (
    WindowDataset,
    get_background_consistency_enabled,
    get_random_mix_mode,
)
from evaluation.metrics import (
    PowerPostprocessConfig,
    apply_power_postprocess_pair,
    background_fpr_table,
    evaluate_bundle,
)
from evaluation.output_format import save_result_json, save_result_table
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
        self.assertFalse(get_background_consistency_enabled({"training": {}}))

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

    def test_paired_background_keeps_real_view_and_exact_labels(self) -> None:
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
            paired_background=True,
        )

        (real_input, swapped_input), paired_targets, paired_states = dataset[0]

        np.testing.assert_allclose(real_input.squeeze(-1).numpy(), inputs[:4])
        np.testing.assert_allclose(
            swapped_input.squeeze(-1).numpy(),
            targets[:4].sum(axis=1) + 200,
        )
        np.testing.assert_allclose(paired_targets.numpy(), targets[:4])
        np.testing.assert_array_equal(paired_states.numpy(), states[:4])

    def test_focal_event_mix_always_contains_on_target_and_samples_background_bins(self) -> None:
        targets = np.zeros((12, 2), dtype=np.float32)
        targets[:4, 0] = 100.0
        targets[4:8, 1] = 200.0
        states = (targets > 0).astype(np.int64)
        background = np.repeat([50.0, 250.0, 900.0], 4).astype(np.float32)
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
            random_mix_mode="full",
            random_mix_focal_event=True,
            random_mix_background_bins_watts=[0, 100, 800],
        )

        self.assertEqual(len(dataset.random_mix_on_starts), 2)
        self.assertEqual(len(dataset.random_mix_background_starts), 3)

        torch.manual_seed(7)
        sampled_backgrounds = set()
        for _ in range(60):
            mixed_input, mixed_targets, mixed_states = dataset[0]
            self.assertTrue(bool(mixed_states.any()))
            residual = mixed_input.squeeze(-1) - mixed_targets.sum(dim=1)
            sampled_backgrounds.add(float(torch.median(residual)))

        self.assertEqual(sampled_backgrounds, {50.0, 250.0, 900.0})

    def test_partial_focal_probability_still_prepares_sampling_pools(self) -> None:
        targets = np.zeros((8, 2), dtype=np.float32)
        targets[:4, 0] = 100.0
        targets[4:, 1] = 200.0
        states = (targets > 0).astype(np.int64)
        background = np.repeat([50.0, 900.0], 4).astype(np.float32)
        dataset = WindowDataset(
            targets.sum(axis=1) + background,
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
            random_mix_mode="full",
            random_mix_focal_event=True,
            random_mix_focal_event_prob=0.25,
            random_mix_background_bins_watts=[0, 800],
        )

        self.assertEqual([len(pool) for pool in dataset.random_mix_on_starts], [1, 1])
        self.assertEqual(
            [len(pool) for pool in dataset.random_mix_background_starts], [1, 1]
        )

    def test_alignment_jitter_changes_only_matching_source_input(self) -> None:
        targets = np.asarray([[0.0], [100.0], [100.0], [0.0]], dtype=np.float32)
        states = (targets > 0).astype(np.int64)
        background = np.full(4, 10.0, dtype=np.float32)
        dataset = WindowDataset(
            targets[:, 0] + background,
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
            random_mix_mode="full",
            source_codes=np.ones(4, dtype=np.int8),
            random_mix_alignment_app_index=0,
            random_mix_alignment_source_code=1,
            random_mix_alignment_offsets=[2],
            random_mix_alignment_probabilities=[1.0],
        )

        torch.manual_seed(3)
        mixed_input, mixed_targets, mixed_states = dataset[0]
        input_appliance = mixed_input.squeeze(-1).numpy() - background

        np.testing.assert_allclose(mixed_targets.numpy(), targets)
        np.testing.assert_array_equal(mixed_states.numpy(), states)
        self.assertGreater(int(np.flatnonzero(input_appliance > 50)[0]), 1)

    def test_alignment_jitter_leaves_other_sources_aligned(self) -> None:
        targets = np.asarray([[0.0], [100.0], [100.0], [0.0]], dtype=np.float32)
        background = np.full(4, 10.0, dtype=np.float32)
        dataset = WindowDataset(
            targets[:, 0] + background,
            targets,
            (targets > 0).astype(np.int64),
            {
                "input_window_length": 4,
                "output_window_length": 4,
                "output_alignment": "end",
            },
            stride=4,
            target_mode="full_input",
            random_mix_prob=1.0,
            random_mix_mode="full",
            source_codes=np.full(4, 2, dtype=np.int8),
            random_mix_alignment_app_index=0,
            random_mix_alignment_source_code=1,
            random_mix_alignment_offsets=[2],
            random_mix_alignment_probabilities=[1.0],
        )

        mixed_input, mixed_targets, _ = dataset[0]
        np.testing.assert_allclose(
            mixed_input.squeeze(-1).numpy() - background,
            mixed_targets.numpy().squeeze(-1),
        )

    def test_alignment_jitter_all_sources(self) -> None:
        targets = np.asarray([[0.0], [100.0], [100.0], [0.0]], dtype=np.float32)
        background = np.full(4, 10.0, dtype=np.float32)
        dataset = WindowDataset(
            targets[:, 0] + background,
            targets,
            (targets > 0).astype(np.int64),
            {
                "input_window_length": 4,
                "output_window_length": 4,
                "output_alignment": "end",
            },
            stride=4,
            target_mode="full_input",
            random_mix_prob=1.0,
            random_mix_mode="full",
            source_codes=np.full(4, 2, dtype=np.int8),
            random_mix_alignment_app_index=0,
            random_mix_alignment_source_code=-1,
            random_mix_alignment_offsets=[1],
            random_mix_alignment_probabilities=[1.0],
        )

        mixed_input, _, _ = dataset[0]
        input_appliance = mixed_input.squeeze(-1).numpy() - background
        self.assertEqual(int(np.flatnonzero(input_appliance > 50)[0]), 2)


class DiagnosticMetricTests(unittest.TestCase):
    def test_event_f1_accepts_onsets_within_tolerance(self) -> None:
        truth = np.asarray([[0], [0], [1], [1], [0], [0]], dtype=np.int32)
        prediction = np.asarray([[0], [0], [0], [0], [1], [1]], dtype=np.int32)
        bundle = _bundle(
            truth.astype(np.float64) * 100.0,
            prediction.astype(np.float64) * 100.0,
            truth,
            prediction,
        )

        strict = evaluate_bundle(
            bundle,
            state_label_source="csv",
            sample_seconds=8,
            event_tolerance_seconds=0,
        ).iloc[0]
        tolerant = evaluate_bundle(
            bundle,
            state_label_source="csv",
            sample_seconds=8,
            event_tolerance_seconds=16,
        ).iloc[0]

        self.assertEqual(strict["event_f1"], 0.0)
        self.assertEqual(tolerant["event_f1"], 1.0)

    def test_power_postprocess_never_changes_ground_truth(self) -> None:
        config = PowerPostprocessConfig(
            enabled=True,
            min_power_watts=5.0,
            max_on_power_watts=np.asarray([600.0]),
        )
        y_true = np.asarray([[2.0], [750.0]])
        y_pred = np.asarray([[2.0], [750.0]])

        processed_true, processed_pred = apply_power_postprocess_pair(
            y_true, y_pred, config
        )

        np.testing.assert_allclose(processed_true, y_true)
        np.testing.assert_allclose(processed_pred[:, 0], [0.0, 600.0])

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


class ResultFormattingTests(unittest.TestCase):
    def test_text_results_use_three_decimal_places(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "metrics.csv"
            json_path = root / "metrics.json"

            save_result_table(
                pd.DataFrame({"metric": [1.23456, 0.00049], "count": [2, 3]}),
                csv_path,
            )
            save_result_json(
                json_path,
                {"metric": 1.23456, "nested": [0.00049, 2]},
            )

            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("1.235,2", csv_text)
            self.assertIn("0.000,3", csv_text)
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8")),
                {"metric": 1.235, "nested": [0.0, 2]},
            )


if __name__ == "__main__":
    unittest.main()

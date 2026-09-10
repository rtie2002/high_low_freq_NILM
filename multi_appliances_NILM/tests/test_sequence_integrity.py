from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset_preprocess"))
sys.path.insert(0, str(ROOT / "multi_appliances_NILM"))

from adapters.config import load_experiment
from adapters.dataloader import WindowDataset, _sequence_ids_from_csv
from dataset_preprocess.ukdale_processing_multi_appliance import make_labels
from scripts.prepare_mixed_ukdale_refit_3week_split import (
    count_events,
    frame_sequence_ids,
)
from ukdale_processing import (
    apply_algorithm1_labeling,
    fill_complete_short_gaps,
    resolve_time_samples,
)


class SequenceIntegrityTests(unittest.TestCase):
    def test_only_complete_short_gaps_are_filled(self) -> None:
        index = pd.date_range("2020-01-01", periods=6, freq="8s", tz="UTC")
        values = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, 6.0], index=index)

        filled = fill_complete_short_gaps(values, max_gap=1)

        self.assertEqual(float(filled.iloc[1]), 2.0)
        self.assertTrue(filled.iloc[3:5].isna().all())

    def test_min_off_duration_counts_off_samples(self) -> None:
        power = np.asarray([100.0, 0.0, 0.0, 100.0])

        labels = apply_algorithm1_labeling(
            power,
            x_threshold=50.0,
            l_window=0,
            remove_spikes=False,
            min_off_duration=2,
            min_on_duration=1,
        )

        np.testing.assert_array_equal(labels, np.ones(4, dtype=int))

    def test_short_gap_segment_keeps_threshold_on(self) -> None:
        """Fridge-like power in a 2-sample gap fragment must not be wiped by min_on=8."""
        power = np.asarray([88.0, 87.0])

        labels = apply_algorithm1_labeling(
            power,
            x_threshold=50.0,
            l_window=0,
            remove_spikes=False,
            min_off_duration=2,
            min_on_duration=8,
        )

        np.testing.assert_array_equal(labels, np.ones(2, dtype=int))

    def test_censored_short_on_is_kept_but_complete_short_on_is_removed(self) -> None:
        boundary_power = np.r_[np.full(3, 100.0), np.zeros(7)]
        interior_power = np.r_[np.zeros(2), np.full(3, 100.0), np.zeros(5)]

        boundary = apply_algorithm1_labeling(
            boundary_power, 50, l_window=0, remove_spikes=False, min_on_duration=5
        )
        interior = apply_algorithm1_labeling(
            interior_power, 50, l_window=0, remove_spikes=False, min_on_duration=5
        )

        np.testing.assert_array_equal(boundary[:3], np.ones(3, dtype=int))
        self.assertEqual(int(interior.sum()), 0)

    def test_physical_duration_is_stable_across_sampling_rates(self) -> None:
        config = {"min_on_seconds": 1800, "resample_gap_fill_seconds": 18}

        self.assertEqual(resolve_time_samples(config, "min_on_duration", 1, 6), 300)
        self.assertEqual(resolve_time_samples(config, "min_on_duration", 1, 8), 225)
        self.assertEqual(resolve_time_samples(config, "resample_gap_fill", 1, 6), 3)
        self.assertEqual(resolve_time_samples(config, "resample_gap_fill", 1, 8), 2)

    def test_appliance_spike_override_preserves_short_event(self) -> None:
        power = np.asarray([0.0, 1000.0, 0.0])
        appliance = {
            "on_power_threshold": 200,
            "min_off_duration": 0,
            "min_on_duration": 1,
            "remove_spikes": False,
        }
        algorithm = {
            "remove_spikes": True,
            "spike_window": 3,
            "spike_threshold": 1.5,
            "background_threshold": 50,
        }

        labels = make_labels(power, appliance, algorithm, house=1, sample_seconds=8)

        np.testing.assert_array_equal(labels, np.asarray([0, 1, 0]))

    def test_windows_do_not_cross_segments(self) -> None:
        values = np.arange(12, dtype=np.float32)
        targets = values[:, None]
        states = np.zeros((12, 1), dtype=np.int64)
        segments = np.asarray([0] * 6 + [1] * 6)

        dataset = WindowDataset(
            values,
            targets,
            states,
            {"input_window_length": 4, "output_window_length": 4},
            stride=2,
            segment_ids=segments,
        )

        np.testing.assert_array_equal(dataset.indices, np.asarray([0, 2, 6, 8]))
        self.assertEqual(dataset.n_rejected_windows, 1)

    def test_window_stride_restarts_at_each_segment(self) -> None:
        values = np.arange(12, dtype=np.float32)
        targets = values[:, None]
        states = np.zeros((12, 1), dtype=np.int64)
        segments = np.asarray([0] * 5 + [1] * 7)

        dataset = WindowDataset(
            values,
            targets,
            states,
            {"input_window_length": 4, "output_window_length": 4},
            stride=3,
            segment_ids=segments,
        )

        np.testing.assert_array_equal(dataset.indices, np.asarray([0, 5, 8]))

    def test_csv_metadata_creates_monotonic_segments(self) -> None:
        frame = pd.DataFrame(
            {
                "dataset": ["a", "a", "a", "b", "b"],
                "house": [1, 1, 1, 1, 1],
                "sequence_id": [0, 0, 0, 0, 0],
                "unix_time": [0, 8, 24, 0, 8],
            }
        )

        segments = _sequence_ids_from_csv(frame, sample_seconds=8)

        np.testing.assert_array_equal(segments, np.asarray([0, 0, 1, 2, 2]))

    def test_event_count_treats_each_house_as_a_new_sequence(self) -> None:
        frame = pd.DataFrame(
            {
                "dataset": ["ukdale", "ukdale", "refit", "refit"],
                "house": [1, 1, 2, 2],
                "sequence_id": [0, 0, 0, 0],
            }
        )
        labels = np.ones(4, dtype=np.int8)

        segments = frame_sequence_ids(frame)

        np.testing.assert_array_equal(segments, np.asarray([0, 0, 1, 1]))
        self.assertEqual(count_events(labels, segments), 2)

    def test_experiment_loads_generated_normalization(self) -> None:
        stats = {
            "aggregate": {"mean": 10.0, "std": 2.0},
            "appliances": {"kettle": {"mean": 1.0, "std": 3.0}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
            (root / "experiment.yaml").write_text(
                "normalization_file: stats.json\n", encoding="utf-8"
            )

            experiment = load_experiment(root / "experiment.yaml")

        self.assertEqual(experiment["normalization"], stats)


if __name__ == "__main__":
    unittest.main()

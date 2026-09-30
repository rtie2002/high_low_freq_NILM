from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.plots import save_background_range_waveforms


class BackgroundRangePlotTests(unittest.TestCase):
    def test_saves_one_diagnostic_figure_per_appliance(self) -> None:
        backgrounds = np.repeat([50.0, 150.0, 300.0, 600.0, 900.0], 20)
        n_points = len(backgrounds)
        true_watts = np.zeros((n_points, 1), dtype=np.float64)
        pred_watts = np.zeros_like(true_watts)
        true_on = np.zeros_like(true_watts, dtype=np.int32)
        pred_on = np.zeros_like(true_watts, dtype=np.int32)

        for block_start in range(0, n_points, 20):
            true_watts[block_start + 4 : block_start + 9, 0] = 100.0
            pred_watts[block_start + 4 : block_start + 9, 0] = 90.0
            true_on[block_start + 4 : block_start + 9, 0] = 1
            pred_on[block_start + 4 : block_start + 9, 0] = 1

            pred_watts[block_start + 12 : block_start + 15, 0] = 60.0
            pred_on[block_start + 12 : block_start + 15, 0] = 1

        aggregate = backgrounds + true_watts[:, 0]
        with tempfile.TemporaryDirectory() as tmp_dir:
            saved = save_background_range_waveforms(
                tmp_dir,
                appliances=["test_appliance"],
                aggregate_watts=aggregate,
                y_true_watts=true_watts,
                y_pred_watts=pred_watts,
                y_true_on=true_on,
                y_pred_on=pred_on,
                sample_seconds=8,
                csv_timesteps=np.arange(n_points),
                segment_ids=np.zeros(n_points, dtype=np.int64),
                margin_samples=2,
                max_samples=20,
                dpi=40,
            )

            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].name, "test_appliance_background_ranges.png")
            self.assertTrue(Path(saved[0]).is_file())
            self.assertGreater(Path(saved[0]).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

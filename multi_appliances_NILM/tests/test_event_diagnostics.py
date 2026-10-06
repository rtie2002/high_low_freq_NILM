from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from evaluation.event_diagnostics import (
    event_waveform_table,
    paired_event_differences,
    run_event_summary,
    snr_summary_table,
)


def _bundle(label: str, prediction: np.ndarray, pred_on: np.ndarray) -> SimpleNamespace:
    target = np.asarray([[0], [10], [10], [0], [0], [20], [20], [0]], dtype=float)
    true_on = target > 0
    return SimpleNamespace(
        experiment_id=label,
        model_name="test",
        split="house",
        appliances=["appliance"],
        sample_index=np.arange(len(target)),
        y_true_watts=target,
        y_pred_watts=np.asarray(prediction, dtype=float).reshape(-1, 1),
        y_true_on=true_on,
        y_pred_on=np.asarray(pred_on, dtype=bool).reshape(-1, 1),
        csv_timesteps=np.arange(len(target)),
        segment_ids=np.zeros(len(target), dtype=int),
    )


class EventDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.aggregate = np.asarray([5, 15, 15, 5, 5, 25, 25, 5], dtype=float)
        self.target = np.asarray([[0], [10], [10], [0], [0], [20], [20], [0]], dtype=float)
        self.true_on = self.target > 0
        self.segments = np.zeros(len(self.target), dtype=int)

    def _table(self, bundle: SimpleNamespace, label: str):
        return event_waveform_table(
            bundle,
            run_label=label,
            aggregate_watts=self.aggregate,
            true_appliance_watts=self.target,
            true_on=self.true_on,
            segment_ids=self.segments,
            sample_seconds=8,
        )

    def test_perfect_events_have_unit_iou_and_zero_error(self) -> None:
        bundle = _bundle("perfect", self.target[:, 0], self.true_on[:, 0])
        table = self._table(bundle, "perfect")
        self.assertEqual(len(table), 2)
        np.testing.assert_allclose(table["event_iou"], 1.0)
        np.testing.assert_allclose(table["event_nrmse"], 0.0)
        np.testing.assert_allclose(table["energy_error_pct"], 0.0)
        np.testing.assert_allclose(table["start_error_seconds"], 0.0)
        np.testing.assert_allclose(table["end_error_seconds"], 0.0)

    def test_missed_event_and_false_event_are_counted(self) -> None:
        prediction = np.asarray([30, 0, 0, 0, 0, 20, 20, 0], dtype=float)
        pred_on = prediction > 0
        bundle = _bundle("mixed", prediction, pred_on)
        table = self._table(bundle, "mixed")
        self.assertEqual(table.iloc[0]["detected"], 0)
        self.assertEqual(table.iloc[0]["event_iou"], 0.0)
        summary = run_event_summary(
            table,
            {"mixed": bundle},
            true_on=self.true_on,
            segment_ids=self.segments,
            sample_seconds=8,
        )
        self.assertEqual(summary.iloc[0]["false_event_count"], 1)

    def test_snr_summary_and_paired_differences(self) -> None:
        perfect = _bundle("a", self.target[:, 0], self.true_on[:, 0])
        weak = _bundle("b", self.target[:, 0] * 0.5, self.true_on[:, 0])
        import pandas as pd

        events = pd.concat([self._table(perfect, "a"), self._table(weak, "b")], ignore_index=True)
        summary = snr_summary_table(events)
        self.assertFalse(summary.empty)
        paired = paired_event_differences(events, "a", "b")
        self.assertEqual(len(paired), 2)
        self.assertTrue((paired["delta_event_nrmse_b_minus_a"] > 0).all())


if __name__ == "__main__":
    unittest.main()

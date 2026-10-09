import unittest

import numpy as np

from scripts.prepare_refit_microwave_aligned_dataset import align_sequence


class TestRefitMicrowaveAlignment(unittest.TestCase):
    def test_moves_complete_event_to_supported_later_edge(self):
        aggregate = np.full(14, 100.0)
        aggregate[6:] += 900.0
        power = np.zeros(14)
        power[4:8] = 800.0
        state = (power >= 100.0).astype(np.int8)

        aligned_power, aligned_state, audit = align_sequence(
            aggregate, power, state
        )

        np.testing.assert_array_equal(np.flatnonzero(aligned_state), np.arange(6, 10))
        np.testing.assert_allclose(aligned_power[6:10], 800.0)
        self.assertEqual(audit["lag_counts"]["2"], 1)
        self.assertEqual(audit["changed_events"], 1)

    def test_rejects_shift_without_sufficient_aggregate_edge(self):
        aggregate = np.full(14, 100.0)
        power = np.zeros(14)
        power[4:8] = 800.0
        state = (power >= 100.0).astype(np.int8)

        aligned_power, aligned_state, audit = align_sequence(
            aggregate, power, state
        )

        np.testing.assert_array_equal(aligned_power, power)
        np.testing.assert_array_equal(aligned_state, state)
        self.assertEqual(audit["accepted_events"], 0)
        self.assertEqual(audit["rejected_events"], 1)


if __name__ == "__main__":
    unittest.main()

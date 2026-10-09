import unittest

from runner import _epoch_score, _is_better, _resolve_checkpoint_monitor


class CheckpointMonitorTests(unittest.TestCase):
    def test_validation_ap_is_maximized(self):
        key, mode, initial = _resolve_checkpoint_monitor(
            {"checkpoint_monitor": "val_ap"}
        )

        self.assertEqual(key, "val_ap")
        self.assertEqual(mode, "max")
        self.assertEqual(initial, float("-inf"))
        self.assertEqual(_epoch_score(key, {"val_ap": 0.81}), 0.81)
        self.assertTrue(_is_better(0.81, 0.79, mode))
        self.assertFalse(_is_better(0.79, 0.81, mode))


if __name__ == "__main__":
    unittest.main()

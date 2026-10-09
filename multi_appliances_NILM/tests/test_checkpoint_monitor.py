import unittest

import torch

from runner import (
    _configure_cuda,
    _epoch_score,
    _is_better,
    _resolve_checkpoint_monitor,
    seed_everything,
)


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

    def test_seed_controls_model_initialization(self):
        seed_everything(2026)
        first = torch.nn.Linear(4, 3).weight.detach().clone()
        seed_everything(2026)
        second = torch.nn.Linear(4, 3).weight.detach().clone()
        self.assertTrue(torch.equal(first, second))

    def test_deterministic_flag_controls_torch_algorithms(self):
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            _configure_cuda({"deterministic": True, "cudnn_benchmark": False})
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=True)


if __name__ == "__main__":
    unittest.main()

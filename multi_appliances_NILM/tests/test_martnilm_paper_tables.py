from __future__ import annotations

import unittest

from config import load_model_config
from scripts.run_mart_nilm_paper_tables import (
    MODEL_CONFIGS,
    TRAINING_VARIANTS,
    _paper_training_variant,
)


class MARTNILMTrainingAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = load_model_config(MODEL_CONFIGS["mart_nilm"])

    def test_two_by_two_variants_change_only_training_controls(self) -> None:
        variants = {
            name: _paper_training_variant(self.base, name, seed=2026)
            for name in TRAINING_VARIANTS
        }
        expected = {
            "standard_joint_training": (False, "none"),
            "focal_event_mixing": (True, "none"),
            "appliance_wise_balancing": (False, "per_appliance_clipped"),
            "complete": (True, "per_appliance_clipped"),
        }
        for name, (use_mix, balance) in expected.items():
            with self.subTest(name=name):
                cfg = variants[name]
                self.assertEqual(cfg["training"]["random_mix"]["enabled"], use_mix)
                self.assertEqual(
                    cfg["training"]["random_mix"]["focal_event_sampling"]["enabled"],
                    use_mix,
                )
                self.assertEqual(cfg["loss"]["task_balance"], balance)
                self.assertEqual(cfg["architecture"], self.base["architecture"])
                self.assertEqual(cfg["loss"]["lambda_state"], 0.8)


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.run_multinilm_simplification import (
    FEATURE_VARIANTS,
    _architecture_candidate,
    _feature_candidate,
    _loss_candidate,
    _repro_candidate,
)


class SimplificationFeatureConfigTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "experiment_id": "base",
            "fractional": {
                "k": 4,
                "include_raw": True,
                "include_delta": True,
                "include_abs_delta": True,
                "include_rolling_mean": True,
                "include_rolling_std": True,
                "rolling_windows": [8, 23, 45],
                "memory": 24,
                "channel_normalize": "none",
            },
        }

    def test_feature_candidates_have_one_to_four_channels(self):
        expected = {
            "feature_1": 1,
            "feature_2": 2,
            "feature_3": 3,
            "feature_4": 4,
            "feature_2_mean": 2,
        }
        for name in FEATURE_VARIANTS:
            with self.subTest(name=name):
                cfg = _feature_candidate(self.base, name)["fractional"]
                channels = (
                    int(cfg["include_raw"])
                    + int(cfg["include_delta"])
                    + len(cfg["rolling_windows"]) * int(cfg["include_rolling_mean"])
                    + len(cfg["rolling_windows"]) * int(cfg["include_rolling_std"])
                    + len(cfg["alphas"])
                )
                self.assertEqual(channels, expected[name])

    def test_candidate_does_not_mutate_base_config(self):
        _feature_candidate(self.base, "feature_1")
        self.assertEqual(self.base["fractional"]["k"], 4)
        self.assertTrue(self.base["fractional"]["include_abs_delta"])

    def test_core_loss_removes_only_auxiliary_terms(self):
        self.base["loss"] = {
            "task_balance": "per_appliance_clipped",
            "lambda_state": 0.8,
            "power_on_weight": 1.0,
            "power_off_weight": 0.5,
            "power_delta_weight": 0.15,
            "power_energy_relative_weight": 0.25,
            "state_fp_weight": 1.0,
        }
        cfg = _loss_candidate(self.base, "loss_core", "feature_1")

        self.assertEqual(cfg["loss"]["task_balance"], "per_appliance_clipped")
        self.assertEqual(cfg["loss"]["lambda_state"], 0.8)
        self.assertEqual(cfg["loss"]["power_on_weight"], 1.0)
        for name in (
            "power_off_weight",
            "power_delta_weight",
            "power_energy_relative_weight",
            "state_fp_weight",
        ):
            self.assertEqual(cfg["loss"][name], 0.0)

    def test_ap_monitor_changes_selection_not_loss(self):
        self.base["loss"] = {"power_on_weight": 1.0}
        self.base["training"] = {
            "checkpoint_monitor": "val_mae_plus_one_minus_ap",
            "learning_rate": 1e-4,
        }
        cfg = _loss_candidate(self.base, "loss_core_ap_monitor", "feature_1")

        self.assertEqual(cfg["training"]["checkpoint_monitor"], "val_ap")
        self.assertEqual(cfg["training"]["learning_rate"], 1e-4)
        self.assertNotIn("checkpoint_monitor", cfg["loss"])

    def test_plain_relation_removes_nested_refinements_only(self):
        self.base["architecture"] = {
            "use_multiscale_stem": True,
            "stem_norm_type": "ibn",
            "num_blocks": 5,
            "head_local_layers": 2,
            "task_attention": {"enabled": True, "reduction": 4},
            "cross_appliance": {"enabled": True, "mode": "relation_attention"},
        }
        cfg = _architecture_candidate(self.base, "plain_relation")
        architecture = cfg["architecture"]

        self.assertFalse(architecture["use_multiscale_stem"])
        self.assertEqual(architecture["stem_norm_type"], "batch")
        self.assertEqual(architecture["head_local_layers"], 1)
        self.assertFalse(architecture["task_attention"]["enabled"])
        self.assertEqual(architecture["num_blocks"], 5)
        self.assertTrue(architecture["cross_appliance"]["enabled"])

    def test_seeded_reference_and_raw_change_only_features(self):
        baseline = _repro_candidate(self.base, "seeded_baseline")
        raw = _repro_candidate(self.base, "seeded_raw")

        self.assertTrue(baseline["fractional"]["include_abs_delta"])
        self.assertFalse(raw["fractional"]["include_abs_delta"])
        self.assertEqual(raw["fractional"]["alphas"], [])


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.run_multinilm_simplification import (
    FEATURE_VARIANTS,
    _architecture_candidate,
    _feature_candidate,
    _hyperparameter_candidate,
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

    def test_compact_loss_keeps_edge_and_false_positive_terms(self):
        self.base["loss"] = {
            "power_off_weight": 0.5,
            "power_delta_weight": 0.15,
            "power_energy_relative_weight": 0.25,
            "state_fp_weight": 1.0,
        }
        cfg = _loss_candidate(self.base, "loss_compact", "feature_1")

        self.assertEqual(cfg["loss"]["power_off_weight"], 0.0)
        self.assertEqual(cfg["loss"]["power_energy_relative_weight"], 0.0)
        self.assertEqual(cfg["loss"]["power_delta_weight"], 0.15)
        self.assertEqual(cfg["loss"]["state_fp_weight"], 1.0)

    def test_lambda_state_candidate_changes_one_weight(self):
        self.base["loss"] = {"lambda_state": 0.8, "power_on_weight": 1.0}
        cfg = _loss_candidate(self.base, "lambda_state_1", "feature_1")

        self.assertEqual(cfg["loss"]["lambda_state"], 1.0)
        self.assertEqual(cfg["loss"]["power_on_weight"], 1.0)

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

    def test_no_task_attention_preserves_other_architecture(self):
        self.base["architecture"] = {
            "use_multiscale_stem": True,
            "stem_norm_type": "ibn",
            "num_blocks": 5,
            "head_local_layers": 2,
            "task_attention": {"enabled": True, "reduction": 4},
            "cross_appliance": {"enabled": True, "mode": "relation_attention"},
        }
        cfg = _architecture_candidate(self.base, "no_task_attention")
        architecture = cfg["architecture"]

        self.assertFalse(architecture["task_attention"]["enabled"])
        self.assertTrue(architecture["use_multiscale_stem"])
        self.assertEqual(architecture["stem_norm_type"], "ibn")
        self.assertEqual(architecture["head_local_layers"], 2)
        self.assertTrue(architecture["cross_appliance"]["enabled"])

    def test_one_head_block_preserves_attention(self):
        self.base["architecture"] = {
            "head_local_layers": 2,
            "task_attention": {"enabled": True, "reduction": 4},
            "cross_appliance": {"enabled": True, "mode": "relation_attention"},
        }
        cfg = _architecture_candidate(self.base, "one_head_block")
        architecture = cfg["architecture"]

        self.assertEqual(architecture["head_local_layers"], 1)
        self.assertTrue(architecture["task_attention"]["enabled"])
        self.assertTrue(architecture["cross_appliance"]["enabled"])

    def test_batch_stem_norm_changes_only_stem_norm(self):
        self.base["architecture"] = {
            "stem_norm_type": "ibn",
            "temporal_norm_type": "batch",
            "head_norm_type": "batch",
            "task_attention": {"enabled": True},
        }
        cfg = _architecture_candidate(self.base, "batch_stem_norm")
        architecture = cfg["architecture"]

        self.assertEqual(architecture["stem_norm_type"], "batch")
        self.assertEqual(architecture["temporal_norm_type"], "batch")
        self.assertEqual(architecture["head_norm_type"], "batch")
        self.assertTrue(architecture["task_attention"]["enabled"])

    def test_three_tcn_blocks_preserves_heads_and_attention(self):
        self.base["architecture"] = {
            "num_blocks": 5,
            "max_dilation": 16,
            "head_local_layers": 2,
            "task_attention": {"enabled": True},
            "cross_appliance": {"enabled": True},
        }
        cfg = _architecture_candidate(self.base, "three_tcn_blocks")
        architecture = cfg["architecture"]

        self.assertEqual(architecture["num_blocks"], 3)
        self.assertEqual(architecture["max_dilation"], 4)
        self.assertEqual(architecture["head_local_layers"], 2)
        self.assertTrue(architecture["task_attention"]["enabled"])
        self.assertTrue(architecture["cross_appliance"]["enabled"])

    def test_no_train_power_gate_preserves_evaluation_and_heads(self):
        self.base["architecture"] = {
            "gate_mode": "soft",
            "head_local_layers": 2,
            "task_attention": {"enabled": True},
            "cross_appliance": {"enabled": True},
        }
        self.base["evaluation"] = {
            "state_calibration": {"apply_to_power": True}
        }
        cfg = _architecture_candidate(self.base, "no_train_power_gate")

        self.assertEqual(cfg["architecture"]["gate_mode"], "none")
        self.assertEqual(cfg["architecture"]["head_local_layers"], 2)
        self.assertTrue(cfg["architecture"]["task_attention"]["enabled"])
        self.assertTrue(cfg["architecture"]["cross_appliance"]["enabled"])
        self.assertTrue(cfg["evaluation"]["state_calibration"]["apply_to_power"])

    def test_detached_train_gate_changes_only_gradient_path(self):
        self.base["architecture"] = {
            "gate_mode": "soft",
            "head_local_layers": 2,
            "task_attention": {"enabled": True},
            "cross_appliance": {"enabled": True},
        }
        cfg = _architecture_candidate(self.base, "detached_train_power_gate")

        self.assertEqual(cfg["architecture"]["gate_mode"], "soft_detached")
        self.assertEqual(cfg["architecture"]["head_local_layers"], 2)
        self.assertTrue(cfg["architecture"]["task_attention"]["enabled"])
        self.assertTrue(cfg["architecture"]["cross_appliance"]["enabled"])

    def test_half_gradient_gate_preserves_soft_forward_mode(self):
        self.base["architecture"] = {
            "gate_mode": "soft",
            "gate_gradient_scale": 1.0,
            "head_local_layers": 2,
            "cross_appliance": {"enabled": True},
        }
        cfg = _architecture_candidate(self.base, "half_gradient_train_power_gate")

        self.assertEqual(cfg["architecture"]["gate_mode"], "soft")
        self.assertEqual(cfg["architecture"]["gate_gradient_scale"], 0.5)
        self.assertEqual(cfg["architecture"]["head_local_layers"], 2)
        self.assertTrue(cfg["architecture"]["cross_appliance"]["enabled"])

    def test_seeded_reference_and_raw_change_only_features(self):
        baseline = _repro_candidate(self.base, "seeded_baseline")
        raw = _repro_candidate(self.base, "seeded_raw")

        self.assertTrue(baseline["fractional"]["include_abs_delta"])
        self.assertFalse(raw["fractional"]["include_abs_delta"])
        self.assertEqual(raw["fractional"]["alphas"], [])

        ap_selected = _repro_candidate(self.base, "seeded_raw_ap_monitor")
        self.assertEqual(ap_selected["training"]["checkpoint_monitor"], "val_ap")

    def test_dropout_candidate_changes_one_shared_value(self):
        self.base["architecture"] = {
            "dropout": 0.25,
            "num_blocks": 5,
            "cross_appliance": {"enabled": True},
        }
        cfg = _hyperparameter_candidate(self.base, "dropout_035")

        self.assertEqual(cfg["architecture"]["dropout"], 0.35)
        self.assertEqual(cfg["architecture"]["num_blocks"], 5)
        self.assertTrue(cfg["architecture"]["cross_appliance"]["enabled"])

    def test_relation_scale_candidate_preserves_relation_topology(self):
        self.base["architecture"] = {
            "dropout": 0.25,
            "cross_appliance": {
                "enabled": True,
                "mode": "relation_attention",
                "attention_channels": 16,
                "residual_scale": 0.25,
            },
        }
        cfg = _hyperparameter_candidate(self.base, "relation_scale_05")
        relation = cfg["architecture"]["cross_appliance"]

        self.assertEqual(relation["residual_scale"], 0.5)
        self.assertEqual(relation["mode"], "relation_attention")
        self.assertEqual(relation["attention_channels"], 16)
        self.assertEqual(cfg["architecture"]["dropout"], 0.25)


if __name__ == "__main__":
    unittest.main()

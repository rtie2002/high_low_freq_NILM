from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from model.MultiNILM import (
    FractionalFrontEnd,
    MultiNILM,
    MultiNILMAdapter,
    build_multinilm_fractional,
    multinilm_config,
    state_gate,
)


class MultiNILMDualExpertTests(unittest.TestCase):
    def test_detached_soft_gate_keeps_value_and_blocks_state_gradient(self) -> None:
        state_logit = torch.tensor([0.4], requires_grad=True)
        regression = torch.tensor([2.0], requires_grad=True)
        probability = torch.sigmoid(state_logit)

        gate = state_gate(
            probability,
            mode="soft_detached",
            training=True,
        )
        self.assertTrue(torch.allclose(gate, probability))

        (gate * regression).sum().backward()
        self.assertIsNone(state_logit.grad)
        self.assertIsNotNone(regression.grad)

    def test_soft_gate_gradient_scale_changes_gradient_not_forward_value(self) -> None:
        full_logit = torch.tensor([0.4], requires_grad=True)
        half_logit = torch.tensor([0.4], requires_grad=True)
        full_prob = torch.sigmoid(full_logit)
        half_prob = torch.sigmoid(half_logit)

        full_gate = state_gate(
            full_prob,
            mode="soft",
            training=True,
            gradient_scale=1.0,
        )
        half_gate = state_gate(
            half_prob,
            mode="soft",
            training=True,
            gradient_scale=0.5,
        )
        self.assertTrue(torch.allclose(full_gate, half_gate))

        full_gate.sum().backward()
        half_gate.sum().backward()
        self.assertTrue(torch.allclose(half_logit.grad, 0.5 * full_logit.grad))

    def test_early_relational_frontend_has_signed_delta_and_thirteen_channels(self) -> None:
        frontend = FractionalFrontEnd(
            alphas=[0.25, 0.5, 0.75, 1.0],
            include_raw=True,
            include_delta=True,
            include_abs_delta=True,
            include_rolling_mean=True,
            include_rolling_std=True,
            rolling_windows=[8, 23, 45],
            memory=24,
            channel_normalize="none",
        )
        self.assertEqual(frontend.out_channels, 13)

        x = torch.tensor([[[1.0, 3.0, 2.0, 5.0]]])
        features, alpha_one = frontend(x, return_alpha_one=True)
        expected_delta = torch.tensor([[[0.0, 2.0, -1.0, 3.0]]])
        expected_alpha_one = torch.tensor([[[1.0, 2.0, -1.0, 3.0]]])
        torch.testing.assert_close(features[:, 1:2], expected_delta)
        torch.testing.assert_close(alpha_one, expected_alpha_one, atol=1e-6, rtol=1e-6)

    def test_local_contrast_checkpoint_frontend_has_thirteen_channels(self) -> None:
        frontend = FractionalFrontEnd(
            alphas=[0.25, 0.5, 0.75, 1.0],
            include_raw=True,
            include_abs_delta=True,
            include_local_contrast=True,
            local_contrast_span=45,
            include_rolling_mean=True,
            include_rolling_std=True,
            rolling_windows=[8, 23, 45],
            memory=24,
            channel_normalize="none",
        )
        self.assertEqual(frontend.out_channels, 13)

        x = torch.zeros(1, 1, 64)
        x[..., 32:] = 1.0
        features = frontend(x)
        self.assertEqual(tuple(features.shape), (1, 13, 64))
        self.assertTrue(torch.isfinite(features).all())

    def test_dual_expert_shapes_gate_initialization_and_relation(self) -> None:
        model = MultiNILM(
            input_channels=4,
            num_appliances=3,
            output_length=64,
            hidden_channels=16,
            channel_schedule=[8, 16],
            num_blocks=2,
            kernel_size=5,
            temporal_dropout=0.25,
            head_dropout=0.10,
            dual_expert_enabled=True,
            dual_expert_local_channels=8,
            dual_expert_local_kernel_size=5,
            dual_expert_local_dilations=[1, 2, 4],
            dual_expert_dropout=0.10,
            dual_expert_gate_hidden_channels=8,
            dual_expert_gate_initial_local_weight=0.1,
            cross_appliance_enabled=True,
            cross_appliance_mode="relation_attention",
            cross_appliance_attention_channels=8,
            cross_appliance_dropout=0.0,
        )
        encoded = torch.randn(2, 4, 64)
        raw = torch.randn(2, 1, 64)
        alpha_one = torch.randn(2, 1, 64)

        power, logits = model(encoded, raw_input=raw, alpha_one=alpha_one)

        self.assertEqual(tuple(power.shape), (2, 64, 3))
        self.assertEqual(tuple(logits.shape), (2, 64, 3))
        self.assertEqual(model.temporal_encoder[0].dropout.p, 0.25)
        self.assertEqual(model.local_expert.temporal_blocks[0].dropout.p, 0.10)
        self.assertEqual(model.appliance_heads[0].dropout.p, 0.10)
        self.assertEqual(model.cross_appliance_distill.dropout.p, 0.0)
        self.assertEqual(tuple(model.last_expert_gates.shape), (2, 3, 2, 64))
        torch.testing.assert_close(
            model.last_expert_gates.sum(dim=2),
            torch.ones(2, 3, 64),
        )
        self.assertAlmostEqual(
            float(model.last_expert_gates[:, :, 0, :].mean()),
            0.1,
            delta=0.01,
        )

        (power.square().mean() + logits.square().mean()).backward()
        self.assertIsNotNone(model.local_expert.input_projection[0].weight.grad)
        self.assertGreater(
            float(model.local_expert.input_projection[0].weight.grad.abs().sum()),
            0.0,
        )
        self.assertIsNotNone(model.expert_gates[0].network[2].weight.grad)
        self.assertGreater(
            float(model.expert_gates[0].network[2].weight.grad.abs().sum()),
            0.0,
        )
        self.assertIsNotNone(model.cross_appliance_distill.query.weight.grad)

    def test_fractional_wrapper_routes_alpha_one_to_local_expert(self) -> None:
        architecture = {
            "hidden_channels": 16,
            "channel_schedule": [8, 16],
            "num_blocks": 2,
            "kernel_size": 5,
            "temporal_dropout": 0.0,
            "head_dropout": 0.0,
            "fractional": {
                "k": 2,
                "include_raw": True,
                "include_abs_delta": True,
                "memory": 4,
                "channel_normalize": "none",
            },
            "dual_expert": {
                "enabled": True,
                "local_channels": 8,
                "local_kernel_size": 5,
                "local_dilations": [1, 2, 4],
                "local_norm_type": "group",
                "dropout": 0.0,
                "gate_hidden_channels": 8,
                "gate_initial_local_weight": 0.1,
            },
            "cross_appliance": {
                "enabled": True,
                "mode": "relation_attention",
                "attention_channels": 8,
                "residual_scale": 0.25,
                "dropout": 0.0,
            },
        }
        model = build_multinilm_fractional(
            architecture,
            num_appliances=2,
            output_length=32,
            appliance_off_norm=[0.0, 0.0],
        )

        aggregate = torch.randn(2, 32)
        captured = {}

        def capture_local_inputs(_module, inputs):
            captured["raw"], captured["alpha_one"] = inputs

        handle = model.backbone.local_expert.register_forward_pre_hook(capture_local_inputs)
        power, logits = model(aggregate)
        handle.remove()

        self.assertEqual(tuple(power.shape), (2, 32, 2))
        self.assertEqual(tuple(logits.shape), (2, 32, 2))
        self.assertEqual(tuple(model.last_expert_gates.shape), (2, 2, 2, 32))
        expected_alpha_one = torch.cat(
            [aggregate[:, :1], aggregate[:, 1:] - aggregate[:, :-1]], dim=1
        ).unsqueeze(1)
        torch.testing.assert_close(captured["raw"], aggregate.unsqueeze(1))
        torch.testing.assert_close(captured["alpha_one"], expected_alpha_one, atol=1e-6, rtol=1e-6)

    def test_private_local_experts_are_distinct_and_all_receive_gradients(self) -> None:
        model = MultiNILM(
            input_channels=4,
            num_appliances=3,
            output_length=32,
            hidden_channels=16,
            channel_schedule=[8, 16],
            num_blocks=1,
            kernel_size=3,
            temporal_dropout=0.0,
            head_dropout=0.0,
            dual_expert_enabled=True,
            dual_expert_share_local=False,
            dual_expert_local_channels=8,
            dual_expert_local_dilations=[1, 2],
            dual_expert_dropout=0.0,
            dual_expert_gate_hidden_channels=8,
        )

        self.assertIsNone(model.local_expert)
        self.assertEqual(len(model.local_experts), 3)
        weights = [expert.input_projection[0].weight for expert in model.local_experts]
        self.assertEqual(len({weight.data_ptr() for weight in weights}), 3)

        encoded = torch.randn(2, 4, 32)
        raw = torch.randn(2, 1, 32)
        alpha_one = torch.randn(2, 1, 32)
        power, logits = model(encoded, raw_input=raw, alpha_one=alpha_one)

        self.assertEqual(tuple(power.shape), (2, 32, 3))
        self.assertEqual(tuple(logits.shape), (2, 32, 3))
        self.assertEqual(tuple(model.last_expert_gates.shape), (2, 3, 2, 32))
        (power.square().mean() + logits.square().mean()).backward()
        for weight in weights:
            self.assertIsNotNone(weight.grad)
            self.assertGreater(float(weight.grad.abs().sum()), 0.0)

    def test_private_local_expert_config_is_opt_in(self) -> None:
        shared = multinilm_config({
            "hidden_channels": 8,
            "dual_expert": {"enabled": True},
        })
        private = multinilm_config({
            "hidden_channels": 8,
            "dual_expert": {
                "enabled": True,
                "share_local_expert": False,
            },
        })
        self.assertTrue(shared.dual_expert_share_local)
        self.assertFalse(private.dual_expert_share_local)

    def test_old_configuration_keeps_single_context_path(self) -> None:
        cfg = multinilm_config({"hidden_channels": 8})
        self.assertFalse(cfg.dual_expert_enabled)

        model = MultiNILM(
            input_channels=1,
            num_appliances=2,
            output_length=24,
            hidden_channels=8,
            num_blocks=1,
            kernel_size=3,
            temporal_dropout=0.0,
            head_dropout=0.0,
        )
        power, logits = model(torch.randn(2, 24))

        self.assertEqual(tuple(power.shape), (2, 24, 2))
        self.assertEqual(tuple(logits.shape), (2, 24, 2))
        self.assertIsNone(model.last_expert_gates)

    def test_adapter_logs_local_gate_for_on_and_off_samples(self) -> None:
        adapter = object.__new__(MultiNILMAdapter)
        adapter.cfg = {"appliances": ["first", "second"]}
        adapter.model_cfg = {"evaluation": {"pred_on_source": "state_head"}}
        model = build_multinilm_fractional(
            {
                "hidden_channels": 8,
                "num_blocks": 1,
                "kernel_size": 3,
                "temporal_dropout": 0.0,
                "head_dropout": 0.0,
                "fractional": {"k": 1, "include_raw": True, "memory": 4},
                "dual_expert": {
                    "enabled": True,
                    "local_channels": 4,
                    "dropout": 0.0,
                    "gate_hidden_channels": 4,
                },
            },
            num_appliances=2,
            output_length=24,
            appliance_off_norm=[0.0, 0.0],
        )

        def fake_loss(power, logits, true_power, true_state, **kwargs):
            loss = power.square().mean() + logits.square().mean()
            per_app = power.square().mean(dim=(0, 1))
            return SimpleNamespace(
                loss=loss,
                loss_power=power.square().mean(),
                loss_state=logits.square().mean(),
                loss_state_term=logits.square().mean(),
                loss_energy_relative=loss * 0.0,
                mae=(power - true_power).abs().mean(),
                loss_power_per_appliance=per_app,
                loss_state_per_appliance=logits.square().mean(dim=(0, 1)),
            )

        x = torch.randn(2, 24)
        y = torch.zeros(2, 24, 2)
        z = torch.zeros(2, 24, 2)
        z[:, 5:10, :] = 1
        output = adapter.step(model, fake_loss, (x, y, z))

        for appliance in adapter.cfg["appliances"]:
            self.assertIn(f"gate_local_{appliance}", output.logs)
            self.assertIn(f"gate_local_on_{appliance}", output.logs)
            self.assertIn(f"gate_local_off_{appliance}", output.logs)

    def test_adapter_adds_paired_background_consistency(self) -> None:
        adapter = object.__new__(MultiNILMAdapter)
        adapter.cfg = {"appliances": ["first", "second"]}
        adapter.model_cfg = {
            "evaluation": {"pred_on_source": "state_head"},
            "training": {
                "background_consistency": {"enabled": True, "weight": 0.1}
            },
        }
        model = MultiNILM(
            input_channels=1,
            num_appliances=2,
            output_length=24,
            hidden_channels=8,
            num_blocks=1,
            kernel_size=3,
            temporal_dropout=0.0,
            head_dropout=0.0,
            gate_mode="soft",
        )

        def fake_loss(power, logits, true_power, true_state, **kwargs):
            power_loss = power.square().mean()
            state_loss = logits.square().mean()
            loss = power_loss + state_loss
            return SimpleNamespace(
                loss=loss,
                loss_power=power_loss,
                loss_state=state_loss,
                loss_state_term=state_loss,
                loss_energy_relative=loss * 0.0,
                mae=(power - true_power).abs().mean(),
                loss_power_per_appliance=power.square().mean(dim=(0, 1)),
                loss_state_per_appliance=logits.square().mean(dim=(0, 1)),
            )

        x_real = torch.randn(2, 24)
        x_swapped = x_real + torch.linspace(0.0, 1.0, 24)
        y = torch.zeros(2, 24, 2)
        z = torch.zeros(2, 24, 2)
        output = adapter.step(model, fake_loss, ((x_real, x_swapped), y, z))

        self.assertIn("loss_supervised", output.logs)
        self.assertIn("loss_background_consistency", output.logs)
        self.assertIn("loss_background_consistency_state", output.logs)
        self.assertIn("loss_background_consistency_power", output.logs)
        self.assertGreaterEqual(output.logs["loss_background_consistency"], 0.0)

        output.loss.backward()
        self.assertIsNotNone(model.appliance_heads[0].state_head.weight.grad)
        self.assertIsNotNone(model.appliance_heads[0].power_head.weight.grad)


if __name__ == "__main__":
    unittest.main()

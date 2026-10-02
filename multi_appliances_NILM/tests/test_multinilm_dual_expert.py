from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from model.MultiNILM import (
    MultiNILM,
    MultiNILMAdapter,
    build_multinilm_fractional,
    multinilm_config,
)


class MultiNILMDualExpertTests(unittest.TestCase):
    def test_dual_expert_shapes_gate_initialization_and_relation(self) -> None:
        model = MultiNILM(
            input_channels=4,
            num_appliances=3,
            output_length=64,
            hidden_channels=16,
            channel_schedule=[8, 16],
            num_blocks=2,
            kernel_size=5,
            dropout=0.0,
            dual_expert_enabled=True,
            dual_expert_local_channels=8,
            dual_expert_local_kernel_size=5,
            dual_expert_local_dilations=[1, 2, 4],
            dual_expert_gate_hidden_channels=8,
            dual_expert_gate_initial_local_weight=0.1,
            cross_appliance_enabled=True,
            cross_appliance_mode="relation_attention",
            cross_appliance_attention_channels=8,
        )
        encoded = torch.randn(2, 4, 64)
        raw = torch.randn(2, 1, 64)

        power, logits = model(encoded, raw_input=raw)

        self.assertEqual(tuple(power.shape), (2, 64, 3))
        self.assertEqual(tuple(logits.shape), (2, 64, 3))
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

    def test_fractional_wrapper_passes_raw_signal_to_local_expert(self) -> None:
        architecture = {
            "hidden_channels": 16,
            "channel_schedule": [8, 16],
            "num_blocks": 2,
            "kernel_size": 5,
            "dropout": 0.0,
            "fractional": {
                "k": 2,
                "include_raw": True,
                "include_delta": True,
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
                "gate_hidden_channels": 8,
                "gate_initial_local_weight": 0.1,
            },
            "cross_appliance": {
                "enabled": True,
                "mode": "relation_attention",
                "attention_channels": 8,
                "residual_scale": 0.25,
            },
        }
        model = build_multinilm_fractional(
            architecture,
            num_appliances=2,
            output_length=32,
            appliance_off_norm=[0.0, 0.0],
        )

        power, logits = model(torch.randn(2, 32))

        self.assertEqual(tuple(power.shape), (2, 32, 2))
        self.assertEqual(tuple(logits.shape), (2, 32, 2))
        self.assertEqual(tuple(model.last_expert_gates.shape), (2, 2, 2, 32))

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
            dropout=0.0,
        )
        power, logits = model(torch.randn(2, 24))

        self.assertEqual(tuple(power.shape), (2, 24, 2))
        self.assertEqual(tuple(logits.shape), (2, 24, 2))
        self.assertIsNone(model.last_expert_gates)

    def test_adapter_logs_local_gate_for_on_and_off_samples(self) -> None:
        adapter = object.__new__(MultiNILMAdapter)
        adapter.cfg = {"appliances": ["first", "second"]}
        adapter.model_cfg = {"evaluation": {"pred_on_source": "state_head"}}
        model = MultiNILM(
            input_channels=1,
            num_appliances=2,
            output_length=24,
            hidden_channels=8,
            num_blocks=1,
            kernel_size=3,
            dropout=0.0,
            dual_expert_enabled=True,
            dual_expert_local_channels=4,
            dual_expert_gate_hidden_channels=4,
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
                loss_state_smooth=None,
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


if __name__ == "__main__":
    unittest.main()

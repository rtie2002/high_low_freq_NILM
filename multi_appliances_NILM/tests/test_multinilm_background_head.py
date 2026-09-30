from __future__ import annotations

import unittest

import torch

from model.MultiNILM import MultiNILM, build_multinilm_fractional
from model.MultiNILM_loss import MultiNILMLoss


class MultiNILMBackgroundHeadTests(unittest.TestCase):
    def test_background_head_is_separate_from_five_appliance_outputs(self) -> None:
        model = MultiNILM(
            input_channels=1,
            num_appliances=5,
            output_length=32,
            hidden_channels=16,
            channel_schedule=[8, 16],
            num_blocks=1,
            kernel_size=3,
            dropout=0.0,
            background_head_enabled=True,
            cross_appliance_enabled=True,
            cross_appliance_mode="relation_attention",
            cross_appliance_attention_channels=8,
        )

        power, logits = model(torch.randn(2, 32))

        self.assertEqual(tuple(power.shape), (2, 32, 5))
        self.assertEqual(tuple(logits.shape), (2, 32, 5))
        self.assertEqual(tuple(model.last_background_pred.shape), (2, 32, 1))

        model.last_background_pred.square().mean().backward()
        self.assertIsNotNone(model.background_head.weight.grad)
        self.assertIsNone(model.cross_appliance_distill.query.weight.grad)

    def test_fractional_wrapper_exposes_background_prediction(self) -> None:
        model = build_multinilm_fractional(
            {
                "hidden_channels": 16,
                "channel_schedule": [8, 16],
                "num_blocks": 1,
                "kernel_size": 3,
                "dropout": 0.0,
                "fractional": {"k": 2, "include_raw": True},
                "background_head": {"enabled": True},
            },
            num_appliances=2,
            output_length=24,
            appliance_off_norm=[0.0, 0.0],
        )

        power, logits = model(torch.randn(2, 24))

        self.assertEqual(tuple(power.shape), (2, 24, 2))
        self.assertEqual(tuple(logits.shape), (2, 24, 2))
        self.assertEqual(tuple(model.last_background_pred.shape), (2, 24, 1))

    def test_exact_decomposition_has_zero_auxiliary_loss(self) -> None:
        loss_fn = MultiNILMLoss(
            lambda_state=0.0,
            power_scale=[1.0, 1.0],
            target_mean=[0.0, 0.0],
            task_balance="none",
            background_weight=0.1,
            reconstruction_weight=0.05,
            background_huber_beta=0.1,
            aggregate_mean=0.0,
            aggregate_scale=1.0,
        )
        power_true = torch.tensor([[[2.0, 3.0], [1.0, 4.0]]])
        background_true = torch.tensor([[[5.0], [6.0]]])
        aggregate_true = power_true.sum(dim=-1, keepdim=True) + background_true
        state_true = torch.zeros_like(power_true)
        state_logits = torch.zeros_like(power_true)

        out = loss_fn(
            power_true,
            state_logits,
            power_true,
            state_true,
            background_pred=background_true,
            aggregate_true=aggregate_true,
        )

        torch.testing.assert_close(out.loss_background, torch.tensor(0.0))
        torch.testing.assert_close(out.loss_reconstruction, torch.tensor(0.0))
        torch.testing.assert_close(out.loss, torch.tensor(0.0))

    def test_background_error_contributes_with_configured_weights(self) -> None:
        loss_fn = MultiNILMLoss(
            lambda_state=0.0,
            power_scale=[1.0],
            target_mean=[0.0],
            task_balance="none",
            background_weight=0.1,
            reconstruction_weight=0.05,
            background_huber_beta=0.1,
        )
        power = torch.zeros(1, 2, 1)
        state = torch.zeros_like(power)
        aggregate = torch.zeros(1, 2, 1)
        background_pred = torch.ones(1, 2, 1)

        out = loss_fn(
            power,
            state,
            power,
            state,
            background_pred=background_pred,
            aggregate_true=aggregate,
        )

        expected_smooth_l1 = torch.tensor(0.95)
        torch.testing.assert_close(out.loss_background, expected_smooth_l1)
        torch.testing.assert_close(out.loss_reconstruction, expected_smooth_l1)
        torch.testing.assert_close(out.loss, 0.15 * expected_smooth_l1)


if __name__ == "__main__":
    unittest.main()

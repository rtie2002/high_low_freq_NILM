from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from model.MultiNILM_loss import MultiNILMLoss


class MultiNILMHardNegativeTests(unittest.TestCase):
    def _loss(self, **overrides) -> MultiNILMLoss:
        kwargs = {
            "lambda_state": 0.0,
            "task_balance": "none",
            "power_scale": [1.0, 1.0, 1.0],
            "hard_negative_weight": 0.05,
            "hard_negative_fraction": 0.25,
            "hard_negative_appliance_indices": [1, 2],
        }
        kwargs.update(overrides)
        return MultiNILMLoss(**kwargs)

    def test_selects_top_fraction_only_for_configured_appliances(self) -> None:
        loss_fn = self._loss()
        logits = torch.tensor(
            [[[9.0, -3.0, 0.0], [8.0, 2.0, 1.0], [7.0, -1.0, 4.0], [6.0, 0.5, -2.0]]]
        )
        states = torch.zeros_like(logits)

        per_app = loss_fn._hard_negative_loss(logits, states)

        self.assertEqual(float(per_app[0]), 0.0)
        torch.testing.assert_close(per_app[1], F.softplus(torch.tensor(2.0)))
        torch.testing.assert_close(per_app[2], F.softplus(torch.tensor(4.0)))

    def test_temporal_guard_excludes_off_points_next_to_true_on(self) -> None:
        loss_fn = self._loss(
            hard_negative_fraction=0.5,
            hard_negative_appliance_indices=[1],
            hard_negative_exclusion_samples=1,
        )
        logits = torch.zeros(1, 5, 3)
        logits[0, :, 1] = torch.tensor([1.0, 10.0, 0.0, 9.0, 2.0])
        states = torch.zeros_like(logits)
        states[0, 2, 1] = 1.0

        per_app = loss_fn._hard_negative_loss(logits, states)

        # Indices 1..3 are protected; among eligible indices 0 and 4, top 50%
        # selects only index 4 (logit 2), not the misleading logits 10 or 9.
        torch.testing.assert_close(per_app[1], F.softplus(torch.tensor(2.0)))

    def test_gradient_reaches_only_mined_logits(self) -> None:
        loss_fn = self._loss(hard_negative_appliance_indices=[1])
        logits = torch.tensor(
            [[[9.0, -3.0, 0.0], [8.0, 2.0, 1.0], [7.0, -1.0, 4.0], [6.0, 0.5, -2.0]]],
            requires_grad=True,
        )
        states = torch.zeros_like(logits)

        loss_fn._hard_negative_loss(logits, states).sum().backward()

        self.assertEqual(int(torch.count_nonzero(logits.grad[..., 0])), 0)
        self.assertEqual(int(torch.count_nonzero(logits.grad[..., 1])), 1)
        self.assertGreater(float(logits.grad[0, 1, 1]), 0.0)
        self.assertEqual(int(torch.count_nonzero(logits.grad[..., 2])), 0)

    def test_warmup_and_ramp_schedule(self) -> None:
        loss_fn = self._loss(
            hard_negative_weight=0.05,
            hard_negative_warmup_epochs=20,
            hard_negative_ramp_epochs=10,
        )

        expected = {1: 0.0, 20: 0.0, 21: 0.005, 25: 0.025, 30: 0.05, 80: 0.05}
        for epoch, weight in expected.items():
            loss_fn.set_epoch(epoch)
            self.assertAlmostEqual(loss_fn._hard_negative_effective_weight(), weight)

    def test_fixed_term_does_not_change_existing_balanced_state_term(self) -> None:
        common = {
            "lambda_state": 0.8,
            "task_balance": "equal",
            "power_scale": [1.0, 1.0, 1.0],
            "state_fp_weight": 1.0,
        }
        baseline = MultiNILMLoss(**common)
        mined = MultiNILMLoss(
            **common,
            hard_negative_weight=0.05,
            hard_negative_fraction=0.25,
            hard_negative_appliance_indices=[1, 2],
        )
        power_pred = torch.ones(1, 4, 3)
        power_true = torch.zeros_like(power_pred)
        logits = torch.tensor(
            [[[0.0, -3.0, 0.0], [0.0, 2.0, 1.0], [0.0, -1.0, 4.0], [0.0, 0.5, -2.0]]]
        )
        states = torch.zeros_like(logits)

        base_out = baseline(power_pred, logits, power_true, states)
        mined_out = mined(power_pred, logits, power_true, states)

        torch.testing.assert_close(mined_out.loss_state, base_out.loss_state)
        torch.testing.assert_close(mined_out.loss_state_term, base_out.loss_state_term)
        torch.testing.assert_close(
            mined_out.loss - base_out.loss,
            0.05 * mined_out.loss_hard_negative,
        )


if __name__ == "__main__":
    unittest.main()

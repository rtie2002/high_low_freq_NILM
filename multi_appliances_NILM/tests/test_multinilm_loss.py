from __future__ import annotations

import unittest

import torch

from model.MultiNILM_loss import MultiNILMLoss


class MultiNILMLossBalanceTests(unittest.TestCase):
    def test_per_appliance_balance_has_expected_value_and_local_scales(self) -> None:
        criterion = MultiNILMLoss(lambda_state=0.8, task_balance="per_appliance_equal")
        power_per_app = torch.tensor([2.0, 18.0], requires_grad=True)
        state_per_app = torch.tensor([4.0, 1.0], requires_grad=True)

        state_term = criterion._balanced_state_term(
            power_per_app.sum(),
            state_per_app.sum(),
            power_per_app,
            state_per_app,
        )
        state_term.backward()

        self.assertAlmostEqual(float(state_term.detach()), 16.0, places=6)
        torch.testing.assert_close(state_per_app.grad, torch.tensor([0.4, 14.4]))
        self.assertIsNone(power_per_app.grad)

    def test_global_and_per_appliance_balance_match_in_scalar_value(self) -> None:
        power_per_app = torch.tensor([2.0, 18.0])
        state_per_app = torch.tensor([4.0, 1.0])
        global_loss = MultiNILMLoss(lambda_state=0.8, task_balance="equal")
        local_loss = MultiNILMLoss(lambda_state=0.8, task_balance="per_appliance_equal")

        global_term = global_loss._balanced_state_term(
            power_per_app.sum(), state_per_app.sum()
        )
        local_term = local_loss._balanced_state_term(
            power_per_app.sum(),
            state_per_app.sum(),
            power_per_app,
            state_per_app,
        )

        torch.testing.assert_close(global_term, local_term)

    def test_clipped_per_appliance_balance_limits_extreme_state_gradients(self) -> None:
        criterion = MultiNILMLoss(
            lambda_state=0.8,
            task_balance="per_appliance_clipped",
            task_balance_ratio_limit=2.0,
        )
        power_per_app = torch.tensor([2.0, 18.0], requires_grad=True)
        state_per_app = torch.tensor([4.0, 1.0], requires_grad=True)

        state_term = criterion._balanced_state_term(
            power_per_app.sum(),
            state_per_app.sum(),
            power_per_app,
            state_per_app,
        )
        state_term.backward()

        # global ratio = 4; local ratios [0.5, 18] are clipped to [2, 8].
        self.assertAlmostEqual(float(state_term.detach()), 12.8, places=6)
        torch.testing.assert_close(state_per_app.grad, torch.tensor([1.6, 6.4]))
        self.assertIsNone(power_per_app.grad)


if __name__ == "__main__":
    unittest.main()

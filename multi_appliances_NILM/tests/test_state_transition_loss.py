from __future__ import annotations

import unittest

import torch

from model.MultiNILM_loss import MultiNILMLoss


class StateTransitionLossTest(unittest.TestCase):
    def test_correct_boundary_has_lower_loss_than_wrong_boundary(self) -> None:
        target = torch.tensor([[[0.0], [0.0], [1.0], [1.0]]])
        correct = torch.tensor([[[-6.0], [-6.0], [6.0], [6.0]]])
        late = torch.tensor([[[-6.0], [-6.0], [-6.0], [6.0]]])
        loss_fn = MultiNILMLoss(
            task_balance="none",
            state_transition_weight=[1.0],
            state_steady_weight=0.1,
        )

        correct_loss = loss_fn._per_appliance_state_loss(correct, target)[0]
        late_loss = loss_fn._per_appliance_state_loss(late, target)[0]

        self.assertLess(float(correct_loss), float(late_loss))

    def test_zero_weight_preserves_plain_state_loss(self) -> None:
        target = torch.tensor([[[0.0], [1.0], [1.0]]])
        logits = torch.tensor([[[-1.0], [1.0], [0.5]]])
        loss_fn = MultiNILMLoss(
            task_balance="none",
            state_transition_weight=[0.0],
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[..., 0], target[..., 0]
        )
        actual = loss_fn._per_appliance_state_loss(logits, target)[0]
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()

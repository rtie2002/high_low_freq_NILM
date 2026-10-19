"""Loss for the TransferNILM multiappliance-task network.

Sun et al., Eq. (16), define the objective as the sum over appliance power
mean-squared error and state binary cross-entropy. The released trainer uses
PyTorch's mean reductions over batch, time, and appliance dimensions and sets
the state coefficient to one. This implementation follows that released
training code while retaining ``lambda_state`` to expose the paper's symbol.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TransferNILMLossOutput:
    """Scalar objective and its two reported components."""

    loss: torch.Tensor
    loss_power: torch.Tensor
    loss_state: torch.Tensor
    mae: torch.Tensor


class TransferNILMLoss(nn.Module):
    """Global mean MSE plus weighted global mean BCE."""

    def __init__(
        self,
        lambda_state: float = 1.0,
        power_scale: float | list[float] | torch.Tensor = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_state = float(lambda_state)
        self.register_buffer(
            "power_scale", torch.as_tensor(power_scale, dtype=torch.float32)
        )

    def forward(
        self,
        power_pred: torch.Tensor,
        state_prob: torch.Tensor,
        power_true: torch.Tensor,
        state_true: torch.Tensor,
    ) -> TransferNILMLossOutput:
        power_true = power_true.to(dtype=power_pred.dtype)
        state_true = state_true.to(dtype=state_prob.dtype)

        # The public implementation optimizes nn.MSELoss() + nn.BCELoss().
        loss_power = F.mse_loss(power_pred, power_true)
        loss_state = F.binary_cross_entropy(state_prob, state_true)
        loss = loss_power + self.lambda_state * loss_state

        scale = self.power_scale.to(device=power_pred.device, dtype=power_pred.dtype)
        mae = torch.mean(torch.abs((power_pred - power_true) * scale))
        return TransferNILMLossOutput(
            loss=loss,
            loss_power=loss_power,
            loss_state=loss_state,
            mae=mae,
        )

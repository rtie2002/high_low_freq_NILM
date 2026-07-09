"""Transfer-learning baseline loss — global mean MSE + global mean BCE.

Matches NILM_model/baseline/transfer_learning_multi-appliance/trainer.py:

    loss = MSE(power_pred, power_true) + BCE(state_prob, state_true)

The model outputs sigmoid state probabilities, so BCELoss is used (not BCEWithLogits).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TransferNILMLossOutput:
    loss: torch.Tensor
    loss_power: torch.Tensor
    loss_state: torch.Tensor
    mae: torch.Tensor


class TransferNILMLoss(nn.Module):
    def __init__(self, power_scale: float | list[float] | torch.Tensor = 1.0):
        super().__init__()
        self.register_buffer("power_scale", torch.as_tensor(power_scale, dtype=torch.float32))
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(
        self,
        power_pred: torch.Tensor,
        state_prob: torch.Tensor,
        power_true: torch.Tensor,
        state_true: torch.Tensor,
    ) -> TransferNILMLossOutput:
        device_type = power_pred.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            power_pred_f = power_pred.float()
            state_prob_f = state_prob.float()
            power_true_f = power_true.float()
            state_true_f = state_true.float()
            loss_power = self.mse(power_pred_f, power_true_f)
            loss_state = self.bce(state_prob_f, state_true_f)
        loss = loss_power + loss_state

        scale = self.power_scale.to(device=power_pred.device, dtype=power_pred.dtype)
        mae = torch.mean(torch.abs((power_pred.float() - power_true.float()) * scale))
        return TransferNILMLossOutput(
            loss=loss,
            loss_power=loss_power,
            loss_state=loss_state,
            mae=mae,
        )

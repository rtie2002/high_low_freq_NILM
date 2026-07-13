"""Transfer-learning baseline loss — global mean MSE + global mean BCE.

Matches NILM_model/baseline/transfer_learning_multi-appliance/trainer.py:

    loss = MSE(power_pred, power_true) + BCE(state_prob, state_true)

The model outputs sigmoid state probabilities, so BCELoss is used (not BCEWithLogits).

Gated power uses OFF-norm blend in CNNApplianceHead (see docs/multinilm_off_norm_gate.md).
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
        power_true = power_true.to(dtype=power_pred.dtype)
        state_true = state_true.to(dtype=state_prob.dtype)
        device_type = power_pred.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            loss_power = self.mse(power_pred, power_true)
            loss_state = self.bce(state_prob, state_true)
        loss = loss_power + loss_state

        scale = self.power_scale.to(device=power_pred.device, dtype=power_pred.dtype)
        mae = torch.mean(torch.abs((power_pred - power_true) * scale))
        return TransferNILMLossOutput(
            loss=loss,
            loss_power=loss_power,
            loss_state=loss_state,
            mae=mae,
        )

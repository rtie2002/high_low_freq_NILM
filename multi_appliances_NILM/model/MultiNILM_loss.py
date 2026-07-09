"""MultiNILM loss.

Author-style implementation:

    total_loss = power_loss + lambda_state * state_loss

where:

    power_loss = MSE(power_pred, power_true)
    state_loss = BCE(state_pred, state_true)

MultiNILM outputs raw state logits, so we use BCEWithLogitsLoss instead of
BCELoss. This is the same BCE idea, but numerically safer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MultiNILMLossOutput:
    loss: torch.Tensor
    loss_power: torch.Tensor
    loss_state: torch.Tensor
    mae: torch.Tensor


class MultiNILMLoss(nn.Module):
    """MSE power loss + BCE ON/OFF loss.

    Formula:

        L = L_power + lambda_state * L_state

        L_power = mean((y_true - y_pred)^2)

        L_state = -mean(
            o_true * log(o_pred)
            + (1 - o_true) * log(1 - o_pred)
        )

    Symbol meaning:

        y_true / power_true   = true appliance power
        y_pred / power_pred   = predicted appliance power
        o_true / state_true   = true ON/OFF state, 0 or 1
        o_pred                = predicted ON probability
        state_logits          = raw model output before sigmoid
        lambda_state          = weight for ON/OFF loss
    """

    def __init__(
        self,
        lambda_state: float = 0.1,
        pos_weight: torch.Tensor | list[float] | None = None,
        power_scale: float | list[float] | torch.Tensor = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_state = float(lambda_state)
        self.register_buffer("power_scale", torch.as_tensor(power_scale, dtype=torch.float32))

        if pos_weight is not None:
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)

        # L_power = mean((power_true - power_pred)^2)
        self.mse = nn.MSELoss()

        # L_state = BCE(state prediction, true ON/OFF state)
        #
        # The paper formula writes BCE using probability o_pred:
        #   o_pred = sigmoid(state_logits)
        #
        # Instead of doing sigmoid manually and then BCELoss, PyTorch recommends
        # BCEWithLogitsLoss. It combines sigmoid + BCE in one stable operation.
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        power_pred: torch.Tensor,
        state_logits: torch.Tensor,
        power_true: torch.Tensor,
        state_true: torch.Tensor,
    ) -> MultiNILMLossOutput:
        power_pred = power_pred.float()
        state_logits = state_logits.float()
        power_true = power_true.float()
        state_true = state_true.float()

        # Power regression term:
        #   L_power = mean((y_true - y_pred)^2)
        #
        # Shape of power_pred and power_true:
        #   (batch, output_length, num_appliances)
        #
        # nn.MSELoss() averages over batch, time, and appliances.
        loss_power = self.mse(power_pred, power_true)

        # State classification term:
        #   L_state = BCE(o_pred, o_true)
        #
        # state_logits are raw model outputs:
        #   state probability = sigmoid(state_logits)
        #
        # state_true contains binary labels:
        #   1 = appliance ON
        #   0 = appliance OFF
        loss_state = self.bce(state_logits, state_true)

        # Final multitask loss:
        #   L = L_power + lambda_state * L_state
        loss = loss_power + self.lambda_state * loss_state

        # MAE is only for logging/monitoring.
        # It is not used to update the model in this loss.
        #
        # If targets are normalized by a scale, power_scale converts MAE back
        # toward real power units for easier interpretation.
        scale = self.power_scale.to(device=power_pred.device, dtype=power_pred.dtype)
        mae = torch.mean(torch.abs((power_pred - power_true) * scale))

        return MultiNILMLossOutput(
            loss=loss,
            loss_power=loss_power,
            loss_state=loss_state,
            mae=mae,
        )

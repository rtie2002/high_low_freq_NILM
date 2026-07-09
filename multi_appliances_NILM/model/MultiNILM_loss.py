"""MultiNILM loss.

Paper-style multitask loss (equation 16):

    L = sum_{i=1}^{n} ( L_power^i + lambda * L_state^i )

where each appliance i has its own power MSE and state BCE, computed over
batch and time, then summed across appliances.

MultiNILM outputs raw state logits, so we use BCEWithLogitsLoss instead of
BCELoss. This is the same BCE idea, but numerically safer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultiNILMLossOutput:
    loss: torch.Tensor
    loss_power: torch.Tensor
    loss_state: torch.Tensor
    mae: torch.Tensor
    loss_power_per_appliance: torch.Tensor
    loss_state_per_appliance: torch.Tensor


class MultiNILMLoss(nn.Module):
    """Per-appliance MSE power loss + per-appliance BCE ON/OFF loss.

    Formula:

        L = sum_i L_power^i + lambda_state * sum_i L_state^i

        L_power^i = mean_{batch,time} (y_true^i - y_pred^i)^2

        L_state^i = mean_{batch,time} BCEWithLogits(o_pred^i, o_true^i)

    This matches the paper structure where every appliance contributes its own
    regression and classification terms before the final sum.
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
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def _per_appliance_power_loss(
        self,
        power_pred: torch.Tensor,
        power_true: torch.Tensor,
    ) -> torch.Tensor:
        # Shape: (batch, time, appliances) -> one MSE per appliance.
        return torch.mean((power_pred - power_true) ** 2, dim=(0, 1))

    def _per_appliance_state_loss(
        self,
        state_logits: torch.Tensor,
        state_true: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        n_apps = state_logits.shape[-1]
        for app_i in range(n_apps):
            logits_i = state_logits[..., app_i]
            target_i = state_true[..., app_i]
            weight_i = None
            if self.pos_weight is not None:
                weight_i = self.pos_weight[app_i] if self.pos_weight.ndim > 0 else self.pos_weight
            losses.append(
                F.binary_cross_entropy_with_logits(
                    logits_i,
                    target_i,
                    pos_weight=weight_i,
                )
            )
        return torch.stack(losses)

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

        loss_power_per_app = self._per_appliance_power_loss(power_pred, power_true)
        loss_state_per_app = self._per_appliance_state_loss(state_logits, state_true)

        # Paper equation (16a): sum appliance losses, not one global average.
        loss_power = loss_power_per_app.sum()
        loss_state = loss_state_per_app.sum()
        loss = loss_power + self.lambda_state * loss_state

        scale = self.power_scale.to(device=power_pred.device, dtype=power_pred.dtype)
        if scale.ndim > 0:
            mae_per_app = torch.mean(torch.abs(power_pred - power_true), dim=(0, 1)) * scale
            mae = mae_per_app.mean()
        else:
            mae = torch.mean(torch.abs((power_pred - power_true) * scale))

        return MultiNILMLossOutput(
            loss=loss,
            loss_power=loss_power,
            loss_state=loss_state,
            mae=mae,
            loss_power_per_appliance=loss_power_per_app.detach(),
            loss_state_per_appliance=loss_state_per_app.detach(),
        )

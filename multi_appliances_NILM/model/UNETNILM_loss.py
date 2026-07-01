"""UNet-NILM loss functions (Faustine et al., BuildSys 2020).

State head:  NLL over (B, 2, M) logits vs (B, M) ON/OFF labels  — Eq. 5
Power head:  pinball / quantile loss over (B, Q, M) vs (B, M)   — Eq. 3–4
Total:       loss_state + loss_power                            — Eq. 6

Hyperparameters (quantiles, etc.) come from model/UNETNILM.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileLoss(nn.Module):
    """Multi-target pinball loss (paper Eq. 3–4)."""

    def __init__(self, quantiles: list[float] | None = None):
        super().__init__()
        self.quantiles = quantiles or [0.0025, 0.1, 0.5, 0.9, 0.975]

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  (B, Q, M) predicted quantiles
            targets: (B, M)    observed appliance power
        """
        targets = targets.unsqueeze(1).expand_as(inputs)
        q = torch.tensor(self.quantiles, dtype=inputs.dtype, device=inputs.device)
        error = (targets - inputs).permute(0, 2, 1)
        loss = torch.max(q * error, (q - 1) * error)
        return loss.mean()


@dataclass
class UNETNILMLossOutput:
    loss: torch.Tensor
    loss_state: torch.Tensor
    loss_power: torch.Tensor
    mae: torch.Tensor


class UNETNILMLoss(nn.Module):
    """
    Combined UNet-NILM training objective matching the reference implementation.

    Expects model outputs from UNETNiLM.forward():
        states_logits: (B, 2, M)
        power_logits:  (B, Q, M) when n_quantiles > 1, else (B, M)
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        single_appliance: bool = False,
    ):
        super().__init__()
        self.quantiles = quantiles or [0.0025, 0.1, 0.5, 0.9, 0.975]
        self.single_appliance = single_appliance
        self.quantile_loss = QuantileLoss(self.quantiles)

    def _prepare_targets(
        self,
        power: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.single_appliance:
            power = power.unsqueeze(-1)
            state = state.unsqueeze(-1)
        return power, state

    def state_loss(self, states_logits: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return F.nll_loss(F.log_softmax(states_logits, dim=1), state.long())

    def power_loss(self, power_logits: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
        if power_logits.dim() == 3 and len(self.quantiles) > 1:
            return self.quantile_loss(power_logits, power)
        return F.mse_loss(power_logits, power)

    def forward(
        self,
        states_logits: torch.Tensor,
        power_logits: torch.Tensor,
        power: torch.Tensor,
        state: torch.Tensor,
    ) -> UNETNILMLossOutput:
        power, state = self._prepare_targets(power, state)

        loss_state = self.state_loss(states_logits, state)
        loss_power = self.power_loss(power_logits, power)
        loss = loss_state + loss_power

        if power_logits.dim() == 3 and len(self.quantiles) > 1:
            mae = F.l1_loss(power_logits, power.unsqueeze(1).expand_as(power_logits))
        else:
            mae = F.l1_loss(power_logits, power)

        return UNETNILMLossOutput(
            loss=loss,
            loss_state=loss_state,
            loss_power=loss_power,
            mae=mae,
        )


__all__ = ["QuantileLoss", "UNETNILMLoss", "UNETNILMLossOutput"]

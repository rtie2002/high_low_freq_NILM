"""Loss functions from the original UNet-NILM paper.

Paper:
    A. Faustine et al., "UNet-NILM: A Deep Neural Network for Multi-tasks
    Appliances State Detection and Power Estimation in NILM", NILM 2020.
    DOI: 10.1145/3427771.3427859

The paper optimizes two equally weighted objectives (Equation 8):

    total_loss = state_cross_entropy + power_pinball_loss

State logits have shape ``(batch, 2, appliances)`` because every appliance
has two classes: OFF and ON. Power predictions have shape
``(batch, quantiles, appliances)`` and estimate the conditional power
quantiles for every appliance simultaneously.

No additional loss weighting, class weighting, gating, or appliance-specific
penalty is part of the published UNet-NILM objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


# Section 2.3 of the paper: 2.5%, 10%, 50%, 90%, and 97.5%.
# The official repository contains 0.0025 for the first value, which conflicts
# with the paper text. We follow the published paper and use 0.025.
PAPER_QUANTILES = (0.025, 0.10, 0.50, 0.90, 0.975)


@dataclass
class UNETNILMLossOutput:
    """Scalar losses returned to the shared training pipeline."""

    loss: torch.Tensor
    loss_state: torch.Tensor
    loss_power: torch.Tensor
    mae: torch.Tensor


class QuantileLoss(nn.Module):
    """Multi-target pinball loss from UNet-NILM Equations 3 and 4.

    For residual ``error = target - prediction`` and quantile ``q``:

        pinball(error, q) = max(q * error, (q - 1) * error)

    The final value is averaged over the batch, quantiles, and appliances,
    matching the mean reduction used in the authors' official implementation.
    """

    def __init__(self, quantiles: Sequence[float] = PAPER_QUANTILES) -> None:
        super().__init__()

        quantile_tensor = torch.as_tensor(tuple(quantiles), dtype=torch.float32)
        if quantile_tensor.ndim != 1 or quantile_tensor.numel() == 0:
            raise ValueError("quantiles must be a non-empty one-dimensional sequence")
        if torch.any((quantile_tensor <= 0) | (quantile_tensor >= 1)):
            raise ValueError("every quantile must be strictly between 0 and 1")

        self.register_buffer("quantiles", quantile_tensor)

    def forward(
        self,
        power_quantiles: torch.Tensor,
        power_target: torch.Tensor,
    ) -> torch.Tensor:
        """Return mean pinball loss.

        Args:
            power_quantiles: Predicted power, shape ``(B, Q, A)``.
            power_target: Ground-truth power, shape ``(B, A)``.
        """
        if power_quantiles.ndim != 3:
            raise ValueError(
                "power_quantiles must have shape (batch, quantiles, appliances)"
            )
        if power_target.ndim != 2:
            raise ValueError("power_target must have shape (batch, appliances)")
        if power_quantiles.shape[0] != power_target.shape[0]:
            raise ValueError("power prediction and target batch sizes do not match")
        if power_quantiles.shape[2] != power_target.shape[1]:
            raise ValueError("power prediction and target appliance counts do not match")
        if power_quantiles.shape[1] != self.quantiles.numel():
            raise ValueError(
                f"model predicts {power_quantiles.shape[1]} quantiles, "
                f"but the loss expects {self.quantiles.numel()}"
            )

        target = power_target.to(
            device=power_quantiles.device,
            dtype=power_quantiles.dtype,
        ).unsqueeze(1)
        quantiles = self.quantiles.to(
            device=power_quantiles.device,
            dtype=power_quantiles.dtype,
        ).view(1, -1, 1)

        error = target - power_quantiles
        pinball = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
        return pinball.mean()


class UNETNILMLoss(nn.Module):
    """Original UNet-NILM joint state-and-power objective.

    Expected tensors follow the authors' model output layout:

    - ``state_logits``: ``(B, 2, A)``
    - ``power_quantiles``: ``(B, Q, A)``
    - ``state_target``: ``(B, A)``, with integer values 0 (OFF) or 1 (ON)
    - ``power_target``: ``(B, A)``

    The reported MAE uses the median (q=0.5) prediction, as in the paper's
    inference and evaluation procedure. MAE is a metric only; it is not added
    to the training loss.
    """

    def __init__(self, quantiles: Sequence[float] = PAPER_QUANTILES) -> None:
        super().__init__()
        quantiles = tuple(float(value) for value in quantiles)
        self.power_loss = QuantileLoss(quantiles)

        median_matches = [
            index for index, value in enumerate(quantiles) if abs(float(value) - 0.5) < 1e-8
        ]
        if len(median_matches) != 1:
            raise ValueError("UNet-NILM quantiles must contain exactly one median (0.5)")
        self.median_index = median_matches[0]

    def forward(
        self,
        state_logits: torch.Tensor,
        power_quantiles: torch.Tensor,
        power_target: torch.Tensor,
        state_target: torch.Tensor,
    ) -> UNETNILMLossOutput:
        if state_logits.ndim != 3 or state_logits.shape[1] != 2:
            raise ValueError("state_logits must have shape (batch, 2, appliances)")
        if state_target.ndim != 2:
            raise ValueError("state_target must have shape (batch, appliances)")
        if state_logits.shape[0] != state_target.shape[0]:
            raise ValueError("state prediction and target batch sizes do not match")
        if state_logits.shape[2] != state_target.shape[1]:
            raise ValueError("state prediction and target appliance counts do not match")

        state_target = state_target.to(device=state_logits.device)
        if torch.any((state_target != 0) & (state_target != 1)):
            raise ValueError("state_target values must be binary: 0 (OFF) or 1 (ON)")
        state_target = state_target.to(dtype=torch.long)

        # Equivalent to the official implementation:
        # F.nll_loss(F.log_softmax(state_logits, dim=1), state_target)
        loss_state = F.cross_entropy(state_logits, state_target)
        loss_power = self.power_loss(power_quantiles, power_target)
        loss = loss_state + loss_power

        median_power = power_quantiles[:, self.median_index, :]
        mae = F.l1_loss(
            median_power,
            power_target.to(device=median_power.device, dtype=median_power.dtype),
        )

        return UNETNILMLossOutput(
            loss=loss,
            loss_state=loss_state,
            loss_power=loss_power,
            mae=mae,
        )

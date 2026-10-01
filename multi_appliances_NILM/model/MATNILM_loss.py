"""MATNilm training objective.

Paper:
    J. Xiong, T. Hong, D. Zhao, and Y. Zhang, "MATNilm: Multi-Appliance-Task
    Non-Intrusive Load Monitoring With Limited Labeled Data", IEEE Trans.
    Industrial Informatics, 2024. DOI: 10.1109/TII.2023.3301026

Code:
    https://github.com/jxiong22/MATNilm

Equation (13) writes a sum of per-appliance MSE and BCE terms. The released
``main.py`` optimizes ``nn.MSELoss() + nn.BCELoss()``, which is the mean of
those terms over the batch, time, and appliances. This module follows that
training loop so the published learning rate of 0.001 has the same scale.

The regression target is the final gated power, ``p_hat * o_hat``, not the
ungated regression head. Both tensors are in the paper's divide-by-612 units.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MATNILMLossOutput:
    """Scalar losses returned to the shared training pipeline."""

    loss: torch.Tensor
    loss_power: torch.Tensor
    loss_state: torch.Tensor


class MATNILMLoss(nn.Module):
    """Mean MSE on gated power plus mean BCE on the on-off probability."""

    def forward(
        self,
        power_pred: torch.Tensor,
        state_pred: torch.Tensor,
        power_target: torch.Tensor,
        state_target: torch.Tensor,
    ) -> MATNILMLossOutput:
        """Return the official unweighted sum of the two mean losses.

        Args:
            power_pred: Gated power, shape ``(B, T, A)``, units of watts/612.
            state_pred: Sigmoid on-off probability, shape ``(B, T, A)``.
            power_target: Ground-truth power, shape ``(B, T, A)``, watts/612.
            state_target: Ground-truth on-off labels, shape ``(B, T, A)``.
        """
        loss_power = F.mse_loss(power_pred, power_target)
        loss_state = F.binary_cross_entropy(state_pred, state_target)
        return MATNILMLossOutput(
            loss=loss_power + loss_state,
            loss_power=loss_power,
            loss_state=loss_state,
        )

"""MATNILM loss — MSE (scaled power) + BCE (on/off)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MATNILMLossOutput:
    loss: torch.Tensor
    loss_power: torch.Tensor
    loss_state: torch.Tensor
    mae: torch.Tensor


class MATNILMLoss(nn.Module):
    def __init__(self, power_scale: float = 1.0):
        super().__init__()
        self.power_scale = float(power_scale)
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(
        self,
        y_pred_r: torch.Tensor,
        y_pred_c: torch.Tensor,
        y_true_r: torch.Tensor,
        y_true_c: torch.Tensor,
    ) -> MATNILMLossOutput:
        y_true_c = y_true_c.float()
        loss_r = self.mse(y_pred_r, y_true_r)
        loss_c = self.bce(y_pred_c, y_true_c)
        loss = loss_r + loss_c
        mae = torch.mean(torch.abs((y_pred_r - y_true_r) * self.power_scale))
        return MATNILMLossOutput(loss=loss, loss_power=loss_r, loss_state=loss_c, mae=mae)

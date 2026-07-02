"""MATNILM loss — MSE (scaled power) + BCELoss (on/off probabilities)."""

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
        # Author MATNILM uses sigmoid probabilities + BCELoss. PyTorch AMP refuses
        # BCELoss under autocast, so keep the same formula but compute it in FP32.
        device_type = y_pred_c.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            y_pred_r_f = y_pred_r.float()
            y_pred_c_f = y_pred_c.float()
            y_true_r_f = y_true_r.float()
            y_true_c_f = y_true_c.float()
            loss_r = self.mse(y_pred_r_f, y_true_r_f)
            loss_c = self.bce(y_pred_c_f, y_true_c_f)
        loss = loss_r + loss_c
        mae = torch.mean(torch.abs((y_pred_r.float() - y_true_r.float()) * self.power_scale))
        return MATNILMLossOutput(loss=loss, loss_power=loss_r, loss_state=loss_c, mae=mae)

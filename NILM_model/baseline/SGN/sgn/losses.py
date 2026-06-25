import torch
from torch import nn
import torch.nn.functional as F


class SGNLoss(nn.Module):
    """SGN objective: output regression loss plus on/off classification loss.

    Loss terms:
      L_output  — MSE on gated_power vs true_power  (paper Eq. 7a)
      L_on      — BCE on on_prob vs true on/off label  (paper Eq. 7c)
      L_reg_on  — MSE on raw regression vs true_power, restricted to true-ON
                  timesteps only.  NOT in the paper, but needed when training
                  cross-house: soft-gate attenuates regression gradient by
                  on_prob (≤0.5 early in training) while 99% OFF windows
                  pull regression → 0.  L_reg_on gives the regression head a
                  direct full-strength gradient on ON samples, bypassing the
                  gate.  Set reg_on_weight=0 to reproduce the paper exactly.
    """

    def __init__(
        self,
        output_weight: float = 1.0,
        on_weight: float = 1.0,
        label_smoothing: float = 0.0,
        reg_on_weight: float = 0.0,
        bce_pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.output_weight = output_weight
        self.on_weight = on_weight
        self.label_smoothing = label_smoothing
        self.reg_on_weight = float(reg_on_weight)
        self.bce_pos_weight = float(bce_pos_weight)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        target_power: torch.Tensor,
        target_on: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.label_smoothing > 0:
            smooth_on = target_on * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smooth_on = target_on

        output_error = (predictions["gated_power"] - target_power).pow(2)
        if self.bce_pos_weight > 1.0:
            pos_w = smooth_on.new_tensor(self.bce_pos_weight)
            on_error = F.binary_cross_entropy(
                predictions["on_prob"], smooth_on,
                weight=torch.where(smooth_on >= 0.5, pos_w, torch.ones_like(smooth_on)),
                reduction="none",
            )
        else:
            on_error = F.binary_cross_entropy(predictions["on_prob"], smooth_on, reduction="none")

        output_loss = output_error.mean()
        on_loss = on_error.mean()

        # ON-restricted direct regression loss
        on_mask = target_on >= 0.5
        if self.reg_on_weight > 0.0 and on_mask.any():
            reg_error = (predictions["power"] - target_power).pow(2)
            reg_on_loss = reg_error[on_mask].mean()
        else:
            reg_on_loss = output_loss.detach() * 0.0

        reduce_dims = (0,) if output_error.ndim == 2 else (0, 2)
        output_loss_per_appliance = output_error.mean(dim=reduce_dims)
        on_loss_per_appliance = on_error.mean(dim=reduce_dims)

        total = (
            self.output_weight * output_loss
            + self.on_weight * on_loss
            + self.reg_on_weight * reg_on_loss
        )
        loss_per_appliance = (
            self.output_weight * output_loss_per_appliance
            + self.on_weight * on_loss_per_appliance
        )
        return {
            "loss": total,
            "output_loss": output_loss.detach(),
            "on_loss": on_loss.detach(),
            "reg_on_loss": reg_on_loss.detach(),
            "output_loss_per_appliance": output_loss_per_appliance.detach(),
            "on_loss_per_appliance": on_loss_per_appliance.detach(),
            "loss_per_appliance": loss_per_appliance.detach(),
        }

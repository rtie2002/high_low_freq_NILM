import torch
from torch import nn
import torch.nn.functional as F


class SGNLoss(nn.Module):
    """SGN objective: output regression loss plus on/off classification loss."""

    def __init__(self, output_weight: float = 1.0, on_weight: float = 1.0) -> None:
        super().__init__()
        self.output_weight = output_weight
        self.on_weight = on_weight

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        target_power: torch.Tensor,
        target_on: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output_error = (predictions["gated_power"] - target_power).pow(2)
        on_error = F.binary_cross_entropy(predictions["on_prob"], target_on, reduction="none")
        output_loss = output_error.mean()
        on_loss = on_error.mean()
        if output_error.ndim == 2:
            output_loss_per_appliance = output_error.mean(dim=(0,))
            on_loss_per_appliance = on_error.mean(dim=(0,))
        else:
            output_loss_per_appliance = output_error.mean(dim=(0, 2))
            on_loss_per_appliance = on_error.mean(dim=(0, 2))
        total = self.output_weight * output_loss + self.on_weight * on_loss
        loss_per_appliance = (
            self.output_weight * output_loss_per_appliance
            + self.on_weight * on_loss_per_appliance
        )
        return {
            "loss": total,
            "output_loss": output_loss.detach(),
            "on_loss": on_loss.detach(),
            "output_loss_per_appliance": output_loss_per_appliance.detach(),
            "on_loss_per_appliance": on_loss_per_appliance.detach(),
            "loss_per_appliance": loss_per_appliance.detach(),
        }

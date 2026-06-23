import torch
from torch import nn


class SGNLoss(nn.Module):
    """SGN objective: output regression loss plus on/off classification loss."""

    def __init__(self, output_weight: float = 1.0, on_weight: float = 1.0) -> None:
        super().__init__()
        self.output_weight = output_weight
        self.on_weight = on_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        target_power: torch.Tensor,
        target_on: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output_loss = self.mse(predictions["gated_power"], target_power)
        on_loss = self.bce(predictions["on_prob"], target_on)
        total = self.output_weight * output_loss + self.on_weight * on_loss
        return {
            "loss": total,
            "output_loss": output_loss.detach(),
            "on_loss": on_loss.detach(),
        }


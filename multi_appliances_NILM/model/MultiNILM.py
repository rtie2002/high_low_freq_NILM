"""Simple multi-appliance NILM model.

MultiNILM is intentionally small and easy to debug:

    aggregate window -> shared temporal CNN/TCN encoder -> power + ON/OFF heads

The model predicts all appliances at the same time. It returns:

    power_pred:   (B, output_length, num_appliances)
    state_logits: (B, output_length, num_appliances)

The state head returns logits, so the matching loss should use
``BCEWithLogitsLoss``. Do not apply sigmoid before the loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ResidualTCNBlock(nn.Module):
    """Small residual temporal block with dilated Conv1d."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MultiNILM(nn.Module):
    """Basic multi-output CNN/TCN for multi-appliance NILM."""

    def __init__(
        self,
        input_channels: int = 1,
        num_appliances: int = 5,
        output_length: int = 64,
        hidden_channels: int = 64,
        num_blocks: int = 5,
        kernel_size: int = 5,
        dropout: float = 0.1,
        nonnegative_power: bool = False,
    ) -> None:
        super().__init__()
        if num_appliances <= 0:
            raise ValueError("num_appliances must be positive.")
        if output_length <= 0:
            raise ValueError("output_length must be positive.")

        self.input_channels = int(input_channels)
        self.num_appliances = int(num_appliances)
        self.output_length = int(output_length)
        self.nonnegative_power = bool(nonnegative_power)

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )

        blocks = []
        for idx in range(num_blocks):
            dilation = 2 ** idx
            blocks.append(
                ResidualTCNBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.encoder = nn.Sequential(*blocks)

        self.power_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, num_appliances, kernel_size=1),
        )
        self.state_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, num_appliances, kernel_size=1),
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[-1] == self.input_channels:
            x = x.permute(0, 2, 1)

        if x.dim() != 3:
            raise ValueError(
                "Expected input shape (B, T), (B, C, T), or (B, T, C); "
                f"got {tuple(x.shape)}."
            )
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channel(s), got {x.shape[1]}."
            )
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._prepare_input(x.float())
        features = self.encoder(self.stem(x))
        features = F.interpolate(
            features,
            size=self.output_length,
            mode="linear",
            align_corners=False,
        )

        power = self.power_head(features)
        if self.nonnegative_power:
            power = F.relu(power)
        state_logits = self.state_head(features)

        return power.permute(0, 2, 1), state_logits.permute(0, 2, 1)


@dataclass
class MultiNILMConfig:
    input_channels: int = 1
    num_appliances: int = 5
    output_length: int = 64
    hidden_channels: int = 64
    num_blocks: int = 5
    kernel_size: int = 5
    dropout: float = 0.1
    nonnegative_power: bool = False


def multinilm_config(architecture: dict[str, Any]) -> MultiNILMConfig:
    return MultiNILMConfig(
        input_channels=int(architecture.get("input_channels", architecture.get("input_size", 1))),
        num_appliances=int(architecture.get("num_appliances", 5)),
        output_length=int(architecture.get("output_length", 64)),
        hidden_channels=int(architecture.get("hidden_channels", architecture.get("hidden", 64))),
        num_blocks=int(architecture.get("num_blocks", 5)),
        kernel_size=int(architecture.get("kernel_size", 5)),
        dropout=float(architecture.get("dropout", 0.1)),
        nonnegative_power=bool(architecture.get("nonnegative_power", False)),
    )

"""Predict an aggregate waveform with a shallow 1D residual network."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Refine each time step while retaining the original input through a skip path."""

    def __init__(self, channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)

    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x + residual)


class ResNet_NILM(nn.Module):
    """Map appliance features [B, T, C] to aggregate power [B, T].

    This is a shallow adaptation of the residual refinement idea in
    Chen et al.'s convolutional sequence-to-sequence NILM model. Unlike the
    original disaggregation task, this forward model maps appliance powers to
    the total-meter waveform. Stride is always 1, so no time samples are lost.
    """

    def __init__(self, input_size, hidden_size, num_blocks, kernel_size, head_size):
        super().__init__()
        self.input_layer = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.blocks = nn.Sequential(
            *(ResidualBlock(hidden_size, kernel_size) for _ in range(num_blocks))
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.ReLU(),
            nn.Linear(head_size, 1),
        )

    def forward(self, appliance_power):
        # Conv1d expects [batch, channels, time], while the loader returns [B, T, C].
        x = appliance_power.transpose(1, 2)
        x = self.input_layer(x)
        x = self.blocks(x)
        x = x.transpose(1, 2)  # [B, T, hidden_size]
        aggregate = self.output_head(x)  # [B, T, 1]
        return aggregate.squeeze(-1)  # [B, T]

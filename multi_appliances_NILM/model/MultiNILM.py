"""MultiNILM: a clear multi-appliance NILM baseline.

This model is intentionally written in a beginner-readable style.

Task:
    Input  : aggregate power window
    Output : appliance power + appliance ON/OFF state for every appliance

Expected shapes:
    x            : (batch, input_length) or (batch, 1, input_length)
    power_pred   : (batch, output_length, num_appliances)
    state_logits : (batch, output_length, num_appliances)

Important:
    state_logits are raw logits. Use BCEWithLogitsLoss for ON/OFF loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ResidualTemporalBlock(nn.Module):
    """One temporal convolution block.

    The block keeps the same tensor shape:

        input : (batch, channels, time)
        output: (batch, channels, time)

    We add the input back to the output so the model can learn a small
    correction instead of relearning the full signal at every layer.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # Same-length padding for odd kernel sizes.
        padding = ((kernel_size - 1) * dilation) // 2

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        y = self.conv(x)
        y = self.norm(y)
        y = self.activation(y)
        y = self.dropout(y)

        return residual + y


class StagedFeatureExtractor(nn.Module):
    """Gradually widen channel depth before the shared TCN (seq2point-style).

    Example schedule [16, 32, 64]:
        Conv1d 1→16  k=7  + BN + GELU
        Conv1d 16→32 k=5  + BN + GELU
        Conv1d 32→64 k=5  + BN + GELU
    """

    def __init__(
        self,
        input_channels: int,
        channel_schedule: list[int],
        stem_kernel_size: int = 7,
        stage_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if not channel_schedule:
            raise ValueError("channel_schedule must contain at least one width.")

        layers: list[nn.Module] = []
        in_channels = int(input_channels)
        for stage_index, out_channels in enumerate(channel_schedule):
            kernel_size = stem_kernel_size if stage_index == 0 else stage_kernel_size
            padding = kernel_size // 2
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=int(out_channels),
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(int(out_channels)),
                    nn.GELU(),
                ]
            )
            in_channels = int(out_channels)
        self.stages = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(x)


class ApplianceHead(nn.Module):
    """One appliance-specific decoder on top of the shared TCN features.

    Each appliance gets its own small head instead of sharing one multi-channel
    Conv1d. This reduces competition between appliances with very different
    power scales and ON/OFF patterns.
    """

    def __init__(self, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.feature_refine = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.power_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.state_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def forward(self, shared_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_refine(shared_features)
        power_raw = self.power_head(features)
        state_logits = self.state_head(features)

        # Match transfer-learning baseline: gate power by predicted ON probability.
        # State logits stay unbounded for BCEWithLogitsLoss.
        state_prob = torch.sigmoid(state_logits)
        power = power_raw * state_prob
        return power, state_logits


class MultiNILM(nn.Module):
    """Simple CNN/TCN model for multi-appliance NILM.

    Layer-by-layer architecture:

        Input aggregate window
            Shape: (B, T) or (B, 1, T)

        1. _format_input
            Convert input to Conv1d format.
            Output: (B, 1, T)

        2. aggregate_feature_extractor
            Either staged Conv1d widening (channel_schedule) or one Conv1d jump.
            Output: (B, hidden_channels, T)

        3. temporal_encoder
            ResidualTemporalBlock x num_blocks
            Default dilation sequence: 1, 2, 4, 8, 16
            Output: (B, hidden_channels, T)

        4. temporal alignment
            Center-crop (or pad) features to output_length so each output step
            matches the same CSV timestep as the dataloader center targets.
            Output: (B, hidden_channels, output_length)

        5. appliance_heads (one per appliance)
            Each head has its own power + state 1x1 conv decoders.
            Outputs are concatenated back to:
            (B, output_length, num_appliances)

    Notes:
        - Shared TCN learns aggregate patterns once.
        - Per-appliance heads specialize power/state decoding per device.
        - Use BCEWithLogitsLoss for state_logits; apply sigmoid only for inference.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_appliances: int = 5,
        output_length: int = 64,
        hidden_channels: int = 64,
        channel_schedule: list[int] | None = None,
        stem_kernel_size: int = 7,
        stage_kernel_size: int = 5,
        num_blocks: int = 5,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_channels = int(input_channels)
        self.num_appliances = int(num_appliances)
        self.output_length = int(output_length)
        self.hidden_channels = int(hidden_channels)

        # Step 1: widen aggregate power into temporal feature maps.
        if channel_schedule:
            schedule = [int(width) for width in channel_schedule]
            if schedule[-1] != self.hidden_channels:
                raise ValueError(
                    "hidden_channels must match the last entry in channel_schedule; "
                    f"got hidden_channels={self.hidden_channels}, schedule={schedule}."
                )
            self.aggregate_feature_extractor = StagedFeatureExtractor(
                input_channels=self.input_channels,
                channel_schedule=schedule,
                stem_kernel_size=int(stem_kernel_size),
                stage_kernel_size=int(stage_kernel_size),
            )
        else:
            self.aggregate_feature_extractor = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.input_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=int(stem_kernel_size),
                    padding=int(stem_kernel_size) // 2,
                ),
                nn.BatchNorm1d(self.hidden_channels),
                nn.GELU(),
            )

        # Step 2: process temporal features with dilated convolution blocks.
        # Dilation values 1, 2, 4, 8, ... let the model see short and longer
        # appliance patterns without making the network very deep.
        temporal_blocks = []
        for block_index in range(num_blocks):
            dilation = 2 ** block_index
            temporal_blocks.append(
                ResidualTemporalBlock(
                    channels=self.hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.temporal_encoder = nn.Sequential(*temporal_blocks)

        # Step 3: one decoder head per appliance (dynamic count from experiment).
        self.appliance_heads = nn.ModuleList(
            [
                ApplianceHead(hidden_channels=self.hidden_channels, dropout=dropout)
                for _ in range(self.num_appliances)
            ]
        )

    def _format_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to Conv1d format: (batch, channels, time)."""

        # Common dataloader format: (batch, time)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Alternative format: (batch, time, channels)
        elif x.dim() == 3 and x.shape[-1] == self.input_channels:
            x = x.permute(0, 2, 1)

        if x.dim() != 3:
            raise ValueError(
                "MultiNILM expected x with shape (B, T), (B, C, T), or (B, T, C); "
                f"got {tuple(x.shape)}."
            )

        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"MultiNILM expected {self.input_channels} input channel(s), "
                f"got {x.shape[1]}."
            )

        return x.float()

    def _align_output_time(self, features: torch.Tensor) -> torch.Tensor:
        """Crop or pad features on the time axis to match label alignment."""
        time_len = features.shape[-1]
        if time_len == self.output_length:
            return features
        if time_len > self.output_length:
            offset = (time_len - self.output_length) // 2
            return features[:, :, offset : offset + self.output_length]
        pad_total = self.output_length - time_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return F.pad(features, (pad_left, pad_right))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the full MultiNILM architecture.

        Flow:
            aggregate input
            -> aggregate_feature_extractor
            -> temporal_encoder
            -> resize to output_length
            -> resize to output_length
            -> per-appliance heads
            -> return (B, output_length, num_appliances)
        """

        x = self._format_input(x)

        # Step 1-2:
        # Raw aggregate waveform -> hidden temporal representation.
        features = self.aggregate_feature_extractor(x)

        # Step 3:
        # Learn appliance-related temporal patterns from the aggregate signal.
        features = self.temporal_encoder(features)

        # Step 4:
        # Keep the same time indices as the dataloader center targets.
        # Do NOT interpolate the full window into output_length — that misaligns
        # labels and creates repeating pulse artifacts per window.
        output_features = self._align_output_time(features)

        # Step 5:
        # Each appliance head predicts one power channel and one state channel.
        # Power is gated by sigmoid(state) like the transfer-learning baseline.
        power_parts: list[torch.Tensor] = []
        state_parts: list[torch.Tensor] = []
        for head in self.appliance_heads:
            power_i, state_i = head(output_features)
            power_parts.append(power_i)
            state_parts.append(state_i)

        power_pred = torch.cat(power_parts, dim=1)
        state_logits = torch.cat(state_parts, dim=1)

        # Step 6:
        # Convert (B, num_appliances, output_length) -> (B, output_length, num_appliances)
        # so predictions match dataloader target layout.
        power_pred = power_pred.permute(0, 2, 1)
        state_logits = state_logits.permute(0, 2, 1)

        return power_pred, state_logits


@dataclass
class MultiNILMConfig:
    input_channels: int = 1
    num_appliances: int = 5
    output_length: int = 64
    hidden_channels: int = 64
    channel_schedule: list[int] | None = None
    stem_kernel_size: int = 7
    stage_kernel_size: int = 5
    num_blocks: int = 5
    kernel_size: int = 5
    dropout: float = 0.1


def multinilm_config(architecture: dict[str, Any]) -> MultiNILMConfig:
    """Read MultiNILM settings from the model YAML architecture section."""

    return MultiNILMConfig(
        input_channels=int(architecture.get("input_channels", architecture.get("input_size", 1))),
        num_appliances=int(architecture.get("num_appliances", 5)),
        output_length=int(architecture.get("output_length", 64)),
        hidden_channels=int(architecture.get("hidden_channels", architecture.get("hidden", 64))),
        channel_schedule=architecture.get("channel_schedule"),
        stem_kernel_size=int(architecture.get("stem_kernel_size", 7)),
        stage_kernel_size=int(architecture.get("stage_kernel_size", 5)),
        num_blocks=int(architecture.get("num_blocks", 5)),
        kernel_size=int(architecture.get("kernel_size", 5)),
        dropout=float(architecture.get("dropout", 0.1)),
    )

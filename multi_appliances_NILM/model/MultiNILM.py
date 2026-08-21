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

from dataclasses import dataclass, field
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


# Named hooks for domain-adaptation feature collection (MMD / CORAL).
# Analogous to Lin et al. selecting fc6–fc8 by layer index.
DOMAIN_FEATURE_LAYER_ALIASES = {
    "shared": "aligned",
    "encoder": "temporal",
    "aggregate": "stem",
}


class IBN1d(nn.Module):
    """Split channels between InstanceNorm and BatchNorm (IBN-Net style)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.instance_channels = channels // 2
        self.batch_channels = channels - self.instance_channels
        self.instance_norm = nn.InstanceNorm1d(
            self.instance_channels,
            affine=True,
        )
        self.batch_norm = nn.BatchNorm1d(self.batch_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_instance, x_batch = torch.split(
            x,
            [self.instance_channels, self.batch_channels],
            dim=1,
        )
        return torch.cat(
            [self.instance_norm(x_instance), self.batch_norm(x_batch)],
            dim=1,
        )


def make_norm_1d(channels: int, norm_type: str = "batch") -> nn.Module:
    """Build a 1D normalization layer while keeping old checkpoints compatible."""
    kind = str(norm_type or "batch").lower()
    if kind in {"batch", "batchnorm", "bn"}:
        return nn.BatchNorm1d(int(channels))
    if kind in {"instance", "instancenorm", "in"}:
        return nn.InstanceNorm1d(int(channels), affine=True)
    if kind in {"ibn", "ibn1d"}:
        if int(channels) < 2:
            return nn.BatchNorm1d(int(channels))
        return IBN1d(int(channels))
    if kind in {"group", "groupnorm", "gn"}:
        groups = min(8, int(channels))
        while groups > 1 and int(channels) % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, int(channels))
    raise ValueError(
        "norm_type must be batch|instance|ibn|group, "
        f"got {norm_type!r}"
    )


def normalize_domain_feature_layers(layers: list[str] | None) -> list[str]:
    """Normalize yaml names; default is post-align shared features."""
    if not layers:
        return ["aligned"]
    out: list[str] = []
    for raw in layers:
        name = str(raw).strip().lower()
        name = DOMAIN_FEATURE_LAYER_ALIASES.get(name, name)
        if name not in out:
            out.append(name)
    return out or ["aligned"]


def pool_domain_feature_map(features: torch.Tensor) -> torch.Tensor:
    """Collapse (B, C, T) → (B, C) for CORAL/MMD on vectors (Lin-style)."""
    if features.dim() != 3:
        raise ValueError(
            f"Expected domain features (B, C, T), got shape {tuple(features.shape)}"
        )
    return features.mean(dim=-1)


def state_gate(
    state_prob: torch.Tensor,
    *,
    mode: str = "soft",
    threshold: float = 0.5,
    training: bool = False,
) -> torch.Tensor:
    """Gate power by predicted ON probability (soft) or binary mask (hard).

    Modes:
      soft:
        Always use σ(state) in (0, 1). Smooth edges (can blunt waveforms).
      hard:
        Binary 1{σ >= thr}. During training uses STE so gradients still flow
        through soft probabilities; eval is pure hard.
      soft_train_hard_eval (aliases: train_soft_eval_hard, soft_hard):
        Soft while ``training=True`` (stable BCE+power gradients);
        hard threshold while ``training=False`` (sharper val/test/plots).
    """
    gate_mode = str(mode or "soft").lower()
    thr = float(threshold)

    def _hard_mask() -> torch.Tensor:
        return (state_prob >= thr).to(dtype=state_prob.dtype)

    if gate_mode in {"soft", "sigmoid", "prob", "probability"}:
        return state_prob

    if gate_mode in {
        "soft_train_hard_eval",
        "train_soft_eval_hard",
        "soft_hard",
    }:
        if training:
            return state_prob
        return _hard_mask()

    if gate_mode in {"hard", "binary", "threshold"}:
        hard = _hard_mask()
        if training and state_prob.requires_grad:
            # Straight-through: forward hard, backward through soft probs.
            return hard - state_prob.detach() + state_prob
        return hard

    raise ValueError(
        "gate_mode must be soft | hard | soft_train_hard_eval, "
        f"got {mode!r}"
    )


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
        norm_type: str = "batch",
    ) -> None:
        super().__init__()

        # Same-length padding only works for odd kernels:
        # L_out = L + 2*pad - dil*(k-1); need 2*pad == dil*(k-1).
        k = int(kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError(
                f"ResidualTemporalBlock kernel_size must be odd positive, got {k}. "
                "Even k (e.g. 10) shrinks length by 1 and breaks residual add."
            )
        padding = ((k - 1) * dilation) // 2

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=k,
            padding=padding,
            dilation=dilation,
        )
        self.norm = make_norm_1d(channels, norm_type)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        y = self.conv(x)
        y = self.norm(y)
        y = self.activation(y)
        y = self.dropout(y)

        return residual + y


class MultiScaleWaveformStem(nn.Module):
    """Parallel multi-kernel Conv1d for fine + coarse local waveform shape.

    k=3  → sharp edges / small bumps
    k=5–9 → wider ON/OFF shoulders
    Fuse with 1x1 + residual. Adds ~1K params (keeps model small).
    """

    def __init__(
        self,
        input_channels: int,
        out_channels: int,
        kernels: list[int] | tuple[int, ...] = (3, 5, 9),
        branch_channels: int = 12,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        if not kernels:
            raise ValueError("detail_kernels must be non-empty")
        in_ch, out_ch, branch_ch = int(input_channels), int(out_channels), int(branch_channels)
        branches: list[nn.Module] = []
        for kernel_size in kernels:
            k = int(kernel_size)
            if k < 1 or k % 2 == 0:
                raise ValueError(f"detail kernels must be odd positive ints, got {k}")
            branches.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, branch_ch, kernel_size=k, padding=k // 2),
                    make_norm_1d(branch_ch, norm_type),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.Sequential(
            nn.Conv1d(branch_ch * len(branches), out_ch, kernel_size=1),
            make_norm_1d(out_ch, norm_type),
            nn.ReLU(inplace=True),
        )
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1),
                make_norm_1d(out_ch, norm_type),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1)) + self.skip(x)


class StagedFeatureExtractor(nn.Module):
    """Gradually widen channel depth before the shared TCN (seq2point-style).

    Example schedule [16, 32, 64]:
        Conv1d 1→16  k=7  + BN + ReLU
        Conv1d 16→32 k=5  + BN + ReLU
        Conv1d 32→64 k=5  + BN + ReLU
    """

    def __init__(
        self,
        input_channels: int,
        channel_schedule: list[int],
        stem_kernel_size: int = 7,
        stage_kernel_size: int = 5,
        norm_type: str = "batch",
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
                    make_norm_1d(int(out_channels), norm_type),
                    nn.ReLU(inplace=True),
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

    With ``head_local_layers > 0``, a short temporal stack (k=3 by default)
    redraws local waveform shape before the 1x1 power/state readouts.
    """

    def __init__(
        self,
        hidden_channels: int,
        dropout: float,
        *,
        gate_mode: str = "soft_train_hard_eval",
        gate_threshold: float = 0.5,
        off_norm: float = 0.0,
        head_local_layers: int = 2,
        head_kernel_size: int = 3,
        head_use_residual: bool = True,
        norm_type: str = "batch",
        use_task_attention: bool = False,
        task_attention_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.gate_mode = str(gate_mode or "soft").lower()
        self.gate_threshold = float(gate_threshold)
        self.register_buffer("off_norm", torch.tensor(float(off_norm), dtype=torch.float32))

        if use_task_attention:
            attention_channels = max(
                4,
                int(hidden_channels) // max(int(task_attention_reduction), 1),
            )
            self.task_attention: nn.Module | None = nn.Sequential(
                nn.Conv1d(hidden_channels, attention_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(attention_channels, hidden_channels, kernel_size=1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.task_attention[2].weight)
            nn.init.constant_(self.task_attention[2].bias, 2.0)
        else:
            self.task_attention = None

        n_local = int(head_local_layers)
        self.head_use_residual = bool(head_use_residual) and n_local > 0
        if n_local <= 0:
            # Legacy pointwise refine (no local temporal context).
            self.local_decoder = nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
                make_norm_1d(hidden_channels, norm_type),
                nn.ReLU(inplace=True),
            )
        else:
            k = int(head_kernel_size)
            if k < 1 or k % 2 == 0:
                raise ValueError(f"head_kernel_size must be odd positive, got {k}")
            blocks: list[nn.Module] = []
            for _ in range(n_local):
                blocks.extend(
                    [
                        nn.Conv1d(
                            hidden_channels,
                            hidden_channels,
                            kernel_size=k,
                            padding=k // 2,
                        ),
                        make_norm_1d(hidden_channels, norm_type),
                        nn.ReLU(inplace=True),
                    ]
                )
            self.local_decoder = nn.Sequential(*blocks)

        self.dropout = nn.Dropout(dropout)
        self.power_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.state_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        # Alias for feature-map hooks / older docs that say feature_refine.
        self.feature_refine = self.local_decoder

    def encode_features(self, shared_features: torch.Tensor) -> torch.Tensor:
        """Head body only: shared TCN map → per-appliance features ``F`` (B, C, T)."""
        attended = shared_features
        if self.task_attention is not None:
            attended = attended * self.task_attention(attended)
        features = self.local_decoder(attended)
        if self.head_use_residual:
            features = features + attended
        return self.dropout(features)

    def decode_from_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Final 1×1 power/state + gate from features ``F`` or ``F^dist``."""
        power_raw = self.power_head(features)
        state_logits = self.state_head(features)

        # State logits stay unbounded for BCEWithLogitsLoss.
        state_prob = torch.sigmoid(state_logits)
        gate = state_gate(
            state_prob,
            mode=self.gate_mode,
            threshold=self.gate_threshold,
            training=self.training,
        )
        # Blend ON power with the normalized OFF level (0 W -> -mean/std), not 0.
        # denorm(0) equals the dataset mean and causes constant watt spikes in plots.
        power = gate * power_raw + (1.0 - gate) * self.off_norm
        return power, state_logits

    def forward(self, shared_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Independent head path (no cross-appliance distill)."""
        return self.decode_from_features(self.encode_features(shared_features))


class TaskTemporalScaleFusion(nn.Module):
    """Fuse short- and long-context TCN maps with one gate per appliance.

    The gate is scalar across feature channels but varies over time. Channel
    selection remains the responsibility of each appliance's task attention.

        gate_i = sigmoid(G_i([F_short, F_long]))       # (B, 1, T)
        F_i    = F_short + gate_i * (F_long - F_short) # (B, C, T)
    """

    def __init__(
        self,
        num_appliances: int,
        channels: int,
        *,
        hidden_channels: int = 16,
        gate_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_appliances = int(num_appliances)
        self.channels = int(channels)
        hidden = max(1, int(hidden_channels))
        initial_gate = float(gate_init)
        if not 0.0 < initial_gate < 1.0:
            raise ValueError(
                "temporal_scale_fusion.gate_init must satisfy 0 < value < 1, "
                f"got {gate_init}."
            )
        initial_bias = math.log(initial_gate / (1.0 - initial_gate))

        gates: list[nn.Module] = []
        for _ in range(self.num_appliances):
            gate = nn.Sequential(
                nn.Conv1d(2 * self.channels, hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            # Begin as a stable fixed interpolation. The final projection then
            # learns whether each appliance needs more short or long context.
            nn.init.zeros_(gate[2].weight)
            nn.init.constant_(gate[2].bias, initial_bias)
            gates.append(gate)
        self.gates = nn.ModuleList(gates)

    def forward(
        self,
        short_features: torch.Tensor,
        long_features: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if short_features.shape != long_features.shape:
            raise ValueError(
                "Temporal scale maps must have identical shapes, got "
                f"{tuple(short_features.shape)} and {tuple(long_features.shape)}."
            )
        if short_features.dim() != 3 or short_features.shape[1] != self.channels:
            raise ValueError(
                "TaskTemporalScaleFusion expected (B,C,T) with "
                f"C={self.channels}, got {tuple(short_features.shape)}."
            )

        joined = torch.cat([short_features, long_features], dim=1)
        delta = long_features - short_features
        fused: list[torch.Tensor] = []
        gate_values: list[torch.Tensor] = []
        for gate_layer in self.gates:
            gate = gate_layer(joined)
            fused.append(short_features + gate * delta)
            gate_values.append(gate)
        return fused, torch.cat(gate_values, dim=1)  # (B, A, T)


class CrossApplianceDistill(nn.Module):
    """PAD-lite residual mix: ``F_k^dist = F_k + α · Mix_k(F_1..F_K)``.

    Bottleneck ``(K·C) → mid → (K·C)`` (default ``mid = 2·C``). Not PAD-Net Module C.
    """

    def __init__(
        self,
        num_appliances: int,
        channels: int,
        *,
        residual_scale: float = 0.5,
        dropout: float = 0.0,
        mid_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.num_appliances = int(num_appliances)
        self.channels = int(channels)
        self.residual_scale = float(residual_scale)
        stacked = self.num_appliances * self.channels
        mid = int(mid_channels) if mid_channels is not None else max(2 * self.channels, 64)
        mid = max(1, min(mid, stacked))
        self.mix = nn.Sequential(
            nn.Conv1d(stacked, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(mid, stacked, kernel_size=1, bias=True),
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != self.num_appliances:
            raise ValueError(
                f"CrossApplianceDistill expected {self.num_appliances} maps, got {len(features)}"
            )
        stacked = torch.stack(features, dim=1)  # (B, K, C, T)
        bsz, _, channels, time_len = stacked.shape
        mixed = self.mix(stacked.reshape(bsz, self.num_appliances * channels, time_len))
        mixed = mixed.reshape(bsz, self.num_appliances, channels, time_len)
        alpha = self.residual_scale
        return [features[k] + alpha * mixed[:, k] for k in range(self.num_appliances)]


class CrossApplianceRelationAttention(nn.Module):
    """Attention-gated message passing across appliances at every timestep.

    TCN blocks already model the time axis. This module treats the appliance
    heads as a short token sequence (K is normally 5), so it can learn dynamic
    co-occurrence and confusion relations without quadratic attention over the
    2048-sample time window.
    """

    def __init__(
        self,
        num_appliances: int,
        channels: int,
        *,
        residual_scale: float = 0.5,
        dropout: float = 0.0,
        attention_channels: int = 16,
    ) -> None:
        super().__init__()
        self.num_appliances = int(num_appliances)
        self.channels = int(channels)
        self.residual_scale = float(residual_scale)
        relation_channels = max(4, min(int(attention_channels), self.channels))

        self.query = nn.Conv1d(self.channels, relation_channels, kernel_size=1)
        self.key = nn.Conv1d(self.channels, relation_channels, kernel_size=1)
        self.value = nn.Conv1d(self.channels, relation_channels, kernel_size=1)
        self.out = nn.Conv1d(relation_channels, self.channels, kernel_size=1)
        self.message_gate = nn.Sequential(
            nn.Conv1d(2 * self.channels, self.channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.scale = math.sqrt(float(relation_channels))

    def _project(self, layer: nn.Module, stacked: torch.Tensor) -> torch.Tensor:
        batch, appliances, channels, time_len = stacked.shape
        flat = stacked.reshape(batch * appliances, channels, time_len)
        projected = layer(flat)
        relation_channels = projected.shape[1]
        return projected.reshape(
            batch,
            appliances,
            relation_channels,
            time_len,
        ).permute(0, 3, 1, 2)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != self.num_appliances:
            raise ValueError(
                "CrossApplianceRelationAttention expected "
                f"{self.num_appliances} maps, got {len(features)}"
            )

        stacked = torch.stack(features, dim=1)  # (B, K, C, T)
        query = self._project(self.query, stacked)  # (B, T, K, D)
        key = self._project(self.key, stacked)
        value = self._project(self.value, stacked)
        scores = torch.einsum("btkd,btjd->btkj", query, key) / self.scale
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("btkj,btjd->btkd", weights, value)

        batch, time_len, appliances, relation_channels = context.shape
        context = context.permute(0, 2, 3, 1).reshape(
            batch * appliances,
            relation_channels,
            time_len,
        )
        message = self.out(context).reshape(
            batch,
            appliances,
            self.channels,
            time_len,
        )

        outputs: list[torch.Tensor] = []
        for app_i, feature in enumerate(features):
            message_i = message[:, app_i]
            gate_i = self.message_gate(torch.cat([feature, message_i], dim=1))
            outputs.append(
                feature
                + self.residual_scale * gate_i * self.dropout(message_i)
            )
        return outputs


class MultiNILM(nn.Module):
    """Simple CNN/TCN model for multi-appliance NILM.

    Layer-by-layer architecture:

        Input aggregate window
            Shape: (B, T) or (B, 1, T)

        1. _format_input
            Convert input to Conv1d format.
            Output: (B, 1, T)

        2. aggregate_feature_extractor
            Optional multi-scale stem (k=3/5/9) then staged Conv1d widening
            (channel_schedule), or one Conv1d jump.
            Output: (B, hidden_channels, T)

        3. temporal_encoder
            ResidualTemporalBlock x num_blocks
            Default dilation sequence: 1, 2, 4, 8, 16
            Output: (B, hidden_channels, T)

        4. temporal alignment
            Center-crop (or pad) features to output_length so each output step
            matches the same CSV timestep as the dataloader center targets.
            Output: (B, hidden_channels, output_length)

        5. optional task temporal-scale fusion
            Preserve block-4 short features and final long features, align both
            to the output timeline, then learn one interpolation gate per
            appliance and timestep.

        6. appliance_heads (one per appliance)
            Optional local temporal decoder (k=3 x N) + residual → F_k.
            Optional CrossApplianceDistill (PAD-lite): F → F^dist across appliances.
            Then 1x1 power/state heads with state-gated power.
            Outputs: (B, output_length, num_appliances)

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
        max_dilation: int = 128,
        gate_mode: str = "soft_train_hard_eval",
        gate_threshold: float = 0.5,
        appliance_off_norm: list[float] | None = None,
        domain_feature_layers: list[str] | None = None,
        head_local_layers: int = 2,
        head_kernel_size: int = 3,
        head_use_residual: bool = True,
        use_multiscale_stem: bool = False,
        detail_kernels: list[int] | None = None,
        detail_branch_channels: int = 12,
        stem_norm_type: str = "batch",
        temporal_norm_type: str = "batch",
        head_norm_type: str = "batch",
        task_attention_enabled: bool = False,
        task_attention_reduction: int = 4,
        temporal_scale_fusion_enabled: bool = False,
        temporal_scale_short_block: int = 4,
        temporal_scale_gate_hidden_channels: int = 16,
        temporal_scale_gate_init: float = 0.5,
        cross_appliance_enabled: bool = False,
        cross_appliance_mode: str = "bottleneck",
        cross_appliance_residual_scale: float = 0.5,
        cross_appliance_mid_channels: int | None = None,
        cross_appliance_attention_channels: int = 16,
    ) -> None:
        super().__init__()

        self.input_channels = int(input_channels)
        self.num_appliances = int(num_appliances)
        self.output_length = int(output_length)
        self.hidden_channels = int(hidden_channels)
        self.gate_mode = str(gate_mode or "soft").lower()
        self.gate_threshold = float(gate_threshold)
        self.domain_feature_layers = normalize_domain_feature_layers(domain_feature_layers)
        off_norms = list(appliance_off_norm or [0.0] * self.num_appliances)
        if len(off_norms) != self.num_appliances:
            raise ValueError(
                f"appliance_off_norm length {len(off_norms)} != num_appliances {self.num_appliances}"
            )

        detail_ks = [int(k) for k in (detail_kernels or [3, 5, 9])]

        # Step 1: widen aggregate power into temporal feature maps.
        # Multi-scale stem (optional) replaces the first coarse k=7 layer.
        if channel_schedule:
            schedule = [int(width) for width in channel_schedule]
            if schedule[-1] != self.hidden_channels:
                raise ValueError(
                    "hidden_channels must match the last entry in channel_schedule; "
                    f"got hidden_channels={self.hidden_channels}, schedule={schedule}."
                )
            if use_multiscale_stem:
                stem_out = schedule[0]
                stages: list[nn.Module] = [
                    MultiScaleWaveformStem(
                        input_channels=self.input_channels,
                        out_channels=stem_out,
                        kernels=detail_ks,
                        branch_channels=int(detail_branch_channels),
                        norm_type=stem_norm_type,
                    )
                ]
                rest = schedule[1:]
                if rest:
                    stages.append(
                        StagedFeatureExtractor(
                            input_channels=stem_out,
                            channel_schedule=rest,
                            stem_kernel_size=int(stage_kernel_size),
                            stage_kernel_size=int(stage_kernel_size),
                            norm_type=stem_norm_type,
                        )
                    )
                self.aggregate_feature_extractor = nn.Sequential(*stages)
            else:
                self.aggregate_feature_extractor = StagedFeatureExtractor(
                    input_channels=self.input_channels,
                    channel_schedule=schedule,
                    stem_kernel_size=int(stem_kernel_size),
                    stage_kernel_size=int(stage_kernel_size),
                    norm_type=stem_norm_type,
                )
        elif use_multiscale_stem:
            self.aggregate_feature_extractor = MultiScaleWaveformStem(
                input_channels=self.input_channels,
                out_channels=self.hidden_channels,
                kernels=detail_ks,
                branch_channels=int(detail_branch_channels),
                norm_type=stem_norm_type,
            )
        else:
            self.aggregate_feature_extractor = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.input_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=int(stem_kernel_size),
                    padding=int(stem_kernel_size) // 2,
                ),
                make_norm_1d(self.hidden_channels, stem_norm_type),
                nn.ReLU(inplace=True),
            )

        # Step 2: residual TCN. Dilations cycle 1,2,4,...,max_dilation so deeper
        # stacks keep local scales instead of exploding past the window length.
        max_dil = max(1, int(max_dilation))
        cycle = int(max_dil).bit_length()  # e.g. 128 → 8 steps: 1..128
        temporal_blocks = []
        for block_index in range(num_blocks):
            dilation = 2 ** (block_index % cycle)
            temporal_blocks.append(
                ResidualTemporalBlock(
                    channels=self.hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    norm_type=temporal_norm_type,
                )
            )
        self.temporal_encoder = nn.Sequential(*temporal_blocks)

        self.temporal_scale_short_block = int(temporal_scale_short_block)
        self.last_temporal_scale_gate_means: torch.Tensor | None = None
        if temporal_scale_fusion_enabled:
            if not 1 <= self.temporal_scale_short_block < len(self.temporal_encoder):
                raise ValueError(
                    "temporal_scale_fusion.short_block must leave at least one "
                    "later TCN block for long context; expected "
                    f"1 <= short_block < {len(self.temporal_encoder)}, got "
                    f"{self.temporal_scale_short_block}."
                )
            self.temporal_scale_fusion: TaskTemporalScaleFusion | None = (
                TaskTemporalScaleFusion(
                    num_appliances=self.num_appliances,
                    channels=self.hidden_channels,
                    hidden_channels=int(temporal_scale_gate_hidden_channels),
                    gate_init=float(temporal_scale_gate_init),
                )
            )
        else:
            self.temporal_scale_fusion = None

        # Step 6: one decoder head per appliance (dynamic count from experiment).
        self.appliance_heads = nn.ModuleList(
            [
                ApplianceHead(
                    hidden_channels=self.hidden_channels,
                    dropout=dropout,
                    gate_mode=self.gate_mode,
                    gate_threshold=self.gate_threshold,
                    off_norm=off_norms[app_i],
                    head_local_layers=int(head_local_layers),
                    head_kernel_size=int(head_kernel_size),
                    head_use_residual=bool(head_use_residual),
                    norm_type=head_norm_type,
                    use_task_attention=bool(task_attention_enabled),
                    task_attention_reduction=int(task_attention_reduction),
                )
                for app_i in range(self.num_appliances)
            ]
        )

        # Optional PAD-lite: mix head-body features across appliances, then final 1×1.
        if cross_appliance_enabled:
            cross_mode = str(cross_appliance_mode or "bottleneck").lower()
            if cross_mode in {"relation_attention", "attention", "relational"}:
                self.cross_appliance_distill: nn.Module | None = (
                    CrossApplianceRelationAttention(
                        num_appliances=self.num_appliances,
                        channels=self.hidden_channels,
                        residual_scale=float(cross_appliance_residual_scale),
                        dropout=float(dropout),
                        attention_channels=int(cross_appliance_attention_channels),
                    )
                )
            elif cross_mode in {"bottleneck", "distill", "pad_lite"}:
                self.cross_appliance_distill = CrossApplianceDistill(
                    num_appliances=self.num_appliances,
                    channels=self.hidden_channels,
                    residual_scale=float(cross_appliance_residual_scale),
                    dropout=float(dropout),
                    mid_channels=cross_appliance_mid_channels,
                )
            else:
                raise ValueError(
                    "cross_appliance.mode must be bottleneck|relation_attention, "
                    f"got {cross_appliance_mode!r}"
                )
        else:
            self.cross_appliance_distill = None

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
        """Match encoder time length to ``output_length`` for the appliance heads.

        Where this sits in the forward pass::

            aggregate (B, 1, T_in)
              → aggregate_feature_extractor   # still length T_in
              → temporal_encoder             # still length T_in  (same-pad convs)
              → _align_output_time           # → length T_out = self.output_length
              → appliance_heads

        ``features`` shape: ``(batch, channels, time_len)``.
        Return shape: ``(batch, channels, output_length)``.

        Why it exists
        -------------
        The dataloader can supervise a *shorter* label window than the input
        (e.g. old setup T_in=864, T_out=256 with center targets). The encoder
        still runs on the full input, so we must cut (or pad) the time axis so
        each head output timestep lines up with the CSV / label timestep.

        Important: this is a *slice or pad*, never interpolate. Interpolating
        the full window into ``output_length`` would warp time and misalign
        labels (and caused repeating pulse artifacts in earlier experiments).

        Three cases
        -----------
        1) ``time_len == output_length`` (current yaml: 480 in / 480 out)
           Identity. No crop, no pad. Shared features Z have the same length
           as the labels → domain-adaptation hook can use this tensor as-is.

        2) ``time_len > output_length`` (e.g. 864 → 256)
           **Center crop**: drop equal context on both sides::

               offset = (time_len - output_length) // 2
               features[:, :, offset : offset + output_length]

           Example 864 → 256: offset = 304, keep indices [304, 560).
           That matches dataloader ``output_alignment: center`` targets.

           Note: this implementation always center-crops. It does *not* read
           yaml ``output_alignment: end``. If you use end-aligned labels with
           T_in > T_out, either change this crop to a right-end slice or keep
           center alignment in the experiment config.

        3) ``time_len < output_length`` (rare)
           Symmetric zero-pad on the left/right so length becomes
           ``output_length`` (``F.pad`` on the last dim).

        Domain adaptation
        -----------------
        The tensor returned here is the recommended shared representation Z
        for CORAL/MMD (after temporal encoder, before appliance heads).
        """
        time_len = features.shape[-1]
        if time_len == self.output_length:
            # Case 1: full_input / equal windows — nothing to do.
            return features
        if time_len > self.output_length:
            # Case 2: center-crop longer encoder features to label length.
            offset = (time_len - self.output_length) // 2
            return features[:, :, offset : offset + self.output_length]
        # Case 3: pad shorter features up to label length.
        pad_total = self.output_length - time_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return F.pad(features, (pad_left, pad_right))

    def available_domain_feature_layers(self) -> list[str]:
        """Layer names you can put in ``domain_feature_layers`` (yaml / ctor).

        Paper analogue: Lin et al. set ``l1=6, l2=8`` on fc layers.
        Here you select by name, e.g. ``["aligned"]`` or
        ``["temporal_2", "temporal_4", "aligned"]`` (indices < num_blocks).
        """
        names = ["stem", "temporal", "aligned"]
        for i in range(len(self.temporal_encoder)):
            names.append(f"temporal_{i}")
        return names

    def _validate_domain_feature_layers(self, layers: list[str]) -> list[str]:
        layers = normalize_domain_feature_layers(layers)
        allowed = set(self.available_domain_feature_layers())
        unknown = [name for name in layers if name not in allowed]
        if unknown:
            raise ValueError(
                f"Unknown domain_feature_layers {unknown}. "
                f"Choose from {sorted(allowed)}."
            )
        return layers

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_domain_features: bool = False,
        domain_feature_layers: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Run MultiNILM; optionally also return selected encoder features for DA.

        Default (training / eval today)::

            power_pred, state_logits = model(x)

        Domain-adaptation collection (like selecting fc6–fc8 in Lin et al.)::

            power, state, feats = model(x, return_domain_features=True)
            # feats = {"aligned": (B, C, T_out), ...}   # names from yaml

        Or override layers for one call::

            ..., feats = model(
                x,
                return_domain_features=True,
                domain_feature_layers=["temporal_4", "aligned"],
            )

        Flow:
            aggregate input
            -> aggregate_feature_extractor          # hook: stem
            -> temporal_encoder blocks              # hooks: temporal_i, temporal
            -> _align_output_time                   # hook: aligned, default DA Z
            -> optional per-appliance temporal-scale fusion
            -> per-appliance head bodies
            -> optional cross-appliance relation module
            -> final 1x1 + state gate
            -> (B, output_length, num_appliances)
        """
        collect_layers: list[str] = []
        if return_domain_features:
            collect_layers = self._validate_domain_feature_layers(
                domain_feature_layers
                if domain_feature_layers is not None
                else self.domain_feature_layers
            )
        need_block = any(name.startswith("temporal_") for name in collect_layers)
        want = set(collect_layers)
        domain_feats: dict[str, torch.Tensor] = {}

        x = self._format_input(x)

        # Step 1-2: raw aggregate → hidden maps.
        features = self.aggregate_feature_extractor(x)
        if "stem" in want:
            domain_feats["stem"] = features

        # Step 3: dilated residual temporal stack. Scale fusion also needs the
        # exact map after ``short_block`` while retaining the final long map.
        short_features: torch.Tensor | None = None
        if need_block or "temporal" in want or self.temporal_scale_fusion is not None:
            for block_index, block in enumerate(self.temporal_encoder):
                features = block(features)
                if (
                    self.temporal_scale_fusion is not None
                    and block_index + 1 == self.temporal_scale_short_block
                ):
                    short_features = features
                key = f"temporal_{block_index}"
                if key in want:
                    domain_feats[key] = features
            if "temporal" in want:
                domain_feats["temporal"] = features
        else:
            features = self.temporal_encoder(features)

        # Step 4: align time to labels / heads.
        output_features = self._align_output_time(features)
        if "aligned" in want:
            domain_feats["aligned"] = output_features

        # Step 5: optionally choose short/long context per appliance, then run
        # the existing task attention and local appliance decoders.
        if self.temporal_scale_fusion is not None:
            if short_features is None:
                raise RuntimeError("Temporal scale fusion did not capture short features.")
            short_output_features = self._align_output_time(short_features)
            head_inputs, scale_gates = self.temporal_scale_fusion(
                short_output_features,
                output_features,
            )
            self.last_temporal_scale_gate_means = scale_gates.detach().mean(
                dim=(0, 2)
            )
        else:
            head_inputs = [output_features] * self.num_appliances
            self.last_temporal_scale_gate_means = None

        head_feats = [
            head.encode_features(head_input)
            for head, head_input in zip(self.appliance_heads, head_inputs)
        ]
        if self.cross_appliance_distill is not None:
            head_feats = self.cross_appliance_distill(head_feats)

        power_parts: list[torch.Tensor] = []
        state_parts: list[torch.Tensor] = []
        for head, feat in zip(self.appliance_heads, head_feats):
            power_i, state_i = head.decode_from_features(feat)
            power_parts.append(power_i)
            state_parts.append(state_i)

        power_pred = torch.cat(power_parts, dim=1)
        state_logits = torch.cat(state_parts, dim=1)

        # Step 6: (B, A, T) → (B, T, A).
        power_pred = power_pred.permute(0, 2, 1)
        state_logits = state_logits.permute(0, 2, 1)

        if return_domain_features:
            return power_pred, state_logits, domain_feats
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
    max_dilation: int = 128
    # soft | hard | soft_train_hard_eval (train soft, val/test/plots hard)
    gate_mode: str = "soft_train_hard_eval"
    gate_threshold: float = 0.5
    # Per-appliance local temporal decoder (0 = legacy 1x1 refine only).
    head_local_layers: int = 2
    head_kernel_size: int = 3
    head_use_residual: bool = True
    # Multi-scale front-end (shape-oriented); no shape loss required.
    use_multiscale_stem: bool = False
    detail_kernels: list[int] = field(default_factory=lambda: [3, 5, 9])
    detail_branch_channels: int = 12
    stem_norm_type: str = "batch"
    temporal_norm_type: str = "batch"
    head_norm_type: str = "batch"
    task_attention_enabled: bool = False
    task_attention_reduction: int = 4
    # Per-appliance interpolation between an intermediate and final TCN map.
    temporal_scale_fusion_enabled: bool = False
    temporal_scale_short_block: int = 4
    temporal_scale_gate_hidden_channels: int = 16
    temporal_scale_gate_init: float = 0.5
    # PAD-lite cross-appliance distill (off = skip mix, still encode→decode).
    cross_appliance_enabled: bool = False
    cross_appliance_mode: str = "bottleneck"
    cross_appliance_residual_scale: float = 0.5
    cross_appliance_mid_channels: int | None = None
    cross_appliance_attention_channels: int = 16
    # Lin-style multi-layer DA hooks (late TCN + pre-head), analogous to fc6–fc8.
    domain_feature_layers: list[str] = field(
        default_factory=lambda: ["temporal_2", "temporal_4", "aligned"]
    )


def _parse_cross_appliance(
    architecture: dict[str, Any],
) -> tuple[bool, str, float, int | None, int]:
    """Read ``architecture.cross_appliance`` from model yaml."""
    block = architecture.get("cross_appliance")
    if not isinstance(block, dict):
        return False, "bottleneck", 0.5, None, 16
    enabled = bool(block.get("enabled", False))
    mode = str(block.get("mode", "bottleneck"))
    scale = float(block.get("residual_scale", 0.5))
    mid = block.get("mid_channels", None)
    mid_i = None if mid is None else int(mid)
    attention_channels = int(block.get("attention_channels", 16))
    return enabled, mode, scale, mid_i, attention_channels


def multinilm_config(architecture: dict[str, Any]) -> MultiNILMConfig:
    """Read MultiNILM settings from the model YAML architecture section."""

    detail_kernels = architecture.get("detail_kernels", [3, 5, 9])
    ca_enabled, ca_mode, ca_scale, ca_mid, ca_attention = _parse_cross_appliance(
        architecture
    )
    task_attention = architecture.get("task_attention", {})
    if not isinstance(task_attention, dict):
        task_attention = {}
    temporal_scale_fusion = architecture.get("temporal_scale_fusion", {})
    if not isinstance(temporal_scale_fusion, dict):
        temporal_scale_fusion = {}
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
        max_dilation=int(architecture.get("max_dilation", 128)),
        gate_mode=str(architecture.get("gate_mode", "soft_train_hard_eval")),
        gate_threshold=float(architecture.get("gate_threshold", 0.5)),
        head_local_layers=int(architecture.get("head_local_layers", 2)),
        head_kernel_size=int(architecture.get("head_kernel_size", 3)),
        head_use_residual=bool(architecture.get("head_use_residual", True)),
        use_multiscale_stem=bool(architecture.get("use_multiscale_stem", False)),
        detail_kernels=[int(k) for k in detail_kernels],
        detail_branch_channels=int(architecture.get("detail_branch_channels", 12)),
        stem_norm_type=str(architecture.get("stem_norm_type", "batch")),
        temporal_norm_type=str(architecture.get("temporal_norm_type", "batch")),
        head_norm_type=str(architecture.get("head_norm_type", "batch")),
        task_attention_enabled=bool(task_attention.get("enabled", False)),
        task_attention_reduction=int(task_attention.get("reduction", 4)),
        temporal_scale_fusion_enabled=bool(
            temporal_scale_fusion.get("enabled", False)
        ),
        temporal_scale_short_block=int(
            temporal_scale_fusion.get("short_block", 4)
        ),
        temporal_scale_gate_hidden_channels=int(
            temporal_scale_fusion.get("gate_hidden_channels", 16)
        ),
        temporal_scale_gate_init=float(
            temporal_scale_fusion.get("gate_init", 0.5)
        ),
        cross_appliance_enabled=ca_enabled,
        cross_appliance_mode=ca_mode,
        cross_appliance_residual_scale=ca_scale,
        cross_appliance_mid_channels=ca_mid,
        cross_appliance_attention_channels=ca_attention,
        domain_feature_layers=normalize_domain_feature_layers(
            architecture.get("domain_feature_layers")
        ),
    )

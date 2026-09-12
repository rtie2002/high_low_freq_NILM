"""MultiNILM architecture (one file).

Symbols
    B = batch size
    T = input window length (samples)
    T_out = label / output length
    C = hidden channels (yaml hidden_channels, usually 128)
    A = number of appliances
    x = aggregate power, already normalized by the dataloader

Forward (read MultiNILM.forward)
    x                  (B, T) or (B, 1, T)
    optional GL front  (B, C_in, T)
    stem + TCN         (B, C, T)
    time align         (B, C, T_out)
    appliance heads    A tensors of (B, C, T_out)
    relation attention A tensors of (B, C, T_out)
    power, state       (B, T_out, A), (B, T_out, A)

state_logits are raw logits for BCEWithLogitsLoss.
Loss, metrics, and the train loop are not in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from model.preprocess_feature.fractional import (
    FractionalFrontEnd,
    parse_fractional_architecture,
)


# Named hooks for domain-adaptation feature collection (MMD / CORAL).
# Analogous to Lin et al. selecting fc6–fc8 by layer index.
DOMAIN_FEATURE_LAYER_ALIASES = {
    "shared": "aligned",
    "encoder": "temporal",
    "aggregate": "stem",
}


class IBN1d(nn.Module):
    """Instance-Batch Normalization on a 1D feature map ``(B, C, T)``.

    IBN-Net (Pan et al., ECCV 2018) splits channels rather than averaging two
    normalizers. Half the channels use per-window InstanceNorm so house-style
    offset/scale is suppressed; the other half use BatchNorm so absolute
    amplitude (100 W fridge vs 2 kW kettle) is not fully washed out.

        X_IN = X[:, :C/2, :],   X_BN = X[:, C/2:, :]
        Y    = concat(IN(X_IN), BN(X_BN))

    Shape is unchanged: ``(B, C, T) → (B, C, T)``. Used in the early stem only.
    """

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
    """Choose how predicted appliance state controls the power estimate.

    Modes:
      none (alias: ungated):
        Return an all-one gate, so the power head learns directly from the
        power loss. The state head remains supervised independently and an
        evaluation-time calibrated state mask may gate the final watts.
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

    if gate_mode in {"none", "ungated"}:
        return torch.ones_like(state_prob)

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
        "gate_mode must be none | soft | hard | soft_train_hard_eval, "
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
        # (B, C, T) -> (B, C, T)
        return x + self.dropout(self.activation(self.norm(self.conv(x))))


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
        """Shared map Z → appliance feature F.  Both (B, C, T)."""
        z = shared_features
        if self.task_attention is not None:
            # M = σ(A(Z));  Z̃ = Z ⊙ M
            z = z * self.task_attention(z)
        f = self.local_decoder(z)
        if self.head_use_residual:
            f = f + z
        return self.dropout(f)

    def decode_from_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """F → raw power r, state logit s, gated ŷ.  Each (B, 1, T)."""
        r = self.power_head(features)
        s = self.state_head(features)
        p = torch.sigmoid(s)
        g = state_gate(
            p,
            mode=self.gate_mode,
            threshold=self.gate_threshold,
            training=self.training,
        )
        # ŷ = g · r + (1-g) · y_off
        # y_off is 0 W in normalized space (-mean/std), not 0.
        y_hat = g * r + (1.0 - g) * self.off_norm
        return y_hat, s

    def forward(self, shared_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Independent head path (no cross-appliance distill)."""
        return self.decode_from_features(self.encode_features(shared_features))


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
    """Per-timestep attention over K appliance tokens, not over time.

    TCN already covers the time axis. Here each appliance head is one token, so
    a high-power pulse can compare kettle / microwave / dishwasher evidence
    without ``O(T^2)`` attention over the window.

    For features ``F_i(t) ∈ R^C`` at one timestep:

        α_{ij}(t) = softmax_j( q_i(t)^T k_j(t) / √d )
        C_i(t)    = Σ_j α_{ij}(t) v_j(t)
        F'_i      = F_i + ρ · G_i ⊙ W_o C_i

    ``G_i = σ([F_i, message_i])`` is a learned gate: unused messages can be
    shut off. ``ρ`` is ``residual_scale`` (yaml default 0.25).

    Shapes:
        in  : K tensors of ``(B, C, T)``
        attn: ``(B, T, K, K)``  — one K×K matrix per timestep
        out : K tensors of ``(B, C, T)``
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

        self.relation_channels = relation_channels
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

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != self.num_appliances:
            raise ValueError(
                f"expected {self.num_appliances} appliance maps, got {len(features)}"
            )
        # features[i]: (B, C, T)
        stacked = torch.stack(features, dim=1)
        batch, n_app, channels, time_len = stacked.shape  # (B, A, C, T)
        rel_ch = self.relation_channels
        flat = stacked.reshape(batch * n_app, channels, time_len)  # (B*A, C, T)

        # Q,K,V: (B*A, C, T) -> (B*A, D, T) -> (B, T, A, D)
        query = self.query(flat).reshape(batch, n_app, rel_ch, time_len).permute(0, 3, 1, 2)
        key = self.key(flat).reshape(batch, n_app, rel_ch, time_len).permute(0, 3, 1, 2)
        value = self.value(flat).reshape(batch, n_app, rel_ch, time_len).permute(0, 3, 1, 2)

        # α = softmax(Q K^T / √D)          (B, T, A, A)
        scores = torch.einsum("btkd,btjd->btkj", query, key) / self.scale
        weights = torch.softmax(scores, dim=-1)
        # context = α V                    (B, T, A, D)
        context = torch.einsum("btkj,btjd->btkd", weights, value)

        # message = W_o context            (B, A, C, T)
        message = self.out(
            context.permute(0, 2, 3, 1).reshape(batch * n_app, rel_ch, time_len)
        ).reshape(batch, n_app, self.channels, time_len)

        # F'_i = F_i + ρ · G_i ⊙ message_i
        outputs: list[torch.Tensor] = []
        for app_i, feat in enumerate(features):
            msg_i = message[:, app_i]  # (B, C, T)
            gate_i = self.message_gate(torch.cat([feat, msg_i], dim=1))
            outputs.append(feat + self.residual_scale * gate_i * self.dropout(msg_i))
        return outputs


class MultiNILM(nn.Module):
    """CNN/TCN multi-appliance NILM. See file header for the full shape map."""

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

        # Step 3: one decoder head per appliance (dynamic count from experiment).
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
        """(B, T), (B, C, T), or (B, T, C) → Conv1d layout (B, C, T)."""

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
        """(B, C, T) → (B, C, T_out) by center crop or pad. Never interpolate."""
        time_len = features.shape[-1]
        if time_len == self.output_length:
            return features
        if time_len > self.output_length:
            offset = (time_len - self.output_length) // 2
            return features[:, :, offset : offset + self.output_length]
        pad_total = self.output_length - time_len
        pad_left = pad_total // 2
        return F.pad(features, (pad_left, pad_total - pad_left))

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
        domain_feats: dict[str, torch.Tensor] = {}
        want: set[str] = set()
        if return_domain_features:
            want = set(
                self._validate_domain_feature_layers(
                    domain_feature_layers
                    if domain_feature_layers is not None
                    else self.domain_feature_layers
                )
            )
        need_block = any(name.startswith("temporal_") for name in want)

        # x: (B, T) or (B, C_in, T) or (B, T, C_in)  ->  (B, C_in, T)
        h = self._format_input(x)

        # stem: (B, C_in, T) -> (B, C, T)
        h = self.aggregate_feature_extractor(h)
        if "stem" in want:
            domain_feats["stem"] = h

        # TCN: (B, C, T) -> (B, C, T)   dilations 1,2,4,...
        if need_block or "temporal" in want:
            for block_index, block in enumerate(self.temporal_encoder):
                h = block(h)
                key = f"temporal_{block_index}"
                if key in want:
                    domain_feats[key] = h
            if "temporal" in want:
                domain_feats["temporal"] = h
        else:
            h = self.temporal_encoder(h)

        # align: (B, C, T) -> (B, C, T_out)
        z = self._align_output_time(h)
        if "aligned" in want:
            domain_feats["aligned"] = z

        # heads: Z -> F_i                 each (B, C, T_out)
        feats = [head.encode_features(z) for head in self.appliance_heads]
        # optional mix: F_i -> F'_i
        if self.cross_appliance_distill is not None:
            feats = self.cross_appliance_distill(feats)

        # decode: F'_i -> ŷ_i, s_i        each (B, 1, T_out)
        power_parts: list[torch.Tensor] = []
        state_parts: list[torch.Tensor] = []
        for head, feat in zip(self.appliance_heads, feats):
            power_i, state_i = head.decode_from_features(feat)
            power_parts.append(power_i)
            state_parts.append(state_i)

        # (B, A, T_out) -> (B, T_out, A)
        power_pred = torch.cat(power_parts, dim=1).permute(0, 2, 1)
        state_logits = torch.cat(state_parts, dim=1).permute(0, 2, 1)

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
    # none | soft | hard | soft_train_hard_eval (train soft, val/test hard)
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


def build_multinilm(
    cfg: MultiNILMConfig,
    *,
    num_appliances: int,
    output_length: int,
    appliance_off_norm: list[float] | None = None,
    input_channels: int | None = None,
) -> MultiNILM:
    """Build MultiNILM from a parsed config.

    Adapters and front-end wrappers should call this instead of repeating the
    constructor keyword list. ``input_channels`` overrides yaml when a
    fractional front-end expands ``(B, 1, T)`` into ``(B, C, T)``.
    """
    return MultiNILM(
        input_channels=int(
            cfg.input_channels if input_channels is None else input_channels
        ),
        num_appliances=int(num_appliances),
        output_length=int(output_length),
        hidden_channels=cfg.hidden_channels,
        channel_schedule=cfg.channel_schedule,
        stem_kernel_size=cfg.stem_kernel_size,
        stage_kernel_size=cfg.stage_kernel_size,
        num_blocks=cfg.num_blocks,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
        max_dilation=cfg.max_dilation,
        gate_mode=cfg.gate_mode,
        gate_threshold=cfg.gate_threshold,
        appliance_off_norm=appliance_off_norm,
        domain_feature_layers=cfg.domain_feature_layers,
        head_local_layers=cfg.head_local_layers,
        head_kernel_size=cfg.head_kernel_size,
        head_use_residual=cfg.head_use_residual,
        use_multiscale_stem=cfg.use_multiscale_stem,
        detail_kernels=cfg.detail_kernels,
        detail_branch_channels=cfg.detail_branch_channels,
        stem_norm_type=cfg.stem_norm_type,
        temporal_norm_type=cfg.temporal_norm_type,
        head_norm_type=cfg.head_norm_type,
        task_attention_enabled=cfg.task_attention_enabled,
        task_attention_reduction=cfg.task_attention_reduction,
        cross_appliance_enabled=cfg.cross_appliance_enabled,
        cross_appliance_mode=cfg.cross_appliance_mode,
        cross_appliance_residual_scale=cfg.cross_appliance_residual_scale,
        cross_appliance_mid_channels=cfg.cross_appliance_mid_channels,
        cross_appliance_attention_channels=cfg.cross_appliance_attention_channels,
    )


def multinilm_config(architecture: dict[str, Any]) -> MultiNILMConfig:
    """Read MultiNILM settings from the model YAML architecture section."""

    detail_kernels = architecture.get("detail_kernels", [3, 5, 9])
    ca_enabled, ca_mode, ca_scale, ca_mid, ca_attention = _parse_cross_appliance(
        architecture
    )
    task_attention = architecture.get("task_attention", {})
    if not isinstance(task_attention, dict):
        task_attention = {}
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
        cross_appliance_enabled=ca_enabled,
        cross_appliance_mode=ca_mode,
        cross_appliance_residual_scale=ca_scale,
        cross_appliance_mid_channels=ca_mid,
        cross_appliance_attention_channels=ca_attention,
        domain_feature_layers=normalize_domain_feature_layers(
            architecture.get("domain_feature_layers")
        ),
    )


class MultiNILMFractional(nn.Module):
    """Same architecture as MultiNILM, with a GL channel front-end.

    Kept as a wrapper so old checkpoints still load as
    ``frontend.*`` / ``backbone.*``.
    """

    def __init__(self, *, backbone: MultiNILM, frontend: FractionalFrontEnd) -> None:
        super().__init__()
        self.frontend = frontend
        self.backbone = backbone
        if int(backbone.input_channels) != int(frontend.out_channels):
            raise ValueError(
                "backbone input_channels must match FractionalFrontEnd: "
                f"expected {frontend.out_channels}, got {backbone.input_channels}."
            )
        self.input_channels = 1
        self.feature_channels = int(frontend.out_channels)
        self.num_appliances = backbone.num_appliances
        self.output_length = backbone.output_length
        self.domain_feature_layers = backbone.domain_feature_layers

    def forward(self, x: torch.Tensor, return_domain_features: bool = False):
        # x: (B, T) or (B, 1, T) or (B, T, 1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[-1] == 1:
            x = x.permute(0, 2, 1)
        x = x.float()
        # GL / delta / rolling: (B, 1, T) -> (B, C_in, T)
        x = self.frontend(x)
        return self.backbone(x, return_domain_features=return_domain_features)


def build_multinilm_fractional(
    architecture: dict[str, Any],
    *,
    num_appliances: int,
    output_length: int,
    appliance_off_norm: list[float] | None = None,
) -> MultiNILMFractional:
    """Build the fractional MultiNILM used by ``multinilm_fractional`` yaml."""
    settings = parse_fractional_architecture(architecture)
    frontend = FractionalFrontEnd(
        alphas=settings.resolved_alphas(),
        include_raw=settings.include_raw,
        memory=settings.memory,
        h=settings.h,
        channel_normalize=settings.channel_normalize,
        include_delta=settings.include_delta,
        include_abs_delta=settings.include_abs_delta,
        rolling_windows=settings.rolling_windows,
        include_rolling_mean=settings.include_rolling_mean,
        include_rolling_std=settings.include_rolling_std,
    )
    feature_c = int(frontend.out_channels)
    arch = dict(architecture)
    arch["input_channels"] = feature_c
    backbone = build_multinilm(
        multinilm_config(arch),
        num_appliances=num_appliances,
        output_length=output_length,
        appliance_off_norm=appliance_off_norm,
        input_channels=feature_c,
    )
    return MultiNILMFractional(backbone=backbone, frontend=frontend)

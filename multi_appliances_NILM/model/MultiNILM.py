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
                    nn.BatchNorm1d(branch_ch),
                    nn.GELU(),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.Sequential(
            nn.Conv1d(branch_ch * len(branches), out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1)) + self.skip(x)


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
    ) -> None:
        super().__init__()
        self.gate_mode = str(gate_mode or "soft").lower()
        self.gate_threshold = float(gate_threshold)
        self.register_buffer("off_norm", torch.tensor(float(off_norm), dtype=torch.float32))

        n_local = int(head_local_layers)
        self.head_use_residual = bool(head_use_residual) and n_local > 0
        if n_local <= 0:
            # Legacy pointwise refine (no local temporal context).
            self.local_decoder = nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
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
                        nn.BatchNorm1d(hidden_channels),
                        nn.GELU(),
                    ]
                )
            self.local_decoder = nn.Sequential(*blocks)

        self.dropout = nn.Dropout(dropout)
        self.power_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.state_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def forward(self, shared_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.local_decoder(shared_features)
        if self.head_use_residual:
            features = features + shared_features
        features = self.dropout(features)
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

        5. appliance_heads (one per appliance)
            Optional local temporal decoder (k=3 x N) + residual, then
            1x1 power/state heads with state-gated power.
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
                        )
                    )
                self.aggregate_feature_extractor = nn.Sequential(*stages)
            else:
                self.aggregate_feature_extractor = StagedFeatureExtractor(
                    input_channels=self.input_channels,
                    channel_schedule=schedule,
                    stem_kernel_size=int(stem_kernel_size),
                    stage_kernel_size=int(stage_kernel_size),
                )
        elif use_multiscale_stem:
            self.aggregate_feature_extractor = MultiScaleWaveformStem(
                input_channels=self.input_channels,
                out_channels=self.hidden_channels,
                kernels=detail_ks,
                branch_channels=int(detail_branch_channels),
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
                )
                for app_i in range(self.num_appliances)
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
        ``["temporal_3", "temporal_5", "aligned"]``.
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
            -> _align_output_time                   # hook: aligned  ★ default DA Z
            -> per-appliance heads
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

        # Step 3: dilated residual temporal stack.
        if need_block or "temporal" in want:
            for block_index, block in enumerate(self.temporal_encoder):
                features = block(features)
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

        # Step 5: per-appliance heads.
        power_parts: list[torch.Tensor] = []
        state_parts: list[torch.Tensor] = []
        for head in self.appliance_heads:
            power_i, state_i = head(output_features)
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
    # Lin-style multi-layer DA hooks (late TCN + pre-head), analogous to fc6–fc8.
    domain_feature_layers: list[str] = field(
        default_factory=lambda: ["temporal_4", "temporal_6", "aligned"]
    )

def multinilm_config(architecture: dict[str, Any]) -> MultiNILMConfig:
    """Read MultiNILM settings from the model YAML architecture section."""

    detail_kernels = architecture.get("detail_kernels", [3, 5, 9])
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
        domain_feature_layers=normalize_domain_feature_layers(
            architecture.get("domain_feature_layers")
        ),
    )

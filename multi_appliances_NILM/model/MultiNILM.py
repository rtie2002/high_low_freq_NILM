"""MultiNILM: aggregate power → A appliance powers and ON/OFF logits.

    x (B,T) → optional GL channels (B,C_in,T) → stem+TCN (B,C,T)
      → heads → relation mix → power, state (B,T_out,A)

Fractional checkpoints stay `frontend.*` / `backbone.*`. Loss is MultiNILM_loss.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import appliance_off_norm_normalized
from data.common import BaseNILMAdapter, StepOutput


DOMAIN_FEATURE_LAYER_ALIASES = {"shared": "aligned", "encoder": "temporal", "aggregate": "stem"}


def make_norm_1d(channels: int, norm_type: str = "batch") -> nn.Module:
    kind = str(norm_type or "batch").lower()
    n = int(channels)
    if kind in {"batch", "batchnorm", "bn"}:
        return nn.BatchNorm1d(n)
    if kind in {"instance", "instancenorm", "in"}:
        return nn.InstanceNorm1d(n, affine=True)
    if kind in {"ibn", "ibn1d"}:
        return nn.BatchNorm1d(n) if n < 2 else IBN1d(n)
    if kind in {"group", "groupnorm", "gn"}:
        groups = min(8, n)
        while groups > 1 and n % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, n)
    raise ValueError(f"norm_type must be batch|instance|ibn|group, got {norm_type!r}")


def _conv_norm_relu(in_ch: int, out_ch: int, kernel: int, norm_type: str) -> list[nn.Module]:
    return [
        nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
        make_norm_1d(out_ch, norm_type),
        nn.ReLU(inplace=True),
    ]


class IBN1d(nn.Module):
    """Y = concat(IN(X[:, :C/2]), BN(X[:, C/2:])). Shape (B,C,T) unchanged."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.instance_channels = int(channels) // 2
        self.batch_channels = int(channels) - self.instance_channels
        self.instance_norm = nn.InstanceNorm1d(self.instance_channels, affine=True)
        self.batch_norm = nn.BatchNorm1d(self.batch_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in, x_bn = torch.split(x, [self.instance_channels, self.batch_channels], dim=1)
        return torch.cat([self.instance_norm(x_in), self.batch_norm(x_bn)], dim=1)


def normalize_domain_feature_layers(layers: list[str] | None) -> list[str]:
    if not layers:
        return ["aligned"]
    out: list[str] = []
    for raw in layers:
        name = DOMAIN_FEATURE_LAYER_ALIASES.get(str(raw).strip().lower(), str(raw).strip().lower())
        if name not in out:
            out.append(name)
    return out or ["aligned"]


def pool_domain_feature_map(features: torch.Tensor) -> torch.Tensor:
    if features.dim() != 3:
        raise ValueError(f"Expected (B, C, T), got {tuple(features.shape)}")
    return features.mean(dim=-1)


def state_gate(state_prob, *, mode="soft", threshold=0.5, training=False):
    """none=ungated, soft=σ, hard=1{σ≥thr} with STE in train, soft_train_hard_eval."""
    gate_mode = str(mode or "soft").lower()
    hard = (state_prob >= float(threshold)).to(dtype=state_prob.dtype)
    if gate_mode in {"none", "ungated"}:
        return torch.ones_like(state_prob)
    if gate_mode in {"soft", "sigmoid", "prob", "probability"}:
        return state_prob
    if gate_mode in {"soft_train_hard_eval", "train_soft_eval_hard", "soft_hard"}:
        return state_prob if training else hard
    if gate_mode in {"hard", "binary", "threshold"}:
        if training and state_prob.requires_grad:
            return hard - state_prob.detach() + state_prob
        return hard
    raise ValueError(f"gate_mode must be none|soft|hard|soft_train_hard_eval, got {mode!r}")


class ResidualTemporalBlock(nn.Module):
    """(B,C,T) residual conv. Kernel must be odd so length is unchanged."""

    def __init__(self, channels, kernel_size, dilation, dropout, norm_type="batch"):
        super().__init__()
        k = int(kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError(f"kernel_size must be odd positive, got {k}")
        self.conv = nn.Conv1d(channels, channels, k, padding=((k - 1) * dilation) // 2, dilation=dilation)
        self.norm = make_norm_1d(channels, norm_type)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.activation(self.norm(self.conv(x))))


class MultiScaleWaveformStem(nn.Module):
    """Parallel odd kernels fused with 1×1 + skip."""

    def __init__(self, input_channels, out_channels, kernels=(3, 5, 9), branch_channels=12, norm_type="batch"):
        super().__init__()
        if not kernels:
            raise ValueError("detail_kernels must be non-empty")
        in_ch, out_ch, branch_ch = int(input_channels), int(out_channels), int(branch_channels)
        branches = []
        for kernel_size in kernels:
            k = int(kernel_size)
            if k < 1 or k % 2 == 0:
                raise ValueError(f"detail kernels must be odd positive ints, got {k}")
            branches.append(nn.Sequential(*_conv_norm_relu(in_ch, branch_ch, k, norm_type)))
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.Sequential(*_conv_norm_relu(branch_ch * len(branches), out_ch, 1, norm_type))
        self.skip: nn.Module = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), make_norm_1d(out_ch, norm_type))
        )

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1)) + self.skip(x)


class StagedFeatureExtractor(nn.Module):
    """Widen channels, e.g. 1→32→64→128."""

    def __init__(self, input_channels, channel_schedule, stem_kernel_size=7, stage_kernel_size=5, norm_type="batch"):
        super().__init__()
        if not channel_schedule:
            raise ValueError("channel_schedule must contain at least one width.")
        layers, in_ch = [], int(input_channels)
        for i, out_ch in enumerate(channel_schedule):
            k = stem_kernel_size if i == 0 else stage_kernel_size
            layers.extend(_conv_norm_relu(in_ch, int(out_ch), int(k), norm_type))
            in_ch = int(out_ch)
        self.stages = nn.Sequential(*layers)

    def forward(self, x):
        return self.stages(x)


def _build_stem(*, input_channels, hidden_channels, channel_schedule, use_multiscale_stem,
                detail_kernels, detail_branch_channels, stem_kernel_size, stage_kernel_size, stem_norm_type):
    ms = dict(kernels=detail_kernels, branch_channels=int(detail_branch_channels), norm_type=stem_norm_type)
    if channel_schedule:
        schedule = [int(w) for w in channel_schedule]
        if schedule[-1] != hidden_channels:
            raise ValueError(
                f"hidden_channels must match last channel_schedule entry; "
                f"got {hidden_channels} vs {schedule}."
            )
        if not use_multiscale_stem:
            return StagedFeatureExtractor(
                input_channels, schedule, int(stem_kernel_size), int(stage_kernel_size), stem_norm_type
            )
        stages: list[nn.Module] = [MultiScaleWaveformStem(input_channels, schedule[0], **ms)]
        if schedule[1:]:
            stages.append(StagedFeatureExtractor(
                schedule[0], schedule[1:], int(stage_kernel_size), int(stage_kernel_size), stem_norm_type
            ))
        return nn.Sequential(*stages)
    if use_multiscale_stem:
        return MultiScaleWaveformStem(input_channels, hidden_channels, **ms)
    return nn.Sequential(*_conv_norm_relu(input_channels, hidden_channels, int(stem_kernel_size), stem_norm_type))


class ApplianceHead(nn.Module):
    """Shared Z → appliance features F → gated power and state logit."""

    def __init__(self, hidden_channels, dropout, *, gate_mode="soft_train_hard_eval", gate_threshold=0.5,
                 off_norm=0.0, head_local_layers=2, head_kernel_size=3, head_use_residual=True,
                 norm_type="batch", use_task_attention=False, task_attention_reduction=4):
        super().__init__()
        self.gate_mode = str(gate_mode or "soft").lower()
        self.gate_threshold = float(gate_threshold)
        self.register_buffer("off_norm", torch.tensor(float(off_norm), dtype=torch.float32))
        if use_task_attention:
            att_ch = max(4, int(hidden_channels) // max(int(task_attention_reduction), 1))
            self.task_attention: nn.Module | None = nn.Sequential(
                nn.Conv1d(hidden_channels, att_ch, 1), nn.ReLU(inplace=True),
                nn.Conv1d(att_ch, hidden_channels, 1), nn.Sigmoid(),
            )
            nn.init.zeros_(self.task_attention[2].weight)
            nn.init.constant_(self.task_attention[2].bias, 2.0)
        else:
            self.task_attention = None
        n_local = int(head_local_layers)
        self.head_use_residual = bool(head_use_residual) and n_local > 0
        if n_local <= 0:
            self.local_decoder = nn.Sequential(*_conv_norm_relu(hidden_channels, hidden_channels, 1, norm_type))
        else:
            k = int(head_kernel_size)
            if k < 1 or k % 2 == 0:
                raise ValueError(f"head_kernel_size must be odd positive, got {k}")
            blocks: list[nn.Module] = []
            for _ in range(n_local):
                blocks.extend(_conv_norm_relu(hidden_channels, hidden_channels, k, norm_type))
            self.local_decoder = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.power_head = nn.Conv1d(hidden_channels, 1, 1)
        self.state_head = nn.Conv1d(hidden_channels, 1, 1)
        self.feature_refine = self.local_decoder

    def encode_features(self, z):
        if self.task_attention is not None:
            z = z * self.task_attention(z)
        f = self.local_decoder(z)
        return self.dropout(f + z if self.head_use_residual else f)

    def decode_from_features(self, features):
        r, s = self.power_head(features), self.state_head(features)
        g = state_gate(torch.sigmoid(s), mode=self.gate_mode, threshold=self.gate_threshold, training=self.training)
        # y_off is 0 W in z-score space, not numeric 0.
        return g * r + (1.0 - g) * self.off_norm, s

    def forward(self, shared_features):
        return self.decode_from_features(self.encode_features(shared_features))


class CrossApplianceDistill(nn.Module):
    """F'_k = F_k + α Mix(F). Used when yaml mode is bottleneck."""

    def __init__(self, num_appliances, channels, *, residual_scale=0.5, dropout=0.0, mid_channels=None):
        super().__init__()
        self.num_appliances = int(num_appliances)
        self.channels = int(channels)
        self.residual_scale = float(residual_scale)
        stacked = self.num_appliances * self.channels
        mid = int(mid_channels) if mid_channels is not None else max(2 * self.channels, 64)
        mid = max(1, min(mid, stacked))
        self.mix = nn.Sequential(
            nn.Conv1d(stacked, mid, 1), nn.ReLU(inplace=True), nn.Dropout(float(dropout)), nn.Conv1d(mid, stacked, 1)
        )

    def forward(self, features):
        if len(features) != self.num_appliances:
            raise ValueError(f"expected {self.num_appliances} maps, got {len(features)}")
        stacked = torch.stack(features, dim=1)
        bsz, _, channels, time_len = stacked.shape
        mixed = self.mix(stacked.reshape(bsz, self.num_appliances * channels, time_len))
        mixed = mixed.reshape(bsz, self.num_appliances, channels, time_len)
        return [features[k] + self.residual_scale * mixed[:, k] for k in range(self.num_appliances)]


class CrossApplianceRelationAttention(nn.Module):
    """Per-timestep attention over A appliance tokens, not over time.

        α = softmax(Q K^T / √D)                      (B, T, A, A)
        F'_i = F_i + ρ · σ([F_i, msg_i]) ⊙ msg_i
    """

    def __init__(self, num_appliances, channels, *, residual_scale=0.5, dropout=0.0, attention_channels=16):
        super().__init__()
        self.num_appliances = int(num_appliances)
        self.channels = int(channels)
        self.residual_scale = float(residual_scale)
        d = max(4, min(int(attention_channels), self.channels))
        self.relation_channels = d
        self.query = nn.Conv1d(self.channels, d, 1)
        self.key = nn.Conv1d(self.channels, d, 1)
        self.value = nn.Conv1d(self.channels, d, 1)
        self.out = nn.Conv1d(d, self.channels, 1)
        self.message_gate = nn.Sequential(nn.Conv1d(2 * self.channels, self.channels, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(float(dropout))
        self.scale = math.sqrt(float(d))

    def forward(self, features):
        if len(features) != self.num_appliances:
            raise ValueError(f"expected {self.num_appliances} maps, got {len(features)}")
        stacked = torch.stack(features, dim=1)
        batch, n_app, channels, time_len = stacked.shape
        d = self.relation_channels
        flat = stacked.reshape(batch * n_app, channels, time_len)

        def proj(layer):
            return layer(flat).reshape(batch, n_app, d, time_len).permute(0, 3, 1, 2)

        q, k, v = proj(self.query), proj(self.key), proj(self.value)
        w = torch.softmax(torch.einsum("btkd,btjd->btkj", q, k) / self.scale, dim=-1)
        ctx = torch.einsum("btkj,btjd->btkd", w, v)
        message = self.out(ctx.permute(0, 2, 3, 1).reshape(batch * n_app, d, time_len))
        message = message.reshape(batch, n_app, self.channels, time_len)
        out = []
        for i, feat in enumerate(features):
            msg = message[:, i]
            gate = self.message_gate(torch.cat([feat, msg], dim=1))
            out.append(feat + self.residual_scale * gate * self.dropout(msg))
        return out


class MultiNILM(nn.Module):
    def __init__(
        self,
        input_channels=1, num_appliances=5, output_length=64, hidden_channels=64,
        channel_schedule=None, stem_kernel_size=7, stage_kernel_size=5, num_blocks=5,
        kernel_size=5, dropout=0.1, max_dilation=128, gate_mode="soft_train_hard_eval",
        gate_threshold=0.5, appliance_off_norm=None, domain_feature_layers=None,
        head_local_layers=2, head_kernel_size=3, head_use_residual=True,
        use_multiscale_stem=False, detail_kernels=None, detail_branch_channels=12,
        stem_norm_type="batch", temporal_norm_type="batch", head_norm_type="batch",
        task_attention_enabled=False, task_attention_reduction=4,
        cross_appliance_enabled=False, cross_appliance_mode="bottleneck",
        cross_appliance_residual_scale=0.5, cross_appliance_mid_channels=None,
        cross_appliance_attention_channels=16,
    ):
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
            raise ValueError(f"appliance_off_norm length {len(off_norms)} != {self.num_appliances}")

        self.aggregate_feature_extractor = _build_stem(
            input_channels=self.input_channels, hidden_channels=self.hidden_channels,
            channel_schedule=[int(w) for w in channel_schedule] if channel_schedule else None,
            use_multiscale_stem=bool(use_multiscale_stem),
            detail_kernels=[int(k) for k in (detail_kernels or [3, 5, 9])],
            detail_branch_channels=int(detail_branch_channels),
            stem_kernel_size=int(stem_kernel_size), stage_kernel_size=int(stage_kernel_size),
            stem_norm_type=stem_norm_type,
        )
        cycle = max(1, int(max_dilation)).bit_length()
        self.temporal_encoder = nn.Sequential(*[
            ResidualTemporalBlock(self.hidden_channels, kernel_size, 2 ** (i % cycle), dropout, temporal_norm_type)
            for i in range(num_blocks)
        ])
        self.appliance_heads = nn.ModuleList([
            ApplianceHead(
                self.hidden_channels, dropout, gate_mode=self.gate_mode, gate_threshold=self.gate_threshold,
                off_norm=off_norms[i], head_local_layers=int(head_local_layers),
                head_kernel_size=int(head_kernel_size), head_use_residual=bool(head_use_residual),
                norm_type=head_norm_type, use_task_attention=bool(task_attention_enabled),
                task_attention_reduction=int(task_attention_reduction),
            )
            for i in range(self.num_appliances)
        ])
        self.cross_appliance_distill = None
        if cross_appliance_enabled:
            mode = str(cross_appliance_mode or "bottleneck").lower()
            kw = dict(num_appliances=self.num_appliances, channels=self.hidden_channels,
                      residual_scale=float(cross_appliance_residual_scale), dropout=float(dropout))
            if mode in {"relation_attention", "attention", "relational"}:
                self.cross_appliance_distill = CrossApplianceRelationAttention(
                    attention_channels=int(cross_appliance_attention_channels), **kw
                )
            elif mode in {"bottleneck", "distill", "pad_lite"}:
                self.cross_appliance_distill = CrossApplianceDistill(mid_channels=cross_appliance_mid_channels, **kw)
            else:
                raise ValueError(f"cross_appliance.mode must be bottleneck|relation_attention, got {cross_appliance_mode!r}")

    def _format_input(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[-1] == self.input_channels:
            x = x.permute(0, 2, 1)
        if x.dim() != 3:
            raise ValueError(f"expected (B,T), (B,C,T) or (B,T,C), got {tuple(x.shape)}")
        if x.shape[1] != self.input_channels:
            raise ValueError(f"expected {self.input_channels} input channel(s), got {x.shape[1]}")
        return x.float()

    def _align_output_time(self, features):
        t = features.shape[-1]
        if t == self.output_length:
            return features
        if t > self.output_length:
            off = (t - self.output_length) // 2
            return features[:, :, off:off + self.output_length]
        pad = self.output_length - t
        left = pad // 2
        return F.pad(features, (left, pad - left))

    def available_domain_feature_layers(self):
        names = ["stem", "temporal", "aligned"]
        names.extend(f"temporal_{i}" for i in range(len(self.temporal_encoder)))
        return names

    def _validate_domain_feature_layers(self, layers):
        layers = normalize_domain_feature_layers(layers)
        allowed = set(self.available_domain_feature_layers())
        unknown = [n for n in layers if n not in allowed]
        if unknown:
            raise ValueError(f"Unknown domain_feature_layers {unknown}. Choose from {sorted(allowed)}.")
        return layers

    def forward(self, x, *, return_domain_features=False, domain_feature_layers=None):
        domain_feats, want = {}, set()
        if return_domain_features:
            want = set(self._validate_domain_feature_layers(
                domain_feature_layers if domain_feature_layers is not None else self.domain_feature_layers
            ))
        h = self.aggregate_feature_extractor(self._format_input(x))
        if "stem" in want:
            domain_feats["stem"] = h
        if any(n.startswith("temporal_") for n in want) or "temporal" in want:
            for i, block in enumerate(self.temporal_encoder):
                h = block(h)
                if f"temporal_{i}" in want:
                    domain_feats[f"temporal_{i}"] = h
            if "temporal" in want:
                domain_feats["temporal"] = h
        else:
            h = self.temporal_encoder(h)
        z = self._align_output_time(h)
        if "aligned" in want:
            domain_feats["aligned"] = z
        feats = [head.encode_features(z) for head in self.appliance_heads]
        if self.cross_appliance_distill is not None:
            feats = self.cross_appliance_distill(feats)
        powers, states = zip(*[head.decode_from_features(f) for head, f in zip(self.appliance_heads, feats)])
        power_pred = torch.cat(powers, dim=1).permute(0, 2, 1)
        state_logits = torch.cat(states, dim=1).permute(0, 2, 1)
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
    gate_mode: str = "soft_train_hard_eval"
    gate_threshold: float = 0.5
    head_local_layers: int = 2
    head_kernel_size: int = 3
    head_use_residual: bool = True
    use_multiscale_stem: bool = False
    detail_kernels: list[int] = field(default_factory=lambda: [3, 5, 9])
    detail_branch_channels: int = 12
    stem_norm_type: str = "batch"
    temporal_norm_type: str = "batch"
    head_norm_type: str = "batch"
    task_attention_enabled: bool = False
    task_attention_reduction: int = 4
    cross_appliance_enabled: bool = False
    cross_appliance_mode: str = "bottleneck"
    cross_appliance_residual_scale: float = 0.5
    cross_appliance_mid_channels: int | None = None
    cross_appliance_attention_channels: int = 16
    domain_feature_layers: list[str] = field(default_factory=lambda: ["temporal_2", "temporal_4", "aligned"])


def _parse_cross_appliance(architecture):
    block = architecture.get("cross_appliance")
    if not isinstance(block, dict):
        return False, "bottleneck", 0.5, None, 16
    mid = block.get("mid_channels", None)
    return (bool(block.get("enabled", False)), str(block.get("mode", "bottleneck")),
            float(block.get("residual_scale", 0.5)), None if mid is None else int(mid),
            int(block.get("attention_channels", 16)))


def build_multinilm(cfg, *, num_appliances, output_length, appliance_off_norm=None, input_channels=None):
    kwargs = asdict(cfg)
    kwargs.update(num_appliances=int(num_appliances), output_length=int(output_length),
                  appliance_off_norm=appliance_off_norm)
    if input_channels is not None:
        kwargs["input_channels"] = int(input_channels)
    return MultiNILM(**kwargs)


def multinilm_config(architecture):
    a = architecture
    task = a.get("task_attention") if isinstance(a.get("task_attention"), dict) else {}
    en, mode, scale, mid, attn = _parse_cross_appliance(a)
    return MultiNILMConfig(
        input_channels=int(a.get("input_channels", a.get("input_size", 1))),
        num_appliances=int(a.get("num_appliances", 5)),
        output_length=int(a.get("output_length", 64)),
        hidden_channels=int(a.get("hidden_channels", a.get("hidden", 64))),
        channel_schedule=a.get("channel_schedule"),
        stem_kernel_size=int(a.get("stem_kernel_size", 7)),
        stage_kernel_size=int(a.get("stage_kernel_size", 5)),
        num_blocks=int(a.get("num_blocks", 5)),
        kernel_size=int(a.get("kernel_size", 5)),
        dropout=float(a.get("dropout", 0.1)),
        max_dilation=int(a.get("max_dilation", 128)),
        gate_mode=str(a.get("gate_mode", "soft_train_hard_eval")),
        gate_threshold=float(a.get("gate_threshold", 0.5)),
        head_local_layers=int(a.get("head_local_layers", 2)),
        head_kernel_size=int(a.get("head_kernel_size", 3)),
        head_use_residual=bool(a.get("head_use_residual", True)),
        use_multiscale_stem=bool(a.get("use_multiscale_stem", False)),
        detail_kernels=[int(k) for k in a.get("detail_kernels", [3, 5, 9])],
        detail_branch_channels=int(a.get("detail_branch_channels", 12)),
        stem_norm_type=str(a.get("stem_norm_type", "batch")),
        temporal_norm_type=str(a.get("temporal_norm_type", "batch")),
        head_norm_type=str(a.get("head_norm_type", "batch")),
        task_attention_enabled=bool(task.get("enabled", False)),
        task_attention_reduction=int(task.get("reduction", 4)),
        cross_appliance_enabled=en, cross_appliance_mode=mode,
        cross_appliance_residual_scale=scale, cross_appliance_mid_channels=mid,
        cross_appliance_attention_channels=attn,
        domain_feature_layers=normalize_domain_feature_layers(a.get("domain_feature_layers")),
    )


def gl_binomial_weights(alpha, memory):
    """w[0]=1, w[j]=w[j-1]*(j-1-α)/j.  (D^α p)[t] ≈ Σ_j w[j] p[t-j]."""
    if memory < 0:
        raise ValueError(f"memory must be >= 0, got {memory}")
    w = np.empty(memory + 1, dtype=np.float64)
    w[0] = 1.0
    for j in range(1, memory + 1):
        w[j] = w[j - 1] * (j - 1 - float(alpha)) / j
    return w


def default_schirmer_alphas(k=8):
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return [1.0] if k == 1 else [round((i + 1) / k, 6) for i in range(k)]


class FractionalFrontEnd(nn.Module):
    """Frozen GL bank + optional raw/delta/rolling. (B,1,T) → (B,C,T)."""

    def __init__(self, alphas=None, *, include_raw=True, memory=None, h=1.0, max_memory=256,
                 channel_normalize="mean_std", channel_norm_eps=1e-5, include_delta=False,
                 include_abs_delta=False, rolling_windows=None, include_rolling_mean=False,
                 include_rolling_std=False):
        super().__init__()
        self.alphas = [float(a) for a in alphas] if alphas is not None else default_schirmer_alphas(8)
        if not self.alphas and not include_raw:
            raise ValueError("need at least one alpha or include_raw=True")
        self.include_raw = bool(include_raw)
        self.h = float(h)
        self.memory = int(memory) if memory is not None else int(max_memory)
        if self.memory < 1:
            raise ValueError(f"memory must be >= 1, got {self.memory}")
        self.channel_normalize = str(channel_normalize)
        if self.channel_normalize not in {"mean_std", "none"}:
            raise ValueError(f"channel_normalize must be mean_std|none, got {self.channel_normalize!r}")
        self.channel_norm_eps = float(channel_norm_eps)
        self.include_delta = bool(include_delta)
        self.include_abs_delta = bool(include_abs_delta)
        self.rolling_windows = [int(w) for w in (rolling_windows or [])]
        if any(w < 1 for w in self.rolling_windows):
            raise ValueError(f"rolling_windows must be positive, got {self.rolling_windows}")
        self.include_rolling_mean = bool(include_rolling_mean)
        self.include_rolling_std = bool(include_rolling_std)
        extra = int(self.include_delta) + int(self.include_abs_delta)
        if self.include_rolling_mean:
            extra += len(self.rolling_windows)
        if self.include_rolling_std:
            extra += len(self.rolling_windows)
        self.out_channels = (1 if self.include_raw else 0) + len(self.alphas) + extra
        if self.alphas:
            kernels = []
            for alpha in self.alphas:
                w = gl_binomial_weights(alpha, self.memory) / (self.h ** alpha)
                kernels.append(torch.tensor(np.asarray(w, dtype=np.float64)[::-1].copy(), dtype=torch.float32))
            weight = torch.stack(kernels, dim=0).unsqueeze(1)
            n_a, k_len = len(self.alphas), int(weight.shape[-1])
            self.gl_conv = nn.Conv1d(n_a, n_a, k_len, groups=n_a, bias=False, padding=0)
            with torch.no_grad():
                self.gl_conv.weight.copy_(weight)
            self.gl_conv.weight.requires_grad_(False)
            self.register_buffer("gl_weight", self.gl_conv.weight, persistent=False)
        else:
            self.gl_conv = None
            self.register_buffer("gl_weight", torch.zeros(0), persistent=True)

    def extra_repr(self):
        w_shape = tuple(self.gl_weight.shape) if self.gl_weight.numel() else None
        return (f"alphas={self.alphas}, include_raw={self.include_raw}, memory={self.memory}, "
                f"out_channels={self.out_channels}, channel_normalize={self.channel_normalize!r}, "
                f"delta={self.include_delta}, abs_delta={self.include_abs_delta}, "
                f"rolling_windows={self.rolling_windows}, rolling_mean={self.include_rolling_mean}, "
                f"rolling_std={self.include_rolling_std}, gl_weight={w_shape}")

    @staticmethod
    def _delta(x):
        return torch.cat([torch.zeros_like(x[..., :1]), x[..., 1:] - x[..., :-1]], dim=-1)

    @staticmethod
    def _rolling_mean(x, window):
        if window <= 1:
            return x
        return F.avg_pool1d(F.pad(x, (window - 1, 0), mode="replicate"), window, stride=1)

    def _rolling_std(self, x, window):
        if window <= 1:
            return torch.zeros_like(x)
        mean = self._rolling_mean(x, window)
        var = (self._rolling_mean(x * x, window) - mean * mean).clamp_min(0.0)
        return torch.sqrt(var + self.channel_norm_eps)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"FractionalFrontEnd expected (B,C,T), got {tuple(x.shape)}")
        if x.shape[1] != 1:
            x = x[:, :1, :]
        parts = []
        if self.include_raw:
            parts.append(x)
        if self.include_delta or self.include_abs_delta:
            delta = self._delta(x)
            if self.include_delta:
                parts.append(delta)
            if self.include_abs_delta:
                parts.append(delta.abs())
        for window in self.rolling_windows:
            if self.include_rolling_mean:
                parts.append(self._rolling_mean(x, window))
            if self.include_rolling_std:
                parts.append(self._rolling_std(x, window))
        if self.alphas:
            pad = int(self.gl_conv.weight.shape[-1]) - 1
            parts.append(self.gl_conv(F.pad(x, (pad, 0)).expand(-1, len(self.alphas), -1)))
        out = torch.cat(parts, dim=1)
        if self.channel_normalize == "mean_std":
            mu = out.mean(dim=-1, keepdim=True)
            sigma = out.std(dim=-1, keepdim=True).clamp_min(self.channel_norm_eps)
            out = (out - mu) / sigma
        return out


@dataclass
class FractionalSettings:
    enabled: bool = False
    alphas: list[float] | None = None
    include_raw: bool = True
    memory: int | None = None
    h: float = 1.0
    channel_normalize: str = "mean_std"
    include_delta: bool = False
    include_abs_delta: bool = False
    rolling_windows: list[int] = field(default_factory=list)
    include_rolling_mean: bool = False
    include_rolling_std: bool = False

    def resolved_alphas(self):
        return list(self.alphas) if self.alphas is not None else default_schirmer_alphas(8)


def parse_fractional_architecture(architecture: dict[str, Any]) -> FractionalSettings:
    block = architecture.get("fractional")
    if not isinstance(block, dict):
        return FractionalSettings()
    alphas_raw = block.get("alphas", None)
    alphas = default_schirmer_alphas(int(block.get("k", 8))) if alphas_raw is None else [float(a) for a in alphas_raw]
    memory = block.get("memory", None)
    return FractionalSettings(
        enabled=bool(block.get("enabled", False)), alphas=alphas,
        include_raw=bool(block.get("include_raw", True)),
        memory=None if memory is None else int(memory), h=float(block.get("h", 1.0)),
        channel_normalize=str(block.get("channel_normalize", "mean_std")),
        include_delta=bool(block.get("include_delta", False)),
        include_abs_delta=bool(block.get("include_abs_delta", False)),
        rolling_windows=[int(w) for w in (block.get("rolling_windows") or [])],
        include_rolling_mean=bool(block.get("include_rolling_mean", False)),
        include_rolling_std=bool(block.get("include_rolling_std", False)),
    )


class MultiNILMFractional(nn.Module):
    """Wrapper so checkpoints stay frontend.* / backbone.*."""

    def __init__(self, *, backbone: MultiNILM, frontend: FractionalFrontEnd):
        super().__init__()
        if int(backbone.input_channels) != int(frontend.out_channels):
            raise ValueError(
                f"backbone input_channels must be {frontend.out_channels}, got {backbone.input_channels}."
            )
        self.frontend = frontend
        self.backbone = backbone
        self.input_channels = 1
        self.feature_channels = int(frontend.out_channels)
        self.num_appliances = backbone.num_appliances
        self.output_length = backbone.output_length
        self.domain_feature_layers = backbone.domain_feature_layers

    def forward(self, x, return_domain_features=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[-1] == 1:
            x = x.permute(0, 2, 1)
        return self.backbone(self.frontend(x.float()), return_domain_features=return_domain_features)


def build_multinilm_fractional(architecture, *, num_appliances, output_length, appliance_off_norm=None):
    settings = parse_fractional_architecture(architecture)
    frontend = FractionalFrontEnd(
        alphas=settings.resolved_alphas(), include_raw=settings.include_raw, memory=settings.memory,
        h=settings.h, channel_normalize=settings.channel_normalize, include_delta=settings.include_delta,
        include_abs_delta=settings.include_abs_delta, rolling_windows=settings.rolling_windows,
        include_rolling_mean=settings.include_rolling_mean, include_rolling_std=settings.include_rolling_std,
    )
    feature_c = int(frontend.out_channels)
    arch = dict(architecture)
    arch["input_channels"] = feature_c
    backbone = build_multinilm(
        multinilm_config(arch), num_appliances=num_appliances, output_length=output_length,
        appliance_off_norm=appliance_off_norm, input_channels=feature_c,
    )
    return MultiNILMFractional(backbone=backbone, frontend=frontend)


def _to_numpy(t):
    return t.detach().float().cpu().numpy()


def _resolve_pos_weight(adapter, loss_cfg):
    configured = loss_cfg.get("pos_weight")
    if configured is not None and str(configured).lower() not in {"auto", "null", "none"}:
        return configured
    weights = adapter._data_loader().estimate_state_pos_weights("train")
    cap = loss_cfg.get("pos_weight_cap", None)
    if cap not in (None, "", "none", "null"):
        weights = np.minimum(weights, float(cap))
    return weights.tolist()


def _pred_on_from_config(adapter, power_norm, state_prob):
    source = str(adapter.model_cfg.get("evaluation", {}).get("pred_on_source", "state_head")).lower()
    if source == "state_head":
        return (state_prob >= 0.5).astype(np.int32)
    loader = adapter._data_loader()
    if loader.state_threshold_watts is None:
        raise ValueError("pred_on_source=power_threshold requires threshold training labels")
    power_on = (loader.denorm_to_watts(power_norm) > np.asarray(loader.state_threshold_watts, np.float32)).astype(np.int32)
    if source == "power_threshold":
        return power_on
    if source == "combined":
        return np.maximum((state_prob >= 0.5).astype(np.int32), power_on).astype(np.int32)
    raise ValueError("evaluation.pred_on_source must be state_head, power_threshold, or combined")


class MultiNILMAdapter(BaseNILMAdapter):
    name = "multinilm"

    def build_model(self, device):
        apps = self.cfg["appliances"]
        return build_multinilm(
            multinilm_config(self.model_cfg["architecture"]),
            num_appliances=len(apps),
            output_length=int(self.model_cfg["windowing"].get("output_window_length", 1)),
            appliance_off_norm=appliance_off_norm_normalized(self.experiment, apps),
        ).to(device)

    def build_loss(self):
        from model.MultiNILM_loss import MultiNILMLoss
        cfg = self.model_cfg.get("loss", {})
        loader = self._data_loader()
        mmd = cfg.get("mmd_sigma", None)
        return MultiNILMLoss(
            lambda_state=float(cfg.get("lambda_state", 0.1)),
            task_balance=str(cfg.get("task_balance", "none")),
            pos_weight=_resolve_pos_weight(self, cfg),
            power_scale=loader.loss_scale,
            target_mean=loader.norm.target_mean,
            lambda_domain=float(cfg.get("lambda_domain", 0.0)),
            domain_method=str(cfg.get("domain_method", "coral")),
            domain_mu=float(cfg.get("domain_mu", 0.4)),
            domain_mix=str(cfg.get("domain_mix", "convex")),
            domain_scale=str(cfg.get("domain_scale", "none")),
            power_on_weight=float(cfg.get("power_on_weight", 0.0)),
            power_off_weight=float(cfg.get("power_off_weight", 0.0)),
            power_delta_weight=float(cfg.get("power_delta_weight", 0.0)),
            power_delta_on_only=bool(cfg.get("power_delta_on_only", True)),
            power_energy_weight=float(cfg.get("power_energy_weight", 0.0)),
            state_fp_weight=float(cfg.get("state_fp_weight", 0.0)),
            power_energy_relative_weight=float(cfg.get("power_energy_relative_weight", 0.0)),
            energy_floor_watts=float(cfg.get("energy_floor_watts", 10.0)),
            mmd_sigma=None if mmd in (None, "", "auto") else float(mmd),
        )

    def step(self, model, loss_fn, batch, target_batch=None):
        import torch
        x, y, z = batch
        z = z.float()
        if target_batch is not None and float(getattr(loss_fn, "lambda_domain", 0.0)) != 0.0:
            x_t = target_batch[0] if isinstance(target_batch, (tuple, list)) else target_batch
            power_pred, state_logits, feats_s = model(x, return_domain_features=True)
            _, _, feats_t = model(x_t, return_domain_features=True)
            out = loss_fn(power_pred, state_logits, y, z, domain_feats_S=feats_s, domain_feats_T=feats_t)
        else:
            power_pred, state_logits = model(x)
            out = loss_fn(power_pred, state_logits, y, z)
        state_prob = torch.sigmoid(state_logits)
        pred_state = torch.from_numpy(
            _pred_on_from_config(self, _to_numpy(power_pred), _to_numpy(state_prob))
        ).long()
        app_logs = {
            f"loss_power_{app}": float(out.loss_power_per_appliance[i].detach())
            for i, app in enumerate(self.cfg["appliances"])
        }
        app_logs.update({
            f"loss_state_{app}": float(out.loss_state_per_appliance[i].detach())
            for i, app in enumerate(self.cfg["appliances"])
        })
        return StepOutput(
            loss=out.loss,
            logs={
                "loss": float(out.loss.detach()),
                "loss_power": float(out.loss_power.detach()),
                "loss_state": float(out.loss_state.detach()),
                "loss_state_term": float(out.loss_state_term.detach()),
                "loss_energy_relative": float(out.loss_energy_relative.detach()),
                "loss_domain": float(out.loss_domain.detach()),
                "loss_domain_term": float(out.loss_domain_term.detach()),
                "mae": float(out.mae.detach()),
                **app_logs,
            },
            aux={
                "pred_state": pred_state.detach().cpu(),
                # Keep continuous scores for threshold-free validation AP.
                "state_prob": state_prob.detach().float().cpu(),
                "true_state": z.long().detach().cpu(),
                "pred_power": power_pred.detach().float().cpu(),
                "true_power": y.detach().cpu(),
            },
        )

    @torch.no_grad()
    def predict_dataloader(self, model, loader, device, *, max_batches=None, split="test"):
        import torch
        model.eval()
        pred_power, pred_state, true_power, true_state, sample_indices = [], [], [], [], []
        offset = 0
        for i, (x, y, z) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            power_pred, state_logits = model(x.to(device))
            pred_power.append(_to_numpy(power_pred))
            pred_state.append(_to_numpy(torch.sigmoid(state_logits)))
            true_power.append(y.numpy())
            true_state.append(z.numpy())
            sample_indices.append(self._sample_index(offset, len(x)))
            offset += len(x)
        return self.finalize_prediction_bundle(
            split=split, sample_indices=sample_indices,
            pred_power_batches=pred_power, pred_state_batches=pred_state,
            true_power_batches=true_power, true_state_batches=true_state,
        )


class MultiNILMFractionalAdapter(MultiNILMAdapter):
    name = "multinilm_fractional"

    def build_model(self, device):
        arch = dict(self.model_cfg["architecture"])
        frac = self.model_cfg.get("fractional")
        if isinstance(frac, dict):
            arch["fractional"] = frac
        apps = self.cfg["appliances"]
        return build_multinilm_fractional(
            arch,
            num_appliances=len(apps),
            output_length=int(self.model_cfg["windowing"].get("output_window_length", 1)),
            appliance_off_norm=appliance_off_norm_normalized(self.experiment, apps),
        ).to(device)

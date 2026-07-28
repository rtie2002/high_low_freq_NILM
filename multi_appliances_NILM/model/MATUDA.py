"""
MATUDA temporal encoder + shared FC adaptation tower + per-appliance heads.

Design:
  - Lin TSG 2022: length-preserving TCN + domain loss on FC (fc6–fc8 analogues).
  - Deep CORAL / DAN: align pooled FC activations; multi-layer MMD.
  - Seq2seq: predict full window (B, T, K), not center-only seq2point.
  - MultiNILM-style: one Conv1d head per appliance.
  - EGC-DA in MATUDA_loss.py.

Shapes (seq2seq):
  x:           (B, 1, T)
  da_features: list[(B, D)]  mean-pooled FC maps for domain loss
  states:      (B, T, K)     multi-label logits over the window
  powers:      (B, T, K)     power over the window (normalized space)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """Dilated causal residual block (Lin-style TCN unit)."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def _crop(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(-1) == target_len:
            return x
        return x[..., :target_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(-1)
        y = self.relu(self._crop(self.conv1(x), t))
        y = self.dropout(y)
        y = self.relu(self._crop(self.conv2(y), t))
        y = self.dropout(y)
        return self.relu(x + y)


class MATUDAApplianceHead(nn.Module):
    """One appliance-specific temporal head (MultiNILM-style Conv1d)."""

    def __init__(
        self,
        in_channels: int,
        *,
        head_hidden: int = 64,
        dropout: float = 0.15,
        use_gate: bool = True,
        gate_mode: str = "soft",
        off_norm: float = 0.0,
        head_kernel_size: int = 3,
    ):
        super().__init__()
        self.use_gate = bool(use_gate)
        self.gate_mode = str(gate_mode or "soft").lower()
        self.register_buffer("off_norm", torch.tensor(float(off_norm), dtype=torch.float32))

        hid = int(head_hidden)
        k = int(head_kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError(f"head_kernel_size must be odd positive, got {k}")
        pad = k // 2
        self.refine = nn.Sequential(
            nn.Conv1d(in_channels, hid, kernel_size=k, padding=pad),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hid, hid, kernel_size=k, padding=pad),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.state_head = nn.Conv1d(hid, 1, kernel_size=1)
        self.power_head = nn.Conv1d(hid, 1, kernel_size=1)

    def _gate(self, state_logits: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(state_logits)
        if self.gate_mode in {"hard", "binary"}:
            hard = (p >= 0.5).to(p.dtype)
            return hard + (p - p.detach())
        return p

    def forward(self, shared: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """shared (B, C, T) → power/state_logit each (B, T)."""
        h = self.refine(shared)
        state_logit = self.state_head(h).squeeze(1)
        power_raw = self.power_head(h).squeeze(1)
        if self.use_gate:
            g = self._gate(state_logit)
            power = g * power_raw + (1.0 - g) * self.off_norm.to(
                dtype=power_raw.dtype, device=power_raw.device
            )
        else:
            power = power_raw
        return power, state_logit


class MATUDANet(nn.Module):
    """
    Seq2seq multi-appliance net with Lin-style FC UDA hooks.

    Shared stem + TCN + 1×1 FC tower (length preserved); K Conv1d appliance heads.
    Domain features are mean-pooled over time from each FC map → (B, D).
    """

    def __init__(
        self,
        num_appliances: int,
        seq_len: int = 480,
        conv_channels: int = 96,
        tcn_blocks: int = 8,
        fc_dims: Tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.15,
        use_gate: bool = True,
        stem_kernels: Tuple[int, ...] = (3, 5, 9),
        appliance_off_norm: Tuple[float, ...] | list[float] | None = None,
        gate_mode: str = "soft",
        head_hidden: int = 64,
        head_kernel_size: int = 3,
        use_instance_norm: bool = False,
    ):
        super().__init__()
        self.num_appliances = num_appliances
        self.seq_len = seq_len
        self.use_gate = use_gate
        self.fc_dims = fc_dims
        self.gate_mode = str(gate_mode or "soft").lower()
        self.use_instance_norm = bool(use_instance_norm)

        off = list(appliance_off_norm or [0.0] * num_appliances)
        if len(off) != num_appliances:
            raise ValueError(f"appliance_off_norm length {len(off)} != {num_appliances}")

        branch_ch = max(16, conv_channels // len(stem_kernels))
        self.stem_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, branch_ch, kernel_size=k, padding=k // 2),
                    nn.ReLU(inplace=True),
                )
                for k in stem_kernels
            ]
        )
        stem_out = branch_ch * len(stem_kernels)
        self.stem_proj = nn.Sequential(
            nn.Conv1d(stem_out, conv_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        # InstanceNorm reduces house-level scale shift (AHDA / transfer practice).
        self.stem_norm = (
            nn.InstanceNorm1d(conv_channels, affine=True)
            if self.use_instance_norm
            else nn.Identity()
        )

        blocks = []
        for i in range(tcn_blocks):
            blocks.append(
                TemporalBlock(
                    conv_channels, kernel_size=3, dilation=2**i, dropout=dropout
                )
            )
        self.tcn = nn.Sequential(*blocks)

        # Pointwise FC tower along channels (Lin fc6–fc8), length preserved.
        dims = (conv_channels,) + tuple(fc_dims)
        self.fc_layers = nn.ModuleList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.fc_layers.append(
                nn.Sequential(
                    nn.Conv1d(d_in, d_out, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )

        embed_ch = fc_dims[-1]
        self.appliance_heads = nn.ModuleList(
            [
                MATUDAApplianceHead(
                    embed_ch,
                    head_hidden=int(head_hidden),
                    dropout=dropout,
                    use_gate=use_gate,
                    gate_mode=gate_mode,
                    off_norm=float(off[i]),
                    head_kernel_size=int(head_kernel_size),
                )
                for i in range(num_appliances)
            ]
        )

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Return temporal embed (B,C,T), pooled DA feats (B,D), raw FC maps."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        parts = [b(x) for b in self.stem_branches]
        t_min = min(p.size(-1) for p in parts)
        parts = [p[..., :t_min] for p in parts]
        h = self.stem_proj(torch.cat(parts, dim=1))
        h = self.stem_norm(h)
        h = self.tcn(h)

        da_feats: List[torch.Tensor] = []
        fc_maps: List[torch.Tensor] = []
        for layer in self.fc_layers:
            h = layer(h)
            fc_maps.append(h)
            da_feats.append(h.mean(dim=-1))  # (B, D) for CORAL/MMD
        return h, da_feats, fc_maps

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        embed, da_feats, _ = self.encode(x)
        powers_list: List[torch.Tensor] = []
        state_list: List[torch.Tensor] = []
        for head in self.appliance_heads:
            power_i, state_i = head(embed)
            powers_list.append(power_i)
            state_list.append(state_i)
        # (B, T, K) — matches MultiNILM / pipeline window targets
        powers = torch.stack(powers_list, dim=-1)
        state_logits = torch.stack(state_list, dim=-1)
        return {
            "powers": powers,
            "state_logits": state_logits,
            "da_features": da_feats,
            "embedding": embed,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = MATUDANet(num_appliances=5, seq_len=480)
    x = torch.randn(2, 1, 480)
    out = net(x)
    print("powers", tuple(out["powers"].shape))
    print("states", tuple(out["state_logits"].shape))
    print("heads", len(net.appliance_heads))
    print("da", [tuple(t.shape) for t in out["da_features"]])
    print("params", count_parameters(net))

"""
MATUDA: MultiNILM-matched temporal backbone + compact Lin-style FC DA tower.

Design (aligned with MultiNILM yaml dims):
  - Multi-scale stem (k=3,5,9 → 16 ch) → staged 32→64→128
  - 8× ResidualTemporalBlock (C=128, k=5, max_dilation=64)
  - Compact 1×1 FC tower (default 128→256→192→128) for Lin fc-style DA
  - 5× MultiNILM ApplianceHead (local residual, C=128, hard gate)

Shapes (seq2seq):
  x:           (B, 1, T)
  da_features: list[(B, D)]  mean-pooled FC maps for domain loss
  states:      (B, T, K)
  powers:      (B, T, K)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from model.MultiNILM import (
    ApplianceHead,
    MultiScaleWaveformStem,
    ResidualTemporalBlock,
    StagedFeatureExtractor,
)


class MATUDANet(nn.Module):
    """MultiNILM backbone + compact channel-wise FC tower for UDA hooks."""

    def __init__(
        self,
        num_appliances: int,
        seq_len: int = 480,
        *,
        channel_schedule: Tuple[int, ...] | list[int] = (32, 64, 128),
        hidden_channels: int = 128,
        tcn_blocks: int = 8,
        tcn_kernel_size: int = 5,
        max_dilation: int = 64,
        fc_dims: Tuple[int, ...] = (256, 192, 128),
        dropout: float = 0.15,
        use_gate: bool = True,
        stem_kernels: Tuple[int, ...] = (3, 5, 9),
        detail_branch_channels: int = 16,
        stage_kernel_size: int = 5,
        appliance_off_norm: Tuple[float, ...] | list[float] | None = None,
        gate_mode: str = "hard",
        gate_threshold: float = 0.5,
        head_local_layers: int = 2,
        head_kernel_size: int = 3,
        head_use_residual: bool = True,
        # Legacy aliases (ignored if channel_schedule / hidden_channels set)
        conv_channels: int | None = None,
        head_hidden: int | None = None,
        use_instance_norm: bool = False,
    ):
        super().__init__()
        del use_instance_norm  # kept for yaml/adapter backward compat

        self.num_appliances = int(num_appliances)
        self.seq_len = int(seq_len)
        self.use_gate = bool(use_gate)
        self.fc_dims = tuple(int(d) for d in fc_dims)
        self.gate_mode = str(gate_mode or "hard").lower()

        schedule = [int(c) for c in channel_schedule]
        if conv_channels is not None and not schedule:
            schedule = [int(conv_channels)]
        if not schedule:
            schedule = [32, 64, int(hidden_channels)]
        if schedule[-1] != int(hidden_channels):
            # Prefer explicit hidden_channels as TCN width.
            schedule = list(schedule[:-1]) + [int(hidden_channels)]
        self.hidden_channels = int(schedule[-1])
        self.channel_schedule = schedule

        off = list(appliance_off_norm or [0.0] * num_appliances)
        if len(off) != num_appliances:
            raise ValueError(f"appliance_off_norm length {len(off)} != {num_appliances}")

        # --- MultiNILM front-end: multi-scale stem + staged widen ---
        stem_out = schedule[0]
        front: list[nn.Module] = [
            MultiScaleWaveformStem(
                input_channels=1,
                out_channels=stem_out,
                kernels=tuple(int(k) for k in stem_kernels),
                branch_channels=int(detail_branch_channels),
            )
        ]
        if len(schedule) > 1:
            front.append(
                StagedFeatureExtractor(
                    input_channels=stem_out,
                    channel_schedule=schedule[1:],
                    stem_kernel_size=int(stage_kernel_size),
                    stage_kernel_size=int(stage_kernel_size),
                )
            )
        self.front_end = nn.Sequential(*front)

        # --- MultiNILM TCN ---
        blocks: list[nn.Module] = []
        for i in range(int(tcn_blocks)):
            dilation = min(2**i, int(max_dilation))
            blocks.append(
                ResidualTemporalBlock(
                    channels=self.hidden_channels,
                    kernel_size=int(tcn_kernel_size),
                    dilation=dilation,
                    dropout=float(dropout),
                )
            )
        self.tcn = nn.Sequential(*blocks)

        # --- Compact Lin-style FC tower (1×1 Conv along channels) ---
        dims = (self.hidden_channels,) + self.fc_dims
        self.fc_layers = nn.ModuleList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.fc_layers.append(
                nn.Sequential(
                    nn.Conv1d(d_in, d_out, kernel_size=1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        embed_ch = self.fc_dims[-1] if self.fc_dims else self.hidden_channels

        # --- MultiNILM heads (width follows last FC / TCN) ---
        head_ch = int(head_hidden) if head_hidden is not None else embed_ch
        if head_ch != embed_ch:
            # Project to head width if yaml still sets a different head_hidden.
            self.head_proj = nn.Conv1d(embed_ch, head_ch, kernel_size=1)
        else:
            self.head_proj = nn.Identity()

        self.appliance_heads = nn.ModuleList(
            [
                ApplianceHead(
                    hidden_channels=head_ch,
                    dropout=float(dropout),
                    gate_mode=self.gate_mode,
                    gate_threshold=float(gate_threshold),
                    off_norm=float(off[i]),
                    head_local_layers=int(head_local_layers),
                    head_kernel_size=int(head_kernel_size),
                    head_use_residual=bool(head_use_residual),
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
        h = self.front_end(x)
        h = self.tcn(h)

        da_feats: List[torch.Tensor] = []
        fc_maps: List[torch.Tensor] = []
        if self.fc_layers:
            for layer in self.fc_layers:
                h = layer(h)
                fc_maps.append(h)
                da_feats.append(h.mean(dim=-1))
        else:
            fc_maps.append(h)
            da_feats.append(h.mean(dim=-1))
        return h, da_feats, fc_maps

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        embed, da_feats, _ = self.encode(x)
        shared = self.head_proj(embed)
        powers_list: List[torch.Tensor] = []
        state_list: List[torch.Tensor] = []
        for head in self.appliance_heads:
            # ApplianceHead returns (B, 1, T); squeeze to (B, T)
            power_i, state_i = head(shared)
            powers_list.append(power_i.squeeze(1))
            state_list.append(state_i.squeeze(1))
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
    # Module breakdown
    stem = sum(p.numel() for p in net.front_end.parameters())
    tcn = sum(p.numel() for p in net.tcn.parameters())
    fc = sum(p.numel() for p in net.fc_layers.parameters())
    heads = sum(p.numel() for p in net.appliance_heads.parameters())
    print(f"front={stem} tcn={tcn} fc={fc} heads={heads}")

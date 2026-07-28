"""
MATUDA temporal encoder + shared FC adaptation tower + multi-appliance heads.

Design (literature, not MultiNILM copy):
  - Lin TSG 2022: TCN + domain loss on FC layers (fc6–fc8 analogues).
  - Deep CORAL / DAN: align FC activations; multi-layer MMD.
  - Zhang AAAI 2018: sequence-to-point — predict center of the input window.
  - CDAN+E (Long et al.): entropy-aware / prediction-conditioned transfer
    used in our EGC-DA loss (see matuda_loss.py).

Shapes (seq2point):
  x:           (B, 1, T)   T = input_window_length (default 599)
  da_features: list[(B,D)] FC activations for domain loss
  states:      (B, K)      multi-label logits at window center
  powers:      (B, K)      power at window center (normalized space)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MATUDANet(nn.Module):
    """
    Multi-Appliance multi-Task net with FC-layer UDA hooks (seq2point).

    Forward returns powers, state_logits, da_features (list of FC tensors).
    Power gating blends toward per-appliance OFF-norm (z-score of 0 W), not 0.
    """

    def __init__(
        self,
        num_appliances: int,
        seq_len: int = 599,
        conv_channels: int = 96,
        tcn_blocks: int = 8,
        fc_dims: Tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.15,
        use_gate: bool = True,
        stem_kernels: Tuple[int, ...] = (3, 5, 9),
        appliance_off_norm: Tuple[float, ...] | list[float] | None = None,
        gate_mode: str = "soft",
    ):
        super().__init__()
        self.num_appliances = num_appliances
        self.seq_len = seq_len
        self.use_gate = use_gate
        self.fc_dims = fc_dims
        self.gate_mode = str(gate_mode or "soft").lower()

        off = list(appliance_off_norm or [0.0] * num_appliances)
        if len(off) != num_appliances:
            raise ValueError(f"appliance_off_norm length {len(off)} != {num_appliances}")
        self.register_buffer("off_norm", torch.tensor(off, dtype=torch.float32))

        # Multi-scale stem (capture short edges / waveform detail).
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

        blocks = []
        for i in range(tcn_blocks):
            blocks.append(
                TemporalBlock(
                    conv_channels, kernel_size=3, dilation=2**i, dropout=dropout
                )
            )
        self.tcn = nn.Sequential(*blocks)

        # FC adaptation tower (domain loss attaches here — Lin fc6–fc8).
        dims = (conv_channels,) + tuple(fc_dims)
        self.fc_layers = nn.ModuleList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(d_in, d_out),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )

        embed_dim = fc_dims[-1]
        self.state_head = nn.Linear(embed_dim, num_appliances)
        self.power_head = nn.Linear(embed_dim, num_appliances)

    def encode_fc(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Multi-scale stem → project → TCN → GAP → FC tower.
        parts = [b(x) for b in self.stem_branches]
        # Align lengths if odd kernels differ by 1 sample.
        t_min = min(p.size(-1) for p in parts)
        parts = [p[..., :t_min] for p in parts]
        h = self.stem_proj(torch.cat(parts, dim=1))
        h = self.tcn(h)
        h = h.mean(dim=-1)  # GAP → (B, C)

        da_feats: List[torch.Tensor] = []
        for layer in self.fc_layers:
            h = layer(h)
            da_feats.append(h)
        return h, da_feats

    def _gate(self, state_logits: torch.Tensor) -> torch.Tensor:
        if self.gate_mode in {"hard", "binary"}:
            # Straight-through estimator.
            p = torch.sigmoid(state_logits)
            hard = (p >= 0.5).to(p.dtype)
            return hard + (p - p.detach())
        return torch.sigmoid(state_logits)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        embed, da_feats = self.encode_fc(x)
        state_logits = self.state_head(embed)
        # Linear in normalized space (z-scored targets can be negative).
        powers_raw = self.power_head(embed)
        if self.use_gate:
            g = self._gate(state_logits)
            off = self.off_norm.view(1, -1).to(dtype=powers_raw.dtype, device=powers_raw.device)
            powers = g * powers_raw + (1.0 - g) * off
        else:
            powers = powers_raw
        return {
            "powers": powers,
            "state_logits": state_logits,
            "da_features": da_feats,
            "embedding": embed,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = MATUDANet(num_appliances=5, seq_len=599)
    x = torch.randn(4, 1, 599)
    out = net(x)
    print("powers", tuple(out["powers"].shape))
    print("da", [tuple(t.shape) for t in out["da_features"]])
    print("params", count_parameters(net))

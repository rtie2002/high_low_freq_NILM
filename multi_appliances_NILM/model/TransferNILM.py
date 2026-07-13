"""BERT4NILM transfer-learning baseline — shared encoder + per-appliance CNN heads.

Ported from NILM_model/baseline/transfer_learning_multi-appliance/model.py.

Power head uses OFF-norm blending for z-score targets (see docs/multinilm_off_norm_gate.md).
Author code: power = linear(tanh(x)) * sigmoid(state). Ours when OFF: blend to -mean/std.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TransferNILMConfig:
    window_size: int = 480
    hidden: int = 256
    attn_heads: int = 2
    encoder_layers: int = 2
    head_transformer_layers: int = 1
    dropout: float = 0.1


def transfer_nilm_config(architecture: dict[str, Any], windowing: dict[str, Any]) -> TransferNILMConfig:
    return TransferNILMConfig(
        window_size=int(windowing.get("input_window_length", 480)),
        hidden=int(architecture.get("hidden", 256)),
        attn_heads=int(architecture.get("attn_heads", 2)),
        encoder_layers=int(architecture.get("encoder_layers", 2)),
        head_transformer_layers=int(architecture.get("head_transformer_layers", 1)),
        dropout=float(architecture.get("dropout", architecture.get("drop_out", 0.1))),
    )


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-9):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class Attention(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        dropout: nn.Dropout | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        query, key, value = [
            layer(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (query, key, value))
        ]
        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.activation(self.w_1(x)))


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return self.layer_norm(x + self.dropout(sublayer(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden: int, attn_heads: int, feed_forward_hidden: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


def _truncated_normal_init(module: nn.Module, mean: float = 0, std: float = 0.02, lower: float = -0.04, upper: float = 0.04) -> None:
    for name, param in module.named_parameters():
        if "layer_norm" in name:
            continue
        with torch.no_grad():
            l = (1.0 + math.erf(((lower - mean) / std) / math.sqrt(2.0))) / 2.0
            u = (1.0 + math.erf(((upper - mean) / std) / math.sqrt(2.0))) / 2.0
            param.uniform_(2 * l - 1, 2 * u - 1)
            param.erfinv_()
            param.mul_(std * math.sqrt(2.0))
            param.add_(mean)


class BERT4NILM(nn.Module):
    """Shared encoder from the transfer-learning baseline."""

    def __init__(self, cfg: TransferNILMConfig):
        super().__init__()
        self.cfg = cfg
        self.original_len = cfg.window_size
        self.latent_len = int(self.original_len / 2)
        self.dropout_rate = cfg.dropout
        self.hidden = cfg.hidden
        self.heads = cfg.attn_heads
        self.n_layers = cfg.encoder_layers

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            padding_mode="replicate",
        )
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)
        self.position = PositionalEmbedding(max_len=self.latent_len, d_model=self.hidden)
        self.layer_norm = LayerNorm(self.hidden)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(self.hidden, self.heads, self.hidden * 4, self.dropout_rate)
                for _ in range(self.n_layers)
            ]
        )
        self.deconv = nn.ConvTranspose1d(
            in_channels=self.hidden,
            out_channels=self.hidden,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        _truncated_normal_init(self)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x_token = self.pool(self.conv(sequence.unsqueeze(1))).permute(0, 2, 1)
        embedding = x_token + self.position(sequence)
        x = self.dropout(self.layer_norm(embedding))

        mask = None
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return self.deconv(x.permute(0, 2, 1))


class CNNApplianceHead(nn.Module):
    """Per-appliance head: state-gated power with OFF-norm blend under z-score targets."""

    def __init__(self, cfg: TransferNILMConfig, *, off_norm: float = 0.0):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("off_norm", torch.tensor(float(off_norm), dtype=torch.float32))
        self.dropout_rate = cfg.dropout
        self.hidden = cfg.hidden
        self.heads = cfg.attn_heads

        self.conv = nn.Conv1d(
            in_channels=self.hidden,
            out_channels=self.hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            padding_mode="replicate",
        )
        self.layer_norm = LayerNorm(self.hidden)
        self.layer_norm2 = LayerNorm(self.hidden)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
        self.linear1 = nn.Linear(self.hidden, 128)
        self.linear2 = nn.Linear(128, 1)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(self.hidden, self.heads, self.hidden * 4, self.dropout_rate)
                for _ in range(cfg.head_transformer_layers)
            ]
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        _truncated_normal_init(self)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(sequence).permute(0, 2, 1)
        x = self.dropout(self.layer_norm(x))

        mask = None
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        x = self.dropout2(self.layer_norm2(x))

        state = torch.sigmoid(self.fc(x))
        power_raw = self.linear2(torch.tanh(self.linear1(x)))
        power = state * power_raw + (1.0 - state) * self.off_norm
        return power, state


class TransferMultiApplianceModel(nn.Module):
    """BERT4NILM encoder + one CNNApplianceHead per appliance."""

    def __init__(
        self,
        cfg: TransferNILMConfig,
        num_appliances: int,
        *,
        appliance_off_norm: list[float] | None = None,
    ):
        super().__init__()
        if num_appliances < 1:
            raise ValueError("num_appliances must be >= 1")
        self.cfg = cfg
        self.num_appliances = num_appliances
        off_norms = list(appliance_off_norm or [0.0] * num_appliances)
        if len(off_norms) != num_appliances:
            raise ValueError(
                f"appliance_off_norm length {len(off_norms)} != num_appliances {num_appliances}"
            )
        self.encoder = BERT4NILM(cfg)
        self.heads = nn.ModuleList(
            CNNApplianceHead(cfg, off_norm=off_norms[i]) for i in range(num_appliances)
        )

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.squeeze(-1)

        features = self.encoder(x)
        powers, states = [], []
        for head in self.heads:
            power_i, state_i = head(features)
            powers.append(power_i)
            states.append(state_i)

        return torch.cat(powers, dim=2), torch.cat(states, dim=2)

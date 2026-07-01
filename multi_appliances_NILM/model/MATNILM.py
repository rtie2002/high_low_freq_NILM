"""MATNILM (MATconv) — ported from NILM_model/baseline/MATNILM/modules.py."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn


class ApplSA(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        d_model = 2 * hidden
        self.self_attn = nn.MultiheadAttention(d_model, 2, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

    def _sa_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout1(self.self_attn(x, x, x)[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm1(x + self._sa_block(x))


class ApplFF(nn.Module):
    def __init__(self, hidden: int, dropout: float, dim_feedforward: int = 1024):
        super().__init__()
        d_model = 2 * hidden
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.linear2(self.dropout(torch.relu(self.linear1(x)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(x + self._ff_block(x))


class ApplBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float, *, last: bool = False):
        super().__init__()
        self.multihead_attn_d = ApplSA(hidden, dropout)
        self.multihead_attn_f = ApplSA(hidden, dropout)
        self.multihead_attn_m = ApplSA(hidden, dropout)
        self.multihead_attn_w = ApplSA(hidden, dropout)
        self.multihead_attn_r_g = nn.MultiheadAttention(2 * hidden, 2, batch_first=True)
        self.norm1 = nn.LayerNorm(2 * hidden)

        self.dish = ApplFF(hidden, dropout)
        self.frid = ApplFF(hidden, dropout)
        self.micro = ApplFF(hidden, dropout)
        self.wash = ApplFF(hidden, dropout)
        self.last = last
        if last:
            self.dish_c = ApplFF(hidden, dropout)
            self.frid_c = ApplFF(hidden, dropout)
            self.micro_c = ApplFF(hidden, dropout)
            self.wash_c = ApplFF(hidden, dropout)

    def forward(
        self,
        d_r_a: torch.Tensor,
        f_r_a: torch.Tensor,
        m_r_a: torch.Tensor,
        w_r_a: torch.Tensor,
    ):
        attn_output_d = self.multihead_attn_d(d_r_a)
        attn_output_f = self.multihead_attn_f(f_r_a)
        attn_output_m = self.multihead_attn_m(m_r_a)
        attn_output_w = self.multihead_attn_w(w_r_a)

        global_attn = torch.cat(
            (
                attn_output_d.unsqueeze(3),
                attn_output_f.unsqueeze(3),
                attn_output_m.unsqueeze(3),
                attn_output_w.unsqueeze(3),
            ),
            3,
        )
        global_attn = global_attn.permute(0, 1, 3, 2)
        embed_dim = global_attn.shape[-1]
        global_attn = global_attn.reshape(-1, 4, embed_dim)
        attn_output_r_g, _ = self.multihead_attn_r_g(global_attn, global_attn, global_attn)
        attn_output_r_g = attn_output_r_g.reshape(d_r_a.shape[0], d_r_a.shape[1], 4, embed_dim)

        d_r_a = self.norm1(d_r_a + attn_output_r_g[:, :, 0, :])
        f_r_a = self.norm1(f_r_a + attn_output_r_g[:, :, 1, :])
        m_r_a = self.norm1(m_r_a + attn_output_r_g[:, :, 2, :])
        w_r_a = self.norm1(w_r_a + attn_output_r_g[:, :, 3, :])

        d_r = self.dish(d_r_a)
        f_r = self.frid(f_r_a)
        m_r = self.micro(m_r_a)
        w_r = self.wash(w_r_a)

        if self.last:
            return d_r, f_r, m_r, w_r, self.dish_c(d_r_a), self.frid_c(f_r_a), self.micro_c(m_r_a), self.wash_c(w_r_a)
        return d_r, f_r, m_r, w_r


class MATconv(nn.Module):
    """Multi-appliance transformer-conv NILM (4 appliances, fixed architecture)."""

    NUM_APPLIANCES = 4

    def __init__(
        self,
        *,
        input_size: int = 1,
        hidden: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden = hidden

        self.shared_layer = nn.Sequential(
            nn.Conv1d(1, 30, kernel_size=10, padding="same"),
            nn.ReLU(True),
            nn.Conv1d(30, 30, kernel_size=8, padding="same"),
            nn.ReLU(True),
            nn.Conv1d(30, 40, kernel_size=6, padding="same"),
            nn.ReLU(True),
            nn.Conv1d(40, 50, kernel_size=5, padding="same"),
            nn.ReLU(True),
            nn.Conv1d(50, 50, kernel_size=5, padding="same"),
            nn.ReLU(True),
            nn.Conv1d(50, hidden * 2, kernel_size=5, padding="same"),
            nn.ReLU(True),
        )

        self.block1 = ApplBlock(hidden, dropout)
        self.block2 = ApplBlock(hidden, dropout)
        self.block3 = ApplBlock(hidden, dropout, last=True)

        head = lambda: nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.fc_dr = head()
        self.fc_dc = head()
        self.fc_fr = head()
        self.fc_fc = head()
        self.fc_mr = head()
        self.fc_mc = head()
        self.fc_wr = head()
        self.fc_wc = head()

    def forward(self, input_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.shared_layer(input_data.permute(0, 2, 1)).permute(0, 2, 1)

        d_r, f_r, m_r, w_r = self.block1(encoded, encoded, encoded, encoded)
        d_r, f_r, m_r, w_r = self.block2(d_r, f_r, m_r, w_r)
        d_rr, f_rr, m_rr, w_rr, d_cc, f_cc, m_cc, w_cc = self.block3(d_r, f_r, m_r, w_r)

        dc_logits = self.fc_dc(d_cc)
        fc_logits = self.fc_fc(f_cc)
        mc_logits = self.fc_mc(m_cc)
        wc_logits = self.fc_wc(w_cc)
        dc = torch.sigmoid(dc_logits)
        fc = torch.sigmoid(fc_logits)
        mc = torch.sigmoid(mc_logits)
        wc = torch.sigmoid(wc_logits)

        dr = self.fc_dr(d_rr) * dc
        fr = self.fc_fr(f_rr) * fc
        mr = self.fc_mr(m_rr) * mc
        wr = self.fc_wr(w_rr) * wc

        y_pred_r = torch.cat((dr, fr, mr, wr), dim=2)
        # Logits for BCEWithLogitsLoss (AMP-safe); apply sigmoid at inference.
        y_pred_c = torch.cat((dc_logits, fc_logits, mc_logits, wc_logits), dim=2)
        return y_pred_r, y_pred_c


def matnilm_config(architecture: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        input_size=int(architecture.get("input_size", 1)),
        hidden=int(architecture.get("hidden", 32)),
        dropout=float(architecture.get("dropout", 0.1)),
    )

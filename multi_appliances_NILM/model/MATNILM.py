"""MATNILM (MATconv) — ported from NILM_model/baseline/MATNILM/modules.py."""

from __future__ import annotations

from dataclasses import dataclass
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

        d_r_a = self.norm1(attn_output_r_g[:, :, 0, :] + attn_output_d)
        f_r_a = self.norm1(attn_output_r_g[:, :, 1, :] + attn_output_f)
        m_r_a = self.norm1(attn_output_r_g[:, :, 2, :] + attn_output_m)
        w_r_a = self.norm1(attn_output_r_g[:, :, 3, :] + attn_output_w)

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

        dc = torch.sigmoid(self.fc_dc(d_cc))
        fc = torch.sigmoid(self.fc_fc(f_cc))
        mc = torch.sigmoid(self.fc_mc(m_cc))
        wc = torch.sigmoid(self.fc_wc(w_cc))

        dr = self.fc_dr(d_rr) * dc
        fr = self.fc_fr(f_rr) * fc
        mr = self.fc_mr(m_rr) * mc
        wr = self.fc_wr(w_rr) * wc

        y_pred_r = torch.cat((dr, fr, mr, wr), dim=2)
        y_pred_c = torch.cat((dc, fc, mc, wc), dim=2)
        return y_pred_r, y_pred_c


def matnilm_config(architecture: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        input_size=int(architecture.get("input_size", 1)),
        hidden=int(architecture.get("hidden", 32)),
        dropout=float(architecture.get("dropout", 0.1)),
    )


@dataclass
class MATNILMLossOutput:
    loss: torch.Tensor
    loss_power: torch.Tensor
    loss_state: torch.Tensor
    mae: torch.Tensor


class MATNILMLoss(nn.Module):
    def __init__(self, power_scale: float | list[float] | torch.Tensor = 1.0):
        super().__init__()
        self.register_buffer("power_scale", torch.as_tensor(power_scale, dtype=torch.float32))
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, y_pred_r, y_pred_c, y_true_r, y_true_c) -> MATNILMLossOutput:
        device_type = y_pred_c.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            y_pred_r_f, y_pred_c_f = y_pred_r.float(), y_pred_c.float()
            y_true_r_f, y_true_c_f = y_true_r.float(), y_true_c.float()
            loss_r = self.mse(y_pred_r_f, y_true_r_f)
            loss_c = self.bce(y_pred_c_f, y_true_c_f)
        scale = self.power_scale.to(device=y_pred_r.device, dtype=y_pred_r.dtype)
        mae = torch.mean(torch.abs((y_pred_r.float() - y_true_r.float()) * scale))
        return MATNILMLossOutput(loss=loss_r + loss_c, loss_power=loss_r, loss_state=loss_c, mae=mae)


from data.common import BaseNILMAdapter, StepOutput, center_output_slice
import numpy as np
from torch.utils.data import DataLoader


class MATNILMAdapter(BaseNILMAdapter):
    name = "mat_nilm"

    def __init__(self, merged_cfg: dict[str, Any], data_root: str | None = None):
        super().__init__(merged_cfg, data_root=data_root)
        appliances = self.cfg["appliances"]
        if len(appliances) != MATconv.NUM_APPLIANCES:
            raise ValueError(
                f"MATNILM requires exactly {MATconv.NUM_APPLIANCES} appliances; "
                f"got {len(appliances)}: {appliances}. "
                "Use config/experiment_redd.yaml or another 4-appliance experiment."
            )

    def build_model(self, device: torch.device) -> torch.nn.Module:
        arch = self.model_cfg["architecture"]
        return MATconv(
            input_size=int(arch.get("input_size", 1)),
            hidden=int(arch.get("hidden", 32)),
            dropout=float(arch.get("dropout", 0.1)),
        ).to(device)

    def build_loss(self) -> MATNILMLoss:
        return MATNILMLoss(power_scale=self._data_loader().loss_scale)

    def _align_loss_tensors(self, y_pred_r, y_pred_c, y, z):
        w = self.model_cfg["windowing"]
        out_slice = center_output_slice(w)
        out_len = int(w.get("output_window_length", 1))
        if y_pred_r.shape[1] == y.shape[1]:
            return y_pred_r, y_pred_c, y, z
        if y.dim() == 3 and y.shape[1] == out_len:
            return y_pred_r[:, out_slice, :], y_pred_c[:, out_slice, :], y, z
        if w.get("training_loss_scope") == "center_output":
            return y_pred_r[:, out_slice, :], y_pred_c[:, out_slice, :], y[:, out_slice, :], z[:, out_slice, :]
        return y_pred_r, y_pred_c, y, z

    def step(self, model, loss_fn: MATNILMLoss, batch):
        x, y, z = batch
        z = z.float()
        y_pred_r, y_pred_c = model(x)
        y_pred_r, y_pred_c, y, z = self._align_loss_tensors(y_pred_r, y_pred_c, y, z)
        out = loss_fn(y_pred_r, y_pred_c, y, z)
        pred_state = (y_pred_c >= 0.5).long()
        app_losses = {}
        for app_i, app in enumerate(self.cfg["appliances"]):
            loss_r_i = loss_fn.mse(y_pred_r[..., app_i].float(), y[..., app_i].float())
            loss_c_i = loss_fn.bce(y_pred_c[..., app_i].float(), z[..., app_i].float())
            app_losses[f"loss_{app}"] = float((loss_r_i + loss_c_i).detach())
        return StepOutput(
            loss=out.loss,
            logs={
                "loss": float(out.loss.detach()),
                "loss_power": float(out.loss_power.detach()),
                "loss_state": float(out.loss_state.detach()),
                "mae": float(out.mae.detach()),
                **app_losses,
            },
            aux={
                "pred_state": pred_state.detach().cpu(),
                "true_state": z.long().detach().cpu(),
                "pred_power": y_pred_r.detach().cpu(),
                "true_power": y.detach().cpu(),
            },
        )

    @torch.no_grad()
    def predict_dataloader(self, model, loader: DataLoader, device, *, max_batches=None, split="test"):
        model.eval()
        out_slice = center_output_slice(self.model_cfg["windowing"])
        pred_power, pred_state, true_power, true_state, sample_indices = [], [], [], [], []
        offset = 0
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y, z = batch
            y_pred_r, y_pred_c_prob = model(x.to(device))
            y_pred_r = y_pred_r[:, out_slice, :].cpu().numpy()
            y_pred_c = y_pred_c_prob[:, out_slice, :].cpu().numpy()
            out_len = int(self.model_cfg["windowing"].get("output_window_length", 1))
            if y.dim() == 3 and y.shape[1] == out_len:
                y_true, z_true = y.numpy(), z.numpy()
            else:
                y_true = y[:, out_slice, :].numpy() if y.dim() == 3 else y.numpy()
                z_true = z[:, out_slice, :].numpy() if z.dim() == 3 else z.numpy()
            n_apps = len(self.cfg["appliances"])
            pred_power.append(y_pred_r.reshape(len(x), -1, n_apps))
            pred_state.append((y_pred_c >= 0.5).astype(np.int32).reshape(len(x), -1, n_apps))
            true_power.append(y_true.reshape(len(x), -1, n_apps))
            true_state.append(z_true.reshape(len(x), -1, n_apps))
            sample_indices.append(self._sample_index(offset, len(x)))
            offset += len(x)
        return self.finalize_prediction_bundle(
            split=split, sample_indices=sample_indices,
            pred_power_batches=pred_power, pred_state_batches=pred_state,
            true_power_batches=true_power, true_state_batches=true_state,
        )

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

import numpy as np
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


from data.common import BaseNILMAdapter, StepOutput
from config import appliance_off_norm_normalized


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _batch_x_to_matuda(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3 and x.size(-1) == 1:
        return x.permute(0, 2, 1).contiguous()
    if x.dim() == 2:
        return x.unsqueeze(1)
    return x


def _resolve_pos_weight(adapter, loss_cfg):
    configured = loss_cfg.get("pos_weight")
    if configured is not None and str(configured).lower() not in {"auto", "null", "none"}:
        return torch.tensor(configured, dtype=torch.float32)
    weights = adapter._data_loader().estimate_state_pos_weights("train")
    cap = float(loss_cfg.get("pos_weight_cap", 50.0))
    return torch.clamp(torch.as_tensor(weights, dtype=torch.float32), max=cap)


class MATUDAAdapter(BaseNILMAdapter):
    name = "matuda"

    def build_model(self, device: torch.device) -> torch.nn.Module:
        arch = self.model_cfg.get("architecture", {})
        appliances = self.cfg["appliances"]
        seq_len = int(self.model_cfg["windowing"]["input_window_length"])
        off_norms = appliance_off_norm_normalized(self.experiment, appliances)
        schedule = arch.get("channel_schedule", [32, 64, 128])
        hidden = int(arch.get("hidden_channels", schedule[-1] if schedule else 128))
        return MATUDANet(
            num_appliances=len(appliances),
            seq_len=seq_len,
            channel_schedule=tuple(schedule),
            hidden_channels=hidden,
            tcn_blocks=int(arch.get("tcn_blocks", arch.get("num_blocks", 8))),
            tcn_kernel_size=int(arch.get("tcn_kernel_size", arch.get("kernel_size", 5))),
            max_dilation=int(arch.get("max_dilation", 64)),
            fc_dims=tuple(arch.get("fc_dims", [256, 192, 128])),
            dropout=float(arch.get("dropout", 0.15)),
            use_gate=bool(arch.get("use_gate", True)),
            stem_kernels=tuple(arch.get("stem_kernels", arch.get("detail_kernels", [3, 5, 9]))),
            detail_branch_channels=int(arch.get("detail_branch_channels", 16)),
            stage_kernel_size=int(arch.get("stage_kernel_size", 5)),
            appliance_off_norm=off_norms,
            gate_mode=str(arch.get("gate_mode", "hard")),
            gate_threshold=float(arch.get("gate_threshold", 0.5)),
            head_local_layers=int(arch.get("head_local_layers", 2)),
            head_kernel_size=int(arch.get("head_kernel_size", 3)),
            head_use_residual=bool(arch.get("head_use_residual", True)),
            conv_channels=arch.get("conv_channels"),
            head_hidden=arch.get("head_hidden"),
            use_instance_norm=bool(arch.get("use_instance_norm", False)),
        ).to(device)

    def build_loss(self):
        from model.MATUDA_loss import MATUDACriterion
        loss_cfg = self.model_cfg.get("loss", {})
        da_cfg = self.model_cfg.get("domain_adaptation") or {}
        da_mode = str(loss_cfg.get("da_mode", da_cfg.get("mode", "egc")))
        enabled = bool(da_cfg.get("enabled", False))
        lam = float(loss_cfg.get("lambda_domain", 0.0))
        if (not enabled) or lam <= 0:
            da_mode = "none"
            lam = 0.0
        lambda_state = loss_cfg.get("lambda_state", loss_cfg.get("state_weight", 1.0))
        return MATUDACriterion(
            lambda_domain=lam,
            mu_mmd=float(loss_cfg.get("domain_mu", 0.4)),
            lambda_state=float(lambda_state),
            pos_weight=_resolve_pos_weight(self, loss_cfg),
            power_scale=self._data_loader().loss_scale,
            da_mode=da_mode,
            domain_mix=str(loss_cfg.get("domain_mix", "convex")),
            domain_scale=str(loss_cfg.get("domain_scale", "equal")),
            conditional_weight=float(loss_cfg.get("conditional_weight", 0.5)),
            on_masked_power=bool(loss_cfg.get("on_masked_power", False)),
            pl_weight=float(loss_cfg.get("pl_weight", 0.0)),
            pl_confidence=float(loss_cfg.get("pl_confidence", 0.9)),
            task_balance=str(loss_cfg.get("task_balance", "equal")),
            focal_gamma=float(loss_cfg.get("focal_gamma", 0.0)),
        )

    def step(self, model, loss_fn, batch, target_batch=None):
        x, y, z = batch
        z, y = z.float(), y.float()
        x = _batch_x_to_matuda(x)
        need_target = target_batch is not None and (
            (float(getattr(loss_fn, "lambda_domain", 0.0)) != 0.0 and str(getattr(loss_fn, "da_mode", "none")) != "none")
            or float(getattr(loss_fn, "pl_weight", 0.0)) > 0.0
        )
        out_s = model(x)
        out_t = None
        if need_target:
            x_t = target_batch[0] if isinstance(target_batch, (tuple, list)) else target_batch
            out_t = model(_batch_x_to_matuda(x_t))
        losses = loss_fn(out_s, out_t, y, z)
        power_pred = out_s["powers"]
        pred_state = (torch.sigmoid(out_s["state_logits"]) >= 0.5).long()
        with torch.no_grad():
            mae = float((power_pred - y).abs().mean().detach())
        return StepOutput(
            loss=losses["loss"],
            logs={
                "loss": float(losses["loss"].detach()),
                "loss_power": float(losses["loss_power"].detach()),
                "loss_state": float(losses["loss_state"].detach()),
                "loss_state_term": float(losses["loss_state_term"].detach()),
                "loss_domain": float(losses["loss_domain"].detach()),
                "loss_domain_term": float(losses.get("loss_domain_term", losses["loss_domain"]).detach()),
                "loss_pl": float(losses.get("loss_pl", losses["loss"].new_zeros(())).detach()),
                "lambda_domain": float(losses.get("lambda", 0.0) or 0.0),
                "mae": mae,
            },
            aux={
                "pred_state": pred_state.detach().cpu(),
                "true_state": z.long().detach().cpu(),
                "pred_power": power_pred.detach().float().cpu(),
                "true_power": y.detach().cpu(),
            },
        )

    @torch.no_grad()
    def predict_dataloader(self, model, loader, device, *, max_batches=None, split="test"):
        model.eval()
        pred_power, pred_state, true_power, true_state, sample_indices = [], [], [], [], []
        offset = 0
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y, z = batch
            out = model(_batch_x_to_matuda(x).to(device))
            pred_power.append(_to_numpy(out["powers"]))
            pred_state.append(_to_numpy(torch.sigmoid(out["state_logits"])))
            true_power.append(y.numpy())
            true_state.append(z.numpy())
            sample_indices.append(self._sample_index(offset, len(x)))
            offset += len(x)
        return self.finalize_prediction_bundle(
            split=split, sample_indices=sample_indices,
            pred_power_batches=pred_power, pred_state_batches=pred_state,
            true_power_batches=true_power, true_state_batches=true_state,
        )

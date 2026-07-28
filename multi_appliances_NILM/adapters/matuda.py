"""MATUDA adapter for the shared multi_appliances_NILM train/eval pipeline.

Model/loss live under model/MATUDA*.py.
Seq2seq (full-window) + FC-layer EGC-DA on unlabeled target aggregates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from adapters.common import BaseNILMAdapter, StepOutput
from adapters.config import appliance_off_norm_normalized
from model.MATUDA import MATUDANet
from model.MATUDA_loss import MATUDACriterion


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _batch_x_to_matuda(x: torch.Tensor) -> torch.Tensor:
    """Pipeline dataloader yields (B, T, 1); MATUDA expects (B, 1, T)."""
    if x.dim() == 3 and x.size(-1) == 1:
        return x.permute(0, 2, 1).contiguous()
    if x.dim() == 2:
        return x.unsqueeze(1)
    return x


def _resolve_pos_weight(adapter: "MATUDAAdapter", loss_cfg: dict) -> torch.Tensor | None:
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
        model = MATUDANet(
            num_appliances=len(appliances),
            seq_len=seq_len,
            conv_channels=int(arch.get("conv_channels", 96)),
            tcn_blocks=int(arch.get("tcn_blocks", 8)),
            fc_dims=tuple(arch.get("fc_dims", [512, 256, 128])),
            dropout=float(arch.get("dropout", 0.15)),
            use_gate=bool(arch.get("use_gate", True)),
            stem_kernels=tuple(arch.get("stem_kernels", [3, 5, 9])),
            appliance_off_norm=off_norms,
            gate_mode=str(arch.get("gate_mode", "soft")),
            head_hidden=int(arch.get("head_hidden", 64)),
            head_kernel_size=int(arch.get("head_kernel_size", 3)),
        )
        return model.to(device)

    def build_loss(self) -> MATUDACriterion:
        loss_cfg = self.model_cfg.get("loss", {})
        da_cfg = self.model_cfg.get("domain_adaptation") or {}
        da_mode = str(loss_cfg.get("da_mode", da_cfg.get("mode", "egc")))
        enabled = bool(da_cfg.get("enabled", False))
        lam = float(loss_cfg.get("lambda_domain", 0.0))
        if (not enabled) or lam <= 0:
            da_mode = "none"
            lam = 0.0
        # MultiNILM naming: lambda_state (+ optional legacy state_weight).
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
            # Default False = identical to MultiNILM MSE; set true for ON-only MSE.
            on_masked_power=bool(loss_cfg.get("on_masked_power", False)),
            pl_weight=float(loss_cfg.get("pl_weight", 0.0)),
            pl_confidence=float(loss_cfg.get("pl_confidence", 0.9)),
            task_balance=str(loss_cfg.get("task_balance", "equal")),
        )

    def step(
        self,
        model: torch.nn.Module,
        loss_fn: MATUDACriterion,
        batch: Any,
        target_batch: Any | None = None,
    ) -> StepOutput:
        x, y, z = batch
        z = z.float()
        y = y.float()
        x = _batch_x_to_matuda(x)

        need_target = target_batch is not None and (
            (
                float(getattr(loss_fn, "lambda_domain", 0.0)) != 0.0
                and str(getattr(loss_fn, "da_mode", "none")) != "none"
            )
            or float(getattr(loss_fn, "pl_weight", 0.0)) > 0.0
        )

        out_s = model(x)
        out_t = None
        if need_target:
            x_t = target_batch[0] if isinstance(target_batch, (tuple, list)) else target_batch
            x_t = _batch_x_to_matuda(x_t)
            out_t = model(x_t)

        losses = loss_fn(out_s, out_t, y, z)
        power_pred = out_s["powers"]
        state_logits = out_s["state_logits"]
        state_prob = torch.sigmoid(state_logits)
        pred_state = (state_prob >= 0.5).long()

        with torch.no_grad():
            mae = (power_pred - y).abs().mean()

        return StepOutput(
            loss=losses["loss"],
            logs={
                "loss": float(losses["loss"].detach()),
                "loss_power": float(losses["loss_power"].detach()),
                "loss_state": float(losses["loss_state"].detach()),
                "loss_state_term": float(losses["loss_state_term"].detach()),
                "loss_domain": float(losses["loss_domain"].detach()),
                "loss_domain_term": float(
                    losses.get("loss_domain_term", losses["loss_domain"]).detach()
                ),
                "loss_pl": float(losses.get("loss_pl", losses["loss"].new_zeros(())).detach()),
                "lambda_domain": float(losses.get("lambda", 0.0) or 0.0),
                "mae": float(mae.detach()),
            },
            aux={
                "pred_state": pred_state.detach().cpu(),
                "true_state": z.long().detach().cpu(),
                "pred_power": power_pred.detach().float().cpu(),
                "true_power": y.detach().cpu(),
            },
        )

    @torch.no_grad()
    def predict_dataloader(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        *,
        max_batches: int | None = None,
        split: str = "test",
    ) -> Any:
        model.eval()
        pred_power, pred_state = [], []
        true_power, true_state = [], []
        sample_indices = []
        offset = 0

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y, z = batch
            x = _batch_x_to_matuda(x).to(device)
            out = model(x)
            power_pred = out["powers"]  # (B, T, A)
            state_prob = torch.sigmoid(out["state_logits"])
            pred_power.append(_to_numpy(power_pred))
            # Keep probs for overlap_mean reconstruction (same as MultiNILM).
            pred_state.append(_to_numpy(state_prob))
            true_power.append(y.numpy())
            true_state.append(z.numpy())
            sample_indices.append(self._sample_index(offset, len(x)))
            offset += len(x)

        return self.finalize_prediction_bundle(
            split=split,
            sample_indices=sample_indices,
            pred_power_batches=pred_power,
            pred_state_batches=pred_state,
            true_power_batches=true_power,
            true_state_batches=true_state,
        )

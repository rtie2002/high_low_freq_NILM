"""yaml → MultiNILM → one-batch step → prediction bundle."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from adapters.common import BaseNILMAdapter, StepOutput
from adapters.config import appliance_off_norm_normalized
from model.MultiNILM import build_multinilm, build_multinilm_fractional, multinilm_config
from model.MultiNILM_loss import MultiNILMLoss


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _resolve_pos_weight(adapter: "MultiNILMAdapter", loss_cfg: dict) -> list[float] | None:
    configured = loss_cfg.get("pos_weight")
    if configured is not None and str(configured).lower() not in {"auto", "null", "none"}:
        return configured
    weights = adapter._data_loader().estimate_state_pos_weights("train")
    cap = loss_cfg.get("pos_weight_cap", None)
    if cap not in (None, "", "none", "null"):
        weights = np.minimum(weights, float(cap))
    return weights.tolist()


def _pred_on_from_config(adapter: "MultiNILMAdapter", power_norm, state_prob) -> np.ndarray:
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

    def build_model(self, device: torch.device) -> torch.nn.Module:
        apps = self.cfg["appliances"]
        return build_multinilm(
            multinilm_config(self.model_cfg["architecture"]),
            num_appliances=len(apps),
            output_length=int(self.model_cfg["windowing"].get("output_window_length", 1)),
            appliance_off_norm=appliance_off_norm_normalized(self.experiment, apps),
        ).to(device)

    def build_loss(self) -> MultiNILMLoss:
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
            state_transition_weight=float(cfg.get("state_transition_weight", 0.0)),
            power_energy_relative_weight=float(cfg.get("power_energy_relative_weight", 0.0)),
            energy_floor_watts=float(cfg.get("energy_floor_watts", 10.0)),
            mmd_sigma=None if mmd in (None, "", "auto") else float(mmd),
        )

    def step(self, model, loss_fn: MultiNILMLoss, batch: Any, target_batch: Any | None = None) -> StepOutput:
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
        pred_state = torch.from_numpy(
            _pred_on_from_config(self, _to_numpy(power_pred), _to_numpy(torch.sigmoid(state_logits)))
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
                "loss_state_transition": float(out.loss_state_transition.detach()),
                "loss_energy_relative": float(out.loss_energy_relative.detach()),
                "loss_domain": float(out.loss_domain.detach()),
                "loss_domain_term": float(out.loss_domain_term.detach()),
                "mae": float(out.mae.detach()),
                **app_logs,
            },
            aux={
                "pred_state": pred_state.detach().cpu(),
                "true_state": z.long().detach().cpu(),
                "pred_power": power_pred.detach().float().cpu(),
                "true_power": y.detach().cpu(),
            },
        )

    @torch.no_grad()
    def predict_dataloader(self, model, loader: DataLoader, device, *, max_batches=None, split="test"):
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

    def build_model(self, device: torch.device) -> torch.nn.Module:
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

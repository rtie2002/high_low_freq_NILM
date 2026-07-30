"""MultiNILM adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from adapters.common import BaseNILMAdapter, StepOutput
from adapters.config import appliance_off_norm_normalized
from model.MultiNILM import MultiNILM, multinilm_config
from model.MultiNILM_loss import MultiNILMLoss


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """NumPy does not support bfloat16; cast to float32 when AMP is enabled."""
    return t.detach().float().cpu().numpy()


def _resolve_pos_weight(adapter: "MultiNILMAdapter", loss_cfg: dict) -> list[float] | None:
    """Use yaml pos_weight, or auto-balance rare ON events from the train split."""
    configured = loss_cfg.get("pos_weight")
    if configured is not None and str(configured).lower() not in {"auto", "null", "none"}:
        return configured
    return adapter._data_loader().estimate_state_pos_weights("train").tolist()


def _pred_on_from_config(
    adapter: "MultiNILMAdapter",
    power_norm: np.ndarray,
    state_prob: np.ndarray,
) -> np.ndarray:
    """Build binary ON/OFF predictions for metrics and saved bundles."""
    source = str(adapter.model_cfg.get("evaluation", {}).get("pred_on_source", "state_head")).lower()
    if source == "state_head":
        return (state_prob >= 0.5).astype(np.int32)

    loader = adapter._data_loader()
    power_watts = loader.denorm_to_watts(power_norm)
    threshold = loader.state_threshold_watts
    if threshold is None:
        raise ValueError("pred_on_source=power_threshold requires threshold training labels")
    threshold = np.asarray(threshold, dtype=np.float32)
    power_on = (power_watts > threshold).astype(np.int32)
    if source == "power_threshold":
        return power_on
    if source == "combined":
        state_on = (state_prob >= 0.5).astype(np.int32)
        return np.maximum(state_on, power_on).astype(np.int32)
    raise ValueError(
        "evaluation.pred_on_source must be one of: state_head, power_threshold, combined"
    )


class MultiNILMAdapter(BaseNILMAdapter):
    # This name is used by main.py, logs, run folders, and saved predictions.
    name = "multinilm"

    def build_model(self, device: torch.device) -> torch.nn.Module:
        # Read architecture settings from config/models/multinilm.yaml.
        arch = self.model_cfg["architecture"]
        cfg = multinilm_config(arch)
        appliances = self.cfg["appliances"]
        off_norms = appliance_off_norm_normalized(self.experiment, appliances)

        # Create the MultiNILM neural network.
        model = MultiNILM(
            input_channels=cfg.input_channels,
            num_appliances=len(appliances),
            output_length=int(self.model_cfg["windowing"].get("output_window_length", 1)),
            hidden_channels=cfg.hidden_channels,
            channel_schedule=cfg.channel_schedule,
            stem_kernel_size=cfg.stem_kernel_size,
            stage_kernel_size=cfg.stage_kernel_size,
            num_blocks=cfg.num_blocks,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            max_dilation=cfg.max_dilation,
            gate_mode=cfg.gate_mode,
            gate_threshold=cfg.gate_threshold,
            appliance_off_norm=off_norms,
            domain_feature_layers=cfg.domain_feature_layers,
            head_local_layers=cfg.head_local_layers,
            head_kernel_size=cfg.head_kernel_size,
            head_use_residual=cfg.head_use_residual,
            use_multiscale_stem=cfg.use_multiscale_stem,
            detail_kernels=cfg.detail_kernels,
            detail_branch_channels=cfg.detail_branch_channels,
            cross_appliance_enabled=cfg.cross_appliance_enabled,
            cross_appliance_residual_scale=cfg.cross_appliance_residual_scale,
            cross_appliance_mid_channels=cfg.cross_appliance_mid_channels,
        )

        # Move model to GPU if available, otherwise CPU.
        return model.to(device)

    def build_loss(self) -> MultiNILMLoss:
        # Read loss settings from config/models/multinilm.yaml.
        loss_cfg = self.model_cfg.get("loss", {})

        # Create loss:
        #   L_NILM = L_power + balanced(L_state); see task_balance in yaml
        return MultiNILMLoss(
            # Preference on state vs power after optional equal-balance (1 = equal).
            lambda_state=float(loss_cfg.get("lambda_state", 0.1)),
            task_balance=str(loss_cfg.get("task_balance", "none")),

            # Auto-balance rare ON timesteps when pos_weight is null/auto.
            pos_weight=_resolve_pos_weight(self, loss_cfg),

            # Used only for MAE logging. This comes from dataset normalization
            # stats when available, otherwise from legacy scalar power_scale.
            power_scale=self._data_loader().loss_scale,

            # Domain adaptation (Lin-style).
            lambda_domain=float(loss_cfg.get("lambda_domain", 0.0)),
            domain_method=str(loss_cfg.get("domain_method", "coral")),
            domain_mu=float(loss_cfg.get("domain_mu", 0.4)),
            domain_mix=str(loss_cfg.get("domain_mix", "convex")),
            domain_scale=str(loss_cfg.get("domain_scale", "none")),
            mmd_sigma=(
                None
                if loss_cfg.get("mmd_sigma", None) in (None, "", "auto")
                else float(loss_cfg["mmd_sigma"])
            ),
        )

    def step(
        self,
        model: torch.nn.Module,
        loss_fn: MultiNILMLoss,
        batch: Any,
        target_batch: Any | None = None,
    ) -> StepOutput:
        """Run one batch through MultiNILM and return loss/logs.

        Normal path (validation / DA off):

            power, state = model(x)
            L = L_NILM(power, state, y, z)

        Domain-adaptation path (training + target_batch + lambda_domain > 0):

            power_S, state_S, Z_S = model(x_S, return_domain_features=True)
            _,       _,       Z_T = model(x_T, return_domain_features=True)
            L = (1-?) L_NILM + ? L_domain(Z_S, Z_T)   # domain_mix=convex (Lin)
              or L_NILM + ? L_domain                 # domain_mix=additive

        ``target_batch`` only needs the aggregate ``x_T``; y/z from the target
        split are ignored (unlabeled target domain, Lin-style).
        """
        x, y, z = batch
        z = z.float()

        use_domain = (
            target_batch is not None
            and float(getattr(loss_fn, "lambda_domain", 0.0)) != 0.0
        )

        if use_domain:
            x_t = target_batch[0] if isinstance(target_batch, (tuple, list)) else target_batch
            power_pred, state_logits, feats_s = model(x, return_domain_features=True)
            _, _, feats_t = model(x_t, return_domain_features=True)
            out = loss_fn(
                power_pred,
                state_logits,
                y,
                z,
                domain_feats_S=feats_s,
                domain_feats_T=feats_t,
            )
        else:
            power_pred, state_logits = model(x)
            out = loss_fn(power_pred, state_logits, y, z)

        state_prob = torch.sigmoid(state_logits)

        pred_state = torch.from_numpy(
            _pred_on_from_config(
                self,
                _to_numpy(power_pred),
                _to_numpy(state_prob),
            )
        ).long()

        app_logs = {}
        for app_i, app in enumerate(self.cfg["appliances"]):
            app_logs[f"loss_power_{app}"] = float(out.loss_power_per_appliance[app_i].detach())
            app_logs[f"loss_state_{app}"] = float(out.loss_state_per_appliance[app_i].detach())

        return StepOutput(
            loss=out.loss,
            logs={
                "loss": float(out.loss.detach()),
                "loss_power": float(out.loss_power.detach()),
                "loss_state": float(out.loss_state.detach()),
                "loss_state_term": float(out.loss_state_term.detach()),
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
    def predict_dataloader(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        *,
        max_batches: int | None = None,
        split: str = "test",
    ) -> Any:
        # Evaluation/inference mode.
        # This disables dropout behavior and uses running BatchNorm statistics.
        model.eval()

        # Lists collect batch predictions before concatenating into one array.
        pred_power, pred_state = [], []
        true_power, true_state = [], []
        sample_indices = []
        offset = 0

        for batch_idx, batch in enumerate(loader):
            # Optional limit for quick debugging.
            if max_batches is not None and batch_idx >= max_batches:
                break

            # x/y/z: normalized mains input, appliance power targets, ON/OFF labels
            x, y, z = batch

            # Move already-normalized aggregate input to GPU/CPU.
            x = x.to(device)

            # Predict appliance power and state logits.
            power_pred, state_logits = model(x)

            # Convert logits to probabilities for ON/OFF prediction.
            state_prob = torch.sigmoid(state_logits)

            pred_power.append(_to_numpy(power_pred))

            # Keep state probabilities for overlap_mean timeline reconstruction.
            pred_state.append(_to_numpy(state_prob))

            # y and z are ground truth from dataloader.
            # They are already on CPU because we did not move them to device.
            true_power.append(y.numpy())
            true_state.append(z.numpy())

            # Track window indices so prediction files can align with waveform plots.
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

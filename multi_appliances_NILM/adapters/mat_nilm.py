"""MATNILM adapter — wires MATconv + MSE/BCE loss into the shared pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from adapters.common import (
    AdapterDataMixin,
    build_prediction_bundle,
    center_output_slice,
    denorm_power_array,
    get_power_scale,
    get_state_threshold,
    scale_inputs,
    scale_targets,
    states_from_power,
)
from adapters.types import StepOutput
from model.MATNILM import MATconv
from model.MATNILM_loss import MATNILMLoss


class MATNILMAdapter(AdapterDataMixin):
    name = "mat_nilm"

    def __init__(self, merged_cfg: dict[str, Any], data_root: str | None = None):
        self.cfg = merged_cfg
        self.experiment = merged_cfg["experiment"]
        self.model_cfg = merged_cfg["model"]
        self.data_root = data_root or merged_cfg.get("data_root")
        self._data = None

        appliances = self.cfg["appliances"]
        if len(appliances) != MATconv.NUM_APPLIANCES:
            raise ValueError(
                f"MATNILM requires exactly {MATconv.NUM_APPLIANCES} appliances; "
                f"got {len(appliances)}: {appliances}. "
                "Set data.appliances in config/models/mat_nilm.yaml."
            )

    def build_model(self, device: torch.device) -> torch.nn.Module:
        arch = self.model_cfg["architecture"]
        model = MATconv(
            input_size=int(arch.get("input_size", 1)),
            hidden=int(arch.get("hidden", 32)),
            dropout=float(arch.get("dropout", 0.1)),
        )
        return model.to(device)

    def build_loss(self) -> MATNILMLoss:
        return MATNILMLoss(power_scale=get_power_scale(self.model_cfg))

    def build_dataloader(self, split: str) -> DataLoader:
        return self.build_standard_dataloader(split)

    def _align_loss_tensors(
        self,
        y_pred_r: torch.Tensor,
        y_pred_c: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match model outputs (864) to batch targets (864 train or 64 val/test)."""
        w = self.model_cfg["windowing"]
        out_slice = center_output_slice(w)
        out_len = int(w.get("output_window_length", 1))

        if y_pred_r.shape[1] == y.shape[1]:
            return y_pred_r, y_pred_c, y, z

        # Val/test: dataloader returns center output_window only.
        if y.dim() == 3 and y.shape[1] == out_len:
            return y_pred_r[:, out_slice, :], y_pred_c[:, out_slice, :], y, z

        # Train: optional center-64 loss while keeping full_input targets in batch.
        if w.get("training_loss_scope") == "center_output":
            y_pred_r = y_pred_r[:, out_slice, :]
            y_pred_c = y_pred_c[:, out_slice, :]
            y = y[:, out_slice, :]
            z = z[:, out_slice, :]

        return y_pred_r, y_pred_c, y, z

    def training_step(
        self,
        model: torch.nn.Module,
        loss_fn: MATNILMLoss,
        batch: Any,
    ) -> StepOutput:
        x, y, z = batch
        scale = get_power_scale(self.model_cfg)
        if (thr := get_state_threshold(self.model_cfg)) is not None:
            z = states_from_power(y, thr)
        x = scale_inputs(x, scale)
        y = scale_targets(y, scale)
        z = z.float()

        y_pred_r, y_pred_c = model(x)
        y_pred_r, y_pred_c, y, z = self._align_loss_tensors(y_pred_r, y_pred_c, y, z)
        out = loss_fn(y_pred_r, y_pred_c, y, z)
        return StepOutput(
            loss=out.loss,
            logs={
                "loss": float(out.loss.detach()),
                "loss_power": float(out.loss_power.detach()),
                "loss_state": float(out.loss_state.detach()),
                "mae": float(out.mae.detach()),
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
        scale = get_power_scale(self.model_cfg)
        appliances = self.cfg["appliances"]
        out_slice = center_output_slice(self.model_cfg["windowing"])

        pred_power, pred_state = [], []
        true_power, true_state = [], []
        sample_indices = []
        offset = 0

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y, z = batch
            if (thr := get_state_threshold(self.model_cfg)) is not None:
                z = states_from_power(y, thr)
            x = scale_inputs(x.to(device), scale)
            y_pred_r, y_pred_c_logits = model(x)

            y_pred_r = y_pred_r[:, out_slice, :].cpu().numpy()
            y_pred_c = torch.sigmoid(y_pred_c_logits[:, out_slice, :]).cpu().numpy()
            out_len = int(self.model_cfg["windowing"].get("output_window_length", 1))
            if y.dim() == 3 and y.shape[1] == out_len:
                y_true = y.numpy()
                z_true = z.numpy()
            else:
                y_true = y[:, out_slice, :].numpy() if y.dim() == 3 else y.numpy()
                z_true = z[:, out_slice, :].numpy() if z.dim() == 3 else z.numpy()

            pred_power.append(y_pred_r.reshape(len(x), -1, len(appliances)))
            pred_state.append((y_pred_c >= 0.5).astype(np.int32).reshape(len(x), -1, len(appliances)))
            true_power.append(y_true.reshape(len(x), -1, len(appliances)))
            true_state.append(z_true.reshape(len(x), -1, len(appliances)))
            sample_indices.append(np.arange(offset, offset + len(x)))
            offset += len(x)

        y_pred = np.concatenate(pred_power, axis=0).reshape(-1, len(appliances))
        y_true = np.concatenate(true_power, axis=0).reshape(-1, len(appliances))
        z_pred = np.concatenate(pred_state, axis=0).reshape(-1, len(appliances))
        z_true = np.concatenate(true_state, axis=0).reshape(-1, len(appliances))

        # CSV targets are already in watts; only model outputs are normalized.
        y_pred = denorm_power_array(y_pred, scale)

        return build_prediction_bundle(
            experiment_id=self.experiment["experiment_id"],
            model_name=self.name,
            split=split,
            appliances=appliances,
            sample_index=np.concatenate(sample_indices),
            y_true_watts=y_true,
            y_pred_watts=y_pred,
            y_true_on=z_true,
            y_pred_on=z_pred,
        )

    def configure_optimizer(self, model: torch.nn.Module):
        t = self.model_cfg["training"]
        optim = torch.optim.Adam(
            model.parameters(),
            lr=float(t["learning_rate"]),
            weight_decay=float(t.get("weight_decay", 0.0)),
        )
        return optim, None

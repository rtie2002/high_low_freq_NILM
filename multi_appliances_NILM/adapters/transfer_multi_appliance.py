"""Transfer-learning multi-appliance adapter (BERT4NILM + CNN heads)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from adapters.common import BaseNILMAdapter, StepOutput, center_output_slice
from model.TransferNILM import TransferMultiApplianceModel, transfer_nilm_config
from model.TransferNILM_loss import TransferNILMLoss


class TransferMultiApplianceAdapter(BaseNILMAdapter):
    name = "transfer_multi_appliance"

    def build_model(self, device: torch.device) -> torch.nn.Module:
        arch = self.model_cfg["architecture"]
        windowing = self.model_cfg["windowing"]
        cfg = transfer_nilm_config(arch, windowing)

        model = TransferMultiApplianceModel(
            cfg=cfg,
            num_appliances=len(self.cfg["appliances"]),
        )

        transfer_cfg = self.model_cfg.get("transfer", {})
        if bool(transfer_cfg.get("freeze_encoder", False)):
            model.freeze_encoder()

        return model.to(device)

    def build_loss(self) -> TransferNILMLoss:
        return TransferNILMLoss(power_scale=self._data_loader().loss_scale)

    def _align_loss_tensors(
        self,
        power_pred: torch.Tensor,
        state_prob: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w = self.model_cfg["windowing"]
        out_slice = center_output_slice(w)
        out_len = int(w.get("output_window_length", 1))

        if power_pred.shape[1] == y.shape[1]:
            return power_pred, state_prob, y, z

        if y.dim() == 3 and y.shape[1] == out_len:
            return power_pred[:, out_slice, :], state_prob[:, out_slice, :], y, z

        if w.get("training_loss_scope") == "center_output":
            power_pred = power_pred[:, out_slice, :]
            state_prob = state_prob[:, out_slice, :]
            y = y[:, out_slice, :]
            z = z[:, out_slice, :]

        return power_pred, state_prob, y, z

    def step(
        self,
        model: torch.nn.Module,
        loss_fn: TransferNILMLoss,
        batch: Any,
    ) -> StepOutput:
        x, y, z = batch
        z = z.float()

        power_pred, state_prob = model(x)
        power_pred, state_prob, y, z = self._align_loss_tensors(power_pred, state_prob, y, z)
        out = loss_fn(power_pred, state_prob, y, z)
        pred_state = (state_prob >= 0.5).long()

        app_logs = {}
        for app_i, app in enumerate(self.cfg["appliances"]):
            loss_r_i = loss_fn.mse(power_pred[..., app_i].float(), y[..., app_i].float())
            loss_c_i = loss_fn.bce(state_prob[..., app_i].float(), z[..., app_i].float())
            app_logs[f"loss_power_{app}"] = float(loss_r_i.detach())
            app_logs[f"loss_state_{app}"] = float(loss_c_i.detach())

        return StepOutput(
            loss=out.loss,
            logs={
                "loss": float(out.loss.detach()),
                "loss_power": float(out.loss_power.detach()),
                "loss_state": float(out.loss_state.detach()),
                "mae": float(out.mae.detach()),
                **app_logs,
            },
            aux={
                "pred_state": pred_state.detach().cpu(),
                "true_state": z.long().detach().cpu(),
                "pred_power": power_pred.detach().cpu(),
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
        out_slice = center_output_slice(self.model_cfg["windowing"])
        out_len = int(self.model_cfg["windowing"].get("output_window_length", 1))

        pred_power, pred_state = [], []
        true_power, true_state = [], []
        sample_indices = []
        offset = 0

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x, y, z = batch
            x = x.to(device)
            power_pred, state_prob = model(x)

            power_pred = power_pred[:, out_slice, :].cpu().numpy()
            state_prob = state_prob[:, out_slice, :].cpu().numpy()

            if y.dim() == 3 and y.shape[1] == out_len:
                y_true = y.numpy()
                z_true = z.numpy()
            else:
                y_true = y[:, out_slice, :].numpy() if y.dim() == 3 else y.numpy()
                z_true = z[:, out_slice, :].numpy() if z.dim() == 3 else z.numpy()

            n_apps = len(self.cfg["appliances"])
            pred_power.append(power_pred.reshape(len(x), -1, n_apps))
            pred_state.append((state_prob >= 0.5).astype(np.int32).reshape(len(x), -1, n_apps))
            true_power.append(y_true.reshape(len(x), -1, n_apps))
            true_state.append(z_true.reshape(len(x), -1, n_apps))
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

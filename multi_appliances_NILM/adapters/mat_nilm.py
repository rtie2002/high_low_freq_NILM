"""MATNILM adapter — wires MATconv + MSE/BCE loss into the shared pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from adapters.common import (
    BaseNILMAdapter,
    StepOutput,
    center_output_slice,
)
from model.MATNILM import MATconv
from model.MATNILM_loss import MATNILMLoss


class MATNILMAdapter(BaseNILMAdapter):
    name = "mat_nilm"

    def __init__(self, merged_cfg: dict[str, Any], data_root: str | None = None):
        # BaseNILMAdapter stores cfg, experiment, model_cfg, data_root, and
        # lazy DataLoader setup. MATNILM only adds its own appliance check.
        super().__init__(merged_cfg, data_root=data_root)

        appliances = self.cfg["appliances"]
        if len(appliances) != MATconv.NUM_APPLIANCES:
            raise ValueError(
                f"MATNILM requires exactly {MATconv.NUM_APPLIANCES} appliances; "
                f"got {len(appliances)}: {appliances}. "
                "Choose an experiment yaml that defines 4 appliances "
                "(for example config/experiment_redd.yaml)."
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
        return MATNILMLoss(power_scale=self._data_loader().loss_scale)

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

    def step(
        self,
        model: torch.nn.Module,
        loss_fn: MATNILMLoss,
        batch: Any,
    ) -> StepOutput:
        """Run one batch through MATNILM and return loss/logs.

        runner.py uses this same method for training and validation. The runner
        controls gradients; the adapter only handles model-specific forward and
        loss calculation.
        """
        x, y, z = batch
        # The dataloader now owns normalization and optional state rebuilding,
        # so adapters receive model-ready tensors and only run model logic.
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

        pred_power, pred_state = [], []
        true_power, true_state = [], []
        sample_indices = []
        offset = 0

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y, z = batch
            x = x.to(device)
            y_pred_r, y_pred_c_prob = model(x)

            y_pred_r = y_pred_r[:, out_slice, :].cpu().numpy()
            y_pred_c = y_pred_c_prob[:, out_slice, :].cpu().numpy()
            out_len = int(self.model_cfg["windowing"].get("output_window_length", 1))
            if y.dim() == 3 and y.shape[1] == out_len:
                y_true = y.numpy()
                z_true = z.numpy()
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
            split=split,
            sample_indices=sample_indices,
            pred_power_batches=pred_power,
            pred_state_batches=pred_state,
            true_power_batches=true_power,
            true_state_batches=true_state,
        )


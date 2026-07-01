"""UNet-NILM adapter — wires model, loss, and dataset into the shared pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from adapters.common import build_dataloader, build_prediction_bundle
from adapters.unet_preprocess import denorm_appliance_power
from adapters.types import PredictionBundle, StepOutput
from adapters.dataloader import NILMDataLoader, _resolve_input_length
from model.UNETNILM import UNETNiLM
from model.UNETNILM_loss import UNETNILMLoss


class UNetNILMAdapter:
    name = "unet_nilm"

    def __init__(self, merged_cfg: dict[str, Any], data_root: str | None = None):
        self.cfg = merged_cfg
        self.experiment = merged_cfg["experiment"]
        self.model_cfg = merged_cfg["model"]
        self.data_root = data_root or merged_cfg.get("data_root")
        self._data: NILMDataLoader | None = None

    def _data_loader(self) -> NILMDataLoader:
        if self._data is None:
            self._data = NILMDataLoader(self.experiment, self.model_cfg, self.data_root)
        return self._data

    def build_model(self, device: torch.device) -> torch.nn.Module:
        a = self.model_cfg["architecture"]
        c = a["conv_block"]
        model = UNETNiLM(
            in_size=a["in_size"],
            output_size=a["output_size"],
            seq_len=_resolve_input_length(self.model_cfg["windowing"]),
            d_model=a["encoder"]["d_model"],
            n_layers=a["unet"]["num_layers"],
            n_quantiles=a["heads"]["n_quantiles"],
            features_start=a["unet"]["features_start"],
            pool_filter=a["pool_filter"],
            encoder_n_layers=a["encoder"]["n_layers"],
            mlp_hidden=a["mlp"]["hidden_layers"],
            dropout=a["dropout"],
            kernel_size=c["kernel_size"],
            stride=c["stride"],
            padding=c["padding"],
        )
        return model.to(device)

    def build_loss(self) -> UNETNILMLoss:
        quantiles = self.model_cfg["architecture"]["heads"]["quantiles"]
        return UNETNILMLoss(quantiles=quantiles)

    def build_dataset(self, split: str) -> Dataset:
        return self._data_loader().build_dataset(split)

    def build_dataloader(self, split: str) -> DataLoader:
        return build_dataloader(
            self.build_dataset(split),
            self.model_cfg["training"],
            shuffle=(split == "train"),
        )

    def training_step(
        self,
        model: torch.nn.Module,
        loss_fn: UNETNILMLoss,
        batch: Any,
    ) -> StepOutput:
        x, y, z = batch
        states_logits, power_logits = model(x)
        out = loss_fn(states_logits, power_logits, y, z)
        pred_state = torch.max(F.softmax(states_logits, dim=1), dim=1).indices
        logs = {
            "loss": float(out.loss.detach()),
            "loss_state": float(out.loss_state.detach()),
            "loss_power": float(out.loss_power.detach()),
            "mae": float(out.mae.detach()),
        }
        return StepOutput(
            loss=out.loss,
            logs=logs,
            aux={"pred_state": pred_state.detach().cpu(), "true_state": z.detach().cpu()},
        )

    @torch.no_grad()
    def predict_dataloader(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        *,
        split: str = "validation",
        max_batches: int | None = None,
    ) -> PredictionBundle:
        model.eval()
        quantiles = self.model_cfg["architecture"]["heads"]["quantiles"]
        median_idx = len(quantiles) // 2
        seq2quantile = self.model_cfg["seq2quantile"]
        appliances = self.cfg["appliances"]
        denorm_style = str(self.model_cfg.get("data", {}).get("denorm_style", "standard"))

        pred_power_list, pred_state_list = [], []
        true_power_list, true_state_list = [], []
        sample_indices = []

        offset = 0
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y, z = batch
            x = x.to(device)
            states_logits, power_logits = model(x)
            pred_state = torch.max(F.softmax(states_logits, dim=1), dim=1).indices.cpu().numpy()
            if power_logits.dim() == 3:
                pred_power = power_logits[:, median_idx].cpu().numpy()
            else:
                pred_power = power_logits.cpu().numpy()

            pred_power_list.append(pred_power)
            pred_state_list.append(pred_state)
            true_power_list.append(y.numpy())
            true_state_list.append(z.numpy())
            sample_indices.append(np.arange(offset, offset + len(x)))
            offset += len(x)

        y_true = np.concatenate(true_power_list, axis=0)
        y_pred = np.concatenate(pred_power_list, axis=0)
        z_true = np.concatenate(true_state_list, axis=0)
        z_pred = np.concatenate(pred_state_list, axis=0)

        if self.model_cfg.get("data", {}).get("preprocess") == "unet_nilm":
            y_pred = denorm_appliance_power(y_pred, appliances, seq2quantile, style=denorm_style)
            gt_source = str(
                self.model_cfg.get("data", {}).get("waveform_ground_truth", "csv_raw")
            ).lower()
            if gt_source == "csv_raw":
                data_loader = self._data_loader()
                split_key = "validation" if split in ("val", "validation") else split
                raw_x, raw_y, raw_z = data_loader.get_raw_csv_arrays(split_key)  # type: ignore[arg-type]
                ts = data_loader.window_output_timesteps(split_key, len(y_true))
                y_true = raw_y[ts]
                z_true = raw_z[ts]
            else:
                y_true = denorm_appliance_power(y_true, appliances, seq2quantile, style=denorm_style)

        csv_timesteps = None
        if self.model_cfg.get("data", {}).get("preprocess") == "unet_nilm":
            split_key = "validation" if split in ("val", "validation") else split
            csv_timesteps = self._data_loader().window_output_timesteps(split_key, len(y_true))

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
            csv_timesteps=csv_timesteps,
        )

    def configure_optimizer(self, model: torch.nn.Module):
        t = self.model_cfg["training"]
        optim = torch.optim.Adam(
            model.parameters(),
            lr=float(t["learning_rate"]),
            betas=(float(t["beta_1"]), float(t["beta_2"])),
            eps=float(t.get("eps", 1e-8)),
            weight_decay=float(t.get("weight_decay", 0.0)),
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            patience=int(t.get("patience_scheduler", 5)),
            min_lr=1e-6,
            mode="max",
        )
        return optim, sched

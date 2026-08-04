"""Adapter: MultiNILM backbone + fractional 9-channel front-end."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from adapters.common import build_prediction_bundle
from adapters.config import appliance_off_norm_normalized
from adapters.multinilm import MultiNILMAdapter
from model.MultiNILM_fractional import MultiNILMFractional, build_multinilm_fractional
from model.MultiNILM_loss import MultiNILMLoss
from model.preprocess_feature.fcm import parse_active_state_fcm_config


class MultiNILMFractionalAdapter(MultiNILMAdapter):
    name = "multinilm_fractional"

    def build_model(self, device: torch.device) -> torch.nn.Module:
        arch = dict(self.model_cfg["architecture"])
        if "fractional" in self.model_cfg and isinstance(self.model_cfg["fractional"], dict):
            arch = {**arch, "fractional": self.model_cfg["fractional"]}

        appliances = self.cfg["appliances"]
        off_norms = appliance_off_norm_normalized(self.experiment, appliances)
        fcm_cfg = self.model_cfg.get("active_state_fcm")
        if not isinstance(fcm_cfg, dict):
            fcm_cfg = arch.get("active_state_fcm")

        model = build_multinilm_fractional(
            arch,
            num_appliances=len(appliances),
            output_length=int(self.model_cfg["windowing"].get("output_window_length", 1)),
            appliance_off_norm=off_norms,
            appliances=appliances,
            active_state_fcm_cfg=fcm_cfg if isinstance(fcm_cfg, dict) else None,
        )
        if (
            isinstance(fcm_cfg, dict)
            and bool(fcm_cfg.get("enabled", False))
            and getattr(model, "active_state_fcm", None) is None
        ):
            model.configure_active_state_fcm(
                appliances,
                parse_active_state_fcm_config(fcm_cfg),
                enabled=True,
            )
        return model.to(device)

    def build_loss(self) -> MultiNILMLoss:
        return super().build_loss()

    def _ensure_active_state_fcm_fitted(self, model: torch.nn.Module) -> None:
        """Fit Schirmer FCM centers once from train ground-truth watts."""
        if not isinstance(model, MultiNILMFractional):
            return
        if not model.active_state_fcm_enabled:
            return
        if model._active_state_fcm_fitted:
            return

        train_loader = self.build_dataloader("train")
        chunks: list[np.ndarray] = []
        n_app = len(self.cfg["appliances"])
        for batch in train_loader:
            _x, y, _z = batch
            y_np = np.asarray(y.detach().cpu().numpy(), dtype=np.float64)
            if y_np.ndim == 3:
                y_np = y_np.reshape(-1, y_np.shape[-1])
            elif y_np.ndim != 2 or y_np.shape[-1] != n_app:
                y_np = y_np.reshape(-1, n_app)
            chunks.append(y_np)

        if not chunks:
            return
        y_norm = np.concatenate(chunks, axis=0)
        y_watts = self._data_loader().denorm_to_watts(y_norm)
        summary = model.fit_active_state_fcm(y_watts)
        print(f"[active_state_fcm] fitted centers: {summary}", flush=True)

    @torch.no_grad()
    def predict_dataloader(
        self,
        model: torch.nn.Module,
        loader: Any,
        device: torch.device,
        *,
        max_batches: int | None = None,
        split: str = "test",
    ) -> Any:
        # Fig. 2: fit appliance active centers (source), then predict + post-process.
        self._ensure_active_state_fcm_fitted(model)
        bundle = super().predict_dataloader(
            model,
            loader,
            device,
            max_batches=max_batches,
            split=split,
        )
        if isinstance(model, MultiNILMFractional) and model.active_state_fcm_enabled:
            y_pp = model.postprocess_power_watts(bundle.y_pred_watts)
            bundle = build_prediction_bundle(
                experiment_id=bundle.experiment_id,
                model_name=bundle.model_name,
                split=bundle.split,
                appliances=list(bundle.appliances),
                sample_index=bundle.sample_index,
                y_true_watts=bundle.y_true_watts,
                y_pred_watts=y_pp,
                y_true_on=bundle.y_true_on,
                y_pred_on=bundle.y_pred_on,
                csv_timesteps=getattr(bundle, "csv_timesteps", None),
            )
        return bundle

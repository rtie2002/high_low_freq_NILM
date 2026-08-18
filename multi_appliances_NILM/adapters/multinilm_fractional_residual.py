"""Adapter: MultiNILM-Fractional with all-appliance residual refinement."""

from __future__ import annotations

import numpy as np
import torch

from adapters.config import appliance_off_norm_normalized
from adapters.multinilm import MultiNILMAdapter
from model.MultiNILM_fractional_residual import build_multinilm_fractional_residual
from model.MultiNILM_loss import MultiNILMLoss


class MultiNILMFractionalResidualAdapter(MultiNILMAdapter):
    name = "multinilm_fractional_residual"

    def build_model(self, device: torch.device) -> torch.nn.Module:
        arch = dict(self.model_cfg["architecture"])
        if "fractional" in self.model_cfg and isinstance(self.model_cfg["fractional"], dict):
            arch = {**arch, "fractional": self.model_cfg["fractional"]}

        appliances = self.cfg["appliances"]
        off_norms = appliance_off_norm_normalized(self.experiment, appliances)
        norm = self._data_loader().norm

        app_mean = (
            norm.target_mean.astype(np.float32).tolist()
            if norm.target_mean is not None
            else [0.0] * len(appliances)
        )
        app_std = (
            norm.target_std.astype(np.float32).tolist()
            if norm.target_std is not None
            else [float(norm.legacy_scale)] * len(appliances)
        )

        model = build_multinilm_fractional_residual(
            arch,
            num_appliances=len(appliances),
            output_length=int(self.model_cfg["windowing"].get("output_window_length", 1)),
            appliance_off_norm=off_norms,
            aggregate_mean=float(norm.input_mean or 0.0),
            aggregate_std=float(norm.input_std or norm.legacy_scale or 1.0),
            appliance_mean=app_mean,
            appliance_std=app_std,
        )
        return model.to(device)

    def build_loss(self) -> MultiNILMLoss:
        return super().build_loss()

"""Adapter: MultiNILM backbone + fractional 9-channel front-end."""

from __future__ import annotations

import torch

from adapters.config import appliance_off_norm_normalized
from adapters.multinilm import MultiNILMAdapter
from model.MultiNILM_fractional import build_multinilm_fractional
from model.MultiNILM_loss import MultiNILMLoss


class MultiNILMFractionalAdapter(MultiNILMAdapter):
    name = "multinilm_fractional"

    def build_model(self, device: torch.device) -> torch.nn.Module:
        arch = dict(self.model_cfg["architecture"])
        if "fractional" in self.model_cfg and isinstance(self.model_cfg["fractional"], dict):
            arch = {**arch, "fractional": self.model_cfg["fractional"]}

        appliances = self.cfg["appliances"]
        off_norms = appliance_off_norm_normalized(self.experiment, appliances)
        model = build_multinilm_fractional(
            arch,
            num_appliances=len(appliances),
            output_length=int(self.model_cfg["windowing"].get("output_window_length", 1)),
            appliance_off_norm=off_norms,
        )
        return model.to(device)

    def build_loss(self) -> MultiNILMLoss:
        return super().build_loss()

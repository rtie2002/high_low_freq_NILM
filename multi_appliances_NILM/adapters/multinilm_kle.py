"""Adapter: MultiNILM backbone + KLE 9-channel front-end."""

from __future__ import annotations

import torch

from adapters.config import appliance_off_norm_normalized
from adapters.multinilm import MultiNILMAdapter
from model.MultiNILM_kle import build_multinilm_kle
from model.MultiNILM_loss import MultiNILMLoss


class MultiNILMKLEAdapter(MultiNILMAdapter):
    """Dataloader stays 1D; ``MultiNILMKLE`` expands to raw+8 KLE SC channels."""

    name = "multinilm_kle"

    def build_model(self, device: torch.device) -> torch.nn.Module:
        arch = dict(self.model_cfg["architecture"])
        # Allow top-level ``kle:`` block in yaml (merged into architecture for builder).
        if "kle" in self.model_cfg and isinstance(self.model_cfg["kle"], dict):
            arch = {**arch, "kle": self.model_cfg["kle"]}

        appliances = self.cfg["appliances"]
        off_norms = appliance_off_norm_normalized(self.experiment, appliances)
        model = build_multinilm_kle(
            arch,
            num_appliances=len(appliances),
            output_length=int(self.model_cfg["windowing"].get("output_window_length", 1)),
            appliance_off_norm=off_norms,
        )
        return model.to(device)

    def build_loss(self) -> MultiNILMLoss:
        return super().build_loss()

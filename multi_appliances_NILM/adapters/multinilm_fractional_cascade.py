"""Adapter: paper-style sequential MultiNILM-Fractional cascade."""

from __future__ import annotations

import numpy as np
import torch

from adapters.config import appliance_off_norm_normalized
from adapters.multinilm import MultiNILMAdapter
from model.MultiNILM_fractional_cascade import build_multinilm_fractional_cascade
from model.MultiNILM_loss import MultiNILMLoss


class MultiNILMFractionalCascadeAdapter(MultiNILMAdapter):
    name = "multinilm_fractional_cascade"

    def _cascade_order_indices(self, appliances: list[str]) -> list[int]:
        cascade_cfg = self.model_cfg.get("architecture", {}).get("cascade", {})
        order = cascade_cfg.get("order") if isinstance(cascade_cfg, dict) else None
        if not order:
            order = appliances
        order = [str(name).strip() for name in order]
        missing = [name for name in order if name not in appliances]
        if missing:
            raise ValueError(
                "cascade.order contains appliances not in experiment: "
                + ", ".join(missing)
            )
        if sorted(order) != sorted(appliances):
            raise ValueError(
                "cascade.order must contain every experiment appliance exactly once. "
                f"Expected {appliances}, got {order}"
            )
        return [appliances.index(name) for name in order]

    def build_model(self, device: torch.device) -> torch.nn.Module:
        arch = dict(self.model_cfg["architecture"])
        if "fractional" in self.model_cfg and isinstance(self.model_cfg["fractional"], dict):
            arch = {**arch, "fractional": self.model_cfg["fractional"]}

        appliances = self.cfg["appliances"]
        order_indices = self._cascade_order_indices(appliances)
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

        model = build_multinilm_fractional_cascade(
            arch,
            num_appliances=len(appliances),
            output_length=int(self.model_cfg["windowing"].get("output_window_length", 1)),
            order_indices=order_indices,
            appliance_off_norm=off_norms,
            aggregate_mean=float(norm.input_mean or 0.0),
            aggregate_std=float(norm.input_std or norm.legacy_scale or 1.0),
            appliance_mean=app_mean,
            appliance_std=app_std,
        )
        return model.to(device)

    def build_loss(self) -> MultiNILMLoss:
        return super().build_loss()

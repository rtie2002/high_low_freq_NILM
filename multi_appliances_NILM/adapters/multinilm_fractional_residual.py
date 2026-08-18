"""Adapter: MultiNILM-Fractional with all-appliance residual refinement."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from adapters.config import appliance_off_norm_normalized
from adapters.multinilm import MultiNILMAdapter
from model.MultiNILM_fractional_residual import build_multinilm_fractional_residual
from model.MultiNILM_loss import MultiNILMLoss


ROOT = Path(__file__).resolve().parents[1]


def _load_base_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(f"base_init_checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device)
    source_state = payload.get("model_state_dict", payload)
    base = getattr(model, "base", None)
    if base is None:
        raise ValueError("Residual model has no .base module for base_init_checkpoint")

    target_state = base.state_dict()
    loadable = {}
    skipped = {}
    for name, tensor in source_state.items():
        if name not in target_state:
            skipped[name] = "missing_in_base"
            continue
        if tuple(tensor.shape) != tuple(target_state[name].shape):
            skipped[name] = f"shape {tuple(tensor.shape)} -> {tuple(target_state[name].shape)}"
            continue
        loadable[name] = tensor
    missing, unexpected = base.load_state_dict(loadable, strict=False)
    print(
        f"Initialized residual base from checkpoint: {checkpoint}\n"
        f"  loaded tensors : {len(loadable)}\n"
        f"  skipped tensors: {len(skipped)}\n"
        f"  missing tensors: {len(missing)}\n"
        f"  unexpected     : {len(unexpected)}",
        flush=True,
    )


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
        model = model.to(device)
        train_cfg = self.model_cfg.get("training", {})
        base_init = train_cfg.get("base_init_checkpoint")
        if base_init:
            ckpt = Path(str(base_init))
            if not ckpt.is_absolute():
                ckpt = ROOT / ckpt
            _load_base_checkpoint(model, ckpt, device)
        return model

    def build_loss(self) -> MultiNILMLoss:
        return super().build_loss()

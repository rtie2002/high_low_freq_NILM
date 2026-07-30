"""Adapter for frozen MultiNILM without PAD-lite distill (git 6df298f)."""

from __future__ import annotations

import torch

from adapters.config import appliance_off_norm_normalized
from adapters.multinilm import MultiNILMAdapter
from model.MultiNILM_loss import MultiNILMLoss
from model.MultiNILM_no_distill import MultiNILM, multinilm_config


class MultiNILMNoDistillAdapter(MultiNILMAdapter):
    """Same training/eval path as multinilm, but model = snapshot without distill."""

    name = "multinilm_no_distill"

    def build_model(self, device: torch.device) -> torch.nn.Module:
        arch = self.model_cfg["architecture"]
        cfg = multinilm_config(arch)
        appliances = self.cfg["appliances"]
        off_norms = appliance_off_norm_normalized(self.experiment, appliances)

        model = MultiNILM(
            input_channels=cfg.input_channels,
            num_appliances=len(appliances),
            output_length=int(self.model_cfg["windowing"].get("output_window_length", 1)),
            hidden_channels=cfg.hidden_channels,
            channel_schedule=cfg.channel_schedule,
            stem_kernel_size=cfg.stem_kernel_size,
            stage_kernel_size=cfg.stage_kernel_size,
            num_blocks=cfg.num_blocks,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            max_dilation=cfg.max_dilation,
            gate_mode=cfg.gate_mode,
            gate_threshold=cfg.gate_threshold,
            appliance_off_norm=off_norms,
            domain_feature_layers=cfg.domain_feature_layers,
            head_local_layers=cfg.head_local_layers,
            head_kernel_size=cfg.head_kernel_size,
            head_use_residual=cfg.head_use_residual,
            use_multiscale_stem=cfg.use_multiscale_stem,
            detail_kernels=cfg.detail_kernels,
            detail_branch_channels=cfg.detail_branch_channels,
        )
        return model.to(device)

    def build_loss(self) -> MultiNILMLoss:
        return super().build_loss()

"""
MultiNILM + fractional multi-channel front-end (separate from baseline MultiNILM).

Dataloader still provides 1D aggregate ``(B, T)`` / ``(B, 1, T)``.
``FractionalFrontEnd`` expands to ``(B, C, T)`` (default C=9 = raw + 8 α),
then the standard MultiNILM backbone runs with ``input_channels=C``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.MultiNILM import MultiNILM, MultiNILMConfig, multinilm_config
from model.preprocess_feature.fractional import (
    FractionalFrontEnd,
    default_schirmer_alphas,
    parse_fractional_architecture,
)


class MultiNILMFractional(nn.Module):
    """Wrapper: 1D aggregate → GL channels → MultiNILM backbone."""

    def __init__(
        self,
        *,
        backbone: MultiNILM,
        frontend: FractionalFrontEnd,
    ) -> None:
        super().__init__()
        self.frontend = frontend
        self.backbone = backbone

        if int(backbone.input_channels) != int(frontend.out_channels):
            raise ValueError(
                "MultiNILM backbone input_channels must match FractionalFrontEnd: "
                f"expected {frontend.out_channels}, got {backbone.input_channels}."
            )

        self.input_channels = 1
        self.feature_channels = int(frontend.out_channels)
        self.num_appliances = backbone.num_appliances
        self.output_length = backbone.output_length
        self.domain_feature_layers = backbone.domain_feature_layers

    def _to_b1t(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1).float()
        if x.dim() == 3 and x.shape[1] == 1:
            return x.float()
        if x.dim() == 3 and x.shape[-1] == 1:
            return x.permute(0, 2, 1).float()
        raise ValueError(
            "MultiNILMFractional expects (B, T) or (B, 1, T); "
            f"got {tuple(x.shape)}."
        )

    def forward(self, x: torch.Tensor, return_domain_features: bool = False):
        x_c = self.frontend(self._to_b1t(x))
        return self.backbone(x_c, return_domain_features=return_domain_features)


def build_multinilm_fractional(
    architecture: dict[str, Any],
    *,
    num_appliances: int,
    output_length: int,
    appliance_off_norm: list[float] | None = None,
) -> MultiNILMFractional:
    """
    Build from yaml. Fractional settings live under ``architecture.fractional``
    or top-level ``fractional:`` (adapter merges the latter).
    """
    # Prefer nested block; allow top-level merge by adapter.
    frac_arch = dict(architecture)
    if "fractional" not in frac_arch and isinstance(architecture.get("kle"), dict):
        pass  # unrelated

    enabled, alphas, include_raw, memory, h = parse_fractional_architecture(frac_arch)
    # This builder always enables the front-end (dedicated model).
    if alphas is None:
        alphas = default_schirmer_alphas(8)
    if not enabled and "fractional" not in frac_arch:
        # Dedicated model: default on even if block omitted.
        include_raw = True
        alphas = default_schirmer_alphas(8)
        memory = None
        h = 1.0

    frontend = FractionalFrontEnd(
        alphas=alphas,
        include_raw=include_raw,
        memory=memory,
        h=h,
    )
    feature_c = int(frontend.out_channels)

    arch = dict(architecture)
    arch["input_channels"] = feature_c
    # Avoid accidental re-parse enabling inside baseline MultiNILM (none there).
    cfg: MultiNILMConfig = multinilm_config(arch)

    backbone = MultiNILM(
        input_channels=feature_c,
        num_appliances=int(num_appliances),
        output_length=int(output_length),
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
        appliance_off_norm=appliance_off_norm,
        domain_feature_layers=cfg.domain_feature_layers,
        head_local_layers=cfg.head_local_layers,
        head_kernel_size=cfg.head_kernel_size,
        head_use_residual=cfg.head_use_residual,
        use_multiscale_stem=cfg.use_multiscale_stem,
        detail_kernels=cfg.detail_kernels,
        detail_branch_channels=cfg.detail_branch_channels,
        cross_appliance_enabled=cfg.cross_appliance_enabled,
        cross_appliance_residual_scale=cfg.cross_appliance_residual_scale,
        cross_appliance_mid_channels=cfg.cross_appliance_mid_channels,
    )
    return MultiNILMFractional(backbone=backbone, frontend=frontend)

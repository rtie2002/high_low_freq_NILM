"""
MultiNILM + fractional multi-channel front-end (separate from baseline MultiNILM).

Dataloader still provides 1D aggregate ``(B, T)`` / ``(B, 1, T)``.
``FractionalFrontEnd`` expands to ``(B, C, T)`` (default C=9 = raw + 8 α),
then the standard MultiNILM backbone runs with ``input_channels=C``.

Optional Schirmer Sec. III-C active-state FCM post-process
(``ActiveStateFCMPostProcess``) lives on this wrapper:

  - Fit on **source ground-truth watts** (train labels).
  - Apply on **predicted watts** after denorm (eval / plots), not in ``forward``
    (``forward`` stays in normalized space for training + DA).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from model.MultiNILM import MultiNILM, MultiNILMConfig, multinilm_config
from model.preprocess_feature.fcm import (
    ActiveStateFCMConfig,
    ActiveStateFCMPostProcess,
    parse_active_state_fcm_config,
)
from model.preprocess_feature.fractional import (
    FractionalFrontEnd,
    default_schirmer_alphas,
    parse_fractional_architecture,
)


class MultiNILMFractional(nn.Module):
    """Wrapper: 1D aggregate → GL channels → MultiNILM backbone (+ optional FCM PP)."""

    def __init__(
        self,
        *,
        backbone: MultiNILM,
        frontend: FractionalFrontEnd,
        appliances: Sequence[str] | None = None,
        active_state_fcm: ActiveStateFCMPostProcess | None = None,
        active_state_fcm_enabled: bool = False,
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

        # Schirmer C — not an nn.Module path; eval-only watt snap.
        self.appliances = [str(a) for a in (appliances or [])]
        self.active_state_fcm_enabled = bool(active_state_fcm_enabled)
        self.active_state_fcm = active_state_fcm
        self._active_state_fcm_fitted = False

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
        # Training / DA path: normalized tensors only (no FCM here).
        x_c = self.frontend(self._to_b1t(x))
        return self.backbone(x_c, return_domain_features=return_domain_features)

    # ------------------------------------------------------------------
    # Schirmer active-state FCM (Fig. 2: regression p̂' → post-processing)
    # ------------------------------------------------------------------

    def configure_active_state_fcm(
        self,
        appliances: Sequence[str],
        config: ActiveStateFCMConfig | None = None,
        *,
        enabled: bool = True,
    ) -> ActiveStateFCMPostProcess:
        """Attach / replace the FCM post-process block."""
        self.appliances = [str(a) for a in appliances]
        if len(self.appliances) != int(self.num_appliances):
            raise ValueError(
                f"appliances length {len(self.appliances)} != "
                f"num_appliances {self.num_appliances}"
            )
        self.active_state_fcm = ActiveStateFCMPostProcess(
            self.appliances, config or ActiveStateFCMConfig()
        )
        self.active_state_fcm_enabled = bool(enabled)
        self._active_state_fcm_fitted = False
        return self.active_state_fcm

    def fit_active_state_fcm(self, power_watts: np.ndarray) -> dict[str, list[float]]:
        """
        Fit centers from **source ground-truth watts** ``(T, K)``.

        Call once before eval (e.g. on train split labels).
        """
        if self.active_state_fcm is None:
            if not self.appliances:
                raise RuntimeError(
                    "configure_active_state_fcm(...) before fit_active_state_fcm"
                )
            self.active_state_fcm = ActiveStateFCMPostProcess(
                self.appliances, ActiveStateFCMConfig()
            )
        self.active_state_fcm.fit_from_power(power_watts)
        self._active_state_fcm_fitted = True
        self.active_state_fcm_enabled = True
        return self.active_state_fcm.summary()

    def postprocess_power_watts(self, power_watts: np.ndarray) -> np.ndarray:
        """
        Snap **predicted** watts with Eq. (10).

        No-op if FCM disabled or not fitted. Does not modify near-OFF (≤ ε).
        """
        if (
            not self.active_state_fcm_enabled
            or self.active_state_fcm is None
            or not self._active_state_fcm_fitted
        ):
            return np.asarray(power_watts, dtype=np.float64)
        return self.active_state_fcm.apply(power_watts)


def build_multinilm_fractional(
    architecture: dict[str, Any],
    *,
    num_appliances: int,
    output_length: int,
    appliance_off_norm: list[float] | None = None,
    appliances: Sequence[str] | None = None,
    active_state_fcm_cfg: dict[str, Any] | None = None,
) -> MultiNILMFractional:
    """
    Build from yaml. Fractional settings live under ``architecture.fractional``
    or top-level ``fractional:`` (adapter merges the latter).

    Optional ``active_state_fcm_cfg`` / ``architecture.active_state_fcm`` enables
    Schirmer C (centers fitted later via ``fit_active_state_fcm``).
    """
    frac_arch = dict(architecture)

    enabled, alphas, include_raw, memory, h = parse_fractional_architecture(frac_arch)
    if alphas is None:
        alphas = default_schirmer_alphas(8)
    if not enabled and "fractional" not in frac_arch:
        include_raw = True
        alphas = default_schirmer_alphas(8)
        memory = None
        h = 1.0

    frontend = FractionalFrontEnd(
        alphas=alphas,
        include_raw=include_raw,
        memory=memory,
        h=h,
        channel_normalize=str(
            (architecture.get("fractional") or {}).get("channel_normalize", "mean_std")
            if isinstance(architecture.get("fractional"), dict)
            else "mean_std"
        ),
    )
    feature_c = int(frontend.out_channels)

    arch = dict(architecture)
    arch["input_channels"] = feature_c
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

    fcm_block = active_state_fcm_cfg
    if fcm_block is None and isinstance(architecture.get("active_state_fcm"), dict):
        fcm_block = architecture["active_state_fcm"]  # type: ignore[assignment]
    fcm_enabled = bool(isinstance(fcm_block, dict) and fcm_block.get("enabled", False))
    fcm_pp = None
    app_names = [str(a) for a in (appliances or [])]
    if fcm_enabled:
        if len(app_names) != int(num_appliances):
            raise ValueError(
                "active_state_fcm.enabled requires appliances= list matching num_appliances"
            )
        fcm_pp = ActiveStateFCMPostProcess(
            app_names, parse_active_state_fcm_config(fcm_block)
        )

    return MultiNILMFractional(
        backbone=backbone,
        frontend=frontend,
        appliances=app_names,
        active_state_fcm=fcm_pp,
        active_state_fcm_enabled=fcm_enabled,
    )

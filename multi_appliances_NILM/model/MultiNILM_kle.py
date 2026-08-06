"""
MultiNILM + KLE multi-channel front-end (9-D by default).

Dataloader still provides 1D aggregate ``(B, T)`` / ``(B, 1, T)``.
This wrapper expands each window with ``kle_subspace_channels`` from
``preprocess_feature.kle`` into ``(B, C, T)`` and runs the standard
MultiNILM backbone with ``input_channels=C``.

Default: C = 9 = raw + 8 KLE FIR subspace components (Dinesh-style SCs).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from model.MultiNILM import MultiNILM, MultiNILMConfig, multinilm_config
from model.preprocess_feature.kle import kle_subspace_channels_batch


class MultiNILMKLE(nn.Module):
    """
    Same heads / TCN / DA hooks as MultiNILM; only the input rank changes.

    External contract (dataloader):
        x: (B, T) or (B, 1, T)  — normalized aggregate
    Internal backbone:
        (B, C, T) with C = n_components (+1 if include_raw)
    """

    def __init__(
        self,
        *,
        backbone: MultiNILM,
        kle_n_components: int = 8,
        kle_include_raw: bool = True,
        kle_demean: bool = True,
        kle_channel_normalize: str = "mean_std",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.kle_n_components = int(kle_n_components)
        self.kle_include_raw = bool(kle_include_raw)
        self.kle_demean = bool(kle_demean)
        self.kle_channel_normalize = str(kle_channel_normalize)

        expected_c = self.kle_n_components + (1 if self.kle_include_raw else 0)
        if int(backbone.input_channels) != expected_c:
            raise ValueError(
                "MultiNILM backbone input_channels must match KLE front-end: "
                f"expected {expected_c} "
                f"(n_components={self.kle_n_components}, include_raw={self.kle_include_raw}), "
                f"got {backbone.input_channels}."
            )

        # Expose attributes adapters / DA may read from the top module.
        self.input_channels = 1  # raw aggregate channels from dataloader
        self.feature_channels = expected_c
        self.num_appliances = backbone.num_appliances
        self.output_length = backbone.output_length
        self.domain_feature_layers = backbone.domain_feature_layers

    def _aggregate_bt(self, x: torch.Tensor) -> torch.Tensor:
        """Force shape (B, T) float32 for the numpy KLE front-end."""
        if x.dim() == 2:
            return x.float()
        if x.dim() == 3 and x.shape[1] == 1:
            return x[:, 0, :].float()
        if x.dim() == 3 and x.shape[-1] == 1:
            return x[..., 0].float()
        raise ValueError(
            "MultiNILMKLE expects dataloader aggregate as (B, T) or (B, 1, T); "
            f"got {tuple(x.shape)}."
        )

    def _kle_expand(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, T) → (B, C, T) via ``kle_subspace_channels_batch``.

        Runs on CPU numpy (ACM eigendecomp); result is moved back to x.device.
        Front-end is treated as fixed features (no grad through KLE).
        """
        bt = self._aggregate_bt(x)
        device = bt.device
        arr = bt.detach().cpu().numpy().astype(np.float64, copy=False)
        feats = kle_subspace_channels_batch(
            arr,
            self.kle_n_components,
            demean=self.kle_demean,
            include_raw=self.kle_include_raw,
            channel_normalize=self.kle_channel_normalize,  # type: ignore[arg-type]
        )
        out = torch.from_numpy(feats.astype(np.float32, copy=False)).to(device)
        return out

    def forward(
        self,
        x: torch.Tensor,
        return_domain_features: bool = False,
    ):
        x_c = self._kle_expand(x)
        return self.backbone(x_c, return_domain_features=return_domain_features)

    def encode_shared(self, x: torch.Tensor, **kwargs):
        """Delegate if the training loop calls encode on the root module."""
        x_c = self._kle_expand(x)
        if hasattr(self.backbone, "encode_shared"):
            return self.backbone.encode_shared(x_c, **kwargs)
        raise AttributeError("backbone has no encode_shared")


def build_multinilm_kle(
    architecture: dict[str, Any],
    *,
    num_appliances: int,
    output_length: int,
    appliance_off_norm: list[float] | None = None,
) -> MultiNILMKLE:
    """
    Build MultiNILMKLE from model yaml ``architecture`` (+ optional ``kle`` block).

    Example yaml::

        architecture:
          # stem sees C feature channels (set automatically if omitted)
          input_channels: 9
          ...
        kle:
          n_components: 8
          include_raw: true
          demean: true
          channel_normalize: mean_std
    """
    kle_cfg = architecture.get("kle", {})
    if not isinstance(kle_cfg, dict):
        kle_cfg = {}

    n_comp = int(kle_cfg.get("n_components", 8))
    include_raw = bool(kle_cfg.get("include_raw", True))
    demean = bool(kle_cfg.get("demean", True))
    ch_norm = str(kle_cfg.get("channel_normalize", "mean_std"))
    feature_c = n_comp + (1 if include_raw else 0)

    # Force backbone in_channels to match KLE width (ignore stale yaml=1).
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
        head_style=cfg.head_style,
        use_multiscale_stem=cfg.use_multiscale_stem,
        detail_kernels=cfg.detail_kernels,
        detail_branch_channels=cfg.detail_branch_channels,
        cross_appliance_enabled=cfg.cross_appliance_enabled,
        cross_appliance_residual_scale=cfg.cross_appliance_residual_scale,
        cross_appliance_mid_channels=cfg.cross_appliance_mid_channels,
    )
    return MultiNILMKLE(
        backbone=backbone,
        kle_n_components=n_comp,
        kle_include_raw=include_raw,
        kle_demean=demean,
        kle_channel_normalize=ch_norm,
    )

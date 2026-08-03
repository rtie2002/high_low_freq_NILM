"""
MultiNILM + Schirmer front-end: fractional (1D channels) + KLE spectrogram (2D matrix).

Pipeline
--------
1. ``FractionalFrontEnd``: p → (B, C, T), default C=9 (raw + 8 α) on GPU.
2. ``kle_spectrogram_from_channels`` (GPU): fractional channels → A, Φ ∈ R^{N×K}.
3. ``KleSpectrogramEncoder`` (Conv2d) reads the matrix → embedding.
4. FiLM modulates the C fractional channels with that embedding.
5. Unchanged ``MultiNILM`` backbone (``input_channels=C``).
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
from model.preprocess_feature.kle import kle_spectrogram_from_channels


class KleSpectrogramEncoder(nn.Module):
    """Conv2d encoder for Schirmer A/Φ maps ``(B, 2, N, K)`` → ``(B, out_dim)``."""

    def __init__(
        self,
        *,
        out_dim: int,
        use_phase: bool = True,
        width: int = 32,
    ) -> None:
        super().__init__()
        in_ch = 2 if use_phase else 1
        self.use_phase = bool(use_phase)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 2, int(out_dim)),
        )

    def forward(self, mag: torch.Tensor, phase: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            mag: (B, N, K)
            phase: (B, N, K) if use_phase
        """
        if self.use_phase:
            if phase is None:
                raise ValueError("phase required when use_phase=True")
            x = torch.stack([mag, phase], dim=1)  # (B, 2, N, K)
        else:
            x = mag.unsqueeze(1)
        return self.net(x)


class MultiNILMSchirmer(nn.Module):
    """
    Fractional 1D path + KLE 2D matrix path → FiLM → MultiNILM backbone.
    """

    def __init__(
        self,
        *,
        backbone: MultiNILM,
        frontend: FractionalFrontEnd,
        spec_encoder: KleSpectrogramEncoder,
        film: nn.Linear,
        kle_n_components: int = 64,
        kle_normalize: str = "fundamental",
        kle_include_raw_column: bool = False,
        kle_memory: int | None = None,
        kle_phase_mode: str = "hilbert",
    ) -> None:
        super().__init__()
        self.frontend = frontend
        self.backbone = backbone
        self.spec_encoder = spec_encoder
        self.film = film

        self.kle_n_components = int(kle_n_components)
        self.kle_normalize = str(kle_normalize)
        self.kle_include_raw_column = bool(kle_include_raw_column)
        self.kle_memory = kle_memory
        self.kle_phase_mode = str(kle_phase_mode)
        self.alphas = list(frontend.alphas)

        if int(backbone.input_channels) != int(frontend.out_channels):
            raise ValueError(
                "backbone input_channels must match FractionalFrontEnd.out_channels: "
                f"{backbone.input_channels} vs {frontend.out_channels}"
            )
        if int(film.out_features) != 2 * int(frontend.out_channels):
            raise ValueError(
                "FiLM must map to 2*C (gamma, beta); "
                f"got out_features={film.out_features}, C={frontend.out_channels}"
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
            f"MultiNILMSchirmer expects (B,T) or (B,1,T); got {tuple(x.shape)}"
        )

    def _kle_channels_from_frontend(self, xf: torch.Tensor) -> torch.Tensor:
        """
        Select channels for KLE spectrogram from FractionalFrontEnd output.

        If frontend includes raw and ``kle_include_raw_column`` is False (default),
        use only the α channels (paper K columns). If True, KLE all C channels.
        """
        if self.frontend.include_raw and not self.kle_include_raw_column:
            if xf.shape[1] <= 1:
                return xf
            return xf[:, 1:, :]
        return xf

    def forward(self, x: torch.Tensor, return_domain_features: bool = False):
        b1t = self._to_b1t(x)
        # 1) Fractional multi-channel time series on GPU.
        xf = self.frontend(b1t)  # (B, C, T)

        # 2) KLE spectrogram on GPU (reuse fractional channels; no CPU numpy).
        kle_in = self._kle_channels_from_frontend(xf)
        with torch.no_grad():
            # Fixed front-end features (like numpy path); keeps graph on FiLM/backbone.
            mag, phase = kle_spectrogram_from_channels(
                kle_in.detach(),
                self.kle_n_components,
                normalize=self.kle_normalize,  # type: ignore[arg-type]
                phase_mode=self.kle_phase_mode,  # type: ignore[arg-type]
            )
        emb = self.spec_encoder(mag, phase)  # (B, D)

        # 3) FiLM: modulate each fractional channel with spectral prior.
        gb = self.film(emb)  # (B, 2C)
        c = xf.shape[1]
        gamma, beta = gb[:, :c], gb[:, c:]
        xf = xf * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

        return self.backbone(xf, return_domain_features=return_domain_features)


def build_multinilm_schirmer(
    architecture: dict[str, Any],
    *,
    num_appliances: int,
    output_length: int,
    appliance_off_norm: list[float] | None = None,
) -> MultiNILMSchirmer:
    """
    Yaml blocks:
      fractional: {k, include_raw, memory, ...}
      kle: {n_components, normalize, use_phase, include_raw_column, ...}
    """
    frac_block = architecture.get("fractional", {})
    if not isinstance(frac_block, dict):
        frac_block = {}
    kle_block = architecture.get("kle", {})
    if not isinstance(kle_block, dict):
        kle_block = {}

    # Parse fractional (dedicated model → always on).
    enabled, alphas, include_raw, memory, h = parse_fractional_architecture(
        {"fractional": {**frac_block, "enabled": True}}
    )
    if alphas is None:
        alphas = default_schirmer_alphas(int(frac_block.get("k", 8)))

    frontend = FractionalFrontEnd(
        alphas=alphas,
        include_raw=bool(include_raw if frac_block else True),
        memory=memory if memory is not None else frac_block.get("memory"),
        h=float(h),
    )
    feature_c = int(frontend.out_channels)

    n_comp = int(kle_block.get("n_components", 64))
    use_phase = bool(kle_block.get("use_phase", True))
    kle_norm = str(kle_block.get("normalize", "fundamental"))
    kle_raw_col = bool(kle_block.get("include_raw_column", False))
    kle_phase_mode = str(kle_block.get("phase_mode", "hilbert"))
    kle_memory = kle_block.get("memory", memory)
    kle_memory_i = None if kle_memory is None else int(kle_memory)
    emb_dim = int(kle_block.get("embed_dim", 64))
    conv_width = int(kle_block.get("conv_width", 32))

    spec_encoder = KleSpectrogramEncoder(
        out_dim=emb_dim, use_phase=use_phase, width=conv_width
    )
    film = nn.Linear(emb_dim, 2 * feature_c)

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

    return MultiNILMSchirmer(
        backbone=backbone,
        frontend=frontend,
        spec_encoder=spec_encoder,
        film=film,
        kle_n_components=n_comp,
        kle_normalize=kle_norm,
        kle_include_raw_column=kle_raw_col,
        kle_memory=kle_memory_i,
        kle_phase_mode=kle_phase_mode,
    )

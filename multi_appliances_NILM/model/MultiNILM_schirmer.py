"""
MultiNILM + Schirmer-inspired front-end (honest hybrid).

Not a full Schirmer 2D-CNN regressor: we keep multi-appliance seq2seq MultiNILM
+ DA. The spectrogram path is a *learned spectral prior* with temporal support:

1. ``FractionalFrontEnd``: p → (B, C, T), C=9 (raw + 8 α), per-channel z-score.
2. Sliding-frame KLE → A, Φ ∈ R^{F×N×K} (paper-style framing along the window).
3. Conv2d encoder (+ phase/π + BN) → embeddings (B, D, F) → upsample to T.
4. Time-varying FiLM on **α channels only** (raw untouched); identity-init.
5. Optional residual inject from spectral emb into α channels (zero-init).
6. Unchanged ``MultiNILM`` backbone.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MultiNILM import MultiNILM, MultiNILMConfig, multinilm_config
from model.preprocess_feature.fractional import (
    FractionalFrontEnd,
    default_schirmer_alphas,
    parse_fractional_architecture,
)
from model.preprocess_feature.kle import kle_spectrogram_sliding


class KleSpectrogramEncoder(nn.Module):
    """
    Conv2d encoder for one Schirmer A/Φ map ``(B, 2, N, K)`` → ``(B, out_dim)``.

    Phase is scaled by 1/π; BatchNorm2d balances mag vs phase (paper uses BN).
    """

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
        self.input_bn = nn.BatchNorm2d(in_ch)
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
            mag: (B, N, K) or (B*F, N, K)
            phase: same shape if use_phase
        """
        if self.use_phase:
            if phase is None:
                raise ValueError("phase required when use_phase=True")
            # Mag already normalized; phase ∈ [-π, π] → [-1, 1].
            x = torch.stack([mag, phase / torch.pi], dim=1)
        else:
            x = mag.unsqueeze(1)
        x = self.input_bn(x)
        return self.net(x)


def _zero_init_conv1d(module: nn.Conv1d) -> nn.Conv1d:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class MultiNILMSchirmer(nn.Module):
    """
    Fractional 1D + sliding KLE spectral prior → time-varying FiLM → MultiNILM.
    """

    def __init__(
        self,
        *,
        backbone: MultiNILM,
        frontend: FractionalFrontEnd,
        spec_encoder: KleSpectrogramEncoder,
        film: nn.Conv1d,
        kle_n_components: int = 64,
        kle_normalize: str = "mean_std",
        kle_include_raw_column: bool = False,
        kle_phase_mode: str = "hilbert",
        kle_frame_length: int = 128,
        kle_hop: int = 64,
        film_on_raw: bool = False,
        spec_residual: nn.Conv1d | None = None,
    ) -> None:
        super().__init__()
        self.frontend = frontend
        self.backbone = backbone
        self.spec_encoder = spec_encoder
        self.film = film
        self.spec_residual = spec_residual

        self.kle_n_components = int(kle_n_components)
        self.kle_normalize = str(kle_normalize)
        self.kle_include_raw_column = bool(kle_include_raw_column)
        self.kle_phase_mode = str(kle_phase_mode)
        self.kle_frame_length = int(kle_frame_length)
        self.kle_hop = int(kle_hop)
        self.film_on_raw = bool(film_on_raw)
        self.alphas = list(frontend.alphas)

        feature_c = int(frontend.out_channels)
        if int(backbone.input_channels) != feature_c:
            raise ValueError(
                "backbone input_channels must match FractionalFrontEnd.out_channels: "
                f"{backbone.input_channels} vs {feature_c}"
            )

        n_film = feature_c if self.film_on_raw else max(feature_c - (1 if frontend.include_raw else 0), 0)
        if n_film < 1:
            raise ValueError("need at least one channel to FiLM (α or raw)")
        if int(film.out_channels) != 2 * n_film:
            raise ValueError(
                f"FiLM Conv1d must map to 2*n_film={2 * n_film}; got {film.out_channels}"
            )
        self._n_film_channels = n_film

        self.input_channels = 1
        self.feature_channels = feature_c
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
        if self.frontend.include_raw and not self.kle_include_raw_column:
            if xf.shape[1] <= 1:
                return xf
            return xf[:, 1:, :]
        return xf

    def _film_slice(self, xf: torch.Tensor) -> tuple[slice, torch.Tensor]:
        """Channel slice that receives FiLM (default: α only, leave raw)."""
        if self.film_on_raw or not self.frontend.include_raw:
            return slice(None), xf
        return slice(1, None), xf[:, 1:, :]

    def _encode_sliding_kle(self, kle_in: torch.Tensor, t_len: int) -> torch.Tensor:
        """
        Sliding KLE → (B, D, T) spectral embeddings (trainable encoder; KLE fixed).
        """
        with torch.no_grad():
            mag, phase = kle_spectrogram_sliding(
                kle_in.detach(),
                self.kle_n_components,
                frame_length=self.kle_frame_length,
                hop=self.kle_hop,
                normalize=self.kle_normalize,  # type: ignore[arg-type]
                phase_mode=self.kle_phase_mode,  # type: ignore[arg-type]
            )
        # mag/phase: (B, F, N, K)
        b, n_frames, n_comp, k = mag.shape
        mag_f = mag.reshape(b * n_frames, n_comp, k)
        phase_f = phase.reshape(b * n_frames, n_comp, k)
        emb_f = self.spec_encoder(mag_f, phase_f)  # (B*F, D)
        emb = emb_f.reshape(b, n_frames, -1).transpose(1, 2)  # (B, D, F)
        if n_frames == 1:
            return emb.expand(-1, -1, t_len)
        return F.interpolate(emb, size=t_len, mode="linear", align_corners=False)

    def forward(self, x: torch.Tensor, return_domain_features: bool = False):
        b1t = self._to_b1t(x)
        xf = self.frontend(b1t)  # (B, C, T)
        t_len = int(xf.shape[-1])

        kle_in = self._kle_channels_from_frontend(xf)
        emb_t = self._encode_sliding_kle(kle_in, t_len)  # (B, D, T)

        # Time-varying FiLM on α (or all) channels; identity at init.
        film_slice, xf_mod = self._film_slice(xf)
        gb = self.film(emb_t)  # (B, 2*n_film, T)
        n = self._n_film_channels
        gamma, beta = gb[:, :n], gb[:, n:]
        xf_mod = xf_mod * (1.0 + gamma) + beta
        xf = xf.clone()
        xf[:, film_slice] = xf_mod

        # Optional spectral residual into FiLM'd channels (zero-init → no-op at start).
        if self.spec_residual is not None:
            xf[:, film_slice] = xf[:, film_slice] + self.spec_residual(emb_t)

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
      fractional: {k, include_raw, memory, channel_normalize, ...}
      kle: {n_components, frame_length, hop, normalize, film_on_raw, ...}
    """
    frac_block = architecture.get("fractional", {})
    if not isinstance(frac_block, dict):
        frac_block = {}
    kle_block = architecture.get("kle", {})
    if not isinstance(kle_block, dict):
        kle_block = {}

    _, alphas, include_raw, memory, h = parse_fractional_architecture(
        {"fractional": {**frac_block, "enabled": True}}
    )
    if alphas is None:
        alphas = default_schirmer_alphas(int(frac_block.get("k", 8)))

    ch_norm = str(frac_block.get("channel_normalize", "mean_std"))
    frontend = FractionalFrontEnd(
        alphas=alphas,
        include_raw=bool(include_raw if frac_block else True),
        memory=memory if memory is not None else frac_block.get("memory"),
        h=float(h),
        channel_normalize=ch_norm,
    )
    feature_c = int(frontend.out_channels)

    n_comp = int(kle_block.get("n_components", 64))
    use_phase = bool(kle_block.get("use_phase", True))
    kle_norm = str(kle_block.get("normalize", "mean_std"))
    kle_raw_col = bool(kle_block.get("include_raw_column", False))
    kle_phase_mode = str(kle_block.get("phase_mode", "hilbert"))
    frame_length = int(kle_block.get("frame_length", max(n_comp, 128)))
    hop = int(kle_block.get("hop", max(1, frame_length // 2)))
    film_on_raw = bool(kle_block.get("film_on_raw", False))
    use_spec_residual = bool(kle_block.get("spec_residual", True))
    emb_dim = int(kle_block.get("embed_dim", 64))
    conv_width = int(kle_block.get("conv_width", 32))

    n_film = feature_c if film_on_raw else max(
        feature_c - (1 if frontend.include_raw else 0), 0
    )

    spec_encoder = KleSpectrogramEncoder(
        out_dim=emb_dim, use_phase=use_phase, width=conv_width
    )
    # Identity FiLM at init: γ=0, β=0 → xf unchanged.
    film = _zero_init_conv1d(nn.Conv1d(emb_dim, 2 * n_film, kernel_size=1))
    spec_residual = None
    if use_spec_residual:
        spec_residual = _zero_init_conv1d(nn.Conv1d(emb_dim, n_film, kernel_size=1))

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
        kle_phase_mode=kle_phase_mode,
        kle_frame_length=frame_length,
        kle_hop=hop,
        film_on_raw=film_on_raw,
        spec_residual=spec_residual,
    )

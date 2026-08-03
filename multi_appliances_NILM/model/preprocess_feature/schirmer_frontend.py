"""
Schirmer-style front-end: fractional calculus → per-α KLE magnitude/phase.

Produces either:
  - spectrograms A, Φ ∈ R^{N × K}  (paper 2D path), or
  - multi-channel 1D tensors (B, C, T) for the existing MultiNILM TCN
    (fractional stack only — keep structure, add feature channels).
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from .fractional import (
    default_schirmer_alphas,
    fractional_stack,
    fractional_stack_batch,
)
from .kle import kle_spectrogram_column, normalize_spectrum


def schirmer_kle_maps(
    signal: np.ndarray,
    *,
    alphas: Sequence[float] | None = None,
    n_components: int = 64,
    memory: int | None = None,
    include_raw_as_alpha0: bool = False,
    normalize: Literal["mean_std", "fundamental", "l2", "none"] = "fundamental",
    phase_mode: Literal["hilbert", "coeff_sign"] = "hilbert",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Full paper-style maps for one window.

    Args:
        signal: (T,) aggregate window.
        alphas: K fractional orders; default 8 orders in (0, 1].
        n_components: Ñ (paper: 64 in-house, 256 transfer).
        include_raw_as_alpha0: also KLE the raw p (protocol #5 style extras).

    Returns:
        magnitude (N, K_eff), phase (N, K_eff).
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    alphas_list = list(alphas) if alphas is not None else default_schirmer_alphas(8)

    # Fractional channels as separate 1D series, then KLE each → one spectrogram column.
    frac = fractional_stack(
        x, alphas_list, memory=memory, include_raw=include_raw_as_alpha0
    )  # (C, T)

    mags: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    for row in frac:
        a_col, p_col = kle_spectrogram_column(
            row,
            n_components,
            normalize=normalize,
            phase_mode=phase_mode,
        )
        mags.append(a_col)
        phases.append(p_col)

    magnitude = np.stack(mags, axis=1)  # (N, C)
    phase = np.stack(phases, axis=1)
    return magnitude, phase


def schirmer_kle_maps_batch(
    signals: np.ndarray,
    **kwargs,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Args:
        signals: (B, T)
    Returns:
        magnitude (B, N, K), phase (B, N, K)
    """
    arr = np.asarray(signals, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected (B,T), got {arr.shape}")
    mags = []
    phases = []
    for row in arr:
        a, p = schirmer_kle_maps(row, **kwargs)
        mags.append(a)
        phases.append(p)
    return np.stack(mags, axis=0), np.stack(phases, axis=0)


def fractional_channels_for_tcn(
    signal: np.ndarray,
    *,
    alphas: Sequence[float] | None = None,
    memory: int | None = None,
    include_raw: bool = True,
    channel_normalize: Literal["mean_std", "none"] = "mean_std",
) -> NDArray[np.float64]:
    """
    MultiNILM-friendly path: keep 1D TCN, raise C_in.

    Returns:
        (C, T) float64 — plug in as x with shape (B, C, T).
    """
    alphas_list = list(alphas) if alphas is not None else default_schirmer_alphas(8)
    ch = fractional_stack(
        signal, alphas_list, memory=memory, include_raw=include_raw
    )
    if channel_normalize == "mean_std":
        out = np.empty_like(ch)
        for i in range(ch.shape[0]):
            out[i] = normalize_spectrum(ch[i], mode="mean_std")
        return out
    if channel_normalize == "none":
        return ch
    raise ValueError(f"unknown channel_normalize={channel_normalize!r}")


def fractional_channels_for_tcn_batch(
    signals: np.ndarray,
    *,
    alphas: Sequence[float] | None = None,
    memory: int | None = None,
    include_raw: bool = True,
    channel_normalize: Literal["mean_std", "none"] = "mean_std",
) -> NDArray[np.float64]:
    """Returns (B, C, T)."""
    alphas_list = list(alphas) if alphas is not None else default_schirmer_alphas(8)
    stacked = fractional_stack_batch(
        signals, alphas_list, memory=memory, include_raw=include_raw
    )
    if channel_normalize == "none":
        return stacked
    out = np.empty_like(stacked)
    for b in range(stacked.shape[0]):
        for c in range(stacked.shape[1]):
            out[b, c] = normalize_spectrum(stacked[b, c], mode="mean_std")
    return out

"""
Normalized Karhunen–Loève Expansion (KLE) features for low-frequency NILM.

Schirmer & Mporas (IEEE OAJPE 2022) / Dinesh et al. (TSG 2016):
estimate an autocorrelation matrix (ACM), eigen-decompose it, project a
frame onto the subspace, and read each subspace component (SC) as a
near-sinusoid → magnitude A and phase Φ.

Physical goal (scale invariance): after normalization, different brands of
the same appliance share similar spectral *shape* even if steady-state
watts differ (e.g. 75 W vs 100 W fridge).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import toeplitz
from scipy.signal import hilbert


def _as_1d(frame: np.ndarray) -> NDArray[np.float64]:
    x = np.asarray(frame, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("frame must be non-empty")
    return x


def autocorrelation(
    frame: np.ndarray,
    max_lag: int,
    *,
    demean: bool = True,
    biased: bool = True,
) -> NDArray[np.float64]:
    """
    R[n] for n = 0..max_lag-1 (paper R_PP in Eq. 6).

    Estimated from the full frame (length L >= max_lag) for stabler ACM.
    """
    x = _as_1d(frame)
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")
    if x.size < 2:
        # Degenerate frame: impulse ACF.
        r = np.zeros(max_lag, dtype=np.float64)
        r[0] = float(x[0] ** 2) if x.size else 0.0
        return r

    if demean:
        x = x - x.mean()

    # Full correlation; take non-negative lags.
    corr = np.correlate(x, x, mode="full")
    mid = x.size - 1
    n_take = min(max_lag, x.size)
    r = np.zeros(max_lag, dtype=np.float64)
    seg = corr[mid : mid + n_take]
    if biased:
        # Biased ACF: divide by L (common for Toeplitz PSD / ACM estimates).
        r[:n_take] = seg / float(x.size)
    else:
        # Unbiased: divide by (L - lag).
        denom = np.arange(x.size, x.size - n_take, -1, dtype=np.float64)
        r[:n_take] = seg / denom
    return r


def autocorrelation_matrix(
    frame: np.ndarray,
    order: int,
    *,
    demean: bool = True,
) -> NDArray[np.float64]:
    """
    Toeplitz ACM Θ_PP of size (order, order), paper Eq. (6).
    """
    r = autocorrelation(frame, order, demean=demean, biased=True)
    # scipy.linalg.toeplitz is fine; pure numpy keeps the dependency light.
    return _toeplitz_from_acf(r)


def _toeplitz_from_acf(r: np.ndarray) -> NDArray[np.float64]:
    return np.asarray(toeplitz(r), dtype=np.float64)


def kle_eigensystem(
    frame: np.ndarray,
    n_components: int,
    *,
    demean: bool = True,
    min_eig: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Build ACM and eigen-decompose.

    Returns:
        eigenvalues (ascending, length N),
        Q eigenvectors as columns (N, N),
        ACM (N, N).
    """
    n = int(n_components)
    if n < 1:
        raise ValueError(f"n_components must be >= 1, got {n}")
    x = _as_1d(frame)
    if x.size < n:
        # Pad short frames so transfer-style large N still runs.
        pad = np.zeros(n, dtype=np.float64)
        pad[: x.size] = x
        x = pad

    acm = autocorrelation_matrix(x, n, demean=demean)
    # Symmetrize against tiny numeric asymmetry.
    acm = 0.5 * (acm + acm.T)
    evals, evecs = np.linalg.eigh(acm)
    evals = np.maximum(evals, min_eig)
    return evals, evecs, acm


def _projection_segment(frame: np.ndarray, n: int) -> NDArray[np.float64]:
    """
    Length-N vector for Q^T P (Eq. 7).

    Use the *last* N samples so the projection aligns with the end of the
    window (matches common seq2point / end-aligned NILM framing).
    """
    x = _as_1d(frame)
    if x.size >= n:
        return x[-n:].copy()
    out = np.zeros(n, dtype=np.float64)
    out[-x.size :] = x
    return out


def kle_coefficients(
    frame: np.ndarray,
    n_components: int,
    *,
    demean: bool = True,
    q: NDArray[np.float64] | None = None,
    eigenvalues: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    KLE transform: P̃ = Q^T P  (paper Eq. 7).

    Returns:
        coeffs (N,), Q (N, N) with columns = eigenvectors, eigenvalues (N,).
    """
    if q is None:
        evals, evecs, _ = kle_eigensystem(frame, n_components, demean=demean)
    else:
        evecs = np.asarray(q, dtype=np.float64)
        if evecs.shape != (n_components, n_components):
            raise ValueError(
                f"q must be ({n_components}, {n_components}), got {evecs.shape}"
            )
        if eigenvalues is None:
            evals, _, _ = kle_eigensystem(frame, n_components, demean=demean)
        else:
            evals = np.asarray(eigenvalues, dtype=np.float64)

    segment = _projection_segment(frame, n_components)
    if demean:
        segment = segment - segment.mean()
    coeffs = evecs.T @ segment
    return coeffs, evecs, evals


def kle_magnitude_phase(
    frame: np.ndarray,
    n_components: int,
    *,
    demean: bool = True,
    phase_mode: Literal["hilbert", "coeff_sign"] = "hilbert",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Magnitude A and phase Φ for each subspace component (paper Sec. III-B).

    Each SC is reconstructed as p_i = c_i * q_i (length N). Under the
    sinusoidal assumption we take the Hilbert analytic signal:
      A_i  = mean |analytic|
      Φ_i  = angle(analytic) at the segment center

    phase_mode='coeff_sign' is a cheap ablation: A=|c|, Φ=0 or π from sign(c).
    """
    coeffs, evecs, _ = kle_coefficients(frame, n_components, demean=demean)
    n = int(n_components)
    mag = np.empty(n, dtype=np.float64)
    phase = np.empty(n, dtype=np.float64)

    if phase_mode == "coeff_sign":
        mag[:] = np.abs(coeffs)
        phase[:] = np.where(coeffs >= 0.0, 0.0, np.pi)
        return mag, phase

    if phase_mode != "hilbert":
        raise ValueError(f"unknown phase_mode={phase_mode!r}")

    center = n // 2
    for i in range(n):
        sc = coeffs[i] * evecs[:, i]
        analytic = hilbert(sc)
        mag[i] = float(np.mean(np.abs(analytic)))
        phase[i] = float(np.angle(analytic[center]))
    return mag, phase


def normalize_spectrum(
    values: np.ndarray,
    *,
    mode: Literal["mean_std", "fundamental", "l2", "none"] = "mean_std",
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """
    Scale-invariance helpers (paper Sec. II-A / VI-B).

    - mean_std: |x - mean| / std   (paper Eq. 14 style)
    - fundamental: divide by first bin (proxy for fundamental / largest SC
      energy ordering depends on eigh ascending — we use max(|x|) instead
      when mode='fundamental' for a brand-scale invariant shape)
    - l2: unit ℓ2 norm
    - none: copy
    """
    x = np.asarray(values, dtype=np.float64).copy()
    if mode == "none":
        return x
    if mode == "mean_std":
        mu = float(x.mean())
        sigma = float(x.std())
        return np.abs(x - mu) / (sigma + eps)
    if mode == "fundamental":
        # Use max magnitude as scale proxy (brand wattage ≈ global scale).
        scale = float(np.max(np.abs(x)))
        return x / (scale + eps)
    if mode == "l2":
        return x / (float(np.linalg.norm(x)) + eps)
    raise ValueError(f"unknown normalize mode={mode!r}")


def kle_spectrogram_column(
    frame: np.ndarray,
    n_components: int,
    *,
    demean: bool = True,
    normalize: Literal["mean_std", "fundamental", "l2", "none"] = "fundamental",
    phase_mode: Literal["hilbert", "coeff_sign"] = "hilbert",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    One column of the Schirmer spectrogram: A, Φ ∈ R^{N} for one α-signal.
    """
    mag, phase = kle_magnitude_phase(
        frame, n_components, demean=demean, phase_mode=phase_mode
    )
    mag = normalize_spectrum(mag, mode=normalize)
    # Phase is already in [-π, π]; optional mean_std usually hurts angles.
    return mag, phase


def kle_subspace_channels(
    frame: np.ndarray,
    n_components: int = 8,
    *,
    demean: bool = True,
    include_raw: bool = True,
    channel_normalize: Literal["mean_std", "none"] = "mean_std",
) -> NDArray[np.float64]:
    """
    Expand one aggregate window into multi-channel 1D features via KLE FIRs.

    Dinesh / Schirmer: each eigenvector q_i is an FIR; filtering the full
    frame yields a subspace component (SC) time series. Stacking SCs gives
    a (C, T) tensor for MultiNILM Conv1d — default C = 9 with
    ``include_raw=True`` and ``n_components=8`` (raw + 8 SCs).

    Returns:
        (C, T) float64.
    """
    x = _as_1d(frame)
    t_len = int(x.size)
    n = int(n_components)
    if n < 1:
        raise ValueError(f"n_components must be >= 1, got {n}")

    x_acm = x - x.mean() if demean else x
    _, evecs, _ = kle_eigensystem(x_acm, n, demean=False)

    # Eigenvector columns as causal-ish FIR kernels (same-length conv).
    sc_rows: list[np.ndarray] = []
    for i in range(n):
        kernel = evecs[:, i]
        # mode='same' keeps length T for seq2seq alignment.
        sc_rows.append(np.convolve(x_acm, kernel, mode="same"))

    channels = sc_rows
    if include_raw:
        channels = [x.copy()] + channels

    out = np.stack(channels, axis=0)  # (C, T)
    if channel_normalize == "mean_std":
        for i in range(out.shape[0]):
            out[i] = normalize_spectrum(out[i], mode="mean_std")
    elif channel_normalize != "none":
        raise ValueError(f"unknown channel_normalize={channel_normalize!r}")

    if out.shape[-1] != t_len:
        raise RuntimeError(
            f"KLE channel length {out.shape[-1]} != frame length {t_len}"
        )
    return out


def kle_subspace_channels_batch(
    signals: np.ndarray,
    n_components: int = 8,
    *,
    demean: bool = True,
    include_raw: bool = True,
    channel_normalize: Literal["mean_std", "none"] = "mean_std",
) -> NDArray[np.float64]:
    """
    Args:
        signals: (B, T)
    Returns:
        (B, C, T) with C = n_components (+1 if include_raw).
    """
    arr = np.asarray(signals, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected (B,T), got shape {arr.shape}")
    return np.stack(
        [
            kle_subspace_channels(
                row,
                n_components,
                demean=demean,
                include_raw=include_raw,
                channel_normalize=channel_normalize,
            )
            for row in arr
        ],
        axis=0,
    )

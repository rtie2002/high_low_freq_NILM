"""
Grünwald–Letnikov fractional derivatives for transferable NILM.

Schirmer & Mporas (IEEE OAJPE 2022): multi-order α features for time-shift
robustness. Discrete form (h = 1):

    (D^α p)[t] ≈ Σ_j (-1)^j * C(α, j) * p[t - j]     (paper Eq. 4–5)

Components
----------
1. Core weights / defaults
   - ``gl_binomial_weights``
   - ``default_schirmer_alphas``

2. NumPy API (offline, scripts, KLE / schirmer_frontend)
   - ``fractional_derivative`` / ``fractional_derivative_batch``
   - ``fractional_stack`` / ``fractional_stack_batch``

3. PyTorch API (MultiNILM training front-end)
   - ``FractionalFrontEnd``  — (B,1,T) → (B,C,T)
   - ``parse_fractional_architecture``  — yaml ``architecture.fractional``
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ArrayLike = np.ndarray | Sequence[float]


# ---------------------------------------------------------------------------
# 1. Core weights / defaults
# ---------------------------------------------------------------------------

def gl_binomial_weights(alpha: float, memory: int) -> np.ndarray:
    """
    Weights w[j] = (-1)^j * binom(α, j) for j = 0..memory.

    binom(α, j) = Γ(α+1) / (Γ(j+1) Γ(α-j+1))
    Recurrence: w_0 = 1;  w_j = w_{j-1} * (j - 1 - α) / j
    """
    if memory < 0:
        raise ValueError(f"memory must be >= 0, got {memory}")
    alpha = float(alpha)
    w = np.empty(memory + 1, dtype=np.float64)
    w[0] = 1.0
    for j in range(1, memory + 1):
        w[j] = w[j - 1] * (j - 1 - alpha) / j
    return w


def default_schirmer_alphas(k: int = 8) -> list[float]:
    """
    Evenly spaced orders in (0, 1].

    Paper only reports K=8, not the exact α list — this is our default grid.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k == 1:
        return [1.0]
    return [round((i + 1) / k, 6) for i in range(k)]


# ---------------------------------------------------------------------------
# 2. NumPy API
# ---------------------------------------------------------------------------

def fractional_derivative(
    signal: ArrayLike,
    alpha: float,
    *,
    memory: int | None = None,
    h: float = 1.0,
) -> np.ndarray:
    """
    One GL fractional derivative on a 1D series.

    Args:
        signal: (T,)
        alpha: fractional order (e.g. 0.5, 1.0). α=1 ≈ first difference.
        memory: truncation J; default T-1.
        h: sample step (usually 1).

    Returns:
        (T,) float64, causal.
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    t_len = int(x.shape[0])
    if t_len == 0:
        return x.copy()

    j_max = t_len - 1 if memory is None else int(memory)
    if j_max < 0:
        raise ValueError(f"memory must be >= 0, got {j_max}")
    j_max = min(j_max, t_len - 1)

    w = gl_binomial_weights(alpha, j_max)
    scale = 1.0 / (float(h) ** float(alpha))
    return np.convolve(x, w, mode="full")[:t_len] * scale


def fractional_stack(
    signal: ArrayLike,
    alphas: Sequence[float],
    *,
    memory: int | None = None,
    h: float = 1.0,
    include_raw: bool = False,
) -> np.ndarray:
    """
    Stack K fractional signals as channels.

    Returns:
        (C, T) with C = K or K+1 if ``include_raw``.
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    channels: list[np.ndarray] = []
    if include_raw:
        channels.append(x.copy())
    for alpha in alphas:
        channels.append(
            fractional_derivative(x, float(alpha), memory=memory, h=h)
        )
    return np.stack(channels, axis=0)


def fractional_derivative_batch(
    signals: np.ndarray,
    alpha: float,
    *,
    memory: int | None = None,
    h: float = 1.0,
) -> np.ndarray:
    """``(B, T)`` or ``(T,)`` → same shape."""
    arr = np.asarray(signals, dtype=np.float64)
    if arr.ndim == 1:
        return fractional_derivative(arr, alpha, memory=memory, h=h)
    if arr.ndim != 2:
        raise ValueError(f"expected (T,) or (B,T), got shape {arr.shape}")
    return np.stack(
        [fractional_derivative(row, alpha, memory=memory, h=h) for row in arr],
        axis=0,
    )


def fractional_stack_batch(
    signals: np.ndarray,
    alphas: Sequence[float],
    *,
    memory: int | None = None,
    h: float = 1.0,
    include_raw: bool = False,
) -> np.ndarray:
    """``(B, T)`` → ``(B, C, T)``."""
    arr = np.asarray(signals, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected (B,T), got shape {arr.shape}")
    return np.stack(
        [
            fractional_stack(
                row, alphas, memory=memory, h=h, include_raw=include_raw
            )
            for row in arr
        ],
        axis=0,
    )


# ---------------------------------------------------------------------------
# 3. PyTorch API (MultiNILM front-end)
# ---------------------------------------------------------------------------

class FractionalFrontEnd(nn.Module):
    """
    Fixed multi-order GL filters for training.

    ``(B, 1, T)`` → ``(B, C, T)`` with
    C = (1 if include_raw else 0) + len(alphas).
    Default alphas = ``default_schirmer_alphas(8)`` → C=9 with raw.
    """

    def __init__(
        self,
        alphas: Sequence[float] | None = None,
        *,
        include_raw: bool = True,
        memory: int | None = None,
        h: float = 1.0,
        max_memory: int = 256,
        channel_normalize: str = "mean_std",
        channel_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        alphas_list = (
            list(alphas) if alphas is not None else default_schirmer_alphas(8)
        )
        if not alphas_list and not include_raw:
            raise ValueError("need at least one alpha or include_raw=True")

        self.alphas = [float(a) for a in alphas_list]
        self.include_raw = bool(include_raw)
        self.h = float(h)
        self.memory = int(memory) if memory is not None else int(max_memory)
        if self.memory < 1:
            raise ValueError(f"memory must be >= 1, got {self.memory}")
        self.channel_normalize = str(channel_normalize)
        self.channel_norm_eps = float(channel_norm_eps)
        if self.channel_normalize not in {"mean_std", "none"}:
            raise ValueError(
                f"channel_normalize must be mean_std|none, got {self.channel_normalize!r}"
            )

        self.out_channels = (1 if self.include_raw else 0) + len(self.alphas)

        if self.alphas:
            kernels = []
            for alpha in self.alphas:
                w = gl_binomial_weights(alpha, self.memory)
                scale = 1.0 / (self.h ** alpha)
                # PyTorch conv1d is cross-correlation; flip so it matches
                # np.convolve / causal GL: y[t] = Σ_j w[j] x[t-j].
                w_conv = np.asarray(w * scale, dtype=np.float64)[::-1].copy()
                kernels.append(torch.tensor(w_conv, dtype=torch.float32))
            weight = torch.stack(kernels, dim=0).unsqueeze(1)  # (K, 1, L)
            # Named child so ``print(model)`` shows the GL bank (weights frozen).
            k_len = int(weight.shape[-1])
            n_a = len(self.alphas)
            self.gl_conv = nn.Conv1d(
                in_channels=n_a,
                out_channels=n_a,
                kernel_size=k_len,
                groups=n_a,
                bias=False,
                padding=0,
            )
            with torch.no_grad():
                self.gl_conv.weight.copy_(weight)
            self.gl_conv.weight.requires_grad_(False)
            # Alias buffer for dumps / older code that reads ``gl_weight``.
            self.register_buffer("gl_weight", self.gl_conv.weight, persistent=False)
        else:
            self.gl_conv = None
            self.register_buffer("gl_weight", torch.zeros(0), persistent=True)

    def extra_repr(self) -> str:
        w_shape = tuple(self.gl_weight.shape) if self.gl_weight.numel() else None
        return (
            f"alphas={self.alphas}, include_raw={self.include_raw}, "
            f"memory={self.memory}, out_channels={self.out_channels}, "
            f"channel_normalize={self.channel_normalize!r}, gl_weight={w_shape}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"FractionalFrontEnd expected (B,C,T), got {tuple(x.shape)}"
            )
        if x.shape[1] != 1:
            x = x[:, :1, :]

        parts: list[torch.Tensor] = []
        if self.include_raw:
            parts.append(x)

        if self.alphas:
            assert self.gl_conv is not None
            pad = int(self.gl_conv.weight.shape[-1]) - 1
            x_pad = F.pad(x, (pad, 0))
            x_rep = x_pad.expand(-1, len(self.alphas), -1)
            frac = self.gl_conv(x_rep)
            parts.append(frac)

        out = torch.cat(parts, dim=1)
        if self.channel_normalize == "mean_std":
            # Per-channel, per-window: balances raw vs high-α energy before the stem.
            mu = out.mean(dim=-1, keepdim=True)
            sigma = out.std(dim=-1, keepdim=True).clamp_min(self.channel_norm_eps)
            out = (out - mu) / sigma
        return out


def parse_fractional_architecture(
    architecture: dict[str, Any],
) -> tuple[bool, list[float] | None, bool, int | None, float]:
    """
    Read optional ``architecture.fractional`` from model yaml.

    Returns:
        enabled, alphas, include_raw, memory, h
    """
    block = architecture.get("fractional")
    if not isinstance(block, dict):
        return False, None, True, None, 1.0

    enabled = bool(block.get("enabled", False))
    include_raw = bool(block.get("include_raw", True))
    memory = block.get("memory", None)
    memory_i = None if memory is None else int(memory)
    h = float(block.get("h", 1.0))

    alphas_raw = block.get("alphas", None)
    if alphas_raw is None:
        alphas = default_schirmer_alphas(int(block.get("k", 8)))
    else:
        alphas = [float(a) for a in alphas_raw]

    return enabled, alphas, include_raw, memory_i, h

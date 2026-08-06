"""DILATE-style regression loss (NeurIPS 2019 Le Guen & Thome).

Regression-only shape + temporal distortion (not for BCE/state)::

    L_DILATE = α L_shape + (1 − α) L_temporal

- L_shape  = Soft-DTW_γ (Cuturi & Blondel) on pairwise cost Δ
- L_temporal = ⟨A*_γ, Ω⟩ with A*_γ = ∇_Δ Soft-DTW_γ  (smooth DTW path)
- Ω(h,j) = ((h − j) / k)²   (paper-style diagonal deviation)

Complexity O(k²) per series; callers should downsample long NILM windows.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _pairwise_sq_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean costs between two (B, k) series → (B, k, k).

    Δ[b,h,j] = (a[b,h] − b_series[b,j])²
    Here ``a`` = ground truth, ``b`` = prediction (paper Δ(ŷ, y*)).
    """
    # (B,k,1) - (B,1,k)
    return (a.unsqueeze(-1) - b.unsqueeze(-2)).pow(2)


def soft_dtw(cost: torch.Tensor, gamma: float) -> torch.Tensor:
    """Batched Soft-DTW. ``cost`` (B, n, m) → scalar DTW_γ per batch (B,)."""
    if cost.dim() != 3:
        raise ValueError(f"soft_dtw expects (B,n,m), got {tuple(cost.shape)}")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    batch, n, m = cost.shape
    # Pad with +inf so softmin ignores out-of-grid cells.
    r = cost.new_full((batch, n + 2, m + 2), float("inf"))
    r[:, 0, 0] = 0.0

    # 1-based DP over the cost grid.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r0 = r[:, i - 1, j - 1]
            r1 = r[:, i - 1, j]
            r2 = r[:, i, j - 1]
            # softmin_γ(r0,r1,r2) = −γ log Σ exp(−r/γ)
            stacked = torch.stack((r0, r1, r2), dim=0)
            softmin = -gamma * torch.logsumexp(-stacked / gamma, dim=0)
            r[:, i, j] = cost[:, i - 1, j - 1] + softmin

    return r[:, n, m]


def temporal_omega(k: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Ω(h,j) = ((h−j)/k)²  → (k, k)."""
    idx = torch.arange(k, device=device, dtype=dtype)
    diff = (idx.unsqueeze(1) - idx.unsqueeze(0)) / float(k)
    return diff.pow(2)


def dilate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.5,
    gamma: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DILATE on univariate batch trajectories.

    Args:
        pred, target: (B, k) predicted / ground-truth power (same length).
        alpha: weight on shape (1−alpha on temporal). Paper mixes in [0,1].
        gamma: Soft-DTW temperature (smaller → closer to hard DTW).

    Returns:
        (loss, loss_shape, loss_temporal) — all scalars with grad through ``pred``.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    if pred.dim() != 2:
        raise ValueError(f"dilate_loss expects (B,k), got {tuple(pred.shape)}")

    alpha = float(alpha)
    gamma = float(gamma)
    batch, k = pred.shape
    if k < 2:
        z = pred.sum() * 0.0
        return z, z, z

    # Paper: Δ(ŷ, y*) with rows ↔ target times, cols ↔ pred times.
    cost = _pairwise_sq_1d(target, pred)  # (B,k,k)

    # Soft-DTW path cost grows ~O(k); divide by k so α≈0.5 is usable vs temporal.
    dtw_b = soft_dtw(cost, gamma)
    loss_shape = dtw_b.mean() / float(k)

    if alpha >= 1.0 - 1e-8:
        # Shape-only: skip path / Hessian path.
        z = loss_shape * 0.0
        return loss_shape, loss_shape, z

    # Smooth path A*_γ = ∇_Δ Soft-DTW (needs create_graph for temporal grads).
    path = torch.autograd.grad(
        dtw_b.sum(),
        cost,
        create_graph=True,
        retain_graph=True,
    )[0]
    omega = temporal_omega(k, device=pred.device, dtype=pred.dtype)
    # Official repo: sum(path * Ω) / k²
    loss_temporal = (path * omega.unsqueeze(0)).sum() / float(k * k)

    loss = alpha * loss_shape + (1.0 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal


def dilate_power_loss(
    power_pred: torch.Tensor,
    power_true: torch.Tensor,
    *,
    alpha: float = 0.5,
    gamma: float = 0.01,
    downsample: int = 8,
    appliance_indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-appliance NILM wrapper: (B,T,A) → sum of per-appliance DILATE.

    Downsamples time by ``downsample`` (avg-pool) so Soft-DTW stays O((T/s)²).

    Returns:
        (loss_sum_over_apps, per_app_vector length A) — unused apps are 0.
    """
    if power_pred.shape != power_true.shape:
        raise ValueError(
            f"power shape mismatch {tuple(power_pred.shape)} vs {tuple(power_true.shape)}"
        )
    if power_pred.dim() != 3:
        raise ValueError(f"expected (B,T,A), got {tuple(power_pred.shape)}")

    batch, time, n_app = power_pred.shape
    stride = max(1, int(downsample))
    apps = list(range(n_app)) if appliance_indices is None else list(appliance_indices)

    per_app = power_pred.new_zeros(n_app)
    for i in apps:
        if i < 0 or i >= n_app:
            continue
        yp = power_pred[:, :, i]
        yt = power_true[:, :, i]
        if stride > 1:
            # Avg-pool keeps energy scale roughly comparable to MSE space.
            t_len = time - (time % stride)
            if t_len < stride * 2:
                yp_d = F.adaptive_avg_pool1d(yp.unsqueeze(1), 64).squeeze(1)
                yt_d = F.adaptive_avg_pool1d(yt.unsqueeze(1), 64).squeeze(1)
            else:
                yp_d = yp[:, :t_len].reshape(batch, t_len // stride, stride).mean(dim=-1)
                yt_d = yt[:, :t_len].reshape(batch, t_len // stride, stride).mean(dim=-1)
        else:
            yp_d, yt_d = yp, yt

        loss_i, _, _ = dilate_loss(yp_d, yt_d, alpha=alpha, gamma=gamma)
        per_app[i] = loss_i

    return per_app.sum(), per_app

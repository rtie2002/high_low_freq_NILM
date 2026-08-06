"""DILATE-style regression loss (NeurIPS 2019 Le Guen & Thome).

Regression-only shape + temporal distortion (not for BCE/state)::

    L_DILATE = α L_shape + (1 − α) L_temporal

Soft-DTW uses an anti-diagonal GPU sweep; multi-appliance series are stacked
into one batch. Prefer ``dilate_alpha: 1.0`` (shape-only) for training speed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _pairwise_sq_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Δ[b,h,j] = (a[b,h] − b[b,j])² → (B, k, k)."""
    return (a.unsqueeze(-1) - b.unsqueeze(-2)).pow(2)


def soft_dtw(cost: torch.Tensor, gamma: float) -> torch.Tensor:
    """Batched Soft-DTW. ``cost`` (B, n, m) → (B,). Anti-diagonal vectorized."""
    if cost.dim() != 3:
        raise ValueError(f"soft_dtw expects (B,n,m), got {tuple(cost.shape)}")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    _batch, n, m = cost.shape
    r = cost.new_full((cost.shape[0], n + 1, m + 1), 1.0e9)
    r[:, 0, 0] = 0.0
    inv_gamma = 1.0 / float(gamma)
    gamma_f = float(gamma)

    for s in range(2, n + m + 1):
        i0 = max(1, s - m)
        i1 = min(n, s - 1)
        if i0 > i1:
            continue
        i_idx = torch.arange(i0, i1 + 1, device=cost.device)
        j_idx = s - i_idx
        c = cost[:, i_idx - 1, j_idx - 1]
        r0 = r[:, i_idx - 1, j_idx - 1]
        r1 = r[:, i_idx - 1, j_idx]
        r2 = r[:, i_idx, j_idx - 1]
        stacked = torch.stack((r0, r1, r2), dim=0)
        softmin = -gamma_f * torch.logsumexp(-stacked * inv_gamma, dim=0)
        vals = c + softmin
        for t in range(i_idx.numel()):
            r[:, int(i_idx[t]), int(j_idx[t])] = vals[:, t]

    return r[:, n, m]


def temporal_omega(k: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(k, device=device, dtype=dtype)
    return ((idx.unsqueeze(1) - idx.unsqueeze(0)) / float(k)).pow(2)


def dilate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.5,
    gamma: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DILATE on (B, k) trajectories → (loss, shape, temporal)."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    if pred.dim() != 2:
        raise ValueError(f"dilate_loss expects (B,k), got {tuple(pred.shape)}")

    alpha = float(alpha)
    gamma = float(gamma)
    _batch, k = pred.shape
    if k < 2:
        z = pred.sum() * 0.0
        return z, z, z

    cost = _pairwise_sq_1d(target, pred)
    dtw_b = soft_dtw(cost, gamma)
    loss_shape = dtw_b.mean() / float(k)

    if alpha >= 1.0 - 1e-8:
        z = loss_shape * 0.0
        return loss_shape, loss_shape, z

    path = torch.autograd.grad(
        dtw_b.sum(), cost, create_graph=True, retain_graph=True
    )[0]
    omega = temporal_omega(k, device=pred.device, dtype=pred.dtype)
    loss_temporal = (path * omega.unsqueeze(0)).sum() / float(k * k)
    loss = alpha * loss_shape + (1.0 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal


def _downsample_bt(x: torch.Tensor, stride: int) -> torch.Tensor:
    """(B, T) → (B, T')."""
    batch, time = x.shape
    if stride <= 1:
        return x
    t_len = time - (time % stride)
    if t_len < stride * 2:
        return F.adaptive_avg_pool1d(x.unsqueeze(1), 32).squeeze(1)
    return x[:, :t_len].reshape(batch, t_len // stride, stride).mean(dim=-1)


def dilate_power_loss(
    power_pred: torch.Tensor,
    power_true: torch.Tensor,
    *,
    alpha: float = 0.5,
    gamma: float = 0.01,
    downsample: int = 8,
    appliance_indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,T,A) → (sum_i DILATE_i, per-app vector). One Soft-DTW over B·A_sel."""
    if power_pred.shape != power_true.shape:
        raise ValueError(
            f"power shape mismatch {tuple(power_pred.shape)} vs {tuple(power_true.shape)}"
        )
    if power_pred.dim() != 3:
        raise ValueError(f"expected (B,T,A), got {tuple(power_pred.shape)}")

    batch, _time, n_app = power_pred.shape
    stride = max(1, int(downsample))
    alpha = float(alpha)
    gamma = float(gamma)
    apps = list(range(n_app)) if appliance_indices is None else [
        int(i) for i in appliance_indices if 0 <= int(i) < n_app
    ]
    if not apps:
        z = power_pred.new_zeros(())
        return z, power_pred.new_zeros(n_app)

    yp = power_pred[:, :, apps].permute(0, 2, 1).reshape(batch * len(apps), -1)
    yt = power_true[:, :, apps].permute(0, 2, 1).reshape(batch * len(apps), -1)
    yp_d = _downsample_bt(yp, stride)
    yt_d = _downsample_bt(yt, stride)
    k = yp_d.shape[-1]

    cost = _pairwise_sq_1d(yt_d, yp_d)
    dtw_vec = soft_dtw(cost, gamma) / float(k)  # (B*A_sel,)
    dtw_ba = dtw_vec.view(batch, len(apps)).mean(dim=0)  # (A_sel,)

    per_app = power_pred.new_zeros(n_app)
    for j, app_i in enumerate(apps):
        per_app[app_i] = dtw_ba[j]

    if alpha >= 1.0 - 1e-8:
        return dtw_ba.sum(), per_app

    # Temporal on the stacked batch (one path grad).
    path = torch.autograd.grad(
        dtw_vec.sum() * float(k),  # undo /k for path scale ≈ official Soft-DTW
        cost,
        create_graph=True,
        retain_graph=True,
    )[0]
    omega = temporal_omega(k, device=power_pred.device, dtype=power_pred.dtype)
    loss_temporal = (path * omega.unsqueeze(0)).sum() / float(k * k)
    loss_shape = dtw_ba.sum()
    loss = alpha * loss_shape + (1.0 - alpha) * loss_temporal
    # Log mix: put full loss on apps proportional to shape share.
    w = dtw_ba.detach().clamp_min(1e-8)
    w = w / w.sum()
    per_app = power_pred.new_zeros(n_app)
    for j, app_i in enumerate(apps):
        per_app[app_i] = loss * w[j]
    return loss, per_app

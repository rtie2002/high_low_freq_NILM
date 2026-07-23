"""MultiNILM multitask loss + optional Lin-style domain adaptation.

Pipeline total loss (additive DA)::

    L = L_NILM + lambda_domain * L_domain

Supervised part (see docs/multinilm_task_loss_balance.md)::

    L_power = sum_i MSE_i
    L_state = sum_i BCE_i
    L_shape = sum_i MSE(Δŷ_i, Δy_i)     # first-difference = waveform slope/shape
    L_NILM  = L_power + state_term + shape_term

    task_balance=equal → each *_term matched to L_power scale; λ=1 means equal weight.
    lambda_shape=0 disables shape term.

Domain part (Lin et al., IEEE TSG 2022)::

    L_domain = sum_layer [ mu * MMD^2 + (1-mu) * CORAL ]   # method=both
    Paper convex mix L=(1-λ)L_R + λ L_domain is NOT used here.

Call site: adapters/multinilm.py → MultiNILMLoss(...)
Shapes: power/state (B,T,A); domain hooks dict[str, (B,C,T)|(B,D)]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MultiNILM import pool_domain_feature_map


# ---------------------------------------------------------------------------
# Domain discrepancy (Lin et al.)
# ---------------------------------------------------------------------------


def _pairwise_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """||a_i - b_j||^2 for all pairs → (N_a, N_b)."""
    return (
        a.pow(2).sum(dim=1, keepdim=True)
        + b.pow(2).sum(dim=1).unsqueeze(0)
        - 2.0 * (a @ b.T)
    ).clamp_min(0.0)


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Deep CORAL — align second-order (covariance) statistics.

    Theory (Lin Eqs. 7–9)
    ---------------------
    Given Z_S ∈ R^{n_s×D}, Z_T ∈ R^{n_t×D} (pooled batch features):

        C_S = 1/(n_s-1) [ Z_S^T Z_S − (Z_S^T 1)(1^T Z_S)/n_s ]
        C_T = 1/(n_t-1) [ Z_T^T Z_T − (Z_T^T 1)(1^T Z_T)/n_t ]

        L_CORAL = (1 / (4 D^2)) ||C_S − C_T||_F^2

    Intuition: match feature-cloud *shape* (correlations), not only the mean.
    Requires n_s, n_t ≥ 2 (else return 0).
    """
    source, target = source.float(), target.float()
    if source.dim() != 2 or target.dim() != 2:
        raise ValueError(
            f"CORAL expects (B, D), got {tuple(source.shape)} and {tuple(target.shape)}"
        )
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"CORAL D mismatch: {source.shape[1]} vs {target.shape[1]}"
        )

    n_s, n_t, d = source.shape[0], target.shape[0], source.shape[1]
    if n_s < 2 or n_t < 2:
        return source.new_zeros(())

    # Unbiased sample covariance (Lin Eqs. 8–9).
    ones_s, ones_t = source.new_ones(n_s, 1), target.new_ones(n_t, 1)
    cov_s = (source.T @ source - (source.T @ ones_s) @ (ones_s.T @ source) / n_s) / (n_s - 1)
    cov_t = (target.T @ target - (target.T @ ones_t) @ (ones_t.T @ target) / n_t) / (n_t - 1)
    # Eq. 7: Frobenius distance, scaled by 1/(4 D^2).
    return (cov_s - cov_t).pow(2).sum() / (4.0 * float(d) * float(d))


def mmd_rbf_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    sigma: float | None = None,
) -> torch.Tensor:
    """Squared RBF-MMD — align distributions in RKHS (Lin Eq. 6 style).

    Theory
    ------
    Kernel:  k(u,v) = exp( −||u−v||^2 / (2 σ^2) )

    Empirical MMD²:

        MMD² = E[k(z_S,z_S')] + E[k(z_T,z_T')] − 2 E[k(z_S,z_T)]

    If σ is None: σ = √(median pairwise ||·||² on [Z_S; Z_T])  (median heuristic).

    Intuition: pull source/target feature *clouds* together (stronger than mean-only).
    """
    source, target = source.float(), target.float()
    if source.dim() != 2 or target.dim() != 2:
        raise ValueError(
            f"MMD expects (B, D), got {tuple(source.shape)} and {tuple(target.shape)}"
        )

    # Bandwidth σ: fixed or median heuristic (no grad through σ).
    with torch.no_grad():
        if sigma is None:
            combo = torch.cat([source.detach(), target.detach()], dim=0)
            dist = _pairwise_sq(combo, combo)
            n = dist.shape[0]
            if n > 1:
                med = dist[~torch.eye(n, dtype=torch.bool, device=dist.device)].median()
            else:
                med = dist.new_tensor(1.0)
            sigma_val = torch.sqrt(med.clamp_min(1e-6))
        else:
            sigma_val = source.new_tensor(float(sigma))

    # γ = 1/(2σ²)  →  k = exp(−γ ||·||²)
    gamma = 1.0 / (2.0 * sigma_val.pow(2).clamp_min(1e-12))
    k_ss = torch.exp(-gamma * _pairwise_sq(source, source))
    k_tt = torch.exp(-gamma * _pairwise_sq(target, target))
    k_st = torch.exp(-gamma * _pairwise_sq(source, target))
    return k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()


def _as_feature_matrix(feat: torch.Tensor) -> torch.Tensor:
    """(B,C,T) → mean-pool (B,C); or pass through (B,D)."""
    if feat.dim() == 3:
        return pool_domain_feature_map(feat)
    if feat.dim() == 2:
        return feat
    raise ValueError(f"Domain feature must be (B,C,T) or (B,D), got {tuple(feat.shape)}")


def domain_adaptation_loss(
    feats_source: Mapping[str, torch.Tensor],
    feats_target: Mapping[str, torch.Tensor],
    *,
    method: str = "coral",
    mu: float = 0.4,
    mmd_sigma: float | None = None,
) -> torch.Tensor:
    """Multi-layer domain loss (Lin Eq. 12).

    Theory
    ------
    Per selected hook / layer ℓ:

        L_ℓ = μ · MMD²(Z_S^ℓ, Z_T^ℓ) + (1−μ) · CORAL(Z_S^ℓ, Z_T^ℓ)   # method=both
        L_ℓ = CORAL(...)                                              # method=coral
        L_ℓ = MMD²(...)                                               # method=mmd

    Total (paper sums FC6–8; we sum yaml `domain_feature_layers` keys):

        L_domain = Σ_ℓ L_ℓ

    Default μ = 0.4 (paper).
    """
    method = str(method or "coral").lower()
    if method not in {"coral", "mmd", "both"}:
        raise ValueError(f"domain method must be coral|mmd|both, got {method!r}")
    if not feats_source:
        raise ValueError("feats_source is empty")

    keys = list(feats_source.keys())
    missing = [k for k in keys if k not in feats_target]
    if missing:
        raise ValueError(f"feats_target missing layers {missing}")

    total: torch.Tensor | None = None
    for key in keys:
        zs = _as_feature_matrix(feats_source[key])
        zt = _as_feature_matrix(feats_target[key])
        if method == "coral":
            term = coral_loss(zs, zt)
        elif method == "mmd":
            term = mmd_rbf_loss(zs, zt, sigma=mmd_sigma)
        else:
            # Eq. 12: μ·MMD² + (1−μ)·CORAL
            term = float(mu) * mmd_rbf_loss(zs, zt, sigma=mmd_sigma) + (
                1.0 - float(mu)
            ) * coral_loss(zs, zt)
        total = term if total is None else total + term

    assert total is not None
    return total


# ---------------------------------------------------------------------------
# Multitask NILM loss
# ---------------------------------------------------------------------------


@dataclass
class MultiNILMLossOutput:
    """Scalars / vectors returned for backprop and logging."""

    loss: torch.Tensor                 # L (scalar, has grad)
    loss_power: torch.Tensor           # Σ_i MSE_i  (raw)
    loss_state: torch.Tensor           # Σ_i BCE_i  (raw)
    loss_state_term: torch.Tensor      # balanced state contribution into L_NILM
    loss_shape: torch.Tensor           # Σ_i slope-MSE_i (raw)
    loss_shape_term: torch.Tensor      # balanced shape contribution into L_NILM
    loss_domain: torch.Tensor          # L_domain (0 if DA off)
    mae: torch.Tensor                  # logging only (often denorm scale)
    loss_power_per_appliance: torch.Tensor
    loss_state_per_appliance: torch.Tensor


class MultiNILMLoss(nn.Module):
    """L = L_NILM + λ_domain · L_domain  (DA optional)."""

    def __init__(
        self,
        lambda_state: float = 1.0,
        pos_weight: torch.Tensor | list[float] | None = None,
        power_scale: float | list[float] | torch.Tensor = 1.0,
        *,
        task_balance: str = 'equal',
        lambda_shape: float = 0.0,
        lambda_domain: float = 0.0,
        domain_method: str = 'coral',
        domain_mu: float = 0.4,
        mmd_sigma: float | None = None,
    ) -> None:
        super().__init__()
        self.lambda_state = float(lambda_state)
        self.lambda_shape = float(lambda_shape)
        self.task_balance = str(task_balance or 'none').lower()
        self.lambda_domain = float(lambda_domain)
        self.domain_method = str(domain_method or 'coral').lower()
        self.domain_mu = float(domain_mu)
        self.mmd_sigma = None if mmd_sigma is None else float(mmd_sigma)

        self.register_buffer('power_scale', torch.as_tensor(power_scale, dtype=torch.float32))

        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.as_tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

    def _per_appliance_power_loss(
        self,
        power_pred: torch.Tensor,
        power_true: torch.Tensor,
    ) -> torch.Tensor:
        """MSE_i = mean_{b,t} (ŷ − y)²  → vector length A."""
        return torch.mean((power_pred - power_true) ** 2, dim=(0, 1))

    def _per_appliance_shape_loss(
        self,
        power_pred: torch.Tensor,
        power_true: torch.Tensor,
    ) -> torch.Tensor:
        """Slope / local-shape MSE via first differences (small waveform wiggles).

        Pointwise MSE can match mean level while missing edges. Penalize::

            Δŷ_t = ŷ_{t+1} − ŷ_t ,  Δy_t = y_{t+1} − y_t
            L_shape^i = mean (Δŷ − Δy)²
        """
        if power_pred.shape[1] < 2:
            return power_pred.new_zeros(power_pred.shape[-1])
        d_pred = power_pred[:, 1:, :] - power_pred[:, :-1, :]
        d_true = power_true[:, 1:, :] - power_true[:, :-1, :]
        return torch.mean((d_pred - d_true) ** 2, dim=(0, 1))

    def _per_appliance_state_loss(
        self,
        state_logits: torch.Tensor,
        state_true: torch.Tensor,
    ) -> torch.Tensor:
        """BCEWithLogits per appliance (optional pos_weight) → vector length A."""
        losses: list[torch.Tensor] = []
        for app_i in range(state_logits.shape[-1]):
            weight_i = None
            if self.pos_weight is not None:
                weight_i = (
                    self.pos_weight[app_i] if self.pos_weight.ndim > 0 else self.pos_weight
                )
            losses.append(
                F.binary_cross_entropy_with_logits(
                    state_logits[..., app_i],
                    state_true[..., app_i],
                    pos_weight=weight_i,
                )
            )
        return torch.stack(losses)

    def _balanced_term(
        self,
        loss_power: torch.Tensor,
        loss_raw: torch.Tensor,
        lambda_pref: float,
    ) -> torch.Tensor:
        """Scale loss_raw vs loss_power; λ=1 → equal magnitude when task_balance=equal."""
        if lambda_pref == 0.0:
            return loss_power.new_zeros(())
        if self.task_balance == 'none':
            return float(lambda_pref) * loss_raw
        if self.task_balance == 'equal':
            scale = loss_power.detach() / loss_raw.detach().clamp_min(1e-8)
            return float(lambda_pref) * loss_raw * scale
        raise ValueError(f'task_balance must be none|equal, got {self.task_balance!r}')

    def forward(
        self,
        power_pred: torch.Tensor,
        state_logits: torch.Tensor,
        power_true: torch.Tensor,
        state_true: torch.Tensor,
        *,
        domain_feats_S: Mapping[str, torch.Tensor] | None = None,
        domain_feats_T: Mapping[str, torch.Tensor] | None = None,
    ) -> MultiNILMLossOutput:
        """L_NILM = L_power + state_term + shape_term  (+ optional DA)."""
        power_pred = power_pred.float()
        state_logits = state_logits.float()
        power_true = power_true.float()
        state_true = state_true.float()

        loss_power_per_app = self._per_appliance_power_loss(power_pred, power_true)
        loss_state_per_app = self._per_appliance_state_loss(state_logits, state_true)
        loss_shape_per_app = self._per_appliance_shape_loss(power_pred, power_true)

        loss_power = loss_power_per_app.sum()
        loss_state = loss_state_per_app.sum()
        loss_shape = loss_shape_per_app.sum()

        loss_state_term = self._balanced_term(loss_power, loss_state, self.lambda_state)
        loss_shape_term = self._balanced_term(loss_power, loss_shape, self.lambda_shape)
        loss_nilm = loss_power + loss_state_term + loss_shape_term

        use_da = (
            domain_feats_S is not None
            and domain_feats_T is not None
            and self.lambda_domain != 0.0
        )
        if use_da:
            loss_domain = domain_adaptation_loss(
                domain_feats_S,
                domain_feats_T,
                method=self.domain_method,
                mu=self.domain_mu,
                mmd_sigma=self.mmd_sigma,
            )
            loss = loss_nilm + self.lambda_domain * loss_domain
        else:
            loss_domain = loss_nilm.new_zeros(())
            loss = loss_nilm

        scale = self.power_scale.to(device=power_pred.device, dtype=power_pred.dtype)
        if scale.ndim > 0:
            mae = (torch.mean(torch.abs(power_pred - power_true), dim=(0, 1)) * scale).mean()
        else:
            mae = torch.mean(torch.abs((power_pred - power_true) * scale))

        return MultiNILMLossOutput(
            loss=loss,
            loss_power=loss_power,
            loss_state=loss_state,
            loss_state_term=loss_state_term.detach(),
            loss_shape=loss_shape.detach(),
            loss_shape_term=loss_shape_term.detach(),
            loss_domain=loss_domain.detach(),
            mae=mae,
            loss_power_per_appliance=loss_power_per_app.detach(),
            loss_state_per_appliance=loss_state_per_app.detach(),
        )

"""MultiNILM multitask loss (power + state) and optional domain adaptation.

Paper-style multitask objective (equation 16 style):

    L_NILM = sum_{i=1}^{A} ( L_power^i + lambda_state * L_state^i )

Optional domain adaptation (Lin et al. IEEE TSG 2022):

    Domain features Z come from MultiNILM.forward(..., return_domain_features=True)
    at hooks selected by architecture.domain_feature_layers (default: aligned).

---------------------------------------------------------------------------
DOMAIN LOSS FORMULAS (Lin et al., IEEE TSG 2022)
---------------------------------------------------------------------------

Notation (one selected layer / hook):
    Z_S in R^{n_s x D}   pooled source features  (batch of house(s) with labels)
    Z_T in R^{n_t x D}   pooled target features  (unlabeled target-house aggregates)
    n_s, n_t             batch sizes
    D                    feature dimension after temporal mean-pool
    1                    all-ones column vector

(1) Deep CORAL  (paper Eq. 7–9)  — align second-order (covariance) statistics

    C_S = 1/(n_s - 1) * ( Z_S^T Z_S - (Z_S^T 1)(1^T Z_S)/n_s )
    C_T = 1/(n_t - 1) * ( Z_T^T Z_T - (Z_T^T 1)(1^T Z_T)/n_t )

    L_CORAL(Z_S, Z_T) = (1 / (4 D^2)) * || C_S - C_T ||_F^2

    Intuition: make source/target feature clouds have the same shape / correlations.

(2) Squared RBF-kernel MMD  (paper Eq. 6, kernel form)  — align first-order means

    k(u, v) = exp( - ||u - v||^2 / (2 sigma^2) )

    MMD^2(Z_S, Z_T)
        = mean_{i,i'} k(z_S^i, z_S^{i'})
        + mean_{j,j'} k(z_T^j, z_T^{j'})
        - 2 * mean_{i,j} k(z_S^i, z_T^j)

    If sigma is None: sigma = sqrt( median pairwise squared distance on [Z_S; Z_T] ).

    Intuition: pull source/target feature centers together in RKHS.

(3) Per-layer domain mix  (paper Eq. 12; mu = domain_mu, paper default 0.4)

    L_layer = mu * MMD^2(Z_S, Z_T) + (1 - mu) * L_CORAL(Z_S, Z_T)

    method='coral' -> L_layer = L_CORAL
    method='mmd'   -> L_layer = MMD^2
    method='both'  -> L_layer = mu * MMD^2 + (1 - mu) * L_CORAL

(4) Multi-layer sum  (paper sums FC layers l=6..8; we sum selected hook names)

    L_domain = sum_{layer in selected} L_layer

(5) Total training loss

    Paper (convex mix, Eq. 13; paper best lambda = 0.6):
        L = (1 - lambda) * L_R + lambda * L_domain

    This repo (additive; keeps L_NILM scale unchanged when DA is off):
        L = L_NILM + lambda_domain * L_domain

    Mapping tip:
        paper lambda=0.6  <->  domain : task weight ratio  0.6/0.4 = 1.5
        so additive lambda_domain ≈ 1.5 matches that *ratio* only if L_NILM and
        L_domain have similar magnitude; in practice start near 0.6 and tune.
        Keep lambda_domain=0.0 (and domain_adaptation.enabled=false) for the
        original multitask-only baseline.

When domain tensors are omitted or lambda_domain=0, L_domain is zero and
training matches the original multitask-only loss.

---------------------------------------------------------------------------
TENSOR SHAPES (one training / validation batch)
---------------------------------------------------------------------------

    power_pred    : (B, T, A)  model gated power output (normalized watts)
    state_logits  : (B, T, A)  raw ON/OFF logits (sigmoid NOT applied yet)
    power_true    : (B, T, A)  z-score normalized appliance power targets
    state_true    : (B, T, A)  binary ON/OFF targets in {0.0, 1.0}

Optional domain maps (source / target houses):

    domain_feats_S / domain_feats_T : dict[str, Tensor]
        each value (B, C, T) from encoder hooks, or already pooled (B, C)

---------------------------------------------------------------------------
CALL SITE
---------------------------------------------------------------------------

    adapter.step() -> loss_fn(power_pred, state_logits, y, z)

    DA training step:
        loss_fn(..., domain_feats_S=..., domain_feats_T=...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MultiNILM import pool_domain_feature_map


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Deep CORAL loss between source/target feature batches (Lin Eq. 7).

    Formulas
    --------
    Unbiased covariances (Eqs. 8–9)::

        C_S = 1/(n_s-1) * ( Z_S^T Z_S - (Z_S^T 1)(1^T Z_S)/n_s )
        C_T = 1/(n_t-1) * ( Z_T^T Z_T - (Z_T^T 1)(1^T Z_T)/n_t )

    CORAL (Eq. 7)::

        L_CORAL = (1 / (4 D^2)) * ||C_S - C_T||_F^2

    Parameters
    ----------
    source, target : (B, D) feature matrices (already pooled if needed)

    Returns
    -------
    Scalar
        (1 / (4 D^2)) * ||C_S - C_T||_F^2
    """
    source = source.float()
    target = target.float()
    if source.dim() != 2 or target.dim() != 2:
        raise ValueError(
            f"CORAL expects (B, D) features, got {tuple(source.shape)} and {tuple(target.shape)}"
        )
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"CORAL feature dim mismatch: source D={source.shape[1]}, target D={target.shape[1]}"
        )

    n_s = source.shape[0]
    n_t = target.shape[0]
    d = source.shape[1]
    if n_s < 2 or n_t < 2:
        # Covariance undefined for a single sample; treat as zero contribution.
        return source.new_zeros(())

    ones_s = source.new_ones(n_s, 1)
    ones_t = target.new_ones(n_t, 1)
    # Unbiased sample covariance (Lin Eqs. 8–9).
    cov_s = (source.T @ source - (source.T @ ones_s) @ (ones_s.T @ source) / n_s) / (n_s - 1)
    cov_t = (target.T @ target - (target.T @ ones_t) @ (ones_t.T @ target) / n_t) / (n_t - 1)
    return (cov_s - cov_t).pow(2).sum() / (4.0 * float(d) * float(d))


def mmd_rbf_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    sigma: float | None = None,
) -> torch.Tensor:
    """Squared RBF-kernel MMD between source/target batches (Lin Eq. 6 style).

    Formulas
    --------
    Gaussian kernel::

        k(u, v) = exp( -||u - v||^2 / (2 sigma^2) )

    Empirical MMD^2::

        MMD^2 = E[k(z_S, z_S')] + E[k(z_T, z_T')] - 2 E[k(z_S, z_T)]

    If ``sigma`` is None, use median pairwise distance on the concatenated batch
    (standard heuristic; paper uses a Gaussian kernel without fixing one schedule).
    """
    source = source.float()
    target = target.float()
    if source.dim() != 2 or target.dim() != 2:
        raise ValueError(
            f"MMD expects (B, D) features, got {tuple(source.shape)} and {tuple(target.shape)}"
        )

    def _pairwise_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (
            a.pow(2).sum(dim=1, keepdim=True)
            + b.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * (a @ b.T)
        ).clamp_min(0.0)

    with torch.no_grad():
        if sigma is None:
            combo = torch.cat([source.detach(), target.detach()], dim=0)
            dist = _pairwise_sq(combo, combo)
            n = dist.shape[0]
            if n > 1:
                mask = ~torch.eye(n, dtype=torch.bool, device=dist.device)
                med = dist[mask].median()
            else:
                med = dist.new_tensor(1.0)
            sigma_val = torch.sqrt(med.clamp_min(1e-6))
        else:
            sigma_val = source.new_tensor(float(sigma))

    gamma = 1.0 / (2.0 * sigma_val.pow(2).clamp_min(1e-12))
    k_ss = torch.exp(-gamma * _pairwise_sq(source, source))
    k_tt = torch.exp(-gamma * _pairwise_sq(target, target))
    k_st = torch.exp(-gamma * _pairwise_sq(source, target))
    return k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()


def _as_feature_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Accept (B, C, T) maps or (B, D) vectors."""
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
    """Sum domain discrepancy over selected named layers (Lin Eq. 12 style).

    Formulas
    --------
    Per selected layer ``l``::

        L_layer^l = mu * MMD^2(Z_S^l, Z_T^l) + (1 - mu) * L_CORAL(Z_S^l, Z_T^l)

    when ``method='both'``. For ``coral`` / ``mmd``, only that term is used.

    Total::

        L_domain = sum_l L_layer^l

    Paper: sum over FC layers l=6..8 with mu=0.4.
    Here: sum over keys in ``feats_source`` (e.g. ``aligned``).

    Parameters
    ----------
    feats_source / feats_target
        Dicts from MultiNILM ``return_domain_features=True`` (same keys).
    method
        ``coral`` | ``mmd`` | ``both`` (both uses mu * MMD^2 + (1-mu) * CORAL).
    mu
        Weight on MMD when ``method='both'`` (paper default 0.4).
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
            term = float(mu) * mmd_rbf_loss(zs, zt, sigma=mmd_sigma) + (
                1.0 - float(mu)
            ) * coral_loss(zs, zt)
        total = term if total is None else total + term

    assert total is not None
    return total


@dataclass
class MultiNILMLossOutput:
    """All tensors returned by MultiNILMLoss.forward for training and logging."""

    loss: torch.Tensor
    loss_power: torch.Tensor
    loss_state: torch.Tensor
    loss_domain: torch.Tensor
    mae: torch.Tensor
    loss_power_per_appliance: torch.Tensor
    loss_state_per_appliance: torch.Tensor


class MultiNILMLoss(nn.Module):
    """Per-appliance MSE + BCE, plus optional multi-layer domain loss."""

    def __init__(
        self,
        lambda_state: float = 0.1,
        pos_weight: torch.Tensor | list[float] | None = None,
        power_scale: float | list[float] | torch.Tensor = 1.0,
        *,
        lambda_domain: float = 0.0,
        domain_method: str = "coral",
        domain_mu: float = 0.4,
        mmd_sigma: float | None = None,
    ) -> None:
        super().__init__()
        self.lambda_state = float(lambda_state)
        self.lambda_domain = float(lambda_domain)
        self.domain_method = str(domain_method or "coral").lower()
        self.domain_mu = float(domain_mu)
        self.mmd_sigma = None if mmd_sigma is None else float(mmd_sigma)

        self.register_buffer("power_scale", torch.as_tensor(power_scale, dtype=torch.float32))

        if pos_weight is not None:
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def _per_appliance_power_loss(
        self,
        power_pred: torch.Tensor,
        power_true: torch.Tensor,
    ) -> torch.Tensor:
        squared_error = (power_pred - power_true) ** 2
        return torch.mean(squared_error, dim=(0, 1))

    def _per_appliance_state_loss(
        self,
        state_logits: torch.Tensor,
        state_true: torch.Tensor,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        n_apps = state_logits.shape[-1]

        for app_i in range(n_apps):
            logits_i = state_logits[..., app_i]
            target_i = state_true[..., app_i]

            weight_i = None
            if self.pos_weight is not None:
                weight_i = self.pos_weight[app_i] if self.pos_weight.ndim > 0 else self.pos_weight

            losses.append(
                F.binary_cross_entropy_with_logits(
                    logits_i,
                    target_i,
                    pos_weight=weight_i,
                )
            )

        return torch.stack(losses)

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
        """Compute multitask (+ optional domain) loss for one batch.

        Formulas
        --------
        L_NILM = sum_i ( MSE_i + lambda_state * BCE_i )

        If domain features are provided and lambda_domain != 0::

            L = L_NILM + lambda_domain * L_domain(Z_S, Z_T)

        else::

            L = L_NILM   (identical to pre-DA training)

        Paper uses L = (1-lambda)*L_R + lambda*L_domain with lambda=0.6;
        see module docstring for the additive vs convex mapping note.
        """
        power_pred = power_pred.float()
        state_logits = state_logits.float()
        power_true = power_true.float()
        state_true = state_true.float()

        loss_power_per_app = self._per_appliance_power_loss(power_pred, power_true)
        loss_state_per_app = self._per_appliance_state_loss(state_logits, state_true)

        loss_power = loss_power_per_app.sum()
        loss_state = loss_state_per_app.sum()
        loss_nilm = loss_power + self.lambda_state * loss_state

        if (
            domain_feats_S is not None
            and domain_feats_T is not None
            and self.lambda_domain != 0.0
        ):
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
            mae_per_app = torch.mean(torch.abs(power_pred - power_true), dim=(0, 1)) * scale
            mae = mae_per_app.mean()
        else:
            mae = torch.mean(torch.abs((power_pred - power_true) * scale))

        return MultiNILMLossOutput(
            loss=loss,
            loss_power=loss_power,
            loss_state=loss_state,
            loss_domain=loss_domain.detach(),
            mae=mae,
            loss_power_per_appliance=loss_power_per_app.detach(),
            loss_state_per_appliance=loss_state_per_app.detach(),
        )

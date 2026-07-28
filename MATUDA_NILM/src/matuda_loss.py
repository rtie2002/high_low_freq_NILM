"""
Domain losses on fully-connected embeddings (Lin / Deep CORAL / DAN / CDAN+E).

All functions expect features Z of shape (B, D) — after FC layers — NOT (B,C,T).

MATUDA novelty (M0): Entropy-Gated Conditional CORAL/MMD (EGC-DA)
  - Condition alignment on multi-label state predictions (CDAN-style).
  - Down-weight uncertain target windows via prediction entropy (CDAN+E).
  - Mitigates negative transfer observed under uniform FC alignment.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def coral_loss(zs: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    """Deep CORAL on (B, D) activations."""
    assert zs.dim() == 2 and zt.dim() == 2
    d = zs.size(1)
    ns, nt = zs.size(0), zt.size(0)
    if ns < 2 or nt < 2:
        return zs.new_zeros(())
    ones_s = torch.ones(ns, 1, device=zs.device, dtype=zs.dtype)
    ones_t = torch.ones(nt, 1, device=zt.device, dtype=zt.dtype)
    cs = (zs.t() @ zs - (zs.t() @ ones_s) @ (ones_s.t() @ zs) / ns) / (ns - 1 + 1e-8)
    ct = (zt.t() @ zt - (zt.t() @ ones_t) @ (ones_t.t() @ zt) / nt) / (nt - 1 + 1e-8)
    return (cs - ct).pow(2).sum() / (4.0 * d * d)


def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).t()
    dist = (x2 + y2 - 2.0 * (x @ y.t())).clamp_min(0.0)
    return torch.exp(-dist / (2.0 * sigma * sigma + 1e-8))


def mmd_rbf(zs: torch.Tensor, zt: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    assert zs.dim() == 2 and zt.dim() == 2
    if zs.size(0) < 2 or zt.size(0) < 2:
        return zs.new_zeros(())
    with torch.no_grad():
        if sigma is None:
            z = torch.cat([zs, zt], dim=0)
            n = min(z.size(0), 128)
            idx = torch.randperm(z.size(0), device=z.device)[:n]
            zz = z[idx]
            d = torch.pdist(zz, p=2)
            sigma = float(d.median().clamp_min(1e-4).item()) if d.numel() else 1.0
    k_ss = _gaussian_kernel(zs, zs, sigma)
    k_tt = _gaussian_kernel(zt, zt, sigma)
    k_st = _gaussian_kernel(zs, zt, sigma)
    ns, nt = zs.size(0), zt.size(0)
    return (
        (k_ss.sum() - k_ss.diag().sum()) / (ns * (ns - 1) + 1e-8)
        + (k_tt.sum() - k_tt.diag().sum()) / (nt * (nt - 1) + 1e-8)
        - 2.0 * k_st.mean()
    )


def _l2_normalize(z: torch.Tensor) -> torch.Tensor:
    return F.normalize(z, p=2, dim=1)


def prediction_entropy_weights(state_logits: torch.Tensor) -> torch.Tensor:
    """
    Per-sample transferability weight in (0,1], CDAN+E-inspired.
    Low entropy (confident multi-label preds) -> higher weight.
    state_logits: (B, K)
    """
    p = torch.sigmoid(state_logits).clamp(1e-6, 1 - 1e-6)
    # Binary entropy averaged over K heads, normalized to [0,1].
    h = -(p * p.log() + (1 - p) * (1 - p).log()).mean(dim=1)  # (B,)
    h_norm = h / 0.693147  # ln(2)
    return (1.0 - h_norm).clamp(0.05, 1.0)


def _apply_sample_weights(
    z: torch.Tensor, weights: Optional[torch.Tensor]
) -> torch.Tensor:
    """Scale L2-normalized rows by √w. Do **not** re-normalize afterward.

    Weighting before L2-norm is a no-op (direction unchanged → unit vector again).
    Weighting after L2-norm changes each sample's contribution to CORAL/MMD.
    """
    if weights is None:
        return z
    if weights.dim() != 1 or weights.size(0) != z.size(0):
        raise ValueError(
            f"sample weights must be (B,), got {tuple(weights.shape)} for Z {tuple(z.shape)}"
        )
    return z * weights.sqrt().unsqueeze(1)


def multilayer_domain_loss(
    feats_s: Sequence[torch.Tensor],
    feats_t: Sequence[torch.Tensor],
    mu: float = 0.5,
    weights_s: Optional[torch.Tensor] = None,
    weights_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Hybrid MMD+CORAL summed over FC layers (Lin Eq. 12).
    Optional EGC sample weights: L2-normalize first, then scale by √w (no re-norm).
    """
    assert len(feats_s) == len(feats_t) and len(feats_s) > 0
    total = feats_s[0].new_zeros(())
    for zs, zt in zip(feats_s, feats_t):
        zs_n = _apply_sample_weights(_l2_normalize(zs), weights_s)
        zt_n = _apply_sample_weights(_l2_normalize(zt), weights_t)
        total = total + mu * mmd_rbf(zs_n, zt_n) + (1.0 - mu) * coral_loss(zs_n, zt_n)
    # Lin Eq. 12: sum over layers (not mean).
    return total


def conditional_appliance_domain_loss(
    feats_s: Sequence[torch.Tensor],
    feats_t: Sequence[torch.Tensor],
    state_logits_s: torch.Tensor,
    state_logits_t: torch.Tensor,
    mu: float = 0.5,
    min_mass: float = 2.0,
) -> torch.Tensor:
    """
    Appliance-conditional alignment on the *last* FC embedding:
    soft-weight rows by predicted ON probability per appliance, then MMD+CORAL.
    Averages over appliances with enough predicted ON mass on both domains.
    """
    z_s = _l2_normalize(feats_s[-1])
    z_t = _l2_normalize(feats_t[-1])
    p_s = torch.sigmoid(state_logits_s).detach()
    p_t = torch.sigmoid(state_logits_t).detach()
    if p_s.dim() == 3:
        p_s = p_s.mean(dim=1)
        p_t = p_t.mean(dim=1)
    k = p_s.size(1)
    total = z_s.new_zeros(())
    used = 0
    for a in range(k):
        ws, wt = p_s[:, a], p_t[:, a]
        if float(ws.sum()) < min_mass or float(wt.sum()) < min_mass:
            continue
        zs = _apply_sample_weights(z_s, ws)
        zt = _apply_sample_weights(z_t, wt)
        total = total + mu * mmd_rbf(zs, zt) + (1.0 - mu) * coral_loss(zs, zt)
        used += 1
    if used == 0:
        return multilayer_domain_loss(feats_s, feats_t, mu=mu)
    return total / float(used)


class MATUDACriterion(nn.Module):
    """
    Supervised multi-task + unsupervised FC domain adaptation.

    Modes:
      none      — L = L_sup
      global    — uniform multilayer MMD+CORAL (B1)
      egc       — entropy-gated + appliance-conditional (M0 / MATUDA)

    Mix:
      additive  — L = L_sup + λ L_domain
      convex    — L = (1-λ) L_sup + λ L_domain   (Lin)
    Scale:
      none | equal (match |L_domain| to |L_sup| with stop-grad)
    """

    def __init__(
        self,
        lambda_domain: float = 0.5,
        mu_mmd: float = 0.4,
        power_weight: float = 1.0,
        state_weight: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
        da_mode: str = "global",
        domain_mix: str = "convex",
        domain_scale: str = "equal",
        conditional_weight: float = 0.5,
    ):
        super().__init__()
        self.lambda_domain = lambda_domain
        self.mu_mmd = mu_mmd
        self.power_weight = power_weight
        self.state_weight = state_weight
        self.da_mode = da_mode
        self.domain_mix = domain_mix
        self.domain_scale = domain_scale
        self.conditional_weight = conditional_weight
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def supervised(
        self,
        powers_hat: torch.Tensor,
        powers_gt: torch.Tensor,
        state_logits: torch.Tensor,
        states_gt: torch.Tensor,
    ) -> torch.Tensor:
        # Sparse ON events: optional pos_weight (MATNILM / MultiNILM practice).
        if self.pos_weight is not None:
            loss_cls = F.binary_cross_entropy_with_logits(
                state_logits, states_gt.float(), pos_weight=self.pos_weight
            )
        else:
            loss_cls = F.binary_cross_entropy_with_logits(
                state_logits, states_gt.float()
            )
        loss_reg = F.mse_loss(powers_hat, powers_gt.float())
        return self.state_weight * loss_cls + self.power_weight * loss_reg

    def _domain(
        self,
        out_s: dict,
        out_t: dict,
    ) -> torch.Tensor:
        feats_s, feats_t = out_s["da_features"], out_t["da_features"]
        if self.da_mode == "none":
            return feats_s[0].new_zeros(())

        if self.da_mode == "global":
            return multilayer_domain_loss(feats_s, feats_t, mu=self.mu_mmd)

        # Ablation: entropy-gated global only (no appliance-conditional term).
        if self.da_mode == "egc_no_cond":
            w_s = prediction_entropy_weights(out_s["state_logits"])
            w_t = prediction_entropy_weights(out_t["state_logits"])
            return multilayer_domain_loss(
                feats_s, feats_t, mu=self.mu_mmd, weights_s=w_s, weights_t=w_t
            )

        # Ablation: conditional only (no entropy reweighting on global path).
        if self.da_mode == "egc_no_entropy":
            l_global = multilayer_domain_loss(feats_s, feats_t, mu=self.mu_mmd)
            l_cond = conditional_appliance_domain_loss(
                feats_s,
                feats_t,
                out_s["state_logits"],
                out_t["state_logits"],
                mu=self.mu_mmd,
                min_mass=2.0,
            )
            alpha = self.conditional_weight
            return (1.0 - alpha) * l_global + alpha * l_cond

        # Full EGC-DA: entropy gate + conditional appliance alignment.
        w_s = prediction_entropy_weights(out_s["state_logits"])
        w_t = prediction_entropy_weights(out_t["state_logits"])
        l_global = multilayer_domain_loss(
            feats_s, feats_t, mu=self.mu_mmd, weights_s=w_s, weights_t=w_t
        )
        l_cond = conditional_appliance_domain_loss(
            feats_s,
            feats_t,
            out_s["state_logits"],
            out_t["state_logits"],
            mu=self.mu_mmd,
            min_mass=2.0,
        )
        alpha = self.conditional_weight
        return (1.0 - alpha) * l_global + alpha * l_cond

    def forward(
        self,
        out_s: dict,
        out_t: Optional[dict],
        powers_gt: torch.Tensor,
        states_gt: torch.Tensor,
        lambda_override: Optional[float] = None,
    ) -> dict:
        l_sup = self.supervised(
            out_s["powers"], powers_gt, out_s["state_logits"], states_gt
        )
        lam = self.lambda_domain if lambda_override is None else float(lambda_override)

        if out_t is None or lam <= 0 or self.da_mode == "none":
            return {
                "loss": l_sup,
                "loss_sup": l_sup.detach(),
                "loss_domain": l_sup.new_zeros(()),
                "lambda": 0.0,
            }

        l_dom = self._domain(out_s, out_t)
        if self.domain_scale == "equal":
            # Match magnitude to L_sup so λ is interpretable (MultiNILM practice).
            scale = (l_sup.detach() / (l_dom.detach().abs() + 1e-8)).clamp(0.1, 10.0)
            l_dom_term = l_dom * scale
        else:
            l_dom_term = l_dom

        if self.domain_mix == "convex":
            total = (1.0 - lam) * l_sup + lam * l_dom_term
        else:
            total = l_sup + lam * l_dom_term

        return {
            "loss": total,
            "loss_sup": l_sup.detach(),
            "loss_domain": l_dom.detach(),
            "lambda": lam,
        }

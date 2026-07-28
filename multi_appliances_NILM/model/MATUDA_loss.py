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
    state_logits: (B, K) or (B, T, K)
    """
    p = torch.sigmoid(state_logits).clamp(1e-6, 1 - 1e-6)
    # Binary entropy averaged over appliance (and time if seq2seq).
    reduce_dims = tuple(range(1, p.dim()))
    h = -(p * p.log() + (1 - p) * (1 - p).log()).mean(dim=reduce_dims)  # (B,)
    h_norm = h / 0.693147  # ln(2)
    return (1.0 - h_norm).clamp(0.05, 1.0)


def multilayer_domain_loss(
    feats_s: Sequence[torch.Tensor],
    feats_t: Sequence[torch.Tensor],
    mu: float = 0.5,
    weights_s: Optional[torch.Tensor] = None,
    weights_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Mean hybrid MMD+CORAL over FC layers.
    Optional sample weights: reweight by sqrt(w) before L2-norm (EGC-DA).
    """
    assert len(feats_s) == len(feats_t) and len(feats_s) > 0
    total = feats_s[0].new_zeros(())
    for zs, zt in zip(feats_s, feats_t):
        if weights_s is not None:
            zs = zs * weights_s.sqrt().unsqueeze(1)
        if weights_t is not None:
            zt = zt * weights_t.sqrt().unsqueeze(1)
        zs_n, zt_n = _l2_normalize(zs), _l2_normalize(zt)
        total = total + mu * mmd_rbf(zs_n, zt_n) + (1.0 - mu) * coral_loss(zs_n, zt_n)
    return total / float(len(feats_s))


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
    # Seq2seq (B, T, K): time-average ON probs → (B, K) for sample weighting.
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
        zs = _l2_normalize(z_s * ws.sqrt().unsqueeze(1))
        zt = _l2_normalize(z_t * wt.sqrt().unsqueeze(1))
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
        on_masked_power: bool = True,
        pl_weight: float = 0.0,
        pl_confidence: float = 0.9,
        task_balance: str = "equal",
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
        self.on_masked_power = bool(on_masked_power)
        self.pl_weight = float(pl_weight)
        self.pl_confidence = float(pl_confidence)
        self.task_balance = str(task_balance or "none").lower()
        if self.task_balance not in {"none", "equal"}:
            raise ValueError(f"task_balance must be none|equal, got {self.task_balance!r}")
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
    ) -> dict[str, torch.Tensor]:
        """Return power / state terms (MultiNILM-style logging + optional equal balance)."""
        # Sparse ON events: optional pos_weight (MATNILM / MultiNILM practice).
        # Flatten time for seq2seq so pos_weight (K,) broadcasts like multilabel BCE.
        if state_logits.dim() == 3:
            b, t, k = state_logits.shape
            logits_flat = state_logits.reshape(b * t, k)
            states_flat = states_gt.float().reshape(b * t, k)
        else:
            logits_flat = state_logits
            states_flat = states_gt.float()

        if self.pos_weight is not None:
            loss_state = F.binary_cross_entropy_with_logits(
                logits_flat, states_flat, pos_weight=self.pos_weight
            )
        else:
            loss_state = F.binary_cross_entropy_with_logits(logits_flat, states_flat)

        # ON-masked MSE focuses regression on active events (helps SAE/F1 coupling).
        if self.on_masked_power:
            mask = states_gt.float()
            denom = mask.sum().clamp_min(1.0)
            loss_power = ((powers_hat - powers_gt.float()).pow(2) * mask).sum() / denom
        else:
            loss_power = F.mse_loss(powers_hat, powers_gt.float())

        loss_power = self.power_weight * loss_power
        # MultiNILM equal: rescale state magnitude to match power, then apply state_weight.
        if self.task_balance == "equal":
            scale = loss_power.detach() / loss_state.detach().clamp_min(1e-8)
            loss_state_term = self.state_weight * loss_state * scale
        else:
            loss_state_term = self.state_weight * loss_state

        return {
            "loss_sup": loss_power + loss_state_term,
            "loss_power": loss_power,
            "loss_state": loss_state,
            "loss_state_term": loss_state_term,
        }

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

    def _pseudo_label_state(self, out_t: dict) -> torch.Tensor:
        """Hur et al. (Sensors 2022): confident target multi-label pseudo-labels."""
        logits = out_t["state_logits"]
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        conf = torch.maximum(p, 1.0 - p)
        mask = conf >= self.pl_confidence
        if not bool(mask.any()):
            return logits.new_zeros(())
        pl = (p >= 0.5).float()
        loss = F.binary_cross_entropy_with_logits(logits, pl.detach(), reduction="none")
        return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

    def forward(
        self,
        out_s: dict,
        out_t: Optional[dict],
        powers_gt: torch.Tensor,
        states_gt: torch.Tensor,
        lambda_override: Optional[float] = None,
    ) -> dict:
        parts = self.supervised(
            out_s["powers"], powers_gt, out_s["state_logits"], states_gt
        )
        l_sup = parts["loss_sup"]
        lam = self.lambda_domain if lambda_override is None else float(lambda_override)

        l_pl = l_sup.new_zeros(())
        if out_t is not None and self.pl_weight > 0:
            l_pl = self._pseudo_label_state(out_t)

        base = {
            "loss_sup": l_sup.detach(),
            "loss_power": parts["loss_power"].detach(),
            "loss_state": parts["loss_state"].detach(),
            "loss_state_term": parts["loss_state_term"].detach(),
            "loss_pl": l_pl.detach(),
        }

        if out_t is None or lam <= 0 or self.da_mode == "none":
            total = l_sup + self.pl_weight * l_pl
            return {
                "loss": total,
                "loss_domain": l_sup.new_zeros(()),
                "loss_domain_term": l_sup.new_zeros(()),
                "lambda": 0.0,
                **base,
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
        total = total + self.pl_weight * l_pl

        return {
            "loss": total,
            "loss_domain": l_dom.detach(),
            "loss_domain_term": l_dom_term.detach(),
            "lambda": lam,
            **base,
        }

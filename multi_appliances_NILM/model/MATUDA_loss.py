"""
MATUDA criterion: MultiNILM supervised loss + optional EGC domain / PL.

Supervised L_NILM is exactly ``MultiNILMLoss`` (per-app MSE + BCE, task_balance,
pos_weight). Domain adaptation uses MATUDA EGC (or global MMD+CORAL) on FC
embeddings; mix/scale match MultiNILM (convex + equal).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MultiNILM_loss import MultiNILMLoss


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
    """Per-sample transferability weight in (0,1], CDAN+E-inspired."""
    p = torch.sigmoid(state_logits).clamp(1e-6, 1 - 1e-6)
    reduce_dims = tuple(range(1, p.dim()))
    h = -(p * p.log() + (1 - p) * (1 - p).log()).mean(dim=reduce_dims)
    h_norm = h / 0.693147
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
    """Hybrid MMD+CORAL summed over FC layers (Lin Eq. 12), optional EGC weights."""
    assert len(feats_s) == len(feats_t) and len(feats_s) > 0
    total = feats_s[0].new_zeros(())
    for zs, zt in zip(feats_s, feats_t):
        # 1) unit sphere, 2) √w scale (EGC), 3) CORAL/MMD — never L2 after step 2
        zs_n = _apply_sample_weights(_l2_normalize(zs), weights_s)
        zt_n = _apply_sample_weights(_l2_normalize(zt), weights_t)
        total = total + mu * mmd_rbf(zs_n, zt_n) + (1.0 - mu) * coral_loss(zs_n, zt_n)
    # Lin Eq. 12: L_domain = Σ_ℓ [...], not mean over layers.
    return total


def conditional_appliance_domain_loss(
    feats_s: Sequence[torch.Tensor],
    feats_t: Sequence[torch.Tensor],
    state_logits_s: torch.Tensor,
    state_logits_t: torch.Tensor,
    mu: float = 0.5,
    min_mass: float = 2.0,
) -> torch.Tensor:
    """Appliance-conditional alignment on the last FC embedding."""
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
        # Soft ON weights after L2; do not re-normalize (same as EGC global path).
        zs = _apply_sample_weights(z_s, ws)
        zt = _apply_sample_weights(z_t, wt)
        total = total + mu * mmd_rbf(zs, zt) + (1.0 - mu) * coral_loss(zs, zt)
        used += 1
    if used == 0:
        return multilayer_domain_loss(feats_s, feats_t, mu=mu)
    return total / float(used)


class MATUDACriterion(nn.Module):
    """MultiNILM L_NILM + MATUDA EGC domain (+ optional target PL)."""

    def __init__(
        self,
        lambda_domain: float = 0.5,
        mu_mmd: float = 0.4,
        lambda_state: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
        da_mode: str = "egc",
        domain_mix: str = "convex",
        domain_scale: str = "equal",
        conditional_weight: float = 0.5,
        on_masked_power: bool = False,
        pl_weight: float = 0.0,
        pl_confidence: float = 0.9,
        task_balance: str = "equal",
        power_scale: float | list[float] | torch.Tensor = 1.0,
        focal_gamma: float = 0.0,
        # Legacy aliases (map onto MultiNILM lambda_state)
        power_weight: float | None = None,
        state_weight: float | None = None,
    ):
        super().__init__()
        # Prefer MultiNILM names; fall back to old MATUDA state_weight.
        if state_weight is not None and lambda_state == 1.0:
            lambda_state = float(state_weight)
        _ = power_weight  # unused: MultiNILM does not scale power separately

        self.lambda_domain = float(lambda_domain)
        self.mu_mmd = float(mu_mmd)
        self.da_mode = str(da_mode or "none").lower()
        self.domain_mix = str(domain_mix or "convex").lower()
        self.domain_scale = str(domain_scale or "equal").lower()
        self.conditional_weight = float(conditional_weight)
        self.on_masked_power = bool(on_masked_power)
        self.pl_weight = float(pl_weight)
        self.pl_confidence = float(pl_confidence)
        self.focal_gamma = float(focal_gamma)

        # Same supervised criterion as MultiNILM (DA handled here for EGC).
        self.nilm = MultiNILMLoss(
            lambda_state=float(lambda_state),
            pos_weight=pos_weight,
            power_scale=power_scale,
            task_balance=str(task_balance or "equal"),
            lambda_domain=0.0,
            domain_mix=self.domain_mix,
            domain_scale=self.domain_scale,
        )

    def _on_masked_power_loss(
        self,
        power_pred: torch.Tensor,
        power_true: torch.Tensor,
        state_true: torch.Tensor,
    ) -> torch.Tensor:
        """Optional ON-only MSE (sum over appliances), else MultiNILM MSE."""
        if not self.on_masked_power:
            return self.nilm._per_appliance_power_loss(power_pred, power_true).sum()
        mask = state_true.float()
        denom = mask.sum().clamp_min(1.0)
        return ((power_pred - power_true).pow(2) * mask).sum() / denom

    def supervised(
        self,
        powers_hat: torch.Tensor,
        powers_gt: torch.Tensor,
        state_logits: torch.Tensor,
        states_gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """L_NILM via MultiNILMLoss (optional ON-masked power override)."""
        if self.on_masked_power or self.focal_gamma > 0.0:
            loss_power = self._on_masked_power_loss(powers_hat, powers_gt, states_gt)
            loss_state = self._state_loss(state_logits, states_gt.float())
            loss_state_term = self.nilm._balanced_state_term(loss_power, loss_state)
            loss_sup = loss_power + loss_state_term
            return {
                "loss_sup": loss_sup,
                "loss_power": loss_power,
                "loss_state": loss_state,
                "loss_state_term": loss_state_term,
            }

        out = self.nilm(powers_hat, state_logits, powers_gt, states_gt)
        return {
            "loss_sup": out.loss,
            "loss_power": out.loss_power,
            "loss_state": out.loss_state,
            "loss_state_term": out.loss_state_term,
        }

    def _state_loss(
        self, state_logits: torch.Tensor, states_gt: torch.Tensor
    ) -> torch.Tensor:
        """Per-app BCE (+ optional focal) summed over appliances."""
        if self.focal_gamma <= 0.0:
            return self.nilm._per_appliance_state_loss(state_logits, states_gt).sum()
        # Focal BCE: down-weight easy OFF timesteps (helps rare ON events).
        losses: list[torch.Tensor] = []
        for app_i in range(state_logits.shape[-1]):
            weight_i = None
            if self.nilm.pos_weight is not None:
                weight_i = (
                    self.nilm.pos_weight[app_i]
                    if self.nilm.pos_weight.ndim > 0
                    else self.nilm.pos_weight
                )
            logits = state_logits[..., app_i]
            target = states_gt[..., app_i]
            bce = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=weight_i, reduction="none"
            )
            p = torch.sigmoid(logits).detach()
            p_t = p * target + (1.0 - p) * (1.0 - target)
            focal = (1.0 - p_t).clamp_min(0.0).pow(self.focal_gamma)
            losses.append((focal * bce).mean())
        return torch.stack(losses).sum()

    def _domain(self, out_s: dict, out_t: dict) -> torch.Tensor:
        feats_s, feats_t = out_s["da_features"], out_t["da_features"]
        if self.da_mode == "none":
            return feats_s[0].new_zeros(())

        if self.da_mode == "global":
            return multilayer_domain_loss(feats_s, feats_t, mu=self.mu_mmd)

        if self.da_mode == "egc_no_cond":
            w_s = prediction_entropy_weights(out_s["state_logits"])
            w_t = prediction_entropy_weights(out_t["state_logits"])
            return multilayer_domain_loss(
                feats_s, feats_t, mu=self.mu_mmd, weights_s=w_s, weights_t=w_t
            )

        if self.da_mode == "egc_no_entropy":
            l_global = multilayer_domain_loss(feats_s, feats_t, mu=self.mu_mmd)
            l_cond = conditional_appliance_domain_loss(
                feats_s,
                feats_t,
                out_s["state_logits"],
                out_t["state_logits"],
                mu=self.mu_mmd,
            )
            alpha = self.conditional_weight
            return (1.0 - alpha) * l_global + alpha * l_cond

        # Full EGC-DA
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
        )
        alpha = self.conditional_weight
        return (1.0 - alpha) * l_global + alpha * l_cond

    def _pseudo_label_state(self, out_t: dict) -> torch.Tensor:
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

        # Live tensors for logs (adapter .detach()s) — matches MultiNILM fields.
        base = {
            "loss_sup": l_sup,
            "loss_power": parts["loss_power"],
            "loss_state": parts["loss_state"],
            "loss_state_term": parts["loss_state_term"],
            "loss_pl": l_pl,
        }

        if out_t is None or lam <= 0 or self.da_mode == "none":
            total = l_sup + self.pl_weight * l_pl
            zero = l_sup.new_zeros(())
            return {
                "loss": total,
                "loss_domain": zero,
                "loss_domain_term": zero,
                "lambda": 0.0,
                **base,
            }

        l_dom = self._domain(out_s, out_t).clamp_min(0.0)
        if self.domain_scale == "equal":
            scale = (l_sup.detach() / (l_dom.detach() + 1e-8)).clamp(0.1, 10.0)
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
            "loss_domain": l_dom,
            "loss_domain_term": l_dom_term,
            "lambda": lam,
            **base,
        }

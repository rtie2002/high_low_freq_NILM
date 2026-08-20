"""MultiNILM multitask loss + optional Lin-style domain adaptation.

Pipeline total loss (Lin convex mix when DA on)::

    L = (1 - lambda_domain) * L_NILM + lambda_domain * L_domain

Legacy additive (domain_mix=additive)::

    L = L_NILM + lambda_domain * L_domain

Supervised part (see docs/multinilm_task_loss_balance.md)::

    L_power = sum_i MSE_i
    L_state = sum_i BCE_i
    L_NILM  = L_power + state_term

    task_balance=none :  state_term = lambda_state * L_state
    task_balance=equal:  state_term = lambda_state * L_state
                                       * (L_power / L_state).detach()
                         → lambda_state=1 means equal power ↔ state weight

Domain part (Lin et al., IEEE TSG 2022)::

    L_domain = sum_layer [ mu * MMD^2 + (1-mu) * CORAL ]   # method=both
    Paper: L = (1-λ) L_R + λ L_domain  with best λ=0.6  → domain_mix=convex

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
    loss_state_transition: torch.Tensor
    loss_energy_relative: torch.Tensor
    loss_aggregate_consistency: torch.Tensor
    loss_domain: torch.Tensor          # raw L_domain (0 if DA off)
    loss_domain_term: torch.Tensor     # domain contribution before mix weights
    mae: torch.Tensor                  # logging only (often denorm scale)
    loss_power_per_appliance: torch.Tensor
    loss_state_per_appliance: torch.Tensor


class MultiNILMLoss(nn.Module):
    """Optional DA: Lin convex mix or legacy additive mix."""

    def __init__(
        self,
        lambda_state: float = 1.0,
        pos_weight: torch.Tensor | list[float] | None = None,
        power_scale: float | list[float] | torch.Tensor = 1.0,
        *,
        task_balance: str = "equal",
        lambda_domain: float = 0.0,
        domain_method: str = "coral",
        domain_mu: float = 0.4,
        mmd_sigma: float | None = None,
        domain_mix: str = "convex",
        domain_scale: str = "none",
        power_on_weight: float = 0.0,
        power_off_weight: float = 0.0,
        power_delta_weight: float = 0.0,
        power_delta_on_only: bool = True,
        power_energy_weight: float = 0.0,
        state_fp_weight: float = 0.0,
        state_transition_weight: float = 0.0,
        power_energy_relative_weight: float = 0.0,
        energy_floor_watts: float = 10.0,
        aggregate_consistency_weight: float = 0.0,
        aggregate_tolerance_watts: float = 20.0,
        aggregate_loss_scale_watts: float = 1000.0,
        target_mean: torch.Tensor | list[float] | None = None,
        input_mean: float | None = None,
        input_std: float | None = None,
        output_alignment: str = "end",
    ) -> None:
        super().__init__()
        self.lambda_state = float(lambda_state)
        self.task_balance = str(task_balance or "none").lower()
        self.lambda_domain = float(lambda_domain)
        self.domain_method = str(domain_method or "coral").lower()
        self.domain_mu = float(domain_mu)
        self.mmd_sigma = None if mmd_sigma is None else float(mmd_sigma)
        self.domain_mix = str(domain_mix or "convex").lower()
        if self.domain_mix not in {"convex", "additive"}:
            raise ValueError(f"domain_mix must be convex|additive, got {domain_mix!r}")
        self.domain_scale = str(domain_scale or "none").lower()
        if self.domain_scale not in {"none", "equal"}:
            raise ValueError(f"domain_scale must be none|equal, got {domain_scale!r}")
        self.power_on_weight = float(power_on_weight)
        self.power_off_weight = float(power_off_weight)
        self.power_delta_weight = float(power_delta_weight)
        self.power_delta_on_only = bool(power_delta_on_only)
        self.power_energy_weight = float(power_energy_weight)
        self.state_fp_weight = float(state_fp_weight)
        self.state_transition_weight = float(state_transition_weight)
        self.power_energy_relative_weight = float(power_energy_relative_weight)
        self.energy_floor_watts = float(energy_floor_watts)
        self.aggregate_consistency_weight = float(aggregate_consistency_weight)
        self.aggregate_tolerance_watts = float(aggregate_tolerance_watts)
        self.aggregate_loss_scale_watts = max(
            float(aggregate_loss_scale_watts),
            1e-6,
        )
        self.output_alignment = str(output_alignment or "end").lower()

        # MAE logging scale (watts / std); not used in the training objective.
        self.register_buffer("power_scale", torch.as_tensor(power_scale, dtype=torch.float32))
        target_mean_tensor = (
            torch.as_tensor(target_mean, dtype=torch.float32)
            if target_mean is not None
            else torch.zeros_like(self.power_scale)
        )
        self.register_buffer("target_mean", target_mean_tensor)
        self.register_buffer(
            "input_mean",
            torch.tensor(0.0 if input_mean is None else float(input_mean)),
        )
        self.register_buffer(
            "input_std",
            torch.tensor(1.0 if input_std is None else float(input_std)),
        )
        self.has_physical_stats = (
            target_mean is not None
            and input_mean is not None
            and input_std is not None
        )

        # BCE ON-class weight: pos_weight_i = (1−p_i)/p_i from train ON rate.
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

    def _per_appliance_power_loss(
        self,
        power_pred: torch.Tensor,
        power_true: torch.Tensor,
        state_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MSE_i = mean_{b,t} (ŷ − y)²  → vector length A."""
        err2 = (power_pred - power_true) ** 2
        loss = torch.mean(err2, dim=(0, 1))

        if state_true is not None and self.power_on_weight > 0.0:
            on = state_true.float()
            denom = on.sum(dim=(0, 1)).clamp_min(1.0)
            on_mse = (err2 * on).sum(dim=(0, 1)) / denom
            loss = loss + self.power_on_weight * on_mse

        if state_true is not None and self.power_off_weight > 0.0:
            off = (1.0 - state_true.float()).clamp_min(0.0)
            denom = off.sum(dim=(0, 1)).clamp_min(1.0)
            off_mse = (err2 * off).sum(dim=(0, 1)) / denom
            loss = loss + self.power_off_weight * off_mse

        if power_pred.shape[1] > 1 and self.power_delta_weight > 0.0:
            d_pred = power_pred[:, 1:, :] - power_pred[:, :-1, :]
            d_true = power_true[:, 1:, :] - power_true[:, :-1, :]
            d_err2 = (d_pred - d_true) ** 2
            if state_true is not None and self.power_delta_on_only:
                on_delta = torch.maximum(state_true[:, 1:, :], state_true[:, :-1, :]).float()
                denom = on_delta.sum(dim=(0, 1)).clamp_min(1.0)
                delta_loss = (d_err2 * on_delta).sum(dim=(0, 1)) / denom
            else:
                delta_loss = d_err2.mean(dim=(0, 1))
            loss = loss + self.power_delta_weight * delta_loss

        if self.power_energy_weight > 0.0:
            energy_err = torch.abs(power_pred.sum(dim=1) - power_true.sum(dim=1))
            energy_loss = energy_err.mean(dim=0) / max(float(power_pred.shape[1]), 1.0)
            loss = loss + self.power_energy_weight * energy_loss

        return loss

    def _per_appliance_state_loss(
        self,
        state_logits: torch.Tensor,
        state_true: torch.Tensor,
    ) -> torch.Tensor:
        """BCEWithLogits per appliance (optional pos_weight) → vector length A."""
        losses: list[torch.Tensor] = []
        state_prob = torch.sigmoid(state_logits)
        for app_i in range(state_logits.shape[-1]):
            weight_i = None
            if self.pos_weight is not None:
                weight_i = (
                    self.pos_weight[app_i] if self.pos_weight.ndim > 0 else self.pos_weight
                )
            loss_i = F.binary_cross_entropy_with_logits(
                state_logits[..., app_i],
                state_true[..., app_i],
                pos_weight=weight_i,
            )
            if self.state_fp_weight > 0.0:
                off_i = (1.0 - state_true[..., app_i]).clamp_min(0.0)
                denom = off_i.sum().clamp_min(1.0)
                fp_i = (state_prob[..., app_i].pow(2) * off_i).sum() / denom
                loss_i = loss_i + self.state_fp_weight * fp_i
            losses.append(loss_i)
        return torch.stack(losses)

    def _state_transition_loss(
        self,
        state_logits: torch.Tensor,
        state_true: torch.Tensor,
    ) -> torch.Tensor:
        """Balanced start/stop boundary loss for event width and continuity."""
        if state_logits.shape[1] <= 1:
            return state_logits.new_zeros(state_logits.shape[-1])

        prob = torch.sigmoid(state_logits)
        previous = prob[:, :-1, :]
        current = prob[:, 1:, :]
        # Probability that two adjacent Bernoulli states are different.
        boundary_prob = (
            previous * (1.0 - current)
            + (1.0 - previous) * current
        ).clamp(1e-6, 1.0 - 1e-6)
        boundary_true = torch.abs(
            state_true[:, 1:, :] - state_true[:, :-1, :]
        ).float()

        positive = -torch.log(boundary_prob) * boundary_true
        negative = -torch.log1p(-boundary_prob) * (1.0 - boundary_true)
        positive_loss = positive.sum(dim=(0, 1)) / boundary_true.sum(
            dim=(0, 1)
        ).clamp_min(1.0)
        negative_mask = 1.0 - boundary_true
        negative_loss = negative.sum(dim=(0, 1)) / negative_mask.sum(
            dim=(0, 1)
        ).clamp_min(1.0)
        has_boundary = (boundary_true.sum(dim=(0, 1)) > 0).float()
        return 0.5 * positive_loss * has_boundary + 0.5 * negative_loss

    def _to_watts(self, power: torch.Tensor) -> torch.Tensor:
        scale = self.power_scale.to(device=power.device, dtype=power.dtype)
        mean = self.target_mean.to(device=power.device, dtype=power.dtype)
        return (power * scale + mean).clamp_min(0.0)

    def _relative_energy_loss(
        self,
        power_pred: torch.Tensor,
        power_true: torch.Tensor,
    ) -> torch.Tensor:
        """Per-window relative energy error in physical watt-samples."""
        pred_watts = self._to_watts(power_pred)
        true_watts = self._to_watts(power_true)
        pred_energy = pred_watts.sum(dim=1)
        true_energy = true_watts.sum(dim=1)
        floor = self.energy_floor_watts * max(float(power_pred.shape[1]), 1.0)
        return (
            torch.abs(pred_energy - true_energy) / (true_energy + floor)
        ).mean(dim=0)

    def _align_aggregate(
        self,
        aggregate_input: torch.Tensor,
        output_length: int,
    ) -> torch.Tensor:
        aggregate = aggregate_input.float()
        if aggregate.dim() == 3 and aggregate.shape[-1] == 1:
            aggregate = aggregate[..., 0]
        elif aggregate.dim() == 3 and aggregate.shape[1] == 1:
            aggregate = aggregate[:, 0, :]
        if aggregate.dim() != 2:
            raise ValueError(
                "aggregate_input must be (B,T), (B,T,1), or (B,1,T); "
                f"got {tuple(aggregate_input.shape)}"
            )

        time_len = aggregate.shape[1]
        if time_len == output_length:
            return aggregate
        if time_len < output_length:
            return F.pad(aggregate, (output_length - time_len, 0))
        if self.output_alignment == "center":
            offset = (time_len - output_length) // 2
            return aggregate[:, offset : offset + output_length]
        return aggregate[:, -output_length:]

    def _aggregate_consistency_loss(
        self,
        power_pred: torch.Tensor,
        aggregate_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Penalize only impossible over-allocation; unknown load may remain."""
        if aggregate_input is None or not self.has_physical_stats:
            return power_pred.new_zeros(())
        aggregate = self._align_aggregate(aggregate_input, power_pred.shape[1])
        aggregate_watts = (
            aggregate
            * self.input_std.to(device=aggregate.device, dtype=aggregate.dtype)
            + self.input_mean.to(device=aggregate.device, dtype=aggregate.dtype)
        ).clamp_min(0.0)
        predicted_watts = self._to_watts(power_pred).sum(dim=-1)
        excess = F.relu(
            predicted_watts
            - aggregate_watts
            - self.aggregate_tolerance_watts
        )
        return torch.mean((excess / self.aggregate_loss_scale_watts) ** 2)

    def _balanced_state_term(
        self,
        loss_power: torch.Tensor,
        loss_state: torch.Tensor,
    ) -> torch.Tensor:
        """Build state_term that enters L_NILM = L_power + state_term.

        Why balance?
            MSE and BCE live on different numeric scales. Fixed
            L_power + λ L_state with λ=1 is NOT equal importance.

        task_balance=none
            state_term = λ_state · L_state
            (legacy; you must hand-tune λ_state for scale)

        task_balance=equal
            Rescale state magnitude to match power, then apply λ_state as preference:

                state_term = λ_state · L_state · (L_power / L_state)_stop-grad

            With λ_state=1:  state_term = L_power  → equal weights.
            Example: L_power=2, L_state=8 → ratio=0.25 → state_term=2.

            stop-grad on the ratio: only a magnitude ruler; gradients still
            flow through L_state (and L_power via the other term).
        """
        if self.task_balance == "none":
            return self.lambda_state * loss_state
        if self.task_balance == "equal":
            scale = loss_power.detach() / loss_state.detach().clamp_min(1e-8)
            return self.lambda_state * loss_state * scale
        raise ValueError(f"task_balance must be none|equal, got {self.task_balance!r}")

    def forward(
        self,
        power_pred: torch.Tensor,
        state_logits: torch.Tensor,
        power_true: torch.Tensor,
        state_true: torch.Tensor,
        *,
        aggregate_input: torch.Tensor | None = None,
        domain_feats_S: Mapping[str, torch.Tensor] | None = None,
        domain_feats_T: Mapping[str, torch.Tensor] | None = None,
    ) -> MultiNILMLossOutput:
        """One batch: L_NILM (+ optional L_domain).

        Math
        ----
        L_power = Σ_i MSE_i
        L_state = Σ_i BCE_i
        L_NILM  = L_power + state_term          # see _balanced_state_term

        if DA active (feats given and λ_domain ≠ 0):
            optionally scale L_domain to L_NILM magnitude (domain_scale=equal)
            domain_mix=convex (Lin):
                L = (1-λ) · L_NILM + λ · domain_term
            domain_mix=additive (legacy):
                L = L_NILM + λ · domain_term
        else:
            L = L_NILM
        """
        power_pred = power_pred.float()
        state_logits = state_logits.float()
        power_true = power_true.float()
        state_true = state_true.float()

        # --- supervised NILM ---
        loss_power_per_app = self._per_appliance_power_loss(power_pred, power_true, state_true)
        loss_energy_relative_per_app = self._relative_energy_loss(
            power_pred,
            power_true,
        )
        loss_power_per_app = (
            loss_power_per_app
            + self.power_energy_relative_weight * loss_energy_relative_per_app
        )

        loss_state_per_app = self._per_appliance_state_loss(state_logits, state_true)
        loss_state_transition_per_app = self._state_transition_loss(
            state_logits,
            state_true,
        )
        loss_state_per_app = (
            loss_state_per_app
            + self.state_transition_weight * loss_state_transition_per_app
        )
        loss_power = loss_power_per_app.sum()
        loss_state = loss_state_per_app.sum()
        loss_state_term = self._balanced_state_term(loss_power, loss_state)
        loss_aggregate_consistency = self._aggregate_consistency_loss(
            power_pred,
            aggregate_input,
        )
        loss_aggregate_term = (
            self.aggregate_consistency_weight * loss_aggregate_consistency
        )
        loss_nilm = loss_power + loss_state_term + loss_aggregate_term

        # --- optional domain adaptation ---
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
            # Match magnitudes so λ meaningfully trades off NILM vs DA.
            if self.domain_scale == "equal":
                domain_term = loss_domain * (
                    loss_nilm.detach() / loss_domain.detach().clamp_min(1e-8)
                )
            else:
                domain_term = loss_domain
            lam = self.lambda_domain
            if self.domain_mix == "convex":
                # Lin et al.: L = (1-λ) L_R + λ L_domain
                loss = (1.0 - lam) * loss_nilm + lam * domain_term
            else:
                loss = loss_nilm + lam * domain_term
        else:
            loss_domain = loss_nilm.new_zeros(())
            domain_term = loss_domain
            loss = loss_nilm

        # --- MAE for logs only (not in L) ---
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
            loss_state_transition=loss_state_transition_per_app.sum().detach(),
            loss_energy_relative=loss_energy_relative_per_app.sum().detach(),
            loss_aggregate_consistency=loss_aggregate_consistency.detach(),
            loss_domain=loss_domain.detach(),
            loss_domain_term=domain_term.detach(),
            mae=mae,
            loss_power_per_appliance=loss_power_per_app.detach(),
            loss_state_per_appliance=loss_state_per_app.detach(),
        )

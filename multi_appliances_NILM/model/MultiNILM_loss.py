"""MultiNILM loss. Shapes: power/state (B, T, A).

    L_NILM  = L_power + state_term
    L_state = Σ_i (BCE_i + w_fp FP_i + w_smooth SMOOTH_i)
    none : state_term = λ L_state
    equal: state_term = λ L_state (L_power/L_state).detach()   # λ=1 → equal scale
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    loss_energy_relative: torch.Tensor
    mae: torch.Tensor                  # logging only (often denorm scale)
    loss_power_per_appliance: torch.Tensor
    loss_state_per_appliance: torch.Tensor
    loss_state_smooth: torch.Tensor | None = None  # Σ_i SMOOTH_i, unweighted; None when off
    loss_background: torch.Tensor | None = None
    loss_reconstruction: torch.Tensor | None = None
    loss_hard_negative: torch.Tensor | None = None
    loss_hard_negative_per_appliance: torch.Tensor | None = None
    hard_negative_effective_weight: torch.Tensor | None = None


class MultiNILMLoss(nn.Module):
    """Supervised multi-appliance power and state objective."""

    def __init__(
        self,
        lambda_state: float = 1.0,
        pos_weight: torch.Tensor | list[float] | None = None,
        power_scale: float | list[float] | torch.Tensor = 1.0,
        *,
        task_balance: str = "equal",
        power_on_weight: float = 0.0,
        power_off_weight: float = 0.0,
        power_delta_weight: float = 0.0,
        power_delta_on_only: bool = True,
        state_fp_weight: float = 0.0,
        state_smooth_weight: float = 0.0,
        state_smooth_tau: float = 4.0,
        power_energy_relative_weight: float = 0.0,
        energy_floor_watts: float = 10.0,
        target_mean: torch.Tensor | list[float] | None = None,
        background_weight: float = 0.0,
        reconstruction_weight: float = 0.0,
        background_huber_beta: float = 0.1,
        aggregate_mean: float = 0.0,
        aggregate_scale: float = 1.0,
        hard_negative_weight: float = 0.0,
        hard_negative_fraction: float = 0.05,
        hard_negative_appliance_indices: list[int] | None = None,
        hard_negative_warmup_epochs: int = 0,
        hard_negative_ramp_epochs: int = 0,
        hard_negative_exclusion_samples: int = 0,
    ) -> None:
        super().__init__()
        self.lambda_state = float(lambda_state)
        self.task_balance = str(task_balance or "none").lower()
        self.power_on_weight = float(power_on_weight)
        self.power_off_weight = float(power_off_weight)
        self.power_delta_weight = float(power_delta_weight)
        self.power_delta_on_only = bool(power_delta_on_only)
        self.state_fp_weight = float(state_fp_weight)
        self.state_smooth_weight = float(state_smooth_weight)
        self.state_smooth_tau = float(state_smooth_tau)
        self.power_energy_relative_weight = float(power_energy_relative_weight)
        self.energy_floor_watts = float(energy_floor_watts)
        self.background_weight = float(background_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.background_huber_beta = float(background_huber_beta)
        self.hard_negative_weight = float(hard_negative_weight)
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.hard_negative_appliance_indices = tuple(
            int(i) for i in (hard_negative_appliance_indices or [])
        )
        self.hard_negative_warmup_epochs = int(hard_negative_warmup_epochs)
        self.hard_negative_ramp_epochs = int(hard_negative_ramp_epochs)
        self.hard_negative_exclusion_samples = int(hard_negative_exclusion_samples)
        self.current_epoch = 1
        if self.background_weight < 0.0 or self.reconstruction_weight < 0.0:
            raise ValueError("background and reconstruction weights must be non-negative")
        if self.background_huber_beta <= 0.0:
            raise ValueError("background_huber_beta must be positive")
        if float(aggregate_scale) <= 0.0:
            raise ValueError("aggregate_scale must be positive")
        if self.hard_negative_weight < 0.0:
            raise ValueError("hard_negative_weight must be non-negative")
        if not 0.0 < self.hard_negative_fraction <= 1.0:
            raise ValueError("hard_negative_fraction must be in (0, 1]")
        if self.hard_negative_warmup_epochs < 0 or self.hard_negative_ramp_epochs < 0:
            raise ValueError("hard-negative warmup/ramp epochs must be non-negative")
        if self.hard_negative_exclusion_samples < 0:
            raise ValueError("hard_negative_exclusion_samples must be non-negative")
        # MAE logging scale (watts / std); not used in the training objective.
        self.register_buffer("power_scale", torch.as_tensor(power_scale, dtype=torch.float32))
        target_mean_tensor = (
            torch.as_tensor(target_mean, dtype=torch.float32)
            if target_mean is not None
            else torch.zeros_like(self.power_scale)
        )
        self.register_buffer("target_mean", target_mean_tensor)
        self.register_buffer("aggregate_mean", torch.tensor(float(aggregate_mean), dtype=torch.float32))
        self.register_buffer("aggregate_scale", torch.tensor(float(aggregate_scale), dtype=torch.float32))
        # BCE ON-class weight: pos_weight_i = (1−p_i)/p_i from train ON rate.
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

    def set_epoch(self, epoch: int) -> None:
        """Set the 1-based training epoch used by the hard-negative schedule."""
        self.current_epoch = max(1, int(epoch))

    def _hard_negative_effective_weight(self) -> float:
        """Warm up the base model, then linearly introduce hard negatives."""
        if self.hard_negative_weight <= 0.0:
            return 0.0
        epochs_after_warmup = self.current_epoch - self.hard_negative_warmup_epochs
        if epochs_after_warmup <= 0:
            return 0.0
        if self.hard_negative_ramp_epochs <= 0:
            return self.hard_negative_weight
        progress = min(1.0, epochs_after_warmup / self.hard_negative_ramp_epochs)
        return self.hard_negative_weight * progress

    def _hard_negative_loss(
        self,
        state_logits: torch.Tensor,
        state_true: torch.Tensor,
    ) -> torch.Tensor:
        """Top-k negative BCE per selected appliance, returned as an A-vector.

        Only true-OFF positions are eligible. A small temporal guard around true
        ON labels protects transition samples from label/measurement alignment
        error. For a negative label BCEWithLogits(s, 0) = softplus(s), so
        selecting the largest logits is exactly selecting the hardest negatives.
        """
        num_appliances = state_logits.shape[-1]
        losses = state_logits.new_zeros(num_appliances)
        if self.hard_negative_weight <= 0.0 or not self.hard_negative_appliance_indices:
            return losses

        true_on = state_true >= 0.5
        if self.hard_negative_exclusion_samples > 0:
            # max_pool1d grows each ON region on both sides by `margin` samples.
            margin = self.hard_negative_exclusion_samples
            protected = F.max_pool1d(
                true_on.float().permute(0, 2, 1),
                kernel_size=2 * margin + 1,
                stride=1,
                padding=margin,
            ).permute(0, 2, 1) > 0.0
        else:
            protected = true_on

        eligible_off = ~protected
        for app_i in self.hard_negative_appliance_indices:
            if app_i < 0 or app_i >= num_appliances:
                raise ValueError(
                    f"hard-negative appliance index {app_i} is outside A={num_appliances}"
                )
            negative_logits = state_logits[..., app_i][eligible_off[..., app_i]]
            if negative_logits.numel() == 0:
                continue
            k = max(1, math.ceil(self.hard_negative_fraction * negative_logits.numel()))
            hardest_logits = torch.topk(negative_logits, k=k, sorted=False).values
            losses[app_i] = F.softplus(hardest_logits).mean()
        return losses

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

    def _state_smoothing_loss(self, state_logits: torch.Tensor) -> torch.Tensor:
        """MS-TCN truncated MSE on adjacent log-probabilities → vector length A.

        Abu Farha & Gall, CVPR 2019. Each appliance is a two-class (ON, OFF)
        problem, so log p = (log σ(s), log σ(−s)):

            SMOOTH_i = mean_{b,t,c} min(|log p_c[t] − log p_c[t−1]|, τ)²

        As in the official code, the t−1 term is detached. Beyond τ the term is
        constant with zero gradient, so decisive ON/OFF edges are not smoothed.
        """
        log_p = torch.stack((F.logsigmoid(state_logits), F.logsigmoid(-state_logits)), dim=-1)
        delta2 = (log_p[:, 1:] - log_p[:, :-1].detach()) ** 2   # (B, T-1, A, 2)
        return delta2.clamp(max=self.state_smooth_tau ** 2).mean(dim=(0, 1, 3))

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

    def _background_losses(
        self,
        power_pred: torch.Tensor,
        power_true: torch.Tensor,
        background_pred: torch.Tensor,
        aggregate_true: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Supervise unmodelled load and enforce aggregate reconstruction.

        Background and aggregate tensors use aggregate z-score units. Appliance
        tensors use separate per-appliance z-scores, so the physical sum is
        formed in watts and then converted back to aggregate-normalized units.
        """
        if aggregate_true.dim() == 2:
            aggregate_true = aggregate_true.unsqueeze(-1)
        if background_pred.shape != aggregate_true.shape:
            raise ValueError(
                "background prediction and aggregate target must have the same "
                f"shape, got {tuple(background_pred.shape)} and {tuple(aggregate_true.shape)}"
            )

        aggregate_true = aggregate_true.float()
        background_pred = background_pred.float()
        agg_mean = self.aggregate_mean.to(aggregate_true)
        agg_scale = self.aggregate_scale.to(aggregate_true)
        aggregate_watts = aggregate_true * agg_scale + agg_mean
        true_appliance_watts = self._to_watts(power_true)
        background_true_watts = (
            aggregate_watts - true_appliance_watts.sum(dim=-1, keepdim=True)
        ).clamp_min(0.0)
        background_true = (background_true_watts - agg_mean) / agg_scale

        loss_background = F.smooth_l1_loss(
            background_pred,
            background_true,
            beta=self.background_huber_beta,
        )

        predicted_background_watts = (
            background_pred * agg_scale + agg_mean
        ).clamp_min(0.0)
        predicted_total_watts = (
            self._to_watts(power_pred).sum(dim=-1, keepdim=True)
            + predicted_background_watts
        )
        predicted_total = (predicted_total_watts - agg_mean) / agg_scale
        loss_reconstruction = F.smooth_l1_loss(
            predicted_total,
            aggregate_true,
            beta=self.background_huber_beta,
        )
        return loss_background, loss_reconstruction

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
        background_pred: torch.Tensor | None = None,
        aggregate_true: torch.Tensor | None = None,
    ) -> MultiNILMLossOutput:
        """One supervised batch.

        Math
        ----
        L_power = Σ_i MSE_i
        L_state = Σ_i (BCE_i + w_fp FP_i + w_smooth SMOOTH_i)
        L       = L_power + state_term          # see _balanced_state_term
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
        loss_state_smooth_per_app = None
        if self.state_smooth_weight > 0.0 and state_logits.dim() == 3 and state_logits.shape[1] > 1:
            loss_state_smooth_per_app = self._state_smoothing_loss(state_logits)
            loss_state_per_app = loss_state_per_app + self.state_smooth_weight * loss_state_smooth_per_app
        loss_power = loss_power_per_app.sum()
        loss_state = loss_state_per_app.sum()
        loss_state_term = self._balanced_state_term(loss_power, loss_state)
        loss = loss_power + loss_state_term

        # Keep this outside task_balance so adding hard-negative mining does not
        # silently rescale the existing BCE/FP gradients. Average only across
        # the selected appliances so the configured weight has a stable meaning.
        hard_negative_enabled = (
            self.hard_negative_weight > 0.0
            and bool(self.hard_negative_appliance_indices)
        )
        loss_hard_negative = None
        loss_hard_negative_per_app = None
        hard_negative_effective_weight = None
        if hard_negative_enabled:
            loss_hard_negative_per_app = self._hard_negative_loss(state_logits, state_true)
            hard_negative_effective_weight = self._hard_negative_effective_weight()
            selected = list(self.hard_negative_appliance_indices)
            loss_hard_negative = loss_hard_negative_per_app[selected].mean()
            loss = loss + hard_negative_effective_weight * loss_hard_negative

        loss_background = None
        loss_reconstruction = None
        auxiliary_enabled = self.background_weight > 0.0 or self.reconstruction_weight > 0.0
        if auxiliary_enabled:
            if background_pred is None or aggregate_true is None:
                raise ValueError(
                    "background/reconstruction loss requires background_pred and aggregate_true"
                )
            loss_background, loss_reconstruction = self._background_losses(
                power_pred,
                power_true,
                background_pred,
                aggregate_true,
            )
            loss = (
                loss
                + self.background_weight * loss_background
                + self.reconstruction_weight * loss_reconstruction
            )

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
            loss_energy_relative=loss_energy_relative_per_app.sum().detach(),
            mae=mae,
            loss_power_per_appliance=loss_power_per_app.detach(),
            loss_state_per_appliance=loss_state_per_app.detach(),
            loss_state_smooth=(
                None if loss_state_smooth_per_app is None
                else loss_state_smooth_per_app.sum().detach()
            ),
            loss_background=(
                None if loss_background is None else loss_background.detach()
            ),
            loss_reconstruction=(
                None if loss_reconstruction is None else loss_reconstruction.detach()
            ),
            loss_hard_negative=(
                None if loss_hard_negative is None else loss_hard_negative.detach()
            ),
            loss_hard_negative_per_appliance=(
                None
                if loss_hard_negative_per_app is None
                else loss_hard_negative_per_app.detach()
            ),
            hard_negative_effective_weight=(
                None
                if hard_negative_effective_weight is None
                else state_logits.new_tensor(hard_negative_effective_weight).detach()
            ),
        )

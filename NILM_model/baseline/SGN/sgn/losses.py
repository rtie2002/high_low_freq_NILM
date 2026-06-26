import torch
from torch import nn
import torch.nn.functional as F


class SGNLoss(nn.Module):
    """SGN training objective (Shin et al., AAAI 2019).

    Paper Eq. (8) — default when all extra weights are 0 (sgn_ukdale_paper.json):
        L = L_output + L_on

    Paper Eq. (7a) — L_output (whole-network / gated regression loss):
        L_output = (1/T) * sum_t (y_t - p_hat_t * o_hat_t)^2
                 = MSE(y_hat, y)   where y_hat = gated_power, y = target_power

    Paper Eq. (7c) — L_on (classification subnetwork loss):
        L_on = -sum_t [ o_t*log(o_hat_t) + (1-o_t)*log(1-o_hat_t) ]
             = BCE(o_hat, o)       where o = target_on

    Paper Eq. (7b) L_power on raw p_hat alone is NOT used in Eq. (8).

    Optional extensions (sgn_ukdale_cross_house.json, weights > 0):
      L_reg_on, L_gated_on, weighted BCE — not in the original paper.
    """

    def __init__(
        self,
        output_weight: float = 1.0,
        on_weight: float = 1.0,
        label_smoothing: float = 0.0,
        reg_on_weight: float = 0.0,
        gated_on_weight: float = 0.0,
        on_confidence_weight: float = 0.0,
        on_smooth_weight: float = 0.0,
        bce_pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.output_weight = output_weight
        self.on_weight = on_weight
        self.label_smoothing = label_smoothing
        self.reg_on_weight = float(reg_on_weight)
        self.gated_on_weight = float(gated_on_weight)
        self.on_confidence_weight = float(on_confidence_weight)
        self.on_smooth_weight = float(on_smooth_weight)
        self.bce_pos_weight = float(bce_pos_weight)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        target_power: torch.Tensor,
        target_on: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.label_smoothing > 0:
            smooth_on = target_on * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smooth_on = target_on

        # --- L_output (Eq. 7a): MSE between gated prediction and true power ---
        gated = predictions["gated_power"]   # y_hat = p_hat * o_hat
        on_prob = predictions["on_prob"]     # o_hat
        output_error = (gated - target_power).pow(2)  # per-timestep (y - y_hat)^2

        # --- L_on (Eq. 7c): sigmoid cross-entropy on ON/OFF labels ---
        # Trains the classification branch directly (even when gate blocks regression).
        if self.bce_pos_weight > 1.0:
            pos_w = smooth_on.new_tensor(self.bce_pos_weight)
            on_error = F.binary_cross_entropy(
                on_prob, smooth_on,
                weight=torch.where(smooth_on >= 0.5, pos_w, torch.ones_like(smooth_on)),
                reduction="none",
            )
        else:
            on_error = F.binary_cross_entropy(on_prob, smooth_on, reduction="none")

        output_loss = output_error.mean()   # L_output
        on_loss = on_error.mean()           # L_on

        # --- Optional extra terms (off when reg_on_weight / gated_on_weight == 0) ---
        on_mask = target_on >= 0.5
        zero = output_loss.detach() * 0.0

        if on_mask.any():
            reg_on_loss = (
                (predictions["power"] - target_power).pow(2)[on_mask].mean()
                if self.reg_on_weight > 0.0 else zero
            )
            gated_on_loss = (
                output_error[on_mask].mean()
                if self.gated_on_weight > 0.0 else zero
            )
            # Push on_prob → 1.0 during labeled ON (not just above 0.5)
            on_conf_loss = (
                F.mse_loss(on_prob[on_mask], torch.ones_like(on_prob[on_mask]))
                if self.on_confidence_weight > 0.0 else zero
            )
        else:
            reg_on_loss = zero
            gated_on_loss = zero
            on_conf_loss = zero

        # Penalise jagged gated output inside ON periods (encourage rectangular plateaus)
        if self.on_smooth_weight > 0.0 and target_on.ndim >= 2:
            time_dim = -1
            gated_diff = gated.diff(dim=time_dim)
            on_pairs = (target_on[..., 1:] >= 0.5) & (target_on[..., :-1] >= 0.5)
            if on_pairs.any():
                on_smooth_loss = gated_diff[on_pairs].pow(2).mean()
            else:
                on_smooth_loss = zero
        else:
            on_smooth_loss = zero

        reduce_dims = (0,) if output_error.ndim == 2 else (0, 2)
        output_loss_per_appliance = output_error.mean(dim=reduce_dims)
        on_loss_per_appliance = on_error.mean(dim=reduce_dims)

        # Eq. (8) when output_weight=on_weight=1 and all other weights=0:
        #   total = L_output + L_on
        # runner.py calls loss_parts["loss"].backward() on this scalar.
        total = (
            self.output_weight * output_loss
            + self.on_weight * on_loss
            + self.reg_on_weight * reg_on_loss
            + self.gated_on_weight * gated_on_loss
            + self.on_confidence_weight * on_conf_loss
            + self.on_smooth_weight * on_smooth_loss
        )
        # Paper-only breakdown per appliance (excludes optional reg_on / gated_on terms).
        loss_per_appliance = (
            self.output_weight * output_loss_per_appliance
            + self.on_weight * on_loss_per_appliance
        )
        return {
            "loss": total,  # scalar optimized by Adam; also used for val early stopping when metric=total_loss
            "output_loss": output_loss.detach(),
            "on_loss": on_loss.detach(),
            "reg_on_loss": reg_on_loss.detach(),
            "gated_on_loss": gated_on_loss.detach(),
            "on_conf_loss": on_conf_loss.detach(),
            "on_smooth_loss": on_smooth_loss.detach(),
            "output_loss_per_appliance": output_loss_per_appliance.detach(),
            "on_loss_per_appliance": on_loss_per_appliance.detach(),
            "loss_per_appliance": loss_per_appliance.detach(),
        }

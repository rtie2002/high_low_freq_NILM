"""Loss terms used by Stage-2 Neural-Adjoint optimisation."""

import torch


def inverse_optimization_loss(
    predicted_normalized,
    target_normalized,
    candidate_power,
    power_min,
    power_max,
    boundary_weight,
):
    """Return the differentiable inverse loss for every random restart.

    Meter term:
        L_meter = mean_t[(F(x)_t - y_t)^2]

    NILM boundary adaptation of Ren et al. Eq. (5):
        L_bnd = mean_i,t[ReLU((min_i - p_i) / range_i)^2
                         + ReLU((p_i - max_i) / range_i)^2]

    All quantities are averaged per restart. The returned scalar is the mean
    total loss across restarts, allowing Adam to update them in parallel.
    """
    target = target_normalized.unsqueeze(0).expand_as(predicted_normalized)
    meter_per_restart = (predicted_normalized - target).square().mean(dim=1)

    power_range = (power_max - power_min).clamp_min(1.0)
    below_training_range = torch.relu(
        (power_min - candidate_power) / power_range
    )
    above_training_range = torch.relu(
        (candidate_power - power_max) / power_range
    )
    boundary_per_restart = (
        below_training_range.square() + above_training_range.square()
    ).mean(dim=(1, 2))

    total_per_restart = (
        meter_per_restart
        + boundary_weight * boundary_per_restart
    )
    return (
        total_per_restart.mean(),
        meter_per_restart,
        boundary_per_restart,
        total_per_restart,
    )

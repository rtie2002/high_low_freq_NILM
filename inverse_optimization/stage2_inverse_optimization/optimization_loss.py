"""Loss and box-constraint operations for Stage-2 inverse optimisation."""

import torch


def training_power_bounds(training_power, device):
    """Find the five training-data extrema once for the optional soft loss."""
    return (
        training_power.amin(dim=(0, 1)).to(device),
        training_power.amax(dim=(0, 1)).to(device),
    )


def normalized_power_bounds(appliance_min_w, appliance_max_w, x_mean, x_std):
    """Convert configured watt bounds into the LSTM's z-score coordinates.

    The returned tensors have shape [1, 1, 5], so PyTorch applies the five
    appliance-specific limits to every restart and every time step.
    """
    minimum_w = torch.tensor(
        appliance_min_w, dtype=x_mean.dtype, device=x_mean.device
    ).view(1, 1, -1)
    maximum_w = torch.tensor(
        appliance_max_w, dtype=x_mean.dtype, device=x_mean.device
    ).view(1, 1, -1)
    return (minimum_w - x_mean) / x_std, (maximum_w - x_mean) / x_std


def project_candidate(candidate, candidate_min, candidate_max):
    """Project a z-score candidate into the configured watt-space box.

    This is the projected-Adam operation:
        p_i(t) is constrained to [P_i,min, P_i,max]
    while the optimizer continues to store and pass z-scores to the LSTM.
    """
    with torch.no_grad():
        candidate.copy_(
            torch.maximum(torch.minimum(candidate, candidate_max), candidate_min)
        )


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
    # `y` repeats the one observed aggregate waveform for every restart, so
    # predicted_normalized[r] is compared with exactly the same target y.
    y = target_normalized.unsqueeze(0).expand_as(predicted_normalized) if target_normalized.ndim == 1 else target_normalized

    # L_meter[r] = mean_t[(F(x_r)_t - y_t)^2].
    # It measures how closely restart r reproduces the observed aggregate.
    # Shape: [restarts].
    L_meter = (predicted_normalized - y).square().mean(dim=1)

    # `p_range` scales each appliance penalty by its observed training range.
    # clamp_min(1.0) prevents division by zero for an appliance with no change.
    p_range = (power_max - power_min).clamp_min(1.0)

    # `low` is positive only when a candidate appliance power is below the
    # smallest training value; `high` is positive only above the maximum.
    low = torch.relu((power_min - candidate_power) / p_range)
    high = torch.relu((candidate_power - power_max) / p_range)

    # L_bnd[r] = mean_i,t[low^2 + high^2].
    # This discourages physically implausible powers without clipping them,
    # preserving gradients for the optimizer to follow. Shape: [restarts].
    L_bnd = (low.square() + high.square()).mean(dim=(1, 2))

    # L_total[r] is the complete objective for restart r.
    # The boundary term matters only according to boundary_weight.
    # Shape: [restarts].
    L_total = L_meter + boundary_weight * L_bnd

    # L is one scalar: mean_r[L_total(r)]. Calling L.backward() calculates
    # gradients for all restarts, and Adam uses them to update x.
    L = L_total.mean()

    # Return: scalar optimization loss, then the three per-restart diagnostics.
    return (
        L,  # scalar used by L.backward()
        L_meter,  # one meter loss for each restart
        L_bnd, # one boundary penalty for each restart
        L_total, # one complete loss for each restart
    )

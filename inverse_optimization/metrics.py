"""Simple per-appliance regression and ON/OFF metrics for NILM."""

import torch


def _power(estimated_power, true_power):
    """Return power tensors as [all_time_steps, appliances]."""
    estimated = torch.as_tensor(estimated_power, dtype=torch.float32)
    true = torch.as_tensor(true_power, dtype=torch.float32, device=estimated.device)
    if estimated.shape != true.shape or estimated.ndim < 2:
        raise ValueError("Power inputs must have matching shape [..., time, appliances].")
    return estimated.reshape(-1, estimated.shape[-1]), true.reshape(-1, true.shape[-1])


def _state(estimated_power, true_state):
    """Return estimated power and true ON/OFF labels as 2-D tensors."""
    estimated = torch.as_tensor(estimated_power, dtype=torch.float32)
    state = torch.as_tensor(true_state, dtype=torch.bool, device=estimated.device)
    if estimated.shape != state.shape or estimated.ndim < 2:
        raise ValueError("Power and state inputs must have matching shape [..., time, appliances].")
    return estimated.reshape(-1, estimated.shape[-1]), state.reshape(-1, state.shape[-1])


def mae(estimated_power, true_power):
    """Mean absolute error (W) for each appliance."""
    estimated, true = _power(estimated_power, true_power)
    return (estimated - true).abs().mean(dim=0)


def mse(estimated_power, true_power):
    """Mean squared error (W²) for each appliance."""
    estimated, true = _power(estimated_power, true_power)
    return (estimated - true).square().mean(dim=0)


def rmse(estimated_power, true_power):
    """Root mean squared error (W) for each appliance."""
    return mse(estimated_power, true_power).sqrt()


def sae(estimated_power, true_power):
    """Relative signal aggregate error for each appliance.

    SAE = |sum_t(estimated) - sum_t(true)| / sum_t(|true|).
    """
    estimated, true = _power(estimated_power, true_power)
    numerator = (estimated.sum(dim=0) - true.sum(dim=0)).abs()
    denominator = true.abs().sum(dim=0).clamp_min(1e-6)
    return numerator / denominator


def on_mae(estimated_power, true_power, true_state):
    """MAE (W) only at time steps where the true appliance state is ON."""
    estimated, true = _power(estimated_power, true_power)
    _, state = _state(estimated_power, true_state)
    on_count = state.sum(dim=0)
    error = (estimated - true).abs()
    return torch.where(
        on_count > 0,
        (error * state).sum(dim=0) / on_count.clamp_min(1),
        torch.full_like(error.mean(dim=0), float("nan")),
    )


def off_mae(estimated_power, true_power, true_state):
    """MAE (W) only at time steps where the true appliance state is OFF."""
    estimated, true = _power(estimated_power, true_power)
    _, state = _state(estimated_power, true_state)
    off_state = ~state
    off_count = off_state.sum(dim=0)
    error = (estimated - true).abs()
    return torch.where(
        off_count > 0,
        (error * off_state).sum(dim=0) / off_count.clamp_min(1),
        torch.full_like(error.mean(dim=0), float("nan")),
    )


def average_precision(estimated_power, true_state):
    """Ranking average precision for each appliance, using power as the score."""
    estimated, state = _state(estimated_power, true_state)
    values = []
    for index in range(estimated.shape[1]):
        target = state[:, index].float()
        positives = target.sum()
        if positives == 0:
            values.append(torch.tensor(float("nan"), device=estimated.device))
            continue
        order = estimated[:, index].clamp_min(0).argsort(descending=True)
        ranked_target = target[order]
        rank = torch.arange(1, len(ranked_target) + 1, device=estimated.device)
        precision_at_rank = ranked_target.cumsum(dim=0) / rank
        values.append((precision_at_rank * ranked_target).sum() / positives)
    return torch.stack(values)


def _classification_counts(estimated_power, true_state, on_thresholds):
    """Return true positives, false positives, and false negatives per appliance."""
    estimated, state = _state(estimated_power, true_state)
    thresholds = torch.as_tensor(on_thresholds, dtype=estimated.dtype, device=estimated.device)
    if thresholds.ndim != 1 or len(thresholds) != estimated.shape[1]:
        raise ValueError("on_thresholds must contain one value per appliance.")
    predicted_state = estimated >= thresholds
    true_positive = (predicted_state & state).sum(dim=0).float()
    false_positive = (predicted_state & ~state).sum(dim=0).float()
    false_negative = (~predicted_state & state).sum(dim=0).float()
    return true_positive, false_positive, false_negative


def precision(estimated_power, true_state, on_thresholds):
    """ON/OFF precision for each appliance."""
    true_positive, false_positive, _ = _classification_counts(
        estimated_power, true_state, on_thresholds
    )
    return true_positive / (true_positive + false_positive).clamp_min(1.0)


def recall(estimated_power, true_state, on_thresholds):
    """ON/OFF recall for each appliance."""
    true_positive, _, false_negative = _classification_counts(
        estimated_power, true_state, on_thresholds
    )
    return true_positive / (true_positive + false_negative).clamp_min(1.0)


def f1(estimated_power, true_state, on_thresholds):
    """ON/OFF F1 score for each appliance."""
    value_precision = precision(estimated_power, true_state, on_thresholds)
    value_recall = recall(estimated_power, true_state, on_thresholds)
    return 2 * value_precision * value_recall / (value_precision + value_recall).clamp_min(1e-6)

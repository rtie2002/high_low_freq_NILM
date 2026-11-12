"""Stage-1 supervised losses. Inputs are scaled power [B, 5, T]."""

import torch
from torch import Tensor


def reconstruction_mse(prediction: Tensor, target: Tensor) -> Tensor:
    """L_rec = mean((prediction - target)^2) across batch, appliance and time."""
    return (prediction - target).square().mean()


def event_balanced_mse(prediction: Tensor, target: Tensor, thresholds: Tensor) -> Tensor:
    """Average each appliance's ON and OFF MSE with equal state weight."""
    error = (prediction - target).square()
    on_mask = target >= thresholds[None, :, None]
    per_appliance = []
    for appliance in range(target.shape[1]):
        states = []
        if bool(on_mask[:, appliance].any()):
            states.append(error[:, appliance][on_mask[:, appliance]].mean())
        if bool((~on_mask[:, appliance]).any()):
            states.append(error[:, appliance][~on_mask[:, appliance]].mean())
        per_appliance.append(torch.stack(states).mean())
    return torch.stack(per_appliance).mean()


def inverse_recovery_loss(
    recovered: Tensor,
    target: Tensor,
    *,
    name: str,
    thresholds: Tensor,
) -> Tensor:
    """Apply the configured supervision to waveforms recovered by unrolling."""
    if name == "mse":
        return reconstruction_mse(recovered, target)
    if name == "event_balanced_mse":
        return event_balanced_mse(recovered, target, thresholds)
    raise ValueError(f"Unsupported inverse recovery loss: {name}")

"""Measurement-model terms for Stage 2 latent inverse optimisation."""

from torch import Tensor


def estimate_constant_background(appliance_power: Tensor, observed_meter: Tensor) -> Tensor:
    """b*(z)=max(0, mean_t(y-s(z))) for appliance power [B,5,T]."""
    return (observed_meter - appliance_power.sum(dim=1)).mean(dim=-1).clamp_min(0)


def meter_mse(predicted_meter: Tensor, observed_meter: Tensor) -> Tensor:
    """Mean squared aggregate mismatch for each window, shape [B]."""
    return (predicted_meter - observed_meter).square().mean(dim=-1)


def diagonal_latent_prior(codes: Tensor, center: Tensor, spread: Tensor) -> Tensor:
    """Optional diagonal-Gaussian penalty per window, shape [B]."""
    return ((codes - center) / spread).square().mean(dim=(1, 2, 3))

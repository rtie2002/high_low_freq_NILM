"""Paper-style sequential subtraction for MultiNILM-Fractional.

This model follows the NNAN-style idea more directly:

    residual_0 = aggregate
    for appliance in order:
        predict appliance from current residual
        subtract confident predicted appliance in watts
        feed the remaining residual to the next appliance stage

Each stage is a single-appliance MultiNILM-Fractional model. Outputs are
returned in the original experiment appliance order.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.MultiNILM_fractional import MultiNILMFractional, build_multinilm_fractional


class MultiNILMFractionalCascade(nn.Module):
    """Sequential one-by-one appliance prediction with residual subtraction."""

    def __init__(
        self,
        *,
        stages: list[MultiNILMFractional],
        order_indices: list[int],
        aggregate_mean: float,
        aggregate_std: float,
        appliance_mean: list[float],
        appliance_std: list[float],
        beta: float = 0.8,
        confidence_mode: str = "soft",
        confidence_threshold: float = 0.5,
        clamp_watts: bool = True,
    ) -> None:
        super().__init__()
        if not stages:
            raise ValueError("cascade requires at least one stage")
        if len(stages) != len(order_indices):
            raise ValueError("stages length must match order_indices length")
        self.stages = nn.ModuleList(stages)
        self.order_indices = [int(i) for i in order_indices]
        self.num_appliances = len(order_indices)
        self.output_length = int(stages[0].output_length)
        self.input_channels = 1
        self.feature_channels = int(stages[0].feature_channels)
        self.domain_feature_layers = stages[0].domain_feature_layers

        if sorted(self.order_indices) != list(range(self.num_appliances)):
            raise ValueError(
                "cascade order_indices must contain each appliance index exactly once"
            )
        if len(appliance_mean) != self.num_appliances or len(appliance_std) != self.num_appliances:
            raise ValueError("appliance_mean/appliance_std length must match appliance count")

        self.register_buffer("aggregate_mean", torch.tensor(float(aggregate_mean)))
        self.register_buffer("aggregate_std", torch.tensor(float(aggregate_std)).clamp_min(1e-6))
        self.register_buffer("appliance_mean", torch.tensor(appliance_mean, dtype=torch.float32))
        self.register_buffer("appliance_std", torch.tensor(appliance_std, dtype=torch.float32).clamp_min(1e-6))

        self.beta = float(beta)
        self.confidence_mode = str(confidence_mode or "soft").lower()
        self.confidence_threshold = float(confidence_threshold)
        self.clamp_watts = bool(clamp_watts)

    def _to_bt(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.float()
        if x.dim() == 3 and x.shape[1] == 1:
            return x[:, 0, :].float()
        if x.dim() == 3 and x.shape[-1] == 1:
            return x[:, :, 0].float()
        raise ValueError(
            "MultiNILMFractionalCascade expects (B,T), (B,1,T), or (B,T,1); "
            f"got {tuple(x.shape)}"
        )

    def _confidence(self, state_logits: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(state_logits)
        if self.confidence_mode in {"soft", "prob", "probability"}:
            return prob
        if self.confidence_mode in {"hard", "binary", "threshold"}:
            hard = (prob >= self.confidence_threshold).to(dtype=prob.dtype)
            if self.training and prob.requires_grad:
                return hard - prob.detach() + prob
            return hard
        raise ValueError("cascade confidence_mode must be soft or hard")

    def _norm_aggregate_to_watts(self, x_norm: torch.Tensor) -> torch.Tensor:
        mean = self.aggregate_mean.to(x_norm.device, x_norm.dtype)
        std = self.aggregate_std.to(x_norm.device, x_norm.dtype)
        return x_norm * std + mean

    def _watts_to_norm_aggregate(self, x_watts: torch.Tensor) -> torch.Tensor:
        mean = self.aggregate_mean.to(x_watts.device, x_watts.dtype)
        std = self.aggregate_std.to(x_watts.device, x_watts.dtype)
        return (x_watts - mean) / std

    def _single_power_to_watts(self, power_norm: torch.Tensor, app_idx: int) -> torch.Tensor:
        mean = self.appliance_mean[app_idx].to(power_norm.device, power_norm.dtype)
        std = self.appliance_std[app_idx].to(power_norm.device, power_norm.dtype)
        watts = power_norm * std + mean
        return watts.clamp_min(0.0) if self.clamp_watts else watts

    def forward(self, x: torch.Tensor, return_domain_features: bool = False):
        residual_norm = self._to_bt(x)
        residual_watts = self._norm_aggregate_to_watts(residual_norm)
        power_by_app: list[torch.Tensor | None] = [None] * self.num_appliances
        state_by_app: list[torch.Tensor | None] = [None] * self.num_appliances
        domain_feats: dict[str, torch.Tensor] = {}

        for stage_i, (stage, app_idx) in enumerate(zip(self.stages, self.order_indices)):
            if return_domain_features:
                power_i, state_i, feats_i = stage(residual_norm, return_domain_features=True)
                for key, value in feats_i.items():
                    domain_feats[f"stage_{stage_i}_app_{app_idx}_{key}"] = value
            else:
                power_i, state_i = stage(residual_norm)

            power_by_app[app_idx] = power_i
            state_by_app[app_idx] = state_i

            power_watts = self._single_power_to_watts(power_i, app_idx).squeeze(-1)
            confidence = self._confidence(state_i).squeeze(-1)
            residual_watts = residual_watts - self.beta * confidence * power_watts
            residual_norm = self._watts_to_norm_aggregate(residual_watts)

        if any(part is None for part in power_by_app) or any(part is None for part in state_by_app):
            raise RuntimeError("cascade did not produce every appliance output")

        power_out = torch.cat([part for part in power_by_app if part is not None], dim=-1)
        state_out = torch.cat([part for part in state_by_app if part is not None], dim=-1)
        if return_domain_features:
            return power_out, state_out, domain_feats
        return power_out, state_out


def build_multinilm_fractional_cascade(
    architecture: dict[str, Any],
    *,
    num_appliances: int,
    output_length: int,
    order_indices: list[int],
    appliance_off_norm: list[float],
    aggregate_mean: float,
    aggregate_std: float,
    appliance_mean: list[float],
    appliance_std: list[float],
) -> MultiNILMFractionalCascade:
    """Build one single-appliance fractional model per cascade stage."""
    cascade_cfg = architecture.get("cascade", {})
    if not isinstance(cascade_cfg, dict):
        cascade_cfg = {}

    base_arch = dict(architecture)
    base_arch.pop("cascade", None)
    # Cross-appliance distillation is not meaningful inside a one-appliance stage.
    base_arch["cross_appliance"] = {"enabled": False}

    stages: list[MultiNILMFractional] = []
    for app_idx in order_indices:
        stages.append(
            build_multinilm_fractional(
                base_arch,
                num_appliances=1,
                output_length=output_length,
                appliance_off_norm=[appliance_off_norm[app_idx]],
            )
        )

    return MultiNILMFractionalCascade(
        stages=stages,
        order_indices=order_indices,
        aggregate_mean=aggregate_mean,
        aggregate_std=aggregate_std,
        appliance_mean=appliance_mean,
        appliance_std=appliance_std,
        beta=float(cascade_cfg.get("beta", 0.8)),
        confidence_mode=str(cascade_cfg.get("confidence_mode", "soft")),
        confidence_threshold=float(cascade_cfg.get("confidence_threshold", 0.5)),
        clamp_watts=bool(cascade_cfg.get("clamp_watts", True)),
    )

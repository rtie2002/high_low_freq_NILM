"""MultiNILM-Fractional with all-appliance residual refinement.

This keeps the original MultiNILM-Fractional model as stage 1, then builds an
appliance-specific residual view for every appliance:

    R_a = X - beta * sum_{b != a} C_b * Y_b

Subtraction is done in watts, then each residual is normalized back to aggregate
space and passed through a small appliance refiner. The base model still runs
only once.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.MultiNILM import state_gate
from model.MultiNILM_fractional import MultiNILMFractional, build_multinilm_fractional


class ResidualRefinerHead(nn.Module):
    """Small temporal head that refines one appliance from its residual view."""

    def __init__(
        self,
        *,
        input_channels: int = 4,
        hidden_channels: int = 32,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.05,
        correction_scale: float = 0.5,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError(f"refiner kernel_size must be odd positive, got {k}")
        hidden = int(hidden_channels)
        layers: list[nn.Module] = [
            nn.Conv1d(int(input_channels), hidden, kernel_size=k, padding=k // 2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        ]
        for _ in range(max(0, int(num_layers) - 1)):
            layers.extend(
                [
                    nn.Conv1d(hidden, hidden, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(dropout)),
                ]
            )
        self.body = nn.Sequential(*layers)
        self.delta = nn.Conv1d(hidden, 1, kernel_size=1)
        self.correction_scale = float(correction_scale)

        # Start as "almost baseline": final power begins near initial prediction.
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, z: torch.Tensor, initial_power: torch.Tensor) -> torch.Tensor:
        correction = self.delta(self.body(z))
        return initial_power + self.correction_scale * correction


class MultiNILMFractionalResidual(nn.Module):
    """Two-stage model: MultiNILM-Fractional first, residual refiners second."""

    def __init__(
        self,
        *,
        base: MultiNILMFractional,
        aggregate_mean: float = 0.0,
        aggregate_std: float = 1.0,
        appliance_mean: list[float] | None = None,
        appliance_std: list[float] | None = None,
        residual_beta: float = 0.8,
        confidence_mode: str = "soft",
        confidence_threshold: float = 0.5,
        detach_subtraction: bool = True,
        clamp_watts: bool = True,
        refiner_hidden_channels: int = 32,
        refiner_layers: int = 3,
        refiner_kernel_size: int = 5,
        refiner_dropout: float = 0.05,
        correction_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.base = base
        self.num_appliances = int(base.num_appliances)
        self.output_length = int(base.output_length)
        self.input_channels = int(base.input_channels)
        self.feature_channels = int(base.feature_channels)
        self.domain_feature_layers = base.domain_feature_layers

        app_mean = appliance_mean if appliance_mean is not None else [0.0] * self.num_appliances
        app_std = appliance_std if appliance_std is not None else [1.0] * self.num_appliances
        if len(app_mean) != self.num_appliances or len(app_std) != self.num_appliances:
            raise ValueError(
                "appliance_mean/appliance_std length must match num_appliances "
                f"({self.num_appliances})"
            )

        self.register_buffer("aggregate_mean", torch.tensor(float(aggregate_mean)))
        self.register_buffer("aggregate_std", torch.tensor(float(aggregate_std)).clamp_min(1e-6))
        self.register_buffer("appliance_mean", torch.tensor(app_mean, dtype=torch.float32))
        self.register_buffer("appliance_std", torch.tensor(app_std, dtype=torch.float32).clamp_min(1e-6))

        self.residual_beta = float(residual_beta)
        self.confidence_mode = str(confidence_mode or "soft").lower()
        self.confidence_threshold = float(confidence_threshold)
        self.detach_subtraction = bool(detach_subtraction)
        self.clamp_watts = bool(clamp_watts)

        self.refiners = nn.ModuleList(
            [
                ResidualRefinerHead(
                    input_channels=4,
                    hidden_channels=int(refiner_hidden_channels),
                    num_layers=int(refiner_layers),
                    kernel_size=int(refiner_kernel_size),
                    dropout=float(refiner_dropout),
                    correction_scale=float(correction_scale),
                )
                for _ in range(self.num_appliances)
            ]
        )

    def _to_bt(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.float()
        if x.dim() == 3 and x.shape[1] == 1:
            return x[:, 0, :].float()
        if x.dim() == 3 and x.shape[-1] == 1:
            return x[:, :, 0].float()
        raise ValueError(
            "MultiNILMFractionalResidual expects aggregate as (B,T), (B,1,T), or (B,T,1); "
            f"got {tuple(x.shape)}"
        )

    def _align_aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """Center align aggregate to the base output length."""
        x_bt = self._to_bt(x)
        time_len = x_bt.shape[-1]
        if time_len == self.output_length:
            return x_bt
        if time_len > self.output_length:
            offset = (time_len - self.output_length) // 2
            return x_bt[:, offset : offset + self.output_length]
        pad_total = self.output_length - time_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return F.pad(x_bt, (pad_left, pad_right))

    def _confidence(self, state_logits: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(state_logits)
        if self.confidence_mode in {"soft", "prob", "probability"}:
            return prob
        if self.confidence_mode in {"hard", "binary", "threshold"}:
            hard = (prob >= self.confidence_threshold).to(dtype=prob.dtype)
            if self.training and prob.requires_grad:
                return hard - prob.detach() + prob
            return hard
        raise ValueError("residual confidence_mode must be soft or hard")

    def _norm_power_to_watts(self, power_norm: torch.Tensor) -> torch.Tensor:
        mean = self.appliance_mean.view(1, 1, -1).to(power_norm.device, power_norm.dtype)
        std = self.appliance_std.view(1, 1, -1).to(power_norm.device, power_norm.dtype)
        watts = power_norm * std + mean
        return watts.clamp_min(0.0) if self.clamp_watts else watts

    def _aggregate_norm_to_watts(self, x_norm: torch.Tensor) -> torch.Tensor:
        mean = self.aggregate_mean.to(x_norm.device, x_norm.dtype)
        std = self.aggregate_std.to(x_norm.device, x_norm.dtype)
        return x_norm * std + mean

    def _aggregate_watts_to_norm(self, x_watts: torch.Tensor) -> torch.Tensor:
        mean = self.aggregate_mean.to(x_watts.device, x_watts.dtype)
        std = self.aggregate_std.to(x_watts.device, x_watts.dtype)
        return (x_watts - mean) / std

    def _build_residual_inputs(
        self,
        x: torch.Tensor,
        power_init: torch.Tensor,
        state_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return residuals plus aligned aggregate/state probability.

        Shapes:
            aggregate_norm : (B,T)
            residual_norm  : (B,T,A)
            state_prob     : (B,T,A)
        """
        aggregate_norm = self._align_aggregate(x)
        aggregate_watts = self._aggregate_norm_to_watts(aggregate_norm).unsqueeze(-1)
        power_for_residual = power_init.detach() if self.detach_subtraction else power_init
        state_for_residual = state_logits.detach() if self.detach_subtraction else state_logits
        power_watts = self._norm_power_to_watts(power_for_residual)
        confidence = self._confidence(state_for_residual)
        confident_power = confidence * power_watts

        total_other_base = confident_power.sum(dim=-1, keepdim=True)
        other_power = total_other_base - confident_power
        residual_watts = aggregate_watts - self.residual_beta * other_power
        residual_norm = self._aggregate_watts_to_norm(residual_watts)
        return aggregate_norm, residual_norm, torch.sigmoid(state_logits)

    def _refine_power(
        self,
        x: torch.Tensor,
        power_init: torch.Tensor,
        state_logits: torch.Tensor,
    ) -> torch.Tensor:
        aggregate_norm, residual_norm, state_prob = self._build_residual_inputs(
            x, power_init, state_logits
        )
        refined_parts: list[torch.Tensor] = []
        for app_i, refiner in enumerate(self.refiners):
            z_i = torch.stack(
                [
                    aggregate_norm,
                    residual_norm[:, :, app_i],
                    power_init[:, :, app_i],
                    state_prob[:, :, app_i],
                ],
                dim=1,
            )
            initial_i = power_init[:, :, app_i].unsqueeze(1)
            refined_raw = refiner(z_i, initial_i)
            gate = state_gate(
                state_prob[:, :, app_i].unsqueeze(1),
                mode=self.base.backbone.gate_mode,
                threshold=self.base.backbone.gate_threshold,
                training=self.training,
            )
            off = self.base.backbone.appliance_heads[app_i].off_norm.to(
                refined_raw.device, refined_raw.dtype
            )
            refined_i = gate * refined_raw + (1.0 - gate) * off
            refined_parts.append(refined_i)
        return torch.cat(refined_parts, dim=1).permute(0, 2, 1)

    def forward(self, x: torch.Tensor, return_domain_features: bool = False):
        if return_domain_features:
            power_init, state_logits, feats = self.base(
                x, return_domain_features=True
            )
            power_final = self._refine_power(x, power_init, state_logits)
            return power_final, state_logits, feats
        power_init, state_logits = self.base(x)
        power_final = self._refine_power(x, power_init, state_logits)
        return power_final, state_logits


def build_multinilm_fractional_residual(
    architecture: dict[str, Any],
    *,
    num_appliances: int,
    output_length: int,
    appliance_off_norm: list[float] | None = None,
    aggregate_mean: float = 0.0,
    aggregate_std: float = 1.0,
    appliance_mean: list[float] | None = None,
    appliance_std: list[float] | None = None,
) -> MultiNILMFractionalResidual:
    """Build the residual-refinement variant from yaml architecture."""
    residual_cfg = architecture.get("residual_refinement", {})
    if not isinstance(residual_cfg, dict):
        residual_cfg = {}

    base_arch = dict(architecture)
    base_arch.pop("residual_refinement", None)
    base = build_multinilm_fractional(
        base_arch,
        num_appliances=num_appliances,
        output_length=output_length,
        appliance_off_norm=appliance_off_norm,
    )

    return MultiNILMFractionalResidual(
        base=base,
        aggregate_mean=aggregate_mean,
        aggregate_std=aggregate_std,
        appliance_mean=appliance_mean,
        appliance_std=appliance_std,
        residual_beta=float(residual_cfg.get("beta", 0.8)),
        confidence_mode=str(residual_cfg.get("confidence_mode", "soft")),
        confidence_threshold=float(residual_cfg.get("confidence_threshold", 0.5)),
        detach_subtraction=bool(residual_cfg.get("detach_subtraction", True)),
        clamp_watts=bool(residual_cfg.get("clamp_watts", True)),
        refiner_hidden_channels=int(residual_cfg.get("hidden_channels", 32)),
        refiner_layers=int(residual_cfg.get("layers", 3)),
        refiner_kernel_size=int(residual_cfg.get("kernel_size", 5)),
        refiner_dropout=float(residual_cfg.get("dropout", 0.05)),
        correction_scale=float(residual_cfg.get("correction_scale", 0.5)),
    )

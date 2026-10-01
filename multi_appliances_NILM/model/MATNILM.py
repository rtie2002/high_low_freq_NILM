"""MAT-Conv from the official MATNilm repository.

Paper:
    J. Xiong, T. Hong, D. Zhao, and Y. Zhang, "MATNilm: Multi-Appliance-Task
    Non-Intrusive Load Monitoring With Limited Labeled Data", IEEE Trans.
    Industrial Informatics, 2024. DOI: 10.1109/TII.2023.3301026

Code:
    https://github.com/jxiong22/MATNilm/blob/master/modules.py

The network is the released MATconv: a shared length-preserving convolution
encoder, three appliance blocks, and a last-block split into regression and
classification feed-forward branches. Final power is the regression head
multiplied by the on-off sigmoid. The released code builds four REDD branches
and marks the appliance count with a ``TODO: change 4``. This port uses one
identical branch per appliance, so the UK-DALE/REFIT experiments can pass the
same five channels as the other models: kettle, fridge, dishwasher, washing
machine, and microwave.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from data.common import BaseNILMAdapter, StepOutput
from model.MATNILM_loss import MATNILMLoss


def _paper_config(architecture: dict[str, Any], num_appliances: int) -> dict[str, Any]:
    hidden = int(architecture.get("hidden", 32))
    channels = [int(v) for v in architecture.get("conv_channels", [30, 30, 40, 50, 50])]
    kernels = [int(v) for v in architecture.get("conv_kernels", [10, 8, 6, 5, 5, 5])]
    if len(kernels) != len(channels) + 1:
        raise ValueError("conv_kernels must contain one kernel for every conv, including the final 2*hidden layer")
    return {
        "hidden": hidden,
        "dropout": float(architecture.get("dropout", 0.1)),
        "heads": int(architecture.get("attention_heads", 2)),
        "feedforward": int(architecture.get("feedforward", 1024)),
        "num_blocks": int(architecture.get("num_blocks", 3)),
        "channels": channels,
        "kernels": kernels,
        "power_scale": float(architecture.get("power_scale", 612.0)),
        "on_threshold_watts": float(architecture.get("on_threshold_watts", 15.0)),
        "output_length": int(architecture.get("output_length", 64)),
        "num_appliances": num_appliances,
    }


class ApplSA(nn.Module):
    """Temporal self-attention used inside one appliance branch."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.dropout(self.self_attn(x, x, x)[0])
        return self.norm(x + attended)


class ApplFF(nn.Module):
    """Position-wise feed-forward block, Equation (9)."""

    def __init__(self, width: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(width, feedforward)
        self.linear2 = nn.Linear(feedforward, width)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        updated = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        return self.norm(x + self.dropout2(updated))


class ApplBlock(nn.Module):
    """One decoder block: temporal attention, appliance attention, then FF.

    The last block also keeps a second feed-forward branch for classification.
    That branch reads the attention output, matching Equation (10).
    """

    def __init__(
        self,
        width: int,
        heads: int,
        feedforward: int,
        dropout: float,
        num_appliances: int,
        *,
        last: bool,
    ) -> None:
        super().__init__()
        self.last = last
        self.num_appliances = num_appliances
        self.temporal = nn.ModuleList(
            ApplSA(width, heads, dropout) for _ in range(num_appliances)
        )
        self.appliance_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.regression = nn.ModuleList(
            ApplFF(width, feedforward, dropout) for _ in range(num_appliances)
        )
        self.classification = (
            nn.ModuleList(ApplFF(width, feedforward, dropout) for _ in range(num_appliances))
            if last
            else None
        )

    def forward(self, features: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        if len(features) != self.num_appliances:
            raise ValueError(f"MAT-Conv expects {self.num_appliances} appliance streams")

        local = [block(feature) for block, feature in zip(self.temporal, features)]
        # (B, T, A, D) -> (B*T, A, D), the appliance axis used by modules.py.
        stacked = torch.stack(local, dim=2)
        batch, time, appliances, width = stacked.shape
        appliance_tokens = stacked.reshape(batch * time, appliances, width)
        mixed, _ = self.appliance_attn(appliance_tokens, appliance_tokens, appliance_tokens)
        mixed = mixed.reshape(batch, time, appliances, width)

        attended = [
            self.norm(mixed[:, :, index, :] + local[index])
            for index in range(appliances)
        ]
        regression = [block(feature) for block, feature in zip(self.regression, attended)]
        if self.classification is None:
            return regression, None
        classification = [block(feature) for block, feature in zip(self.classification, attended)]
        return regression, classification


class MATconv(nn.Module):
    """Shared-convolution MAT-Conv network from modules.MATconv."""

    def __init__(self, architecture: dict[str, Any], num_appliances: int) -> None:
        super().__init__()
        if num_appliances < 1:
            raise ValueError("num_appliances must be at least 1")
        cfg = _paper_config(architecture, num_appliances)
        self.num_appliances = num_appliances
        if cfg["num_blocks"] < 1:
            raise ValueError("num_blocks must be at least 1")
        self.output_length = cfg["output_length"]
        hidden = cfg["hidden"]
        width = 2 * hidden
        dropout = cfg["dropout"]

        layers: list[nn.Module] = []
        in_channels = 1
        channel_list = [*cfg["channels"], width]
        for out_channels, kernel in zip(channel_list, cfg["kernels"]):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding="same"))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.shared = nn.Sequential(*layers)

        self.blocks = nn.ModuleList(
            ApplBlock(
                width,
                cfg["heads"],
                cfg["feedforward"],
                dropout,
                num_appliances,
                last=(index == cfg["num_blocks"] - 1),
            )
            for index in range(cfg["num_blocks"])
        )
        self.regression_heads = nn.ModuleList(_output_head(width, hidden) for _ in range(num_appliances))
        self.classification_heads = nn.ModuleList(_output_head(width, hidden) for _ in range(num_appliances))

    def forward(self, aggregate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict gated power and on-off probability.

        Args:
            aggregate: Mains window, shape ``(B, T, 1)``, in watts/612.

        Returns:
            power: Gated power, shape ``(B, T_out, A)``.
            state: Sigmoid probability, shape ``(B, T_out, A)``.
        """
        if aggregate.ndim != 3 or aggregate.shape[-1] != 1:
            raise ValueError("aggregate must have shape (batch, time, 1)")

        encoded = self.shared(aggregate.transpose(1, 2)).transpose(1, 2)
        streams = [encoded for _ in range(self.num_appliances)]
        classification = None
        for block in self.blocks:
            streams, classification = block(streams)
        if classification is None:
            raise RuntimeError("the last MAT-Conv block must return a classification branch")

        # Center crop implements the paper's context window w = (T - T_out) / 2.
        streams = [_center_crop(feature, self.output_length) for feature in streams]
        classification = [_center_crop(feature, self.output_length) for feature in classification]

        state = torch.cat(
            [torch.sigmoid(head(feature)) for head, feature in zip(self.classification_heads, classification)],
            dim=2,
        )
        power = torch.cat(
            [head(feature) for head, feature in zip(self.regression_heads, streams)],
            dim=2,
        )
        return power * state, state


def _output_head(width: int, hidden: int) -> nn.Sequential:
    """Two-layer head used by every regression and classification output."""
    return nn.Sequential(
        nn.Linear(width, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def _center_crop(feature: torch.Tensor, output_length: int) -> torch.Tensor:
    time = feature.shape[1]
    if time == output_length:
        return feature
    if output_length > time:
        raise ValueError(f"output length {output_length} is longer than the encoded window {time}")
    start = (time - output_length) // 2
    return feature[:, start : start + output_length, :]


class MATNILMAdapter(BaseNILMAdapter):
    """Connect MATconv to the shared runner, in the paper's /612 units."""

    name = "mat_nilm"

    def build_model(self, device: torch.device) -> torch.nn.Module:
        architecture = dict(self.model_cfg["architecture"])
        architecture["output_length"] = int(self.model_cfg["windowing"]["output_window_length"])
        return MATconv(architecture, num_appliances=len(self.cfg["appliances"])).to(device)

    def build_loss(self) -> MATNILMLoss:
        return MATNILMLoss()

    def step(self, model: MATconv, loss_fn: MATNILMLoss, batch) -> StepOutput:
        aggregate, power_target, _state_target = batch
        paper_input, paper_power, paper_state = self._paper_batch(aggregate, power_target)
        power_pred, state_pred = model(paper_input)
        output = loss_fn(power_pred, state_pred, paper_power, paper_state)

        logs = {
            "loss": float(output.loss.detach()),
            "loss_power": float(output.loss_power.detach()),
            "loss_state": float(output.loss_state.detach()),
        }
        for index, appliance in enumerate(self.cfg["appliances"]):
            appliance_loss = F.mse_loss(power_pred[:, :, index], paper_power[:, :, index])
            appliance_loss = appliance_loss + F.binary_cross_entropy(
                state_pred[:, :, index],
                paper_state[:, :, index],
            )
            logs[f"loss_{appliance}"] = float(appliance_loss.detach())

        return StepOutput(
            loss=output.loss,
            logs=logs,
            aux={
                "pred_state": (state_pred.detach() >= 0.5).to(torch.int32).cpu(),
                "state_prob": state_pred.detach().cpu(),
                "true_state": paper_state.detach().cpu(),
                "pred_power": self._to_loader_units(power_pred.detach()).cpu(),
                "true_power": power_target.detach().cpu(),
            },
        )

    @torch.no_grad()
    def predict_dataloader(
        self,
        model: MATconv,
        loader: DataLoader,
        device,
        *,
        max_batches=None,
        split="test",
    ):
        model.eval()
        pred_power = []
        pred_state = []
        true_power = []
        true_state = []
        sample_indices = []
        offset = 0

        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            aggregate, power_target, _state_target = batch
            paper_input, _paper_power, paper_state = self._paper_batch(aggregate, power_target)
            power_pred, state_pred = model(paper_input.to(device))
            power_pred = torch.clamp(power_pred, min=0.0)

            batch_size, output_length, _appliances = power_pred.shape
            pred_power.append(self._to_loader_units(power_pred).cpu().numpy())
            pred_state.append(state_pred.cpu().numpy())
            true_power.append(power_target.numpy())
            true_state.append(paper_state.numpy())
            sample_indices.append(np.arange(offset, offset + batch_size * output_length))
            offset += batch_size * output_length

        return self.finalize_prediction_bundle(
            split=split,
            sample_indices=sample_indices,
            pred_power_batches=pred_power,
            pred_state_batches=pred_state,
            true_power_batches=true_power,
            true_state_batches=true_state,
        )

    def _paper_batch(
        self,
        aggregate: torch.Tensor,
        power_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Undo loader normalization, then apply the paper's /612 scale."""
        scale = float(self.model_cfg["architecture"].get("power_scale", 612.0))
        threshold = float(self.model_cfg["architecture"].get("on_threshold_watts", 15.0))
        input_watts = self._denormalize(aggregate, kind="input")
        target_watts = self._denormalize(power_target, kind="target")
        return input_watts / scale, target_watts / scale, (target_watts > threshold).to(target_watts.dtype)

    def _to_loader_units(self, paper_power: torch.Tensor) -> torch.Tensor:
        """Convert nonnegative watts/612 predictions into loader-normalized units."""
        scale = float(self.model_cfg["architecture"].get("power_scale", 612.0))
        watts = paper_power * scale
        norm = self._data_loader().norm
        if norm.target_mean is not None and norm.target_std is not None:
            mean = torch.as_tensor(norm.target_mean, device=watts.device, dtype=watts.dtype)
            std = torch.as_tensor(norm.target_std, device=watts.device, dtype=watts.dtype)
            return (watts - mean) / std
        if norm.legacy_scale != 1.0:
            return watts / float(norm.legacy_scale)
        return watts

    def _denormalize(self, values: torch.Tensor, *, kind: str) -> torch.Tensor:
        norm = self._data_loader().norm
        if kind == "input":
            if norm.input_mean is not None and norm.input_std is not None:
                return values * float(norm.input_std) + float(norm.input_mean)
            if norm.legacy_scale != 1.0:
                return values * float(norm.legacy_scale)
            return values

        if norm.target_mean is not None and norm.target_std is not None:
            mean = torch.as_tensor(norm.target_mean, device=values.device, dtype=values.dtype)
            std = torch.as_tensor(norm.target_std, device=values.device, dtype=values.dtype)
            return values * std + mean
        if norm.legacy_scale != 1.0:
            return values * float(norm.legacy_scale)
        return values

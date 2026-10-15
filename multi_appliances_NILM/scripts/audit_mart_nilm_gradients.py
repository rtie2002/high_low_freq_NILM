#!/usr/bin/env python
"""Measure MART-NILM power/state gradients on the shared encoder.

The audit applies two weighting rules to the same model and validation batches:

1. fixed:       lambda_state * S_i
2. balanced:    lambda_state * r_i * S_i

where r_i is the detached, clipped per-appliance ratio used during training.
This isolates the effect of the weighting rule from changes in learned weights.
The rule rescales gradient magnitudes but, because r_i is a positive detached
scalar, cannot alter the power/state gradient cosine similarity.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_experiment, load_model_config, merge_configs
from main import get_adapter


APPLIANCES = ("kettle", "fridge", "dishwasher", "washingmachine", "microwave")
DEFAULT_EXPERIMENT = ROOT / "config" / "experiment_mixed_ukdale_refit_5w_all_house_split.yaml"
DEFAULT_MODEL = ROOT / "config" / "models" / "mart_nilm.yaml"


def _data_root(merged: dict) -> Path:
    path = Path(merged["data_root"])
    return path if path.is_absolute() else ROOT / path


def _shared_encoder_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    prefixes = (
        "backbone.aggregate_feature_extractor.",
        "backbone.temporal_encoder.",
    )
    params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(prefixes)
    ]
    if not params:
        raise RuntimeError("No shared-encoder parameters matched the expected prefixes")
    return params


def _per_appliance_gradients(
    losses: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[list[torch.Tensor | None], torch.Tensor]:
    """Return one shared-encoder gradient per appliance in two VJP calls.

    ``is_grads_batched`` uses the five rows of an identity matrix as five
    vector-Jacobian products.  It is mathematically identical to calling
    ``autograd.grad(losses[i], ...)`` five times, but avoids redundant backward
    passes through the 1024-sample temporal network.
    """
    num_appliances = int(losses.numel())
    gradients = torch.autograd.grad(
        losses,
        parameters,
        grad_outputs=torch.eye(num_appliances, device=losses.device),
        retain_graph=retain_graph,
        allow_unused=True,
        is_grads_batched=True,
    )
    norm_sq = torch.zeros(num_appliances, device=losses.device, dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            norm_sq += gradient.detach().double().reshape(num_appliances, -1).square().sum(dim=1)
    return list(gradients), torch.sqrt(norm_sq)


def _cosines(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
    left_norm: torch.Tensor,
    right_norm: torch.Tensor,
) -> torch.Tensor:
    dot = torch.zeros_like(left_norm, dtype=torch.float64)
    for grad_left, grad_right in zip(left, right):
        if grad_left is not None and grad_right is not None:
            dot += (
                grad_left.detach()
                .double()
                .mul(grad_right.detach().double())
                .reshape(len(left_norm), -1)
                .sum(dim=1)
            )
    return dot / (left_norm * right_norm).clamp_min(1e-12)


def audit_seed(
    experiment_path: Path,
    model_path: Path,
    checkpoint: Path,
    seed: int,
    max_batches: int,
    batch_size: int,
) -> pd.DataFrame:
    experiment = load_experiment(experiment_path)
    experiment["seed"] = seed
    model_cfg = copy.deepcopy(load_model_config(model_path))
    model_cfg["training"]["batch_size"] = batch_size
    # Build raw per-appliance losses; weighting is applied explicitly below.
    model_cfg["loss"]["task_balance"] = "none"
    merged = merge_configs(experiment, model_cfg)
    adapter = get_adapter(
        merged["model_name"], merged, data_root=str(_data_root(merged))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = adapter.build_model(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    loss_fn = adapter.build_loss().to(device)
    loader = adapter.build_dataloader("validation")
    shared_parameters = _shared_encoder_parameters(model)

    lambda_state = float(model_cfg["loss"].get("lambda_state", 1.0))
    limit = float(model_cfg["loss"].get("task_balance_ratio_limit", 3.0))
    rows: list[dict[str, float | int | str]] = []

    for batch_index, (aggregate, power_true, state_true) in enumerate(loader):
        if batch_index >= max_batches:
            break
        aggregate = aggregate.to(device)
        power_true = power_true.to(device)
        state_true = state_true.float().to(device)
        power_pred, state_logits = model(aggregate)
        power_losses = loss_fn._per_appliance_power_loss(
            power_pred, power_true, state_true
        )
        power_losses = power_losses + loss_fn.power_energy_relative_weight * (
            loss_fn._relative_energy_loss(power_pred, power_true)
        )
        state_losses = loss_fn._per_appliance_state_loss(state_logits, state_true)

        global_scale = (
            power_losses.sum().detach()
            / state_losses.sum().detach().clamp_min(1e-8)
        )
        local_scale = (
            power_losses.detach() / state_losses.detach().clamp_min(1e-8)
        )
        balanced_scale = torch.clamp(
            local_scale,
            min=global_scale / limit,
            max=global_scale * limit,
        )

        power_grads, power_norms = _per_appliance_gradients(
            power_losses, shared_parameters, retain_graph=True
        )
        state_grads, raw_state_norms = _per_appliance_gradients(
            state_losses, shared_parameters, retain_graph=False
        )
        cosines = _cosines(power_grads, state_grads, power_norms, raw_state_norms)

        for appliance_index, appliance in enumerate(APPLIANCES):
            power_norm = float(power_norms[appliance_index].cpu())
            raw_state_norm = float(raw_state_norms[appliance_index].cpu())
            cosine = float(cosines[appliance_index].cpu())
            for rule, coefficient in (
                ("Fixed task weight", lambda_state),
                (
                    "Appliance-wise balance",
                    lambda_state * float(balanced_scale[appliance_index].cpu()),
                ),
            ):
                state_norm = coefficient * raw_state_norm
                rows.append(
                    {
                        "seed": seed,
                        "batch": batch_index,
                        "appliance": appliance,
                        "weighting_rule": rule,
                        "power_gradient_norm": power_norm,
                        "state_gradient_norm": state_norm,
                        "state_to_power_gradient_ratio": state_norm
                        / max(power_norm, 1e-12),
                        "power_state_gradient_cosine": cosine,
                        "state_coefficient": coefficient,
                    }
                )
        del power_pred, state_logits, power_losses, state_losses

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = audit_seed(
        args.experiment,
        args.model_config,
        args.checkpoint,
        args.seed,
        args.max_batches,
        args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.8g")
    print(f"Saved {len(frame)} gradient measurements to {args.output}")


if __name__ == "__main__":
    main()

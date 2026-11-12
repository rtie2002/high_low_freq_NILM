"""Stage 2: recover five appliance waveforms from a whole-house meter window.

Freeze the Stage 1 decoder G. Search for five codes z_i that minimise

    s(t; z) = sum_i G(z_i, i)[t]
    b*(z)   = max(0, mean_t(y(t) - s(t; z)))
    L(z)    = mean_t((s(t; z) + b*(z) - y(t))**2).

All quantities inside the loss use the same scaled power units as Stage 1.
The constant background b* is fitted analytically for each window and each
candidate. The solver uses the whole window; it is an offline estimator.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from models.cnn import ApplianceConditionedCNN, build_forward_model
from stage2_inverse.losses import diagonal_latent_prior, estimate_constant_background, meter_mse
from utils.config import load_yaml
from utils.protocol import load_protocol, save_inverse_evaluation


def solve_windows(
    model: ApplianceConditionedCNN,
    meter: Tensor,
    *,
    steps: int,
    learning_rate: float,
    generator: torch.Generator,
    initial_codes: Tensor | None = None,
    fit_background: bool = True,
    latent_prior: tuple[Tensor, Tensor] | None = None,
    prior_weight: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return selected appliance traces, background, initial and selected meter MSE.

    meter has shape [B, T]. Appliance traces have shape [B, 5, T].
    The background and MSE outputs have one value per window.
    """
    model.eval().requires_grad_(False)
    batch, length = meter.shape
    # 1. Initialise z^(0) in the decoder's expected [B, 5, Z, ceil(T/8)] shape.
    latent_time = math.ceil(length / model.DOWNSAMPLE)
    shape = (batch, len(model.APPLIANCES), model.latent_channels, latent_time)
    if initial_codes is None:
        codes = torch.randn(shape, generator=generator, device=meter.device, dtype=meter.dtype)
    else:
        codes = initial_codes.detach().to(meter).clone()
    codes.requires_grad_()
    optimizer = torch.optim.Adam([codes], lr=learning_rate)

    best_loss = meter.new_full((batch,), float("inf"))
    selected_mse = best_loss.clone()
    best_power = meter.new_zeros((batch, len(model.APPLIANCES), length))
    best_background = meter.new_zeros(batch)

    # Score z^(0) and every updated candidate through z^(steps).
    for iteration in range(steps + 1):
        with torch.set_grad_enabled(iteration < steps):
            # 2. Decode p_hat_i(t) = G(z_i, i)[t], then sum across appliances.
            power = model.decode(codes, output_length=length)
            appliance_sum = power.sum(dim=1)

            # 3. For fixed z, b* minimises mean_t((y - s - b)^2) over b >= 0.
            # A synthetic sum of the five submeters has known b = 0.
            background = (estimate_constant_background(power, meter)
                          if fit_background else meter.new_zeros(batch))

            # 4. Score each window separately. The same b* applies to all T rows.
            loss_per_window = meter_mse(appliance_sum + background[:, None], meter)
            objective = loss_per_window
            if prior_weight:
                center, spread = latent_prior
                # A diagonal Gaussian penalty in training-code standard deviations.
                penalty = diagonal_latent_prior(codes, center, spread)
                objective = objective + prior_weight * penalty

        with torch.no_grad():
            if iteration == 0:
                initial_loss = loss_per_window.detach()
            improved = objective < best_loss
            best_loss = torch.where(improved, objective, best_loss)
            selected_mse = torch.where(improved, loss_per_window, selected_mse)
            best_power = torch.where(improved[:, None, None], power, best_power)
            best_background = torch.where(improved, background, best_background)

        if iteration < steps:
            # 5. Backpropagate dL/dz through the decoder and update only z.
            # Sum independent window losses to keep each gradient batch-size invariant.
            optimizer.zero_grad(set_to_none=True)
            objective.sum().backward()
            optimizer.step()

    return best_power, best_background, initial_loss, selected_mse


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_dir / "configs/stage2.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-windows", type=int, help="Limit windows for a smoke check.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-mean-init", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prior-weight", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in --run-dir.")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_yaml(config_path)
    experiment_config_path = config_path
    if "stage1_config" in config:
        stage1_path = config_path.parent / config["stage1_config"]
        stage1_config = load_yaml(stage1_path)
        stage1_config["inference"] = config["inference"]
        config = stage1_config
        experiment_config_path = stage1_path
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.overwrite:
        parser.error("--run-dir already contains files; choose a new directory or pass --overwrite intentionally.")
    settings = config["inference"]
    for name in ("split", "steps", "learning_rate", "batch_size", "device", "train_mean_init", "prior_weight"):
        if getattr(args, name) is None:
            setattr(args, name, settings[name])
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    seed = int(config["run"]["seed"])
    generator = torch.Generator(device=device).manual_seed(seed)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_forward_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    scale = float(checkpoint["power_scale_watts"])
    length = int(checkpoint["window_length"])

    stride = int(config["data"]["window_stride"])
    protocol = load_protocol(
        experiment_path=experiment_config_path,
        window_length=length, stride=stride, batch_size=args.batch_size,
    )
    latent_prior = None
    initial_code_center = None
    prior_starts = []
    if args.train_mean_init or args.prior_weight:
        model.eval().requires_grad_(False)
        train = protocol.loader.get_split("train")
        all_starts = protocol.loader.build_dataset("train").indices
        count = min(int(settings["center_windows"]), len(all_starts))
        picks = np.linspace(0, len(all_starts) - 1, count, dtype=int)
        prior_starts = all_starts[picks].tolist()
        with torch.no_grad():
            rows = all_starts[picks, None] + np.arange(length)
            training_power = torch.as_tensor(train.targets[rows] / scale, device=device).transpose(1, 2)
            codes = model.encode(training_power)
            initial_code_center = codes.mean((0, 3), keepdim=True)
            spread = codes.std((0, 3), correction=0, keepdim=True)
            latent_prior = (initial_code_center, spread.clamp_min(float(settings["prior_min_std"])))

    # The benchmark loader supplies aligned window indices and mains readings.
    # The solver receives only meter values. Submeters are used by the evaluator.
    dataset = protocol.loader.build_dataset(args.split)
    mains_watts = protocol.loader.get_split(args.split).inputs
    starts = dataset.indices[:args.max_windows]
    offsets = np.arange(length)
    window_outputs = []
    initial_losses = []
    best_losses = []
    started = time.perf_counter()
    for offset in range(0, len(starts), args.batch_size):
        rows = starts[offset:offset + args.batch_size, None] + offsets
        meter = torch.as_tensor(mains_watts[rows] / scale, device=device, dtype=torch.float32)
        initial_codes = (initial_code_center.expand(len(rows), -1, -1, math.ceil(length / model.DOWNSAMPLE))
                         if args.train_mean_init else None)
        power, background, initial_mse, best_mse = solve_windows(
            model, meter, steps=args.steps, learning_rate=args.learning_rate, generator=generator,
            initial_codes=initial_codes, latent_prior=latent_prior, prior_weight=args.prior_weight,
        )
        # Convert back to watts and merge appliance/background traces together.
        background_trace = background[:, None, None].expand(-1, 1, length)
        predictions = torch.cat((power, background_trace), dim=1).transpose(1, 2)
        window_outputs.append(predictions.cpu().numpy() * scale)
        initial_losses.append(initial_mse.cpu().numpy())
        best_losses.append(best_mse.cpu().numpy())
    elapsed = time.perf_counter() - started

    merged, rows = protocol.loader.reconstruct_timeline_from_windows(
        args.split, np.concatenate(window_outputs),
    )
    appliances_watts = merged[:, :-1]
    background_watts = merged[:, -1]
    meter_fit_watts = appliances_watts.sum(axis=1) + background_watts
    observed_watts = mains_watts[rows]
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(config_path),
        "split": args.split,
        "device": str(device),
        "seed": seed,
        "initialisation": "train_mean" if args.train_mean_init else "random",
        "prior_weight": args.prior_weight,
        "prior_train_window_starts": prior_starts,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "power_scale_watts": scale,
        "window_length": length,
        "window_stride": stride,
        "background": "nonnegative_constant_per_window",
        "evaluated_windows": len(starts),
        "available_windows": len(dataset),
        "covered_rows": len(rows),
        "available_rows": len(mains_watts),
        "initial_window_meter_mse_watts2": float(np.concatenate(initial_losses).mean() * scale**2),
        "best_window_meter_mse_watts2": float(np.concatenate(best_losses).mean() * scale**2),
        "selected_window_meter_mse_watts2": float(np.concatenate(best_losses).mean() * scale**2),
        "aggregate_reconstruction_mae_watts": float(np.abs(meter_fit_watts - observed_watts).mean()),
        "inference_seconds": elapsed,
    }
    save_inverse_evaluation(
        protocol=protocol, split=args.split, run_dir=args.run_dir,
        model_name="appliance_conditioned_cnn_inverse",
        csv_timesteps=rows, predicted_appliance_watts=appliances_watts,
        manifest=manifest, random_seed=seed,
        waveform_periods=int(config.get("evaluation", {}).get("on_periods", 5)),
        waveform_dpi=int(config.get("evaluation", {}).get("dpi", 300)),
    )
    np.savez_compressed(
        args.run_dir / "meter_diagnostics.npz", csv_timesteps=rows,
        observed_watts=observed_watts, reconstructed_watts=meter_fit_watts,
        background_watts=background_watts,
        raw_appliance_watts=appliances_watts,
    )
    print(f"Saved {len(rows)} rows of five-appliance predictions to {args.run_dir}")


if __name__ == "__main__":
    main()

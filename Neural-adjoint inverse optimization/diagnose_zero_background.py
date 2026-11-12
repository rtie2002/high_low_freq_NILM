"""Diagnose latent inversion on validation windows with synthetic or measured mains."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forward_model.cnn import ApplianceConditionedCNN, build_forward_model
from step2_inverse_optimization import solve_windows
from utils.config import load_yaml
from utils.metrics import per_appliance_average_precision, per_appliance_f1
from utils.output_format import save_result_json, save_result_table
from utils.protocol import load_protocol


def score(power, truth, states, thresholds, scale):
    """Inputs [B,5,T] use scaled power. Return scores in watts where applicable."""
    prediction = (power.transpose(1, 2).cpu().numpy() * scale).reshape(-1, 5)
    actual = (truth.transpose(1, 2).cpu().numpy() * scale).reshape(-1, 5)
    on = states.reshape(-1, 5).astype(bool)
    pred_on = (prediction >= thresholds).astype(int)
    error = np.abs(prediction - actual)
    return {
        "mae_watts": error.mean(axis=0),
        "on_mae_watts": np.array([error[on[:, i], i].mean() if on[:, i].any() else np.nan for i in range(5)]),
        "off_mae_watts": np.array([error[~on[:, i], i].mean() if (~on[:, i]).any() else np.nan for i in range(5)]),
        "f1": per_appliance_f1(on.astype(int), pred_on),
        "ap": per_appliance_average_precision(on.astype(int), prediction),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "forward_training/train_forward.yaml")
    parser.add_argument("--windows", type=int, default=48)
    parser.add_argument("--train-windows", type=int, default=128)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--measured-mains", action="store_true",
                        help="Use measured mains and fit the baseline constant background.")
    args = parser.parse_args()
    if min(args.windows, args.train_windows, args.steps, args.batch_size) < 1:
        parser.error("Window counts, step count and batch size must be positive.")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("--run-dir must be empty or new.")

    cfg = load_yaml(args.config)
    seed = int(cfg["run"]["seed"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_forward_model(checkpoint["model_config"]).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model_state_dict"])
    scale = float(checkpoint["power_scale_watts"])
    length = int(checkpoint["window_length"])
    if scale != float(cfg["data"]["power_scale_watts"]) or length != int(cfg["data"]["window_length"]):
        raise ValueError("Checkpoint and config must agree on scale and window length.")
    protocol = load_protocol(
        experiment_path=args.config.resolve().parent / cfg["experiment"],
        window_length=length, stride=int(cfg["data"]["window_stride"]), batch_size=args.batch_size,
    )
    if list(model.APPLIANCES) != protocol.appliances or list(checkpoint["appliances"]) != protocol.appliances:
        raise ValueError("Checkpoint, model and experiment must share the same appliance order.")

    def sample_split(split, count):
        starts = protocol.loader.build_dataset(split).indices
        indices = np.linspace(0, len(starts) - 1, min(count, len(starts)), dtype=int)
        selected = starts[indices]
        data = protocol.loader.get_split(split)
        rows = selected[:, None] + np.arange(length)
        scaled = torch.from_numpy(data.targets[rows] / scale).transpose(1, 2)
        return selected, scaled, data.states[rows]

    train_starts, train_power, train_states = sample_split("train", args.train_windows)
    val_starts, truth, states = sample_split("validation", args.windows)
    with torch.no_grad():
        # z_train(i,c) = mean_{window,time}(E(p_i / S)_{c,time}).
        # A single training-derived code per appliance/channel is tiled over time.
        train_center = model.encode(train_power).mean(dim=(0, 3), keepdim=True)
        oracle = model.encode(truth)
        train_init = train_center.expand_as(oracle)
        random_init = torch.randn(oracle.shape, generator=torch.Generator().manual_seed(seed))
        train_reconstruction = model.decode(model.encode(train_power), output_length=length)
    if args.measured_mains:
        rows = val_starts[:, None] + np.arange(length)
        meter = torch.as_tensor(protocol.loader.get_split("validation").inputs[rows] / scale,
                                dtype=torch.float32)
    else:
        meter = truth.sum(dim=1)
    device_rows = []
    window_rows = []
    aggregate_rows = []
    reconstruction_rows = []
    for split, prediction, labels, on in [
        ("train", train_reconstruction, train_power, train_states),
        ("validation", model.decode(oracle, output_length=length), truth, states),
    ]:
        metrics = score(prediction, labels, on, protocol.thresholds_watts, scale)
        for i, appliance in enumerate(protocol.appliances):
            reconstruction_rows.append({
                "split": split, "windows": len(labels), "appliance": appliance,
                "on_samples": int(on[..., i].sum()), **{key: value[i] for key, value in metrics.items()},
            })
    waveforms = {}
    for name, initial in {"random": random_init, "train_mean": train_init, "oracle": oracle}.items():
        with torch.no_grad():
            before = model.decode(initial, output_length=length)
        results = [
            solve_windows(
                model, meter_batch, initial_codes=code_batch, fit_background=args.measured_mains,
                steps=args.steps, learning_rate=args.learning_rate,
                generator=torch.Generator().manual_seed(seed),
            )
            for meter_batch, code_batch in zip(meter.split(args.batch_size), initial.split(args.batch_size))
        ]
        after = torch.cat([result[0] for result in results])
        if not args.measured_mains and any(torch.count_nonzero(result[1]).item() for result in results):
            raise RuntimeError("The zero-background control produced a nonzero background.")
        before_meter = torch.cat([result[2] for result in results])
        after_meter = torch.cat([result[3] for result in results])
        waveforms[name] = (before, after, torch.cat([result[1] for result in results]))
        if not torch.all(after_meter <= before_meter + 1e-9):
            raise RuntimeError("Best candidate must fit the meter at least as well as its initialisation.")

        for phase, powers, meter_mse in [("initial", before, before_meter), ("selected", after, after_meter)]:
            metrics = score(powers, truth, states, protocol.thresholds_watts, scale)
            aggregate_rows.append({
                "initialisation": name, "phase": phase,
                "meter_rmse_watts": float(meter_mse.mean().sqrt() * scale),
                **{key: np.nanmean(value) for key, value in metrics.items()},
            })
            for i, appliance in enumerate(protocol.appliances):
                device_rows.append({
                    "initialisation": name, "phase": phase, "appliance": appliance,
                    **{key: value[i] for key, value in metrics.items()},
                })
        before_mae = (before - truth).abs().mean(dim=(1, 2)) * scale
        after_mae = (after - truth).abs().mean(dim=(1, 2)) * scale
        for i, start in enumerate(val_starts):
            window_rows.append({
                "initialisation": name, "csv_row_start": start,
                "initial_meter_rmse_watts": float(before_meter[i].sqrt() * scale),
                "selected_meter_rmse_watts": float(after_meter[i].sqrt() * scale),
                "initial_appliance_mae_watts": float(before_mae[i]),
                "selected_appliance_mae_watts": float(after_mae[i]),
            })
        print(f"Finished {name} on {len(val_starts)} validation windows.", flush=True)

    summary = pd.DataFrame(aggregate_rows)
    save_result_table(summary, args.run_dir / "summary.csv")
    save_result_table(pd.DataFrame(device_rows), args.run_dir / "per_appliance.csv")
    save_result_table(pd.DataFrame(window_rows), args.run_dir / "per_window.csv")
    save_result_table(pd.DataFrame(reconstruction_rows), args.run_dir / "reconstruction.csv")
    if args.measured_mains:
        before, after, _ = waveforms["oracle"]
        delta = ((after - truth).abs().mean((1, 2)) - (before - truth).abs().mean((1, 2))) * scale
        idx = [int(delta.argmax()), int(delta.argmin())]
        np.savez_compressed(
            args.run_dir / "waveform_examples.npz", appliances=protocol.appliances,
            starts=val_starts[idx], delta=delta[idx].numpy(), truth=(truth[idx] * scale).numpy(),
            reconstruction=(before[idx] * scale).numpy(), meter=(meter[idx] * scale).numpy(),
            names=list(waveforms),
            power=np.stack([(power[idx] * scale).numpy() for _, power, _ in waveforms.values()]),
            fit=np.stack([((power[idx].sum(1) + bg[idx, None]) * scale).numpy()
                          for _, power, bg in waveforms.values()]),
        )
    save_result_json(args.run_dir / "run_manifest.json", {
        "checkpoint": str(args.checkpoint.resolve()), "config": str(args.config.resolve()),
        "best_epoch": checkpoint.get("best_epoch"), "seed": seed, "split": "validation",
        "observation": "measured_mains" if args.measured_mains else "sum_of_five_submeters",
        "background": "nonnegative_constant_per_window" if args.measured_mains else "fixed_zero",
        "train_window_starts": train_starts.tolist(), "validation_window_starts": val_starts.tolist(),
        "steps": args.steps, "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "state_thresholds_watts": dict(zip(protocol.appliances, protocol.thresholds_watts.tolist())),
        "scoring": "raw per-window outputs with fixed watt thresholds, without power gating",
        "oracle": "label-derived initialisation for offline diagnosis only",
    })
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved diagnostics to {args.run_dir.resolve()}")
    if args.measured_mains:
        # Plot in a fresh process to avoid mixing PyTorch and Matplotlib native runtimes.
        subprocess.run([sys.executable, "-c", PLOT_EXAMPLES, str(args.run_dir)], check=True)


PLOT_EXAMPLES = """
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

run_dir = Path(sys.argv[1])
with np.load(run_dir / "waveform_examples.npz") as data:
    for idx, label in enumerate(("largest_mae_increase", "largest_mae_decrease")):
        fig, axes = plt.subplots(6, 1, sharex=True, figsize=(12, 11), layout="constrained")
        for i, appliance in enumerate(data["appliances"]):
            ax = axes[i]
            ax.plot(data["truth"][idx, i], color="black", label="Submeter")
            ax.plot(data["reconstruction"][idx, i], linestyle=":", label="Stage 1 reconstruction")
            for j, name in enumerate(data["names"]):
                ax.plot(data["power"][j, idx, i], label=f"{name} after inversion", alpha=.75)
            ax.set_ylabel(f"{appliance} (W)")
        axes[0].legend(ncol=2)
        axes[-1].plot(data["meter"][idx], color="black", label="Measured mains")
        for j, name in enumerate(data["names"]):
            axes[-1].plot(data["fit"][j, idx], label=name, alpha=.75)
        axes[-1].set(xlabel="Sample in window", ylabel="Mains (W)")
        axes[-1].legend(ncol=2)
        fig.suptitle(f"Validation row {data['starts'][idx]} | oracle appliance MAE change {data['delta'][idx]:+.1f} W")
        fig.savefig(run_dir / f"{label}.png", dpi=150)
        plt.close(fig)
"""


if __name__ == "__main__":
    main()

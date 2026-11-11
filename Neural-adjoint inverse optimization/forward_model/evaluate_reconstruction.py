"""Compare each measured appliance waveform with its raw CNN reconstruction."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_DIR))

from forward_model.cnn import ApplianceConditionedCNN
from utils.config import load_yaml
from utils.output_format import save_result_json, save_result_table
from utils.protocol import load_protocol
from utils.waveform_capture import plot_single_on_period, save_appliance_on_waveforms


def off_rich_crop(
    on: np.ndarray, rows: np.ndarray, segments: np.ndarray, max_samples: int,
) -> tuple[int, int]:
    """Prefer full-length contiguous crops with an ON sample and many OFF rows."""
    cuts = np.flatnonzero((np.diff(rows) != 1) | (np.diff(segments) != 0)) + 1
    bounds = np.r_[0, cuts, len(rows)]
    on_prefix = np.r_[0, np.cumsum(on, dtype=np.int64)]
    best = None
    best_key = None
    for left, right in zip(bounds[:-1], bounds[1:]):
        length = min(max_samples, right - left)
        starts = np.arange(left, right - length + 1)
        on_count = on_prefix[starts + length] - on_prefix[starts]
        # An OFF sample at both edges prevents clipping the selected ON run.
        complete = np.flatnonzero((on_count > 0) & ~on[starts] & ~on[starts + length - 1])
        active = np.flatnonzero(on_count > 0)
        eligible = complete if complete.size else active
        candidate = eligible[np.argmin(on_count[eligible])] if eligible.size else 0
        key = (bool(complete.size), bool(active.size), length, -on_count[candidate])
        if best_key is None or key > best_key:
            best_key = key
            best = (int(starts[candidate]), int(starts[candidate] + length))
    if best is None:
        raise ValueError("No reconstructed rows to plot.")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "forward_training/train_forward.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=("validation", "ukdale_house2", "refit_house20"), default="validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--on-periods", type=int, default=5)
    parser.add_argument("--long-samples", type=int, default=4096)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-windows", type=int, help="Limit validation windows for a smoke check.")
    args = parser.parse_args()
    counts = (args.batch_size, args.on_periods, args.long_samples, args.dpi)
    if min(counts) < 1 or (args.max_windows is not None and args.max_windows < 1):
        parser.error("Window, batch, plot, and resolution counts must be positive.")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("--run-dir must be empty or new.")

    config_path = args.config.resolve()
    config = load_yaml(config_path)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_cfg = dict(checkpoint["model_config"])
    if model_cfg.pop("name") != "appliance_conditioned_cnn":
        raise ValueError("Expected an ApplianceConditionedCNN checkpoint.")
    model = ApplianceConditionedCNN(**model_cfg).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    length = int(checkpoint["window_length"])
    scale = float(checkpoint["power_scale_watts"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Checkpoint power scale must be positive and finite.")
    if length != int(config["data"]["window_length"]) or scale != float(config["data"]["power_scale_watts"]):
        raise ValueError("Config and checkpoint must agree on window length and power scale.")

    split = args.split
    protocol = load_protocol(
        experiment_path=config_path.parent / config["experiment"],
        window_length=length, stride=int(config["data"]["window_stride"]),
        batch_size=args.batch_size, csv_sequence_only=True,
    )
    if list(model.APPLIANCES) != protocol.appliances or list(checkpoint["appliances"]) != protocol.appliances:
        raise ValueError("Appliance channel order must match the checkpoint and experiment.")
    raw = protocol.loader.get_split(split)
    dataset = protocol.loader.build_dataset(split)
    starts = dataset.indices[:args.max_windows]
    offsets = np.arange(length)
    windows = []
    with torch.inference_mode():
        for first in range(0, len(starts), args.batch_size):
            window_rows = starts[first:first + args.batch_size, None] + offsets
            # [B,T,5] in watts -> [B,5,T] in shared scaled power.
            x = torch.as_tensor(raw.targets[window_rows] / scale, device=device).transpose(1, 2)
            # z_i = E(x_i); p_hat_i = S * G(z_i, i).
            reconstructed, _ = model(x)
            windows.append((reconstructed.transpose(1, 2).cpu().numpy() * scale))

    # Average duplicate predictions for each CSV row before scoring/plotting.
    predicted, rows = protocol.loader.reconstruct_timeline_from_windows(split, np.concatenate(windows))
    actual = raw.targets[rows]
    true_on = raw.states[rows].astype(bool)
    segments = raw.segment_ids[rows]
    if not np.isfinite(predicted).all():
        raise ValueError("Checkpoint produced non-finite reconstruction values.")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    for i, appliance in enumerate(protocol.appliances):
        error = predicted[:, i].astype(np.float64) - actual[:, i]
        on = true_on[:, i]
        off = ~on
        # MAE = mean(|p_hat - p|); ON/OFF means use the measured CSV states.
        metrics.append({
            "appliance": appliance,
            "on_samples": int(on.sum()),
            "off_samples": int(off.sum()),
            "mae_watts": np.abs(error).mean(),
            "rmse_watts": np.sqrt(np.mean(error**2)),
            "on_mae_watts": np.abs(error[on]).mean() if on.any() else np.nan,
            "off_mae_watts": np.abs(error[off]).mean() if off.any() else np.nan,
            "off_mean_reconstruction_watts": predicted[off, i].mean() if off.any() else np.nan,
            "zero_baseline_on_mae_watts": np.abs(actual[on, i]).mean() if on.any() else np.nan,
        })
    table = pd.DataFrame(metrics)
    save_result_table(table, args.run_dir / "reconstruction_metrics.csv")
    np.savez_compressed(
        args.run_dir / "reconstruction.npz", appliances=protocol.appliances,
        csv_timesteps=rows, input_watts=actual, reconstructed_watts=predicted,
        true_on=true_on, segment_ids=segments,
    )

    waveform_dir = args.run_dir / "waveforms"
    seed = int(config["run"]["seed"])
    event_plots = save_appliance_on_waveforms(
        waveform_dir, appliances=protocol.appliances,
        y_true_watts=actual, y_pred_watts=predicted, y_true_on=true_on,
        csv_timesteps=rows, segment_ids=segments, n_periods=args.on_periods,
        min_on_duration=1, context_scale=10.0, dpi=args.dpi,
        rng=np.random.default_rng(seed), title_prefix="Stage 1 reconstruction | ",
    )
    long_crops = {}
    for i, appliance in enumerate(protocol.appliances):
        on = true_on[:, i]
        start, end = off_rich_crop(on, rows, segments, args.long_samples)
        off_fraction = float((~on[start:end]).mean())
        plot_single_on_period(
            appliance=appliance, y_true_watts=actual[:, i], y_pred_watts=predicted[:, i],
            y_true_on=on, csv_timesteps=rows, window_start=start, window_end=end,
            output_path=waveform_dir / appliance / "long_off_context.png",
            long_figure=True, dpi=args.dpi,
            title=f"{appliance} | Stage 1 input vs. reconstruction | {off_fraction:.1%} OFF",
        )
        long_crops[appliance] = {
            "first_csv_row": int(rows[start]), "last_csv_row": int(rows[end - 1]),
            "samples": end - start, "on_samples": int(on[start:end].sum()),
            "off_samples": int((~on[start:end]).sum()), "off_fraction": off_fraction,
        }

    save_result_json(args.run_dir / "run_manifest.json", {
        "checkpoint": str(args.checkpoint.resolve()), "config": str(config_path),
        "best_epoch": checkpoint.get("best_epoch"), "split": split, "device": str(device),
        "evaluated_windows": len(starts), "available_windows": len(dataset),
        "covered_rows": len(rows), "available_rows": len(raw.targets),
        "event_plot_files": len(event_plots), "plot_seed": seed, "long_crops": long_crops,
    })
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved raw reconstruction metrics and waveforms to {args.run_dir.resolve()}")


if __name__ == "__main__":
    main()

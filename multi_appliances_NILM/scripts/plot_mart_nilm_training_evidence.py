#!/usr/bin/env python
"""Create publication figures supporting the MART-NILM training strategy.

The script produces two independent, square figures together with the exact
CSV values used to draw them:

1. Event/background coverage in recorded training windows and under the
   focal-event sampler.
2. The normalized appliance-wise state-loss coefficients reconstructed from
   the per-appliance losses saved during training.

No model inference or test-house data are used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
APPLIANCES = ("kettle", "fridge", "dishwasher", "washingmachine", "microwave")
DISPLAY_NAMES = ("Kettle", "Fridge", "Dishwasher", "Washing machine", "Microwave")
BACKGROUND_EDGES = np.asarray([0.0, 100.0, 200.0, 400.0, 800.0, np.inf])
BACKGROUND_LABELS = ("0-100", "100-200", "200-400", "400-800", r"$\geq$800")
SEEDS = (2026, 2027, 2028)


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
        }
    )


def _window_starts(segment_ids: np.ndarray, length: int, stride: int) -> np.ndarray:
    """Match the per-segment stride restart used by the training dataloader."""
    boundaries = np.flatnonzero(
        np.r_[True, segment_ids[1:] != segment_ids[:-1], True]
    )
    starts = [
        np.arange(start, end - length + 1, stride, dtype=np.int64)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end - start >= length
    ]
    return np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)


def event_background_coverage(
    training_csv: Path,
    output_dir: Path,
    window_length: int,
    stride: int,
) -> None:
    power_columns = [f"{name}_power" for name in APPLIANCES]
    state_columns = [f"{name}_on" for name in APPLIANCES]
    columns = [
        "dataset",
        "house",
        "sequence_id",
        "aggregate",
        *power_columns,
        *state_columns,
    ]
    frame = pd.read_csv(training_csv, usecols=columns)

    aggregate = frame["aggregate"].to_numpy(dtype=np.float32)
    appliance_power = frame[power_columns].to_numpy(dtype=np.float32)
    states = frame[state_columns].to_numpy(dtype=np.uint8)
    segment_ids = pd.factorize(
        pd.MultiIndex.from_frame(frame[["dataset", "house", "sequence_id"]])
    )[0]
    starts = _window_starts(segment_ids, window_length, stride)
    ends = starts + window_length
    if len(starts) == 0:
        raise ValueError("No complete training windows were found")

    residual = np.clip(aggregate - appliance_power.sum(axis=1), 0.0, None)
    median_background = np.asarray(
        [np.median(residual[start:end]) for start, end in zip(starts, ends)],
        dtype=np.float64,
    )
    bin_ids = np.searchsorted(BACKGROUND_EDGES[1:-1], median_background, side="right")

    recorded_counts = np.zeros((len(APPLIANCES), len(BACKGROUND_LABELS)), dtype=np.int64)
    for app_i in range(len(APPLIANCES)):
        cumulative = np.pad(np.cumsum(states[:, app_i], dtype=np.int64), (1, 0))
        contains_on = cumulative[ends] - cumulative[starts] > 0
        recorded_counts[app_i] = np.bincount(
            bin_ids[contains_on], minlength=len(BACKGROUND_LABELS)
        )
    recorded_percent = 100.0 * recorded_counts / recorded_counts.sum(axis=1, keepdims=True)

    # The implemented focal sampler selects a non-empty background bin
    # uniformly after selecting the focal appliance. Its expected conditional
    # coverage is therefore uniform over the available bins.
    nonempty_bins = np.bincount(bin_ids, minlength=len(BACKGROUND_LABELS)) > 0
    focal_percent = np.zeros_like(recorded_percent)
    focal_percent[:, nonempty_bins] = 100.0 / float(nonempty_bins.sum())

    records = []
    for app_i, appliance in enumerate(APPLIANCES):
        for bin_i, label in enumerate(BACKGROUND_LABELS):
            records.append(
                {
                    "appliance": appliance,
                    "background_range_w": label,
                    "recorded_on_window_count": int(recorded_counts[app_i, bin_i]),
                    "recorded_row_percent": float(recorded_percent[app_i, bin_i]),
                    "focal_expected_row_percent": float(focal_percent[app_i, bin_i]),
                }
            )
    pd.DataFrame(records).to_csv(
        output_dir / "event_background_coverage.csv", index=False, float_format="%.4f"
    )

    vmax = float(max(recorded_percent.max(), focal_percent.max()))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), constrained_layout=True)
    panels = (
        (recorded_percent, "(a) Recorded ON-window coverage"),
        (focal_percent, "(b) Expected focal-event coverage"),
    )
    image = None
    for ax, (values, title) in zip(axes, panels):
        image = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xticks(np.arange(len(BACKGROUND_LABELS)), BACKGROUND_LABELS)
        ax.set_yticks(np.arange(len(DISPLAY_NAMES)), DISPLAY_NAMES)
        ax.set_xlabel("Median residual background power (W)")
        ax.set_ylabel("Appliance")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                color = "white" if values[row, col] > 0.55 * vmax else "black"
                ax.text(
                    col,
                    row,
                    f"{values[row, col]:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8.5,
                    fontweight="bold" if values[row, col] > 0.55 * vmax else "normal",
                )
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
    colorbar.set_label("Share of ON-containing windows (%)")
    fig.savefig(output_dir / "event_background_coverage.png", dpi=600)
    fig.savefig(output_dir / "event_background_coverage.pdf")
    plt.close(fig)


def appliance_weight_trajectories(
    run_root: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
    ratio_limit: float,
) -> None:
    per_seed = []
    for seed in seeds:
        path = run_root / f"seed_{seed}" / "loss_detail.csv"
        frame = pd.read_csv(path)
        power = frame[[f"train_loss_power_{name}" for name in APPLIANCES]].to_numpy(
            dtype=np.float64
        )
        state = frame[[f"train_loss_state_{name}" for name in APPLIANCES]].to_numpy(
            dtype=np.float64
        )
        global_ratio = power.sum(axis=1) / np.clip(state.sum(axis=1), 1e-8, None)
        local_ratio = power / np.clip(state, 1e-8, None)
        clipped_ratio = np.clip(
            local_ratio,
            global_ratio[:, None] / ratio_limit,
            global_ratio[:, None] * ratio_limit,
        )
        relative_ratio = clipped_ratio / global_ratio[:, None]
        for epoch_i, epoch in enumerate(frame["epoch"].astype(int)):
            for app_i, appliance in enumerate(APPLIANCES):
                per_seed.append(
                    {
                        "seed": seed,
                        "epoch": int(epoch),
                        "appliance": appliance,
                        "global_ratio": float(global_ratio[epoch_i]),
                        "local_ratio": float(local_ratio[epoch_i, app_i]),
                        "clipped_ratio": float(clipped_ratio[epoch_i, app_i]),
                        "normalized_weight": float(relative_ratio[epoch_i, app_i]),
                    }
                )
    raw = pd.DataFrame(per_seed)
    raw.to_csv(
        output_dir / "appliance_weight_trajectories_by_seed.csv",
        index=False,
        float_format="%.6f",
    )
    summary = (
        raw.groupby(["epoch", "appliance"], sort=False)["normalized_weight"]
        .agg(mean="mean", std=lambda values: values.std(ddof=1))
        .reset_index()
    )
    summary.to_csv(
        output_dir / "appliance_weight_trajectories.csv",
        index=False,
        float_format="%.6f",
    )

    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    fig, axes = plt.subplots(
        len(APPLIANCES),
        1,
        figsize=(7.2, 7.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for ax, appliance, display, color in zip(
        axes, APPLIANCES, DISPLAY_NAMES, colors
    ):
        values = summary.loc[summary["appliance"].eq(appliance)].sort_values("epoch")
        epoch = values["epoch"].to_numpy(dtype=float)
        mean = values["mean"].to_numpy(dtype=float)
        std = values["std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(epoch, mean, color=color, linewidth=1.8)
        ax.fill_between(epoch, mean - std, mean + std, color=color, alpha=0.13, linewidth=0)
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=0.9)
        ax.axhline(1.0 / ratio_limit, color="#888888", linestyle=":", linewidth=0.9)
        ax.axhline(ratio_limit, color="#888888", linestyle=":", linewidth=0.9)
        ax.text(
            0.015,
            0.80,
            display,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=color,
            fontweight="bold",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
        ax.set_yticks([1.0 / ratio_limit, 1.0, 2.0, ratio_limit])
        ax.set_ylim(0.2, ratio_limit + 0.15)
        ax.grid(axis="y", color="#E1E1E1", linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Epoch")
    axes[-1].set_xlim(float(summary["epoch"].min()), float(summary["epoch"].max()))
    fig.supylabel(r"Normalized state-loss coefficient  $r_i / r_g$")
    fig.suptitle(
        "Appliance-wise loss balancing during training",
        fontweight="bold",
        fontsize=12,
    )
    fig.savefig(output_dir / "appliance_weight_trajectories.png", dpi=600)
    fig.savefig(output_dir / "appliance_weight_trajectories.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "datasets" / "mixed_ukdale_refit_5w_all_house_split",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs" / "paper_5w" / "table_v" / "mart_nilm",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "runs"
            / "paper_5w"
            / "table_v"
            / "mart_nilm"
            / "training_strategy_figures"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--ratio-limit", type=float, default=3.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _paper_style()
    event_background_coverage(
        args.dataset_root / "training" / "multi_appliance_training.csv",
        args.output_dir,
        args.window_length,
        args.stride,
    )
    appliance_weight_trajectories(
        args.run_root,
        args.output_dir,
        tuple(args.seeds),
        args.ratio_limit,
    )
    print(f"Saved publication figures and source CSV files to {args.output_dir}")


if __name__ == "__main__":
    main()

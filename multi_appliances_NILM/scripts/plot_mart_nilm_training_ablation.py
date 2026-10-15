#!/usr/bin/env python
"""Plot publication evidence for the MART-NILM training ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028)
APPLIANCES = ("kettle", "fridge", "dishwasher", "washingmachine", "microwave")
DISPLAY_NAMES = ("Kettle", "Fridge", "Dishwasher", "Washing\nmachine", "Microwave")
BACKGROUND_EDGES = np.asarray([0.0, 100.0, 200.0, 400.0, 800.0, np.inf])
BACKGROUND_LABELS = ("0--100", "100--200", "200--400", "400--800", r"$\geq$800")


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
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
        }
    )


def _run_dir(run_root: Path, variant: str, seed: int) -> Path:
    if variant == "complete":
        return run_root / "table_v" / "mart_nilm" / f"seed_{seed}"
    return run_root / "table_viii" / variant / f"seed_{seed}"


def _focal_background_effect(
    run_root: Path,
    validation_csv: Path,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    """Paired validation effect of focal mixing within residual-background bins."""
    aggregate = pd.read_csv(validation_csv, usecols=["aggregate"])["aggregate"].to_numpy(
        dtype=float
    )
    by_variant: dict[str, dict[int, dict[tuple[str, str], tuple[float, float, int]]]] = {}
    for variant in ("standard_joint_training", "focal_event_mixing"):
        by_seed = {}
        for seed in seeds:
            data = np.load(_run_dir(run_root, variant, seed) / "validation_predictions.npz")
            names = tuple(str(name) for name in data["appliances"])
            if names != APPLIANCES:
                raise ValueError(f"Unexpected appliance order: {names}")
            timesteps = data["csv_timesteps"].astype(np.int64)
            y_true = data["y_true_watts"].astype(float)
            y_pred = data["y_pred_watts"].astype(float)
            true_on = data["y_true_on"].astype(bool)
            pred_on = data["y_pred_on"].astype(bool)
            background = np.maximum(aggregate[timesteps] - y_true.sum(axis=1), 0.0)
            cells = {}
            for app_i, appliance in enumerate(APPLIANCES):
                for bin_i, (low, high) in enumerate(
                    zip(BACKGROUND_EDGES[:-1], BACKGROUND_EDGES[1:])
                ):
                    mask = true_on[:, app_i] & (background >= low) & (background < high)
                    count = int(mask.sum())
                    recall = float(pred_on[mask, app_i].mean()) if count else np.nan
                    on_mae = (
                        float(np.abs(y_pred[mask, app_i] - y_true[mask, app_i]).mean())
                        if count
                        else np.nan
                    )
                    cells[(appliance, BACKGROUND_LABELS[bin_i])] = (recall, on_mae, count)
            by_seed[seed] = cells
        by_variant[variant] = by_seed

    rows = []
    for seed in seeds:
        standard = by_variant["standard_joint_training"][seed]
        focal = by_variant["focal_event_mixing"][seed]
        for appliance in APPLIANCES:
            for background_bin in BACKGROUND_LABELS:
                std_recall, std_on_mae, count = standard[(appliance, background_bin)]
                focal_recall, focal_on_mae, _ = focal[(appliance, background_bin)]
                rows.append(
                    {
                        "seed": seed,
                        "appliance": appliance,
                        "background_bin_w": background_bin,
                        "on_samples": count,
                        "recall_change_pp": 100.0 * (focal_recall - std_recall),
                        "on_mae_reduction_w": std_on_mae - focal_on_mae,
                    }
                )
    return pd.DataFrame(rows)


def _balance_effect(
    run_root: Path,
    seeds: tuple[int, ...],
    lambda_state: float,
    balance_limit: float,
    warmup_epochs: int,
) -> pd.DataFrame:
    """Effective state-term/power-loss ratio under fixed and balanced weighting."""
    rows = []
    for variant in ("standard_joint_training", "appliance_wise_balancing"):
        for seed in seeds:
            frame = pd.read_csv(_run_dir(run_root, variant, seed) / "loss_detail.csv")
            frame = frame.loc[frame["epoch"].ge(warmup_epochs)]
            power = frame[[f"train_loss_power_{app}" for app in APPLIANCES]].to_numpy(float)
            state = frame[[f"train_loss_state_{app}" for app in APPLIANCES]].to_numpy(float)
            if variant == "standard_joint_training":
                effective_state = lambda_state * state
            else:
                global_scale = power.sum(axis=1) / np.maximum(state.sum(axis=1), 1e-8)
                local_scale = power / np.maximum(state, 1e-8)
                scale = np.clip(
                    local_scale,
                    global_scale[:, None] / balance_limit,
                    global_scale[:, None] * balance_limit,
                )
                effective_state = lambda_state * state * scale
            ratio = effective_state / np.maximum(power, 1e-8)
            for epoch_i, epoch in enumerate(frame["epoch"].astype(int)):
                for app_i, appliance in enumerate(APPLIANCES):
                    rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "epoch": epoch,
                            "appliance": appliance,
                            "state_to_power_ratio": ratio[epoch_i, app_i],
                        }
                    )
    return pd.DataFrame(rows)


def _annotated_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    value_format: str,
) -> None:
    finite = np.abs(values[np.isfinite(values)])
    span = max(float(finite.max(initial=1.0)), 1.0)
    image = ax.imshow(
        values,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span),
        aspect="auto",
    )
    ax.set_xticks(range(len(BACKGROUND_LABELS)), BACKGROUND_LABELS, rotation=28, ha="right")
    ax.set_yticks(range(len(APPLIANCES)), DISPLAY_NAMES)
    ax.set_xlabel("Residual background (W)")
    ax.set_title(title, fontweight="bold", pad=7)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                color = "white" if abs(value) > 0.55 * span else "black"
                ax.text(
                    col,
                    row,
                    value_format.format(value),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.05, pad=0.035)
    colorbar.set_label(colorbar_label, fontsize=8)
    colorbar.ax.tick_params(labelsize=7)


def plot_training_mechanisms(
    run_root: Path,
    validation_csv: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
    lambda_state: float,
    balance_limit: float,
) -> None:
    focal = _focal_background_effect(run_root, validation_csv, seeds)
    focal.to_csv(
        output_dir / "focal_background_effect.csv", index=False, float_format="%.6f"
    )
    focal_mean = focal.groupby(["appliance", "background_bin_w"], sort=False).mean(
        numeric_only=True
    )
    recall = np.asarray(
        [
            [focal_mean.loc[(app, bin_name), "recall_change_pp"] for bin_name in BACKGROUND_LABELS]
            for app in APPLIANCES
        ]
    )
    on_mae = np.asarray(
        [
            [focal_mean.loc[(app, bin_name), "on_mae_reduction_w"] for bin_name in BACKGROUND_LABELS]
            for app in APPLIANCES
        ]
    )
    on_mae[np.abs(on_mae) < 0.5] = 0.0

    balance = _balance_effect(
        run_root,
        seeds,
        lambda_state=lambda_state,
        balance_limit=balance_limit,
        warmup_epochs=20,
    )
    balance.to_csv(
        output_dir / "appliance_balance_effect.csv", index=False, float_format="%.6f"
    )

    # Each mechanism is saved as an independent IEEE single-column figure.
    # Keeping the figures separate makes the cell values legible at print size
    # and lets the paper explain each empirical effect immediately below it.
    fig, ax = plt.subplots(figsize=(3.55, 3.25), constrained_layout=True)
    _annotated_heatmap(
        ax,
        recall,
        "Recall change from focal-event mixing",
        "Focal minus standard (percentage points)",
        "{:+.1f}",
    )
    fig.savefig(output_dir / "focal_recall_by_background.png", dpi=600)
    fig.savefig(output_dir / "focal_recall_by_background.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.55, 3.25), constrained_layout=True)
    _annotated_heatmap(
        ax,
        on_mae,
        "ON-MAE reduction from focal-event mixing",
        "Standard minus focal (W)",
        "{:+.0f}",
    )
    fig.savefig(output_dir / "focal_on_mae_by_background.png", dpi=600)
    fig.savefig(output_dir / "focal_on_mae_by_background.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.55, 3.15), constrained_layout=True)
    x = np.arange(len(APPLIANCES), dtype=float)
    for offset, variant, label, color in (
        (-0.18, "standard_joint_training", "Fixed task weight", "#666666"),
        (0.18, "appliance_wise_balancing", "Appliance-wise balance", "#0072B2"),
    ):
        values = balance.loc[balance["variant"].eq(variant)]
        grouped = values.groupby("appliance")["state_to_power_ratio"]
        median = grouped.median().reindex(APPLIANCES).to_numpy(float)
        q25 = grouped.quantile(0.25).reindex(APPLIANCES).to_numpy(float)
        q75 = grouped.quantile(0.75).reindex(APPLIANCES).to_numpy(float)
        ax.bar(
            x + offset,
            median,
            width=0.32,
            yerr=np.vstack((median - q25, q75 - median)),
            capsize=2,
            color=color,
            label=label,
        )
    ax.axhline(
        1.0, color="black", linewidth=0.9, linestyle="--", label="Equal magnitude"
    )
    ax.set_yscale("log")
    ax.set_ylim(0.005, 2.5)
    ax.set_xticks(x, DISPLAY_NAMES, rotation=22, ha="right")
    ax.set_ylabel("Effective state term / power loss")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, which="major")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=6.3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        labelspacing=0.25,
        handlelength=1.4,
    )

    fig.savefig(output_dir / "appliance_balance_scale.png", dpi=600)
    fig.savefig(output_dir / "appliance_balance_scale.pdf")
    plt.close(fig)


def plot_per_appliance_before_after(summary_path: Path, output_dir: Path) -> None:
    frame = pd.read_csv(summary_path)
    standard = frame.loc[frame["variant"].eq("standard_joint_training")].set_index(
        "appliance"
    )
    complete = frame.loc[frame["variant"].eq("complete")].set_index("appliance")
    x = np.arange(len(APPLIANCES), dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True, constrained_layout=True)
    panels = (
        ("f1", "F1", "(a) State detection", True),
        ("on_mae", "ON MAE (W)", "(b) ON-state power estimation", False),
    )
    for ax, (metric, ylabel, title, bounded) in zip(axes, panels):
        for offset, values, label, color in (
            (-0.08, standard, "Standard joint training", "#666666"),
            (0.08, complete, "Complete MART-NILM", "#0072B2"),
        ):
            mean = values.loc[list(APPLIANCES), f"{metric}_mean"].to_numpy(dtype=float)
            std = values.loc[list(APPLIANCES), f"{metric}_std"].to_numpy(dtype=float)
            ax.errorbar(
                x + offset,
                mean,
                yerr=std,
                marker="o",
                markersize=5,
                linewidth=1.8,
                capsize=3,
                color=color,
                label=label,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        if bounded:
            ax.set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False, loc="lower left")
    axes[-1].set_xticks(x, DISPLAY_NAMES)
    axes[-1].set_xlabel("Appliance")
    fig.suptitle("Per-appliance effect of the complete training strategy", fontweight="bold", fontsize=12)
    fig.savefig(output_dir / "training_strategy_per_appliance.png", dpi=600)
    fig.savefig(output_dir / "training_strategy_per_appliance.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs" / "paper_5w")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=(
            ROOT
            / "runs"
            / "paper_5w"
            / "summaries"
            / "table_viii_per_appliance_mart_nilm.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "paper_5w" / "table_viii" / "figures",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=(
            ROOT
            / "datasets"
            / "mixed_ukdale_refit_5w_all_house_split"
            / "validating"
            / "multi_appliance_validating.csv"
        ),
    )
    parser.add_argument("--lambda-state", type=float, default=0.8)
    parser.add_argument("--balance-limit", type=float, default=3.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _paper_style()
    plot_training_mechanisms(
        args.run_root,
        args.validation_csv,
        args.output_dir,
        tuple(args.seeds),
        args.lambda_state,
        args.balance_limit,
    )
    plot_per_appliance_before_after(args.summary_path, args.output_dir)
    print(f"Saved Table VIII figures and source CSV to {args.output_dir}")


if __name__ == "__main__":
    main()

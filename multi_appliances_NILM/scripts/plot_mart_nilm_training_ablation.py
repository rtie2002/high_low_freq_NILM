#!/usr/bin/env python
"""Plot publication evidence for the MART-NILM training ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028)
APPLIANCES = ("kettle", "fridge", "dishwasher", "washingmachine", "microwave")
DISPLAY_NAMES = ("Kettle", "Fridge", "Dishwasher", "Washing\nmachine", "Microwave")


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


def _history_summary(
    run_root: Path,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for variant in variants:
        for seed in seeds:
            history = pd.read_csv(_run_dir(run_root, variant, seed) / "history.csv")
            for row in history.itertuples(index=False):
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "epoch": int(row.epoch),
                        "val_ap": float(row.val_ap),
                        "val_mae_norm": float(row.val_mae_norm),
                    }
                )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["variant", "epoch"], sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            val_ap_mean=("val_ap", "mean"),
            val_ap_std=("val_ap", lambda values: values.std(ddof=1)),
            val_mae_norm_mean=("val_mae_norm", "mean"),
            val_mae_norm_std=("val_mae_norm", lambda values: values.std(ddof=1)),
        )
        .reset_index()
    )
    return summary.loc[summary["n_seeds"].eq(len(seeds))].copy()


def plot_focal_generalization(
    run_root: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
) -> None:
    variants = ("standard_joint_training", "focal_event_mixing")
    summary = _history_summary(run_root, variants, seeds)
    summary.to_csv(
        output_dir / "focal_event_validation_curves.csv",
        index=False,
        float_format="%.6f",
    )

    styles = {
        "standard_joint_training": ("Standard joint training", "#666666"),
        "focal_event_mixing": ("+ Focal-event mixing", "#0072B2"),
    }
    panels = (
        ("val_ap_mean", "val_ap_std", "Validation macro-AP", "(a) State ranking"),
        (
            "val_mae_norm_mean",
            "val_mae_norm_std",
            "Normalized validation MAE",
            "(b) Power estimation",
        ),
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True, constrained_layout=True)
    for ax, (mean_col, std_col, ylabel, title) in zip(axes, panels):
        for variant in variants:
            values = summary.loc[summary["variant"].eq(variant)].sort_values("epoch")
            epoch = values["epoch"].to_numpy(dtype=float)
            mean = values[mean_col].to_numpy(dtype=float)
            std = values[std_col].fillna(0.0).to_numpy(dtype=float)
            label, color = styles[variant]
            ax.plot(epoch, mean, color=color, linewidth=2.0, label=label)
            ax.fill_between(epoch, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right")
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Validation dynamics of focal-event mixing", fontweight="bold", fontsize=12)
    fig.savefig(output_dir / "focal_event_validation_curves.png", dpi=600)
    fig.savefig(output_dir / "focal_event_validation_curves.pdf")
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
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _paper_style()
    plot_focal_generalization(args.run_root, args.output_dir, tuple(args.seeds))
    plot_per_appliance_before_after(args.summary_path, args.output_dir)
    print(f"Saved Table VIII figures and source CSV to {args.output_dir}")


if __name__ == "__main__":
    main()

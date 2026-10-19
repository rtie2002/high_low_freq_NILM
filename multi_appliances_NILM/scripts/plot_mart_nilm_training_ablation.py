#!/usr/bin/env python
"""Plot publication evidence for the MART-NILM training ablation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _validation_house_rows(dataset_root: Path) -> list[tuple[str, int, int]]:
    summary = pd.read_csv(dataset_root / "selection_summary.csv")
    rows = summary.loc[summary["role"].eq("validation")]
    result = []
    start = 0
    for row in rows.itertuples(index=False):
        stop = start + int(row.n_rows)
        result.append((f"{row.dataset}_house{row.house}", start, stop))
        start = stop
    if len(result) != 2:
        raise ValueError(f"Expected two validation houses, found {len(result)}")
    return result


def paired_complete_changes(
    run_root: Path,
    dataset_root: Path,
    seeds: tuple[int, ...],
    output_path: Path,
) -> pd.DataFrame:
    """Recompute paired changes from predictions using equal-house metrics.

    Each seed's per-appliance metric is first evaluated independently on each
    held-out validation house and then averaged with equal house weight, exactly
    matching Table VIII. The three seed-level paired changes are retained in
    the CSV so the plot's uncertainty is reproducible.
    """
    import yaml

    house_ranges = _validation_house_rows(dataset_root)
    rows = []
    for seed in seeds:
        per_variant: dict[str, dict[str, dict[str, float]]] = {}
        for variant in ("standard_joint_training", "complete"):
            run_dir = _run_dir(run_root, variant, seed)
            bundle = np.load(run_dir / "validation_predictions.npz", allow_pickle=True)
            with (run_dir / "config_merged.yaml").open("r", encoding="utf-8") as handle:
                saved_cfg = yaml.safe_load(handle)
            experiment_cfg = saved_cfg.get("experiment", saved_cfg)
            model_cfg = saved_cfg.get("model", saved_cfg)
            appliances = tuple(str(name) for name in bundle["appliances"])
            if appliances != APPLIANCES:
                raise ValueError(f"Unexpected appliance order in {run_dir}: {appliances}")
            if model_cfg.get("data", {}).get("state_label_source", "auto") != "csv":
                raise ValueError(f"Expected CSV state labels in {run_dir}")
            evaluation = experiment_cfg["evaluation"]
            min_power = float(evaluation.get("power_postprocess", {}).get("min_power_watts", 0.0))
            max_power = np.asarray(
                [evaluation["max_on_power_watts"][app] for app in APPLIANCES], dtype=float
            )
            y_true_power = np.maximum(bundle["y_true_watts"].astype(float), 0.0)
            y_pred_power = np.maximum(bundle["y_pred_watts"].astype(float), 0.0)
            if evaluation.get("power_postprocess", {}).get("enabled", False):
                y_pred_power[y_pred_power < min_power] = 0.0
                y_pred_power = np.clip(y_pred_power, 0.0, max_power[None, :])
            y_true_on = bundle["y_true_on"].astype(bool)
            y_pred_on = bundle["y_pred_on"].astype(bool)
            timesteps = bundle["csv_timesteps"].astype(np.int64)
            house_metrics: dict[str, list[tuple[float, float]]] = {
                appliance: [] for appliance in APPLIANCES
            }
            for house, start, stop in house_ranges:
                mask = (timesteps >= start) & (timesteps < stop)
                if not mask.any():
                    raise ValueError(f"No predictions for {house} in seed {seed}")
                for app_i, appliance in enumerate(APPLIANCES):
                    truth_on = y_true_on[mask, app_i]
                    pred_on = y_pred_on[mask, app_i]
                    tp = np.logical_and(truth_on, pred_on).sum()
                    fp = np.logical_and(~truth_on, pred_on).sum()
                    fn = np.logical_and(truth_on, ~pred_on).sum()
                    f1 = float(2.0 * tp / max(2.0 * tp + fp + fn, 1.0))
                    on_error = np.abs(
                        y_pred_power[mask, app_i] - y_true_power[mask, app_i]
                    )
                    on_mae = float(on_error[truth_on].mean()) if truth_on.any() else np.nan
                    house_metrics[appliance].append((f1, on_mae))
            per_variant[variant] = {}
            for appliance, values in house_metrics.items():
                per_variant[variant][appliance] = {
                    metric: float(np.mean([row[metric_i] for row in values]))
                    for metric, metric_i in (("f1", 0), ("on_mae", 1))
                }

        for appliance in APPLIANCES:
            standard = per_variant["standard_joint_training"][appliance]
            complete = per_variant["complete"][appliance]
            rows.append(
                {
                    "seed": seed,
                    "appliance": appliance,
                    "f1_change": complete["f1"] - standard["f1"],
                    "on_mae_reduction_w": standard["on_mae"] - complete["on_mae"],
                }
            )
    paired = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_path, index=False, float_format="%.8f")
    return paired


def _focal_background_effect(
    run_root: Path,
    validation_csv: Path,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    """Paired validation effect of focal mixing within residual-background bins."""
    import yaml

    from data.common import PredictionBundle
    from data.dataloader import get_state_label_source
    from evaluation.metrics import evaluate_bundle, resolve_power_postprocess

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


def plot_gradient_scale(output_dir: Path) -> None:
    """Plot measured shared-encoder gradient scales, not loss magnitudes.

    A detached positive balance coefficient rescales a state gradient without
    rotating it.  The diagnostic therefore reports the directly measured norm
    ratio ||grad(state)|| / ||grad(power)||, rather than treating loss values as
    a proxy for gradients.
    """
    paths = [output_dir / f"gradient_audit_seed{seed}.csv" for seed in SEEDS]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Gradient-audit files are required: {missing}")
    audit = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    # Average batches within a seed first so every independently trained seed
    # carries equal weight, even if an audit processed a different batch count.
    per_seed = (
        audit.groupby(["seed", "appliance", "weighting_rule"], as_index=False)
        .agg(
            state_to_power_gradient_ratio=("state_to_power_gradient_ratio", "mean"),
            power_state_gradient_cosine=("power_state_gradient_cosine", "mean"),
        )
    )
    summary = (
        per_seed.groupby(["appliance", "weighting_rule"], as_index=False)
        .agg(
            gradient_ratio_mean=("state_to_power_gradient_ratio", "mean"),
            gradient_ratio_std=("state_to_power_gradient_ratio", "std"),
            cosine_mean=("power_state_gradient_cosine", "mean"),
            cosine_std=("power_state_gradient_cosine", "std"),
        )
    )
    summary.to_csv(output_dir / "appliance_gradient_scale.csv", index=False, float_format="%.6f")

    fig, ax = plt.subplots(figsize=(3.55, 3.15), constrained_layout=True)
    x = np.arange(len(APPLIANCES), dtype=float)
    for offset, rule, label, color in (
        (-0.18, "Fixed task weight", "Fixed task weight", "#666666"),
        (0.18, "Appliance-wise balance", "Appliance-wise balance", "#0072B2"),
    ):
        values = summary.loc[summary["weighting_rule"].eq(rule)].set_index("appliance")
        mean = values.loc[list(APPLIANCES), "gradient_ratio_mean"].to_numpy(float)
        std = values.loc[list(APPLIANCES), "gradient_ratio_std"].fillna(0.0).to_numpy(float)
        ax.bar(x + offset, mean, width=0.32, yerr=std, capsize=2, color=color, label=label)
    ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--", label="Matched scale")
    ax.set_yscale("log")
    ax.set_ylim(0.005, 20.0)
    ax.set_xticks(x, DISPLAY_NAMES, rotation=22, ha="right")
    ax.set_ylabel(r"$\Vert g_{\mathrm{state}}\Vert / \Vert g_{\mathrm{power}}\Vert$")
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

    plot_gradient_scale(output_dir)


def plot_per_appliance_change(paired_changes_path: Path, output_dir: Path) -> None:
    """Plot paired validation changes; positive values always mean improvement."""
    if not paired_changes_path.is_file():
        raise FileNotFoundError(f"Missing paired validation summary: {paired_changes_path}")
    paired = pd.read_csv(paired_changes_path)
    summary = paired.groupby("appliance", as_index=False).agg(
        f1_change_mean=("f1_change", "mean"),
        f1_change_std=("f1_change", "std"),
        on_mae_reduction_mean=("on_mae_reduction_w", "mean"),
        on_mae_reduction_std=("on_mae_reduction_w", "std"),
    ).set_index("appliance").loc[list(APPLIANCES)]

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 4.35), sharey=True, constrained_layout=True)
    y = np.arange(len(APPLIANCES), dtype=float)
    panels = (
        ("f1_change_mean", "f1_change_std", r"$\Delta$F1", "(a) State detection"),
        ("on_mae_reduction_mean", "on_mae_reduction_std", r"ON-MAE reduction (W)", "(b) ON-state power estimation"),
    )
    for ax, (mean_col, std_col, xlabel, title) in zip(axes, panels):
        values = summary[mean_col].to_numpy(float)
        errors = summary[std_col].fillna(0.0).to_numpy(float)
        colors = np.where(values >= 0.0, "#0072B2", "#C44E52")
        for pos, value, error, color in zip(y, values, errors, colors):
            ax.errorbar(
                value,
                pos,
                xerr=error,
                fmt="o",
                markersize=4.5,
                capsize=2.5,
                linewidth=1.2,
                color=color,
                zorder=3,
            )
        ax.axvline(0.0, color="black", linewidth=0.8)
        span = max(float(np.ptp(np.concatenate((values - errors, values + errors)))), 0.05)
        label_pad = max(0.035 * span, 0.01 if "f1" in mean_col else 1.0)
        extent = np.concatenate((values - errors, values + errors))
        # The source CSV contains exact seed-level values; omit in-plot number
        # labels so the small negative microwave effect remains uncluttered.
        ax.set_xlim(min(0.0, float(extent.min()) - 2 * label_pad), float(extent.max()) + 2 * label_pad)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, DISPLAY_NAMES)
    axes[0].invert_yaxis()
    axes[1].set_yticks(y, DISPLAY_NAMES)
    axes[1].invert_yaxis()
    fig.savefig(output_dir / "training_strategy_per_appliance.png", dpi=600)
    fig.savefig(output_dir / "training_strategy_per_appliance.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs" / "paper_5w")
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
    parser.add_argument(
        "--paired-changes-path",
        type=Path,
        default=(
            ROOT
            / "runs"
            / "paper_5w"
            / "table_viii"
            / "figures"
            / "complete_strategy_paired_changes.csv"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "datasets" / "mixed_ukdale_refit_5w_all_house_split",
        help="House boundaries used to recompute paired equal-house validation metrics.",
    )
    parser.add_argument(
        "--only-per-appliance",
        action="store_true",
        help="Regenerate only the per-appliance plot, recomputing its source metrics.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _paper_style()
    if not args.only_per_appliance:
        plot_training_mechanisms(
            args.run_root,
            args.validation_csv,
            args.output_dir,
            tuple(args.seeds),
            args.lambda_state,
            args.balance_limit,
        )
    paired_complete_changes(
        args.run_root,
        args.dataset_root,
        tuple(args.seeds),
        args.paired_changes_path,
    )
    plot_per_appliance_change(args.paired_changes_path, args.output_dir)
    print(f"Saved Table VIII figures and source CSV to {args.output_dir}")


if __name__ == "__main__":
    main()

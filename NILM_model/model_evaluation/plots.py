from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_history(
    history: pd.DataFrame | str | Path,
    output_path: str | Path,
    *,
    epoch_col: str = "epoch",
    loss_cols: Iterable[str] = ("train_loss", "val_loss"),
    metric_cols: Iterable[str] = ("val_mae", "val_sae", "val_f1"),
    title: str = "Training History",
    dpi: int = 300,
) -> Path:
    """Plot train/validation loss and validation metrics from a history CSV/DataFrame."""
    if not isinstance(history, pd.DataFrame):
        history = pd.read_csv(history)

    x = history[epoch_col] if epoch_col in history else np.arange(len(history))
    loss_cols = [col for col in loss_cols if col in history]
    metric_cols = [col for col in metric_cols if col in history]
    n_rows = 1 + int(bool(metric_cols))

    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 8), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for col in loss_cols:
        axes[0].plot(x, history[col], marker="o", markersize=2.8, linewidth=1.6, label=col)
    axes[0].set_title(title)
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    if metric_cols:
        for col in metric_cols:
            axes[1].plot(x, history[col], marker="o", markersize=2.8, linewidth=1.6, label=col)
        axes[1].set_ylabel("Metric")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend()

    axes[-1].set_xlabel("Epoch")
    if len(x):
        xmin = int(np.nanmin(x))
        xmax = int(np.nanmax(x))
        pad = max(1, int(0.02 * max(1, xmax - xmin)))
        axes[-1].set_xlim(xmin - pad, xmax + pad)
    fig.tight_layout()
    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _set_epoch_axis(ax, x: pd.Series | np.ndarray) -> None:
    if not len(x):
        return
    xmin = int(np.nanmin(x))
    xmax = int(np.nanmax(x))
    pad = max(1, int(0.02 * max(1, xmax - xmin)))
    ax.set_xlim(xmin - pad, xmax + pad)


def _set_dynamic_y_axis(ax, values: list[np.ndarray]) -> None:
    finite_chunks = [item[np.isfinite(item)] for item in values if len(item)]
    finite_chunks = [item for item in finite_chunks if len(item)]
    if not finite_chunks:
        return
    finite_values = np.concatenate(finite_chunks)
    if len(finite_values) == 0:
        return
    ymin = float(np.nanmin(finite_values))
    ymax = float(np.nanmax(finite_values))
    if np.isclose(ymin, ymax):
        delta = max(1e-6, abs(ymax) * 0.1)
        ymin -= delta
        ymax += delta
    else:
        delta = 0.08 * (ymax - ymin)
        ymin -= delta
        ymax += delta
    ax.set_ylim(max(0.0, ymin), ymax)


def plot_loss_details(
    loss_detail: pd.DataFrame | str | Path,
    output_path: str | Path,
    *,
    epoch_col: str = "epoch",
    title: str = "Detailed Loss Curves",
    dpi: int = 300,
) -> Path:
    """Plot overall and per-appliance train/validation loss components."""
    if not isinstance(loss_detail, pd.DataFrame):
        loss_detail = pd.read_csv(loss_detail)

    x = loss_detail[epoch_col] if epoch_col in loss_detail else np.arange(len(loss_detail))
    appliances = sorted(
        {
            col.removeprefix("train_").removesuffix("_loss")
            for col in loss_detail.columns
            if col.startswith("train_")
            and col.endswith("_loss")
            and col
            not in {
                "train_loss",
                "train_output_loss",
                "train_on_loss",
            }
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
    axes = axes.reshape(-1)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(appliances))))

    panels = [
        (
            axes[0],
            "Overall Total Loss",
            [("train_loss", "train total", "#1f77b4"), ("val_loss", "val total", "#d62728")],
        ),
        (
            axes[1],
            "Overall Components",
            [
                ("train_output_loss", "train regression", "#1f77b4"),
                ("val_output_loss", "val regression", "#17becf"),
                ("train_on_loss", "train classification", "#d62728"),
                ("val_on_loss", "val classification", "#ff7f0e"),
            ],
        ),
    ]

    for ax, panel_title, cols in panels:
        plotted = []
        for col, label, color in cols:
            if col in loss_detail:
                y = loss_detail[col].to_numpy(dtype=float)
                plotted.append(y)
                ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.5, color=color, label=label)
        ax.set_title(panel_title)
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        _set_dynamic_y_axis(ax, plotted)

    for appliance, color in zip(appliances, colors):
        for split, linestyle in (("train", "-"), ("val", "--")):
            col = f"{split}_{appliance}_loss"
            if col in loss_detail:
                axes[2].plot(
                    x,
                    loss_detail[col],
                    linestyle=linestyle,
                    linewidth=1.35,
                    color=color,
                    label=f"{appliance} {split}",
                )
    axes[2].set_title("Per-Appliance Total Loss")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=7, ncol=2)
    _set_dynamic_y_axis(
        axes[2],
        [
            loss_detail[col].to_numpy(dtype=float)
            for col in loss_detail.columns
            if any(col == f"{split}_{app}_loss" for app in appliances for split in ("train", "val"))
        ],
    )

    for appliance, color in zip(appliances, colors):
        for suffix, linestyle in (("output_loss", "-"), ("on_loss", ":")):
            col = f"val_{appliance}_{suffix}"
            if col in loss_detail:
                label = f"{appliance} {'reg' if suffix == 'output_loss' else 'cls'}"
                axes[3].plot(x, loss_detail[col], linestyle=linestyle, linewidth=1.35, color=color, label=label)
    axes[3].set_title("Validation Loss By Appliance")
    axes[3].set_ylabel("Loss")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(fontsize=7, ncol=2)
    _set_dynamic_y_axis(
        axes[3],
        [
            loss_detail[col].to_numpy(dtype=float)
            for col in loss_detail.columns
            if any(col in {f"val_{app}_output_loss", f"val_{app}_on_loss"} for app in appliances)
        ],
    )

    for ax in axes:
        ax.set_xlabel("Epoch")
        _set_epoch_axis(ax, x)

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _plot_single_appliance_event_panels(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    appliance: str,
    true_col: str,
    pred_col: str,
    aggregate_col: str | None,
    time_col: str | None,
    groups: list[np.ndarray],
    n_periods: int,
    title: str,
    dpi: int,
) -> Path:
    groups = sorted(groups, key=len, reverse=True)[:n_periods]
    groups = sorted(groups, key=lambda group: int(group[0]))
    n_rows = len(groups)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 10), sharex=False)
    if n_rows == 1:
        axes = [axes]

    has_aggregate = bool(aggregate_col and aggregate_col in frame)
    for row, group in enumerate(groups):
        ax = axes[row]
        span = int(group[-1] - group[0] + 1)
        margin = max(40, min(240, span))
        panel_samples = max(160, span + 2 * margin)
        center = int((group[0] + group[-1]) // 2)
        start = max(0, center - panel_samples // 2)
        end = min(len(frame), start + panel_samples)
        start = max(0, end - panel_samples)
        view = frame.iloc[start:end]

        if time_col and time_col in view:
            x = pd.to_datetime(view[time_col], errors="coerce")
            if x.isna().all():
                x = np.arange(start, end)
        else:
            x = np.arange(start, end)

        true_values = view[true_col].to_numpy(dtype=float)
        pred_values = view[pred_col].to_numpy(dtype=float)
        ymin = float(np.nanmin([np.nanmin(true_values), np.nanmin(pred_values), 0.0]))
        ymax = float(np.nanmax([np.nanmax(true_values), np.nanmax(pred_values), 1.0]))
        if np.isclose(ymin, ymax):
            ymax = ymin + 1.0

        if has_aggregate:
            agg = view[aggregate_col].to_numpy(dtype=float)
            agg_min = float(np.nanmin(agg))
            agg_max = float(np.nanmax(agg))
            if not np.isclose(agg_min, agg_max):
                agg_scaled = (agg - agg_min) / (agg_max - agg_min)
                agg_scaled = ymin + agg_scaled * (ymax - ymin)
                ax.fill_between(x, ymin, agg_scaled, color="#d9d9d9", alpha=0.35, label="aggregate background")
                ax.plot(x, agg_scaled, color="#8a8a8a", linewidth=0.8, alpha=0.7)

        on_col = f"{appliance}_on"
        if on_col in view:
            on_values = view[on_col].to_numpy(dtype=float) > 0.5
            active = np.flatnonzero(on_values)
            if len(active):
                local_groups = np.split(active, np.where(np.diff(active) != 1)[0] + 1)
                for local_group in local_groups:
                    ax.axvspan(x[local_group[0]], x[local_group[-1]], color="#ffe08a", alpha=0.22, linewidth=0)

        ax.plot(x, true_values, color="#1f77b4", linewidth=1.5, label=f"{appliance} true")
        ax.plot(x, pred_values, color="#d62728", linewidth=1.25, alpha=0.9, label=f"{appliance} pred")
        ax.set_ylabel("Power W")
        ax.set_title(f"ON period {row + 1}", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.12 * (ymax - ymin))
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel(time_col if time_col else "sample index")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_prediction_waveforms(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    time_col: str | None = None,
    aggregate_col: str | None = "aggregate",
    true_pred_pairs: dict[str, tuple[str, str]],
    start: int = 0,
    samples: int = 2000,
    focus_on_activity: bool = True,
    focus_on_periods: int = 5,
    aggregate_background: bool = True,
    title: str = "NILM Prediction Waveforms",
    dpi: int = 300,
) -> Path:
    """Plot aggregate power plus true/predicted appliance waveforms."""
    if focus_on_activity:
        on_cols = [
            f"{appliance}_on"
            for appliance in true_pred_pairs
            if f"{appliance}_on" in frame
        ]
        if on_cols:
            activity = frame[on_cols].fillna(0).to_numpy(dtype=float).sum(axis=1)
            active_indices = np.flatnonzero(activity > 0.5)
            if len(active_indices):
                groups = np.split(active_indices, np.where(np.diff(active_indices) != 1)[0] + 1)
                groups = [group for group in groups if len(group)]
                if len(true_pred_pairs) == 1 and groups and focus_on_periods > 1:
                    appliance, (true_col, pred_col) = next(iter(true_pred_pairs.items()))
                    return _plot_single_appliance_event_panels(
                        frame,
                        output_path,
                        appliance=appliance,
                        true_col=true_col,
                        pred_col=pred_col,
                        aggregate_col=aggregate_col,
                        time_col=time_col,
                        groups=groups,
                        n_periods=min(focus_on_periods, len(groups)),
                        title=title,
                        dpi=dpi,
                    )
                if groups and focus_on_periods > 1:
                    group_count = min(focus_on_periods, len(groups))
                    best_start = 0
                    best_span = None
                    for idx in range(0, len(groups) - group_count + 1):
                        span = int(groups[idx + group_count - 1][-1] - groups[idx][0])
                        if best_span is None or span < best_span:
                            best_span = span
                            best_start = idx
                    selected = groups[best_start : best_start + group_count]
                    focus_start = int(selected[0][0])
                    focus_end = int(selected[-1][-1])
                    focus_span = max(1, focus_end - focus_start + 1)
                    samples = max(samples, int(focus_span * 1.25))
                    center = (focus_start + focus_end) // 2
                else:
                    center = int(active_indices[len(active_indices) // 2])
                start = max(0, center - samples // 2)

    end = min(start + samples, len(frame))
    if start < 0 or start >= end:
        raise ValueError(f"Invalid plot range start={start}, samples={samples}, rows={len(frame)}")
    view = frame.iloc[start:end]

    if time_col and time_col in view:
        x = pd.to_datetime(view[time_col], errors="coerce")
        if x.isna().all():
            x = np.arange(start, end)
    else:
        x = np.arange(start, end)

    has_aggregate = bool(aggregate_col and aggregate_col in frame)
    show_aggregate_row = has_aggregate and not aggregate_background
    n_rows = int(show_aggregate_row) + len(true_pred_pairs)
    figure_size = 8 if n_rows <= 2 else 10
    fig, axes = plt.subplots(n_rows, 1, figsize=(figure_size, figure_size), sharex=True)
    if n_rows == 1:
        axes = [axes]

    row = 0
    if show_aggregate_row:
        axes[row].plot(x, view[aggregate_col], color="#222222", linewidth=1.0, label=aggregate_col)
        axes[row].set_ylabel("Aggregate W")
        axes[row].grid(True, alpha=0.25)
        axes[row].legend(loc="upper right")
        row += 1

    for appliance, (true_col, pred_col) in true_pred_pairs.items():
        ax = axes[row]
        true_values = view[true_col].to_numpy(dtype=float)
        pred_values = view[pred_col].to_numpy(dtype=float)
        ymin = float(np.nanmin([np.nanmin(true_values), np.nanmin(pred_values), 0.0]))
        ymax = float(np.nanmax([np.nanmax(true_values), np.nanmax(pred_values), 1.0]))
        if np.isclose(ymin, ymax):
            ymax = ymin + 1.0

        if has_aggregate and aggregate_background:
            agg = view[aggregate_col].to_numpy(dtype=float)
            agg_min = float(np.nanmin(agg))
            agg_max = float(np.nanmax(agg))
            if not np.isclose(agg_min, agg_max):
                agg_scaled = (agg - agg_min) / (agg_max - agg_min)
                agg_scaled = ymin + agg_scaled * (ymax - ymin)
                ax.fill_between(x, ymin, agg_scaled, color="#d9d9d9", alpha=0.35, label="aggregate background")
                ax.plot(x, agg_scaled, color="#8a8a8a", linewidth=0.8, alpha=0.7)

        on_col = f"{appliance}_on"
        if on_col in view:
            on_values = view[on_col].to_numpy(dtype=float) > 0.5
            active = np.flatnonzero(on_values)
            if len(active):
                groups = np.split(active, np.where(np.diff(active) != 1)[0] + 1)
                for group in groups:
                    ax.axvspan(x[group[0]], x[group[-1]], color="#ffe08a", alpha=0.18, linewidth=0)

        ax.plot(x, true_values, color="#1f77b4", linewidth=1.5, label=f"{appliance} true")
        ax.plot(x, pred_values, color="#d62728", linewidth=1.25, alpha=0.9, label=f"{appliance} pred")
        ax.set_ylabel("Power W")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.10 * (ymax - ymin))
        row += 1

    fig.suptitle(title)
    axes[-1].set_xlabel(time_col if time_col else "sample index")
    fig.tight_layout()
    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path

"""Training loss and per-appliance ON-period waveform plots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

WAVEFORM_DPI = 300


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _set_epoch_axis(ax, x: pd.Series | np.ndarray) -> None:
    if len(x) == 0:
        return
    xmin = int(np.nanmin(x))
    xmax = int(np.nanmax(x))
    pad = max(1, int(0.02 * max(1, xmax - xmin)))
    ax.set_xlim(max(0, xmin - pad), xmax + pad)


def _find_on_events(on: np.ndarray, *, min_duration: int = 10) -> list[tuple[int, int]]:
    mask = np.asarray(on).reshape(-1) >= 0.5
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
    ends = np.flatnonzero(np.diff(padded.astype(int)) == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_duration]


def _pick_random_on_centers(
    on: np.ndarray,
    power: np.ndarray,
    *,
    n_periods: int,
    rng: np.random.Generator,
    min_duration: int = 10,
) -> list[int]:
    events = _find_on_events(on, min_duration=min_duration)
    if events:
        if len(events) <= n_periods:
            picks = events
        else:
            picks = [events[i] for i in rng.choice(len(events), size=n_periods, replace=False)]
        return [int((start + end) // 2) for start, end in picks]

    values = np.asarray(power, dtype=float).reshape(-1)
    if not len(values):
        return []
    order = np.argsort(values)[::-1]
    centers: list[int] = []
    min_sep = max(20, min_duration)
    for idx in order:
        if values[idx] <= 0:
            break
        if all(abs(int(idx) - c) >= min_sep for c in centers):
            centers.append(int(idx))
        if len(centers) >= n_periods:
            break
    return sorted(centers)


def plot_single_on_period(
    *,
    appliance: str,
    y_true_watts: np.ndarray,
    y_pred_watts: np.ndarray,
    center: int,
    output_path: str | Path,
    period_samples: int = 400,
    aggregate: np.ndarray | None = None,
    y_pred_on: np.ndarray | None = None,
    title: str | None = None,
    dpi: int = WAVEFORM_DPI,
) -> Path:
    """High-resolution true vs predicted waveform for one ON period."""
    half = max(period_samples // 2, 10)
    start = max(0, center - half)
    end = min(len(y_true_watts), start + period_samples)
    start = max(0, end - period_samples)

    sl = slice(start, end)
    x = np.arange(start, end)
    true_v = np.asarray(y_true_watts, dtype=float)[sl]
    pred_v = np.maximum(np.asarray(y_pred_watts, dtype=float)[sl], 0.0)

    fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
    agg_view = aggregate[sl] if aggregate is not None and len(aggregate) >= end else None
    if agg_view is not None:
        ax.fill_between(x, 0, agg_view, color="#d9d9d9", alpha=0.14)
        ax.plot(x, agg_view, color="#8a8a8a", linewidth=0.9, alpha=0.55, label="mains (norm)")

    if y_pred_on is not None:
        on_mask = np.asarray(y_pred_on)[sl].astype(bool)
        active = np.flatnonzero(on_mask)
        if len(active):
            groups = np.split(active, np.where(np.diff(active) != 1)[0] + 1)
            for group in groups:
                ax.axvspan(x[group[0]], x[group[-1]], color="#7ad66d", alpha=0.18, linewidth=0)

    ax.plot(x, true_v, color="#1f77b4", linewidth=1.8, label=f"{appliance} true")
    ax.plot(x, pred_v, color="#d62728", linewidth=1.5, alpha=0.92, label=f"{appliance} pred")
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Timestep index")
    ax.grid(True, alpha=0.25)
    ax.set_title(title or f"{appliance} ON period (center={center})")

    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#7ad66d", alpha=0.18, label="pred ON"))
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    ymin = min(0.0, float(np.min(true_v)), float(np.min(pred_v)))
    ymax = max(1.0, float(np.max(true_v)), float(np.max(pred_v)))
    pad = 0.1 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)

    fig.tight_layout()
    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_appliance_on_waveforms(
    output_dir: str | Path,
    *,
    appliances: list[str],
    y_true_watts: np.ndarray,
    y_pred_watts: np.ndarray,
    y_true_on: np.ndarray | None = None,
    y_pred_on: np.ndarray | None = None,
    aggregate: np.ndarray | None = None,
    n_periods: int = 5,
    period_samples: int = 400,
    dpi: int = WAVEFORM_DPI,
    rng: np.random.Generator | None = None,
    file_prefix: str = "on",
    title_prefix: str = "",
) -> list[Path]:
    """Save N random ON-period plots per appliance under output_dir/<appliance>/."""
    output_dir = Path(output_dir)
    rng = rng or np.random.default_rng()
    y_true = np.asarray(y_true_watts, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred_watts, dtype=float), 0.0)
    saved: list[Path] = []

    for idx, app in enumerate(appliances):
        app_dir = output_dir / app
        app_dir.mkdir(parents=True, exist_ok=True)

        true_on = y_true_on[:, idx] if y_true_on is not None else (y_true[:, idx] > 15.0).astype(np.float32)
        pred_on = y_pred_on[:, idx] if y_pred_on is not None else None
        centers = _pick_random_on_centers(true_on, y_true[:, idx], n_periods=n_periods, rng=rng)
        if not centers:
            continue

        for period_i, center in enumerate(centers, start=1):
            path = app_dir / f"{file_prefix}_{period_i:02d}_t{center}.png"
            title = f"{title_prefix}{app} ON period {period_i}".strip()
            plot_single_on_period(
                appliance=app,
                y_true_watts=y_true[:, idx],
                y_pred_watts=y_pred[:, idx],
                center=center,
                output_path=path,
                period_samples=period_samples,
                aggregate=aggregate,
                y_pred_on=pred_on,
                title=title,
                dpi=dpi,
            )
            saved.append(path)
    return saved


def plot_training_history(
    history: pd.DataFrame | str | Path,
    output_path: str | Path,
    *,
    epoch_col: str = "epoch",
    loss_cols: Iterable[str] = ("train_loss", "val_loss"),
    component_cols: Iterable[str] = ("val_loss_state", "val_loss_power"),
    title: str = "Training Loss",
    best_epoch: int | None = None,
    dpi: int = 150,
) -> Path:
    if not isinstance(history, pd.DataFrame):
        history = pd.read_csv(history)

    x = history[epoch_col] if epoch_col in history else np.arange(len(history))
    loss_cols = [c for c in loss_cols if c in history.columns]
    component_cols = [c for c in component_cols if c in history.columns and not history[c].isna().all()]

    n_rows = 1 + int(bool(component_cols))
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 3.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for col in loss_cols:
        axes[0].plot(x, history[col], marker="o", markersize=3, linewidth=1.6, label=col)
    if best_epoch is not None and best_epoch > 0:
        axes[0].axvline(
            best_epoch,
            color="green",
            linestyle="--",
            linewidth=1.2,
            label=f"best epoch {best_epoch}",
        )
    axes[0].set_title(title)
    axes[0].set_ylabel("Total loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    if component_cols:
        for col in component_cols:
            axes[1].plot(x, history[col], marker="o", markersize=3, linewidth=1.5, label=col)
        axes[1].set_ylabel("Component loss")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(fontsize=8)

    axes[-1].set_xlabel("Epoch")
    _set_epoch_axis(axes[-1], x)
    fig.tight_layout()
    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_loss_components(
    loss_detail: pd.DataFrame | str | Path,
    output_path: str | Path,
    *,
    epoch_col: str = "epoch",
    title: str = "Loss Components",
    dpi: int = 150,
) -> Path:
    """Train/val total loss plus state and power branches."""
    if not isinstance(loss_detail, pd.DataFrame):
        loss_detail = pd.read_csv(loss_detail)

    x = loss_detail[epoch_col] if epoch_col in loss_detail else np.arange(len(loss_detail))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    panels = [
        (axes[0], "Total", [("train_loss", "train"), ("val_loss", "val")]),
        (
            axes[1],
            "State / Power",
            [
                ("train_loss_state", "train state"),
                ("val_loss_state", "val state"),
                ("train_loss_power", "train power"),
                ("val_loss_power", "val power"),
            ],
        ),
    ]
    for ax, panel_title, cols in panels:
        for col, label in cols:
            if col in loss_detail.columns:
                ax.plot(x, loss_detail[col], marker="o", markersize=3, linewidth=1.5, label=label)
        ax.set_title(panel_title)
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_xlabel("Epoch")
        _set_epoch_axis(ax, x)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

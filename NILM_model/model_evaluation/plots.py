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
    dpi: int = 180,
) -> Path:
    """Plot train/validation loss and validation metrics from a history CSV/DataFrame."""
    if not isinstance(history, pd.DataFrame):
        history = pd.read_csv(history)

    x = history[epoch_col] if epoch_col in history else np.arange(len(history))
    loss_cols = [col for col in loss_cols if col in history]
    metric_cols = [col for col in metric_cols if col in history]
    n_rows = 1 + int(bool(metric_cols))

    fig, axes = plt.subplots(n_rows, 1, figsize=(10.5, 4.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for col in loss_cols:
        axes[0].plot(x, history[col], marker="o", markersize=3, linewidth=1.5, label=col)
    axes[0].set_title(title)
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    if metric_cols:
        for col in metric_cols:
            axes[1].plot(x, history[col], marker="o", markersize=3, linewidth=1.5, label=col)
        axes[1].set_ylabel("Metric")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend()

    axes[-1].set_xlabel("Epoch")
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
    title: str = "NILM Prediction Waveforms",
    dpi: int = 180,
) -> Path:
    """Plot aggregate power plus true/predicted appliance waveforms."""
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
    n_rows = int(has_aggregate) + len(true_pred_pairs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, max(5, 2.4 * n_rows)), sharex=True)
    if n_rows == 1:
        axes = [axes]

    row = 0
    if has_aggregate:
        axes[row].plot(x, view[aggregate_col], color="#222222", linewidth=1.0, label=aggregate_col)
        axes[row].set_ylabel("Aggregate W")
        axes[row].grid(True, alpha=0.25)
        axes[row].legend(loc="upper right")
        row += 1

    for appliance, (true_col, pred_col) in true_pred_pairs.items():
        ax = axes[row]
        ax.plot(x, view[true_col], color="#1f77b4", linewidth=1.1, label=f"{appliance} true")
        ax.plot(x, view[pred_col], color="#d62728", linewidth=1.0, alpha=0.85, label=f"{appliance} pred")
        ax.set_ylabel("Power W")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        row += 1

    fig.suptitle(title)
    axes[-1].set_xlabel(time_col if time_col else "sample index")
    fig.tight_layout()
    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path

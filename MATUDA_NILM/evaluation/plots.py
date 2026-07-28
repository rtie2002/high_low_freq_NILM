"""MATUDA paper figures: training curves and ON-period power waveforms.

Adapted from multi_appliances_NILM/evaluation/plots.py (same visual language).
"""

from __future__ import annotations

from dataclasses import dataclass
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


def _set_epoch_axis(ax, x) -> None:
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


FULL_CYCLE_APPLIANCES = frozenset({"washingmachine", "dishwasher"})


@dataclass(frozen=True)
class OnPeriodSelection:
    event_start: int
    event_end: int
    crop_start: int
    crop_end: int


def _pick_random_on_events(
    on: np.ndarray,
    power: np.ndarray,
    *,
    n_periods: int,
    rng: np.random.Generator,
    min_duration: int = 10,
    prefer_longest: bool = False,
) -> list[tuple[int, int]]:
    events = _find_on_events(on, min_duration=min_duration)
    if events:
        if prefer_longest:
            events = sorted(events, key=lambda t: t[1] - t[0], reverse=True)
            pool = events[: max(n_periods * 4, n_periods)]
            if len(pool) <= n_periods:
                picks = pool
            else:
                picks = [pool[i] for i in rng.choice(len(pool), size=n_periods, replace=False)]
        elif len(events) <= n_periods:
            picks = events
        else:
            picks = [events[i] for i in rng.choice(len(events), size=n_periods, replace=False)]
        return [(int(s), int(e)) for s, e in picks]

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
    half = max(min_duration // 2, 10)
    n = len(values)
    return [(max(0, c - half), min(n - 1, c + half)) for c in sorted(centers)]


def _window_for_on_event(
    event_start: int,
    event_end: int,
    series_len: int,
    *,
    margin_min: int = 30,
    margin_frac: float = 0.08,
    max_samples: int | None = 1200,
) -> tuple[int, int]:
    event_len = max(1, event_end - event_start + 1)
    margin = max(margin_min, int(margin_frac * event_len))
    start = max(0, event_start - margin)
    end = min(series_len, event_end + margin)
    if max_samples is not None and (end - start) > max_samples:
        end = min(series_len, event_end + margin)
        start = max(0, end - max_samples)
    return start, end


def select_appliance_on_periods(
    appliances: list[str],
    y_true_watts: np.ndarray,
    y_true_on: np.ndarray,
    *,
    n_periods: int,
    period_samples: int | None = None,
    full_cycle_appliances: Iterable[str] | None = None,
    margin_min: int = 40,
    margin_frac: float = 0.08,
    min_on_duration: int = 10,
    rng: np.random.Generator | None = None,
) -> dict[str, list[OnPeriodSelection]]:
    rng = rng or np.random.default_rng()
    y_true = np.asarray(y_true_watts, dtype=float)
    full_cycle = set(full_cycle_appliances or FULL_CYCLE_APPLIANCES)
    series_len = len(y_true)
    out: dict[str, list[OnPeriodSelection]] = {}
    for idx, app in enumerate(appliances):
        true_on = y_true_on[:, idx]
        min_dur = max(min_on_duration, 60 if app in full_cycle else min_on_duration)
        events = _pick_random_on_events(
            true_on,
            y_true[:, idx],
            n_periods=n_periods,
            rng=rng,
            min_duration=min_dur,
            prefer_longest=(app in full_cycle),
        )
        app_cap = None if app in full_cycle else period_samples
        periods: list[OnPeriodSelection] = []
        for ev_start, ev_end in events:
            crop_start, crop_end = _window_for_on_event(
                ev_start,
                ev_end,
                series_len,
                margin_min=margin_min,
                margin_frac=margin_frac,
                max_samples=app_cap if app_cap and app_cap > 0 else None,
            )
            periods.append(
                OnPeriodSelection(
                    event_start=int(ev_start),
                    event_end=int(ev_end),
                    crop_start=int(crop_start),
                    crop_end=int(crop_end),
                )
            )
        out[app] = periods
    return out


def plot_single_on_period(
    *,
    appliance: str,
    y_true_watts: np.ndarray,
    y_pred_watts: np.ndarray,
    output_path: str | Path,
    event_start: int | None = None,
    event_end: int | None = None,
    period_samples: int | None = None,
    margin_min: int = 30,
    margin_frac: float = 0.08,
    figsize: float = 5.5,
    dynamic_figsize: bool = True,
    aggregate: np.ndarray | None = None,
    y_pred_on: np.ndarray | None = None,
    title: str | None = None,
    dpi: int = WAVEFORM_DPI,
) -> Path:
    n = len(y_true_watts)
    if event_start is None or event_end is None:
        raise ValueError("Provide event_start/event_end")
    start, end = _window_for_on_event(
        event_start,
        event_end,
        n,
        margin_min=margin_min,
        margin_frac=margin_frac,
        max_samples=period_samples if period_samples and period_samples > 0 else None,
    )
    sl = slice(start, end)
    x = np.arange(start, end)
    true_v = np.asarray(y_true_watts, dtype=float)[sl]
    pred_v = np.maximum(np.asarray(y_pred_watts, dtype=float)[sl], 0.0)
    on_view = np.asarray(y_pred_on)[sl].astype(bool) if y_pred_on is not None else None
    agg_view = aggregate[sl] if aggregate is not None and len(aggregate) >= end else None

    n_pts = max(1, end - start)
    plot_side = figsize
    if dynamic_figsize and n_pts > 500:
        plot_side = min(figsize * 2.5, figsize * (n_pts / 500) ** 0.45)
    fig, ax = plt.subplots(1, 1, figsize=(plot_side, plot_side * 0.65))
    if agg_view is not None:
        ax_mains = ax.twinx()
        ax_mains.plot(x, agg_view, color="#9a9a9a", linewidth=0.9, alpha=0.45, label="aggregate (W)")
        ax_mains.set_ylabel("Aggregate (W)", color="#777777")
        ax_mains.tick_params(axis="y", colors="#777777", labelsize=8)
        ax_mains.grid(False)
        ax_mains.set_ylim(0.0, max(1.0, float(np.nanmax(agg_view))) * 1.08)

    if on_view is not None:
        active = np.flatnonzero(on_view)
        if len(active):
            groups = np.split(active, np.where(np.diff(active) != 1)[0] + 1)
            for group in groups:
                ax.axvspan(x[group[0]], x[group[-1]], color="#7ad66d", alpha=0.18, linewidth=0)

    ax.plot(x, true_v, color="#1f77b4", linewidth=1.8, label=f"{appliance} true")
    ax.plot(x, pred_v, color="#d62728", linewidth=1.5, alpha=0.92, label=f"{appliance} pred")
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Sample index (seq2point centers)")
    ax.grid(True, alpha=0.25)
    ax.set_title(title or f"{appliance} ON period [{start}:{end}]")

    handles, labels = ax.get_legend_handles_labels()
    if agg_view is not None:
        mh, ml = ax_mains.get_legend_handles_labels()
        handles.extend(mh)
        labels.extend(ml)
    handles.append(Patch(facecolor="#7ad66d", alpha=0.18, label="pred ON"))
    labels.append("pred ON")
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=8)

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
    y_true_on: np.ndarray,
    y_pred_on: np.ndarray | None = None,
    aggregate: np.ndarray | None = None,
    n_periods: int = 2,
    period_samples: int | None = 800,
    dpi: int = WAVEFORM_DPI,
    rng: np.random.Generator | None = None,
    file_prefix: str = "on",
    title_prefix: str = "",
) -> list[Path]:
    output_dir = Path(output_dir)
    rng = rng or np.random.default_rng(2026)
    y_true = np.asarray(y_true_watts, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred_watts, dtype=float), 0.0)
    saved: list[Path] = []
    selections = select_appliance_on_periods(
        appliances,
        y_true,
        y_true_on,
        n_periods=n_periods,
        period_samples=period_samples,
        rng=rng,
    )
    for idx, app in enumerate(appliances):
        app_dir = output_dir / app
        app_dir.mkdir(parents=True, exist_ok=True)
        pred_on = y_pred_on[:, idx] if y_pred_on is not None else None
        for period_i, period in enumerate(selections.get(app, []), start=1):
            path = app_dir / f"{file_prefix}_{period_i:02d}.png"
            plot_single_on_period(
                appliance=app,
                y_true_watts=y_true[:, idx],
                y_pred_watts=y_pred[:, idx],
                event_start=period.event_start,
                event_end=period.event_end,
                output_path=path,
                period_samples=None if app in FULL_CYCLE_APPLIANCES else period_samples,
                aggregate=aggregate,
                y_pred_on=pred_on,
                title=f"{title_prefix}{app} ON period {period_i}".strip(),
                dpi=dpi,
            )
            saved.append(path)
    return saved


def plot_matuda_training_history(
    history: list[dict] | str | Path,
    output_path: str | Path,
    *,
    title: str = "Training curves",
    best_epoch: int | None = None,
    figsize: tuple[float, float] = (10.0, 4.0),
    dpi: int = 200,
) -> Path:
    """MATUDA history.json: loss / domain / val+test F1 & MAE."""
    if not isinstance(history, list):
        import json

        with open(history, "r", encoding="utf-8") as f:
            history = json.load(f)
    df = pd.DataFrame(history)
    x = df["epoch"].to_numpy() if "epoch" in df else np.arange(1, len(df) + 1)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    # Loss
    ax = axes[0]
    ax.plot(x, df["loss"], label="train loss", linewidth=1.6)
    if "loss_sup" in df:
        ax.plot(x, df["loss_sup"], label="L_sup", linewidth=1.2, linestyle="--")
    if "loss_domain" in df and float(df["loss_domain"].max()) > 0:
        ax.plot(x, df["loss_domain"], label="L_domain", linewidth=1.2, linestyle=":")
    ax.set_title("Training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    _set_epoch_axis(ax, x)

    # F1
    ax = axes[1]
    if "val_f1" in df:
        ax.plot(x, df["val_f1"], label="source val F1", linewidth=1.6)
    if "test_f1" in df:
        ax.plot(x, df["test_f1"], label="House-2 F1", linewidth=1.6)
    ax.set_title("Macro-F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    _set_epoch_axis(ax, x)

    # MAE
    ax = axes[2]
    if "val_mae" in df:
        ax.plot(x, df["val_mae"], label="source val MAE", linewidth=1.6)
    if "test_mae" in df:
        ax.plot(x, df["test_mae"], label="House-2 MAE", linewidth=1.6)
    ax.set_title("Power MAE (W)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (W)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    _set_epoch_axis(ax, x)

    if best_epoch is not None and best_epoch > 0:
        for ax in axes:
            ax.axvline(best_epoch, color="green", linestyle="--", linewidth=1.0, alpha=0.7)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_methods_f1_mae_comparison(
    histories: dict[str, list[dict]],
    output_path: str | Path,
    *,
    title: str = "House-2 transfer during training",
    dpi: int = 200,
) -> Path:
    """Overlay House-2 F1 / MAE across methods."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for name, hist in histories.items():
        df = pd.DataFrame(hist)
        x = df["epoch"].to_numpy()
        axes[0].plot(x, df["test_f1"], label=name, linewidth=1.6)
        axes[1].plot(x, df["test_mae"], label=name, linewidth=1.6)
    axes[0].set_title("House-2 macro-F1")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_title("House-2 MAE (W)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (W)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_appliance_grid(
    *,
    appliances: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    n_samples: int = 1500,
    title: str = "Appliance power predictions (House 2)",
    dpi: int = 200,
) -> Path:
    """Compact multi-panel true vs pred for a contiguous slice."""
    n = min(n_samples, len(y_true))
    # Prefer a slice with some ON mass if possible.
    start = 0
    if n < len(y_true):
        on_mass = (y_true > 5).astype(np.float64).sum(axis=1)
        # sliding sum to find active region
        win = n
        csum = np.cumsum(on_mass)
        scores = csum[win - 1 :] - np.concatenate([[0.0], csum[:-win]])
        start = int(np.argmax(scores)) if len(scores) else 0
    sl = slice(start, start + n)
    x = np.arange(n)
    k = len(appliances)
    fig, axes = plt.subplots(k, 1, figsize=(10.0, 1.6 * k), sharex=True)
    if k == 1:
        axes = [axes]
    for i, (app, ax) in enumerate(zip(appliances, axes)):
        ax.plot(x, y_true[sl, i], color="#1f77b4", linewidth=1.2, label="true")
        ax.plot(x, np.maximum(y_pred[sl, i], 0), color="#d62728", linewidth=1.0, alpha=0.9, label="pred")
        ax.set_ylabel(app[:8], fontsize=8)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Sample index")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

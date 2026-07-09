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


FULL_CYCLE_APPLIANCES = frozenset({"washingmachine", "dishwasher"})


def _pick_random_on_events(
    on: np.ndarray,
    power: np.ndarray,
    *,
    n_periods: int,
    rng: np.random.Generator,
    min_duration: int = 10,
    prefer_longest: bool = False,
) -> list[tuple[int, int]]:
    """Return (start, end) inclusive indices for random ON segments."""
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
    """Slice bounds that include the full ON segment plus padding.

    Long events (e.g. washing-machine cycles) are right-anchored when capped so
    the heating / spin tail is not clipped off the right edge.
    """
    event_len = max(1, event_end - event_start + 1)
    margin = max(margin_min, int(margin_frac * event_len))
    start = max(0, event_start - margin)
    end = min(series_len, event_end + margin)

    if max_samples is not None and (end - start) > max_samples:
        # Right-anchor so heating / spin tails are not clipped on long cycles.
        end = min(series_len, event_end + margin)
        start = max(0, end - max_samples)

    return start, end


def plot_single_on_period(
    *,
    appliance: str,
    y_true_watts: np.ndarray,
    y_pred_watts: np.ndarray,
    output_path: str | Path,
    event_start: int | None = None,
    event_end: int | None = None,
    center: int | None = None,
    period_samples: int | None = None,
    margin_min: int = 30,
    margin_frac: float = 0.08,
    figsize: float = 5.5,
    dynamic_figsize: bool = True,
    aggregate: np.ndarray | None = None,
    y_pred_on: np.ndarray | None = None,
    csv_timesteps: np.ndarray | None = None,
    title: str | None = None,
    dpi: int = WAVEFORM_DPI,
) -> Path:
    """True vs predicted waveform for one ON period (full event + padding)."""
    n = len(y_true_watts)
    if event_start is not None and event_end is not None:
        start, end = _window_for_on_event(
            event_start,
            event_end,
            n,
            margin_min=margin_min,
            margin_frac=margin_frac,
            max_samples=period_samples if period_samples and period_samples > 0 else None,
        )
    elif center is not None:
        cap = period_samples if period_samples and period_samples > 0 else 400
        half = max(cap // 2, 10)
        start = max(0, center - half)
        end = min(n, start + cap)
        start = max(0, end - cap)
    else:
        raise ValueError("Provide event_start/event_end or center")

    sl = slice(start, end)
    if csv_timesteps is not None and len(csv_timesteps) >= end:
        x = np.asarray(csv_timesteps[sl], dtype=int)
        x_label = "CSV row index"
        mains_label = "aggregate (W)"
    else:
        x = np.arange(start, end)
        x_label = "Window index"
        mains_label = "aggregate"
    true_v = np.asarray(y_true_watts, dtype=float)[sl]
    pred_v = np.maximum(np.asarray(y_pred_watts, dtype=float)[sl], 0.0)

    n_pts = max(1, end - start)
    plot_side = figsize
    if dynamic_figsize and n_pts > 500:
        plot_side = min(figsize * 2.5, figsize * (n_pts / 500) ** 0.45)
    fig, ax = plt.subplots(1, 1, figsize=(plot_side, plot_side))
    ax.set_box_aspect(1)
    agg_view = aggregate[sl] if aggregate is not None and len(aggregate) >= end else None
    if agg_view is not None:
        ax_mains = ax.twinx()
        ax_mains.plot(x, agg_view, color="#9a9a9a", linewidth=0.9, alpha=0.45, label=mains_label)
        ax_mains.set_ylabel("Aggregate (W)", color="#777777")
        ax_mains.tick_params(axis="y", colors="#777777", labelsize=8)
        ax_mains.grid(False)
        ymax_mains = max(1.0, float(np.nanmax(agg_view)))
        ax_mains.set_ylim(0.0, ymax_mains * 1.08)

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
    ax.set_xlabel(x_label)
    ax.grid(True, alpha=0.25)
    if csv_timesteps is not None and len(csv_timesteps) >= end:
        t0, t1 = int(csv_timesteps[start]), int(csv_timesteps[end - 1])
        title_suffix = f" [{t0}:{t1}] ({end - start} pts)"
    else:
        title_suffix = f" [{start}:{end}] ({end - start} steps)"
    ax.set_title(title or f"{appliance} ON period{title_suffix}")

    handles, labels = ax.get_legend_handles_labels()
    if agg_view is not None:
        mains_handles, mains_labels = ax_mains.get_legend_handles_labels()
        handles.extend(mains_handles)
        labels.extend(mains_labels)
    handles.append(Patch(facecolor="#7ad66d", alpha=0.18, label="pred ON"))
    labels.append("pred ON")
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=9)

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
    csv_timesteps: np.ndarray | None = None,
    n_periods: int = 5,
    period_samples: int | None = None,
    full_cycle_appliances: Iterable[str] | None = None,
    margin_min: int = 40,
    margin_frac: float = 0.08,
    min_on_duration: int = 10,
    figsize: float = 5.5,
    dynamic_figsize: bool = True,
    dpi: int = WAVEFORM_DPI,
    rng: np.random.Generator | None = None,
    file_prefix: str = "on",
    title_prefix: str = "",
) -> list[Path]:
    """Save N random ON-period plots per appliance under output_dir/<appliance>/.

    Waveform plots intentionally use dataset CSV *_on labels for selecting true
    ON periods (y_true_on). This is independent of data.state_label_source in
    model yaml, which may still rebuild labels from power for training/F1.
    """
    output_dir = Path(output_dir)
    rng = rng or np.random.default_rng()
    y_true = np.asarray(y_true_watts, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred_watts, dtype=float), 0.0)
    saved: list[Path] = []
    full_cycle = set(full_cycle_appliances or FULL_CYCLE_APPLIANCES)
    if y_true_on is None:
        raise ValueError("Waveform plots require dataset CSV ON/OFF labels in y_true_on")

    for idx, app in enumerate(appliances):
        app_dir = output_dir / app
        app_dir.mkdir(parents=True, exist_ok=True)

        # Always follow dataset CSV labels for true ON/OFF event selection.
        true_on = y_true_on[:, idx]
        # Predicted ON shading comes from model outputs, not dataset labels.
        pred_on = y_pred_on[:, idx] if y_pred_on is not None else None
        min_dur = max(min_on_duration, 60 if app in full_cycle else min_on_duration)
        events = _pick_random_on_events(
            true_on,
            y_true[:, idx],
            n_periods=n_periods,
            rng=rng,
            min_duration=min_dur,
            prefer_longest=(app in full_cycle),
        )
        if not events:
            continue

        app_cap = None if app in full_cycle else period_samples

        for period_i, (ev_start, ev_end) in enumerate(events, start=1):
            center = (ev_start + ev_end) // 2
            path = app_dir / f"{file_prefix}_{period_i:02d}_t{center}.png"
            title = f"{title_prefix}{app} ON period {period_i}".strip()
            plot_single_on_period(
                appliance=app,
                y_true_watts=y_true[:, idx],
                y_pred_watts=y_pred[:, idx],
                event_start=ev_start,
                event_end=ev_end,
                output_path=path,
                period_samples=app_cap,
                margin_min=margin_min,
                margin_frac=margin_frac,
                figsize=figsize,
                dynamic_figsize=dynamic_figsize,
                aggregate=aggregate,
                y_pred_on=pred_on,
                csv_timesteps=csv_timesteps,
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
    figsize: float = 5.5,
    dpi: int = 150,
) -> Path:
    if not isinstance(history, pd.DataFrame):
        history = pd.read_csv(history)

    x = history[epoch_col] if epoch_col in history else np.arange(len(history))
    loss_cols = [c for c in loss_cols if c in history.columns]
    component_cols = [c for c in component_cols if c in history.columns and not history[c].isna().all()]

    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    ax.set_box_aspect(1)

    for col in loss_cols:
        ax.plot(x, history[col], marker="o", markersize=3, linewidth=1.6, label=col)
    for col in component_cols:
        ax.plot(x, history[col], marker="s", markersize=2, linewidth=1.2, linestyle="--", label=col)
    if best_epoch is not None and best_epoch > 0:
        ax.axvline(
            best_epoch,
            color="green",
            linestyle="--",
            linewidth=1.2,
            label=f"best epoch {best_epoch}",
        )
    ax.set_title(title)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    _set_epoch_axis(ax, x)

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
    figsize: float = 5.5,
    dpi: int = 150,
) -> Path:
    """Train/val total loss plus state and power branches (single square figure)."""
    if not isinstance(loss_detail, pd.DataFrame):
        loss_detail = pd.read_csv(loss_detail)

    x = loss_detail[epoch_col] if epoch_col in loss_detail else np.arange(len(loss_detail))
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    ax.set_box_aspect(1)

    series = [
        ("train_loss", "train"),
        ("val_loss", "val"),
        ("train_loss_state", "train state"),
        ("val_loss_state", "val state"),
        ("train_loss_power", "train power"),
        ("val_loss_power", "val power"),
    ]
    for col, label in series:
        if col in loss_detail.columns:
            ax.plot(x, loss_detail[col], marker="o", markersize=3, linewidth=1.5, label=label)

    ax.set_title(title)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    _set_epoch_axis(ax, x)

    fig.tight_layout()
    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_matnilm_training_losses(
    loss_detail: pd.DataFrame | str | Path,
    output_path: str | Path,
    *,
    appliances: list[str],
    epoch_col: str = "epoch",
    figsize: float = 5.5,
    dpi: int = 200,
) -> Path:
    """Paper-style MATNILM Fig. 2: total and per-appliance training losses."""
    if not isinstance(loss_detail, pd.DataFrame):
        loss_detail = pd.read_csv(loss_detail)

    if epoch_col in loss_detail:
        x = loss_detail[epoch_col].to_numpy(dtype=float) - 1.0
    else:
        x = np.arange(len(loss_detail), dtype=float)

    label_map = {
        "dishwasher": "Dishwasher loss",
        "dish washer": "Dishwasher loss",
        "fridge": "Fridge loss",
        "microwave": "Microwave loss",
        "washingmachine": "Washer dryer loss",
        "washer dryer": "Washer dryer loss",
        "wash": "Washer dryer loss",
    }
    colors = {
        "train_loss": "#1f77b4",
        "dishwasher": "#ff7f0e",
        "fridge": "#2ca02c",
        "microwave": "#d62728",
        "washingmachine": "#9467bd",
    }

    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize * 0.8))
    if "train_loss" in loss_detail.columns:
        ax.plot(x, loss_detail["train_loss"], color=colors["train_loss"], linewidth=1.8, label="Total loss")

    for app in appliances:
        col = f"train_loss_{app}"
        if col not in loss_detail.columns:
            continue
        ax.plot(
            x,
            loss_detail[col],
            linewidth=1.8,
            color=colors.get(app),
            label=label_map.get(app, f"{app} loss"),
        )

    ax.set_xlabel("Epoch number")
    ax.set_ylabel("Training loss")
    ax.grid(True, linestyle="--", linewidth=0.8, color="#9e9e9e", alpha=0.75)
    ax.legend(loc="upper right", frameon=True)
    if len(x):
        ax.set_xlim(max(0, float(np.nanmin(x)) - 1), float(np.nanmax(x)) + 1)
    fig.tight_layout()

    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

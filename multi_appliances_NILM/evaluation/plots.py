"""Training loss and per-appliance ON-period waveform plots."""

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


@dataclass(frozen=True)
class OnPeriodSelection:
    event_start: int
    event_end: int
    crop_start: int
    crop_end: int


def dataset_on_labels_for_bundle(
    data_loader,
    split: str,
    n_points: int,
    csv_timesteps: np.ndarray | None = None,
) -> np.ndarray:
    """Dataset CSV *_on labels aligned with a prediction bundle timeline."""
    if csv_timesteps is not None and len(csv_timesteps) >= n_points:
        return data_loader.csv_on_labels_at_timesteps(split, csv_timesteps[:n_points])
    return data_loader.window_flattened_csv_states(split, n_points)


def bundle_aggregate_watts(
    data_loader,
    split: str,
    *,
    n_points: int,
    csv_timesteps: np.ndarray | None,
) -> np.ndarray | None:
    """Raw CSV aggregate (W) aligned 1:1 with the prediction timeline.

    Must use ``csv_timesteps`` → CSV row lookup. Do not use a sequential offset
    into the series; that misaligns overlap-reconstructed timelines.
    """
    if csv_timesteps is None:
        return None
    ts = np.asarray(csv_timesteps, dtype=np.int64).reshape(-1)
    if len(ts) < int(n_points):
        return None
    try:
        return data_loader.mains_watts_at_timesteps(split, ts[: int(n_points)])
    except Exception:
        return None


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
    """Pick ON periods from dataset CSV labels (same logic as waveform plots)."""
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


def _context_window(
    event_start: int,
    event_end: int,
    focus_start: int,
    focus_end: int,
    series_len: int,
    *,
    scale: float = 10.0,
) -> tuple[int, int]:
    """Wider slice centered on the ON event: ``scale`` × the focused crop length."""
    focus_len = max(1, int(focus_end) - int(focus_start))
    target = max(focus_len + 1, int(round(focus_len * float(scale))))
    center = (int(event_start) + int(event_end)) // 2
    half = target // 2
    start = max(0, center - half)
    end = min(series_len, start + target)
    start = max(0, end - target)
    return int(start), int(end)


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
    window_start: int | None = None,
    window_end: int | None = None,
    figsize: float = 5.5,
    dynamic_figsize: bool = True,
    long_figure: bool = False,
    highlight_start: int | None = None,
    highlight_end: int | None = None,
    aggregate: np.ndarray | None = None,
    y_pred_on: np.ndarray | None = None,
    csv_timesteps: np.ndarray | None = None,
    title: str | None = None,
    dpi: int = WAVEFORM_DPI,
) -> Path:
    """True vs predicted waveform for one ON period (full event + padding).

    ``long_figure=True`` draws a wide context strip (no square aspect). Optional
    ``highlight_start/end`` mark the focused crop on a 10× context plot.
    """
    n = len(y_true_watts)
    if window_start is not None and window_end is not None:
        start, end = int(window_start), int(window_end)
        start = max(0, min(start, n))
        end = max(start + 1, min(end, n))
    elif event_start is not None and event_end is not None:
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
        raise ValueError("Provide event_start/event_end, window_start/window_end, or center")

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
    on_view = np.asarray(y_pred_on)[sl].astype(bool) if y_pred_on is not None else None
    agg_view = aggregate[sl] if aggregate is not None and len(aggregate) >= end else None

    if len(x) > 1 and np.any(np.diff(x) <= 0):
        order = np.argsort(x, kind="stable")
        x = x[order]
        true_v = true_v[order]
        pred_v = pred_v[order]
        if on_view is not None:
            on_view = on_view[order]
        if agg_view is not None:
            agg_view = agg_view[order]

    n_pts = max(1, end - start)
    if long_figure:
        # Wide strip: grow width with timeline length; keep modest height.
        width = figsize * 1.6
        if dynamic_figsize and n_pts > 400:
            width = min(28.0, figsize * (n_pts / 400) ** 0.55)
        height = max(2.8, figsize * 0.55)
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
    else:
        plot_side = figsize
        if dynamic_figsize and n_pts > 500:
            plot_side = min(figsize * 2.5, figsize * (n_pts / 500) ** 0.45)
        fig, ax = plt.subplots(1, 1, figsize=(plot_side, plot_side))
        ax.set_box_aspect(1)
    if agg_view is not None:
        ax_mains = ax.twinx()
        # Twin axis: appliance shape stays readable on the left; aggregate (W) is
        # true CSV mains on the right (may be much larger — that is expected).
        ax_mains.plot(x, agg_view, color="#9a9a9a", linewidth=0.9, alpha=0.55, label=mains_label)
        ax_mains.set_ylabel("Aggregate (W)", color="#777777")
        ax_mains.tick_params(axis="y", colors="#777777", labelsize=8)
        ax_mains.grid(False)
        finite = np.asarray(agg_view, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            ymax_mains = max(1.0, float(np.max(finite)))
            ymin_mains = min(0.0, float(np.min(finite)))
            pad_m = 0.05 * max(ymax_mains - ymin_mains, 1.0)
            ax_mains.set_ylim(ymin_mains - pad_m, ymax_mains + pad_m)

    if highlight_start is not None and highlight_end is not None:
        hs = max(start, int(highlight_start))
        he = min(end - 1, int(highlight_end) - 1 if int(highlight_end) > int(highlight_start) else int(highlight_end))
        if he >= hs and csv_timesteps is not None and len(csv_timesteps) >= end:
            ax.axvspan(int(csv_timesteps[hs]), int(csv_timesteps[he]), color="#f4a261", alpha=0.16, linewidth=0)
        elif he >= hs:
            ax.axvspan(hs, he, color="#f4a261", alpha=0.16, linewidth=0)

    if on_view is not None:
        on_mask = on_view
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
    if highlight_start is not None and highlight_end is not None:
        handles.append(Patch(facecolor="#f4a261", alpha=0.16, label="focused crop"))
        labels.append("focused crop")
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=8 if long_figure else 9)

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
    context_scale: float = 10.0,
    rng: np.random.Generator | None = None,
    file_prefix: str = "on",
    title_prefix: str = "",
) -> list[Path]:
    """Save N random ON-period plots per appliance under output_dir/<appliance>/.

    For each focused situation plot, also saves a long ``*_context10x.png`` with
    ``context_scale`` × wider timeline around the same ON event (set
    ``context_scale <= 1`` to disable).

    Waveform plots intentionally use dataset CSV *_on labels for selecting true
    ON periods (y_true_on). This is independent of data.state_label_source in
    model yaml, which may still rebuild labels from power for training/F1.
    """
    output_dir = Path(output_dir)
    rng = rng or np.random.default_rng()
    y_true = np.asarray(y_true_watts, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred_watts, dtype=float), 0.0)
    saved: list[Path] = []
    if y_true_on is None:
        raise ValueError("Waveform plots require dataset CSV ON/OFF labels in y_true_on")

    selections = select_appliance_on_periods(
        appliances,
        y_true,
        y_true_on,
        n_periods=n_periods,
        period_samples=period_samples,
        full_cycle_appliances=full_cycle_appliances,
        margin_min=margin_min,
        margin_frac=margin_frac,
        min_on_duration=min_on_duration,
        rng=rng,
    )

    series_len = len(y_true)
    save_context = float(context_scale) > 1.0
    full_cycle = set(full_cycle_appliances or FULL_CYCLE_APPLIANCES)

    for idx, app in enumerate(appliances):
        app_dir = output_dir / app
        app_dir.mkdir(parents=True, exist_ok=True)
        pred_on = y_pred_on[:, idx] if y_pred_on is not None else None
        app_cap = None if app in full_cycle else period_samples

        for period_i, period in enumerate(selections.get(app, []), start=1):
            center = (period.event_start + period.event_end) // 2
            path = app_dir / f"{file_prefix}_{period_i:02d}_t{center}.png"
            title = f"{title_prefix}{app} ON period {period_i}".strip()
            plot_single_on_period(
                appliance=app,
                y_true_watts=y_true[:, idx],
                y_pred_watts=y_pred[:, idx],
                event_start=period.event_start,
                event_end=period.event_end,
                output_path=path,
                period_samples=app_cap if app_cap and app_cap > 0 else None,
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

            if not save_context:
                continue
            ctx_start, ctx_end = _context_window(
                period.event_start,
                period.event_end,
                period.crop_start,
                period.crop_end,
                series_len,
                scale=float(context_scale),
            )
            scale_tag = int(round(float(context_scale)))
            ctx_path = app_dir / f"{file_prefix}_{period_i:02d}_t{center}_context{scale_tag}x.png"
            ctx_title = (
                f"{title_prefix}{app} ON period {period_i} "
                f"(×{scale_tag} context)".strip()
            )
            plot_single_on_period(
                appliance=app,
                y_true_watts=y_true[:, idx],
                y_pred_watts=y_pred[:, idx],
                event_start=period.event_start,
                event_end=period.event_end,
                output_path=ctx_path,
                window_start=ctx_start,
                window_end=ctx_end,
                figsize=figsize,
                dynamic_figsize=dynamic_figsize,
                long_figure=True,
                highlight_start=period.crop_start,
                highlight_end=period.crop_end,
                aggregate=aggregate,
                y_pred_on=pred_on,
                csv_timesteps=csv_timesteps,
                title=ctx_title,
                dpi=dpi,
            )
            saved.append(ctx_path)
    return saved


def plot_training_history(
    history: pd.DataFrame | str | Path,
    output_path: str | Path,
    *,
    epoch_col: str = "epoch",
    loss_cols: Iterable[str] = ("train_loss_nilm", "val_loss_nilm", "train_loss", "val_loss"),
    component_cols: Iterable[str] = (),
    title: str = "Training Loss",
    best_epoch: int | None = None,
    figsize: float = 5.5,
    dpi: int = 150,
) -> Path:
    if not isinstance(history, pd.DataFrame):
        history = pd.read_csv(history)

    x = history[epoch_col] if epoch_col in history else np.arange(len(history))
    # Prefer same-scale NILM curves; fall back to total train/val if older CSV.
    preferred = [c for c in ("train_loss_nilm", "val_loss_nilm") if c in history.columns]
    if len(preferred) >= 2:
        loss_cols = list(preferred)
        if "train_loss" in history.columns and not history["train_loss"].isna().all():
            # Optional: show DA-inflated train total as dashed reference later via label.
            loss_cols = preferred + ["train_loss"]
    else:
        loss_cols = [c for c in ("train_loss", "val_loss") if c in history.columns]
    component_cols = [c for c in component_cols if c in history.columns and not history[c].isna().all()]

    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    ax.set_box_aspect(1)

    style = {
        "train_loss_nilm": ("-", 1.8, "train L_NILM"),
        "val_loss_nilm": ("-", 1.8, "val L_NILM"),
        "train_loss": ("--", 1.2, "train L_total (+DA)"),
        "val_loss": ("-", 1.6, "val loss"),
    }
    for col in loss_cols:
        ls, lw, label = style.get(col, ("-", 1.6, col))
        ax.plot(x, history[col], marker="o", markersize=3, linewidth=lw, linestyle=ls, label=label)
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
    """One 2×2 figure (same panel size as the old single square plot).

    Layout::

        [ total  ] [ power  ]
        [ state  ] [ domain ]

    Domain panel shows a note when DA / L_domain is absent.
    """
    if not isinstance(loss_detail, pd.DataFrame):
        loss_detail = pd.read_csv(loss_detail)

    x = loss_detail[epoch_col] if epoch_col in loss_detail else np.arange(len(loss_detail))

    panels: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "L_NILM (same scale, no DA)",
            [
                ("train_loss_nilm", "train L_NILM"),
                ("val_loss_nilm", "val L_NILM"),
                ("train_loss", "train L_total (+DA)"),
            ],
        ),
        (
            "Power (MSE)",
            [
                ("train_loss_power", "train MSE"),
                ("val_loss_power", "val MSE"),
            ],
        ),
        (
            "State (BCE)",
            [
                ("train_loss_state_term", "train term"),
                ("val_loss_state_term", "val term"),
                ("train_loss_state", "train raw"),
                ("val_loss_state", "val raw"),
            ],
        ),
        ("Domain (train only)", [("train_loss_domain", "train L_domain")]),
    ]

    # 2×2 grid; each cell ≈ old single-figure size (figsize × figsize).
    fig, axes = plt.subplots(2, 2, figsize=(figsize * 2, figsize * 2))
    axes_flat = axes.ravel()

    for ax, (panel_title, cols) in zip(axes_flat, panels):
        ax.set_box_aspect(1)
        plotted = False
        for col, label in cols:
            if col not in loss_detail.columns:
                continue
            y = loss_detail[col]
            if y.isna().all():
                continue
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, label=label)
            plotted = True

        ax.set_title(panel_title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        _set_epoch_axis(ax, x)

        if plotted:
            ax.legend(fontsize=7, loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "no data\n(DA off)" if panel_title == "Domain" else "no data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="#666666",
            )

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
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

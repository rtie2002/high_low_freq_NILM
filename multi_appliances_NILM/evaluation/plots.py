"""Training loss and per-appliance ON-period waveform plots."""

from __future__ import annotations

import json
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


def bundle_csv_appliance_watts(
    data_loader,
    split: str,
    *,
    n_points: int,
    csv_timesteps: np.ndarray | None,
) -> np.ndarray | None:
    """Raw CSV appliance powers (W) at the same rows as aggregate (waveform GT)."""
    if csv_timesteps is None:
        return None
    ts = np.asarray(csv_timesteps, dtype=np.int64).reshape(-1)
    if len(ts) < int(n_points):
        return None
    try:
        return data_loader.appliance_watts_at_timesteps(split, ts[: int(n_points)])
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


def _axvspan_bool_mask(
    ax,
    x: np.ndarray,
    mask: np.ndarray,
    *,
    color: str,
    alpha: float,
) -> None:
    """Shade contiguous True runs of ``mask`` along the plot x-axis."""
    on_mask = np.asarray(mask, dtype=bool)
    if on_mask.shape[0] != len(x):
        raise ValueError(f"mask length {on_mask.shape[0]} != x length {len(x)}")
    active = np.flatnonzero(on_mask)
    if not len(active):
        return
    groups = np.split(active, np.where(np.diff(active) != 1)[0] + 1)
    for group in groups:
        ax.axvspan(x[group[0]], x[group[-1]], color=color, alpha=alpha, linewidth=0)


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
    y_true_on: np.ndarray | None = None,
    y_pred_on: np.ndarray | None = None,
    true_on_threshold_watts: float | None = None,
    # Only when state_label_source=threshold: shade via power>thr.
    # If False (csv/auto), always use y_true_on — never power>thr flicker.
    shade_true_on_from_power: bool = False,
    csv_timesteps: np.ndarray | None = None,
    title: str | None = None,
    dpi: int = WAVEFORM_DPI,
) -> Path:
    """True vs predicted waveform for one ON period (full event + padding).

    Background bands (no focused-crop highlight):
      - true ON: CSV ``y_true_on`` by default; ``power > thr`` only if
        ``shade_true_on_from_power`` (threshold label mode)
      - pred ON only  (``y_pred_on``)
      - overlap       (true ∩ pred) in purple

    ``highlight_start/end`` are ignored (kept for call-site compatibility).
    """
    del highlight_start, highlight_end  # focused-crop band removed

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
    # Strict: csv/auto → y_true_on only. power>thr only when explicitly enabled.
    if shade_true_on_from_power and true_on_threshold_watts is not None:
        true_on_view = true_v > float(true_on_threshold_watts)
    elif y_true_on is not None:
        true_on_view = np.asarray(y_true_on)[sl].astype(bool)
    else:
        true_on_view = None
    on_view = np.asarray(y_pred_on)[sl].astype(bool) if y_pred_on is not None else None
    agg_view = aggregate[sl] if aggregate is not None and len(aggregate) >= end else None

    if len(x) > 1 and np.any(np.diff(x) <= 0):
        order = np.argsort(x, kind="stable")
        x = x[order]
        true_v = true_v[order]
        pred_v = pred_v[order]
        if true_on_view is not None:
            true_on_view = true_on_view[order]
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
        # Same watt axis as appliances so heights are comparable. Y-lim follows
        # the appliance (shape focus); aggregate may clip at the top when huge.
        ax.plot(x, agg_view, color="#9a9a9a", linewidth=1.0, alpha=0.55, label=mains_label)

    # ON bands: true-only (blue), pred-only (green), overlap (purple).
    true_only = pred_only = both = None
    if true_on_view is not None and on_view is not None:
        both = true_on_view & on_view
        true_only = true_on_view & ~on_view
        pred_only = on_view & ~true_on_view
        _axvspan_bool_mask(ax, x, true_only, color="#6baed6", alpha=0.20)
        _axvspan_bool_mask(ax, x, pred_only, color="#7ad66d", alpha=0.20)
        _axvspan_bool_mask(ax, x, both, color="#9b59b6", alpha=0.28)
    elif true_on_view is not None:
        true_only = true_on_view
        _axvspan_bool_mask(ax, x, true_only, color="#6baed6", alpha=0.20)
    elif on_view is not None:
        pred_only = on_view
        _axvspan_bool_mask(ax, x, pred_only, color="#7ad66d", alpha=0.20)

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
    if true_on_view is not None:
        thr_note = " (power>thr)" if shade_true_on_from_power else " (csv)"
        handles.append(Patch(facecolor="#6baed6", alpha=0.20, label=f"true ON{thr_note}"))
        labels.append(f"true ON{thr_note}")
    if on_view is not None:
        handles.append(Patch(facecolor="#7ad66d", alpha=0.20, label="pred ON"))
        labels.append("pred ON")
    if true_on_view is not None and on_view is not None:
        handles.append(Patch(facecolor="#9b59b6", alpha=0.28, label="ON overlap"))
        labels.append("ON overlap")
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=8 if long_figure else 9)

    # Focus y-range on appliance shape; leave headroom so modest aggregate shows.
    app_max = max(1.0, float(np.max(true_v)), float(np.max(pred_v)))
    if agg_view is not None and np.isfinite(agg_view).any():
        agg_max = float(np.nanmax(agg_view))
        # If aggregate is only slightly above appliance, show both fully.
        # If aggregate dwarfs appliance, clip view ~1.25× appliance peak.
        if agg_max <= app_max * 1.5:
            ymax = max(app_max, agg_max)
        else:
            ymax = app_max * 1.25
    else:
        ymax = app_max
    ymin = min(0.0, float(np.min(true_v)), float(np.min(pred_v)))
    pad = 0.1 * max(ymax - ymin, 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)

    fig.tight_layout()
    output_path = _ensure_parent(output_path)
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    except FileNotFoundError:
        # Windows delayed-delete race after rmtree — recreate parent and retry once.
        output_path.parent.mkdir(parents=True, exist_ok=True)
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
    on_thresholds_watts: float | np.ndarray | None = None,
    state_label_source: str = "csv",
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

    True-ON shading follows ``state_label_source`` strictly:
      - ``csv`` / ``auto`` → always ``y_true_on`` (CSV Algorithm-1). Ignores
        ``on_thresholds_watts`` even if passed (avoids WM flicker on dips).
      - ``threshold`` → shade with ``power > on_thresholds_watts``.
    Period selection always uses ``y_true_on``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = rng or np.random.default_rng()
    y_true = np.asarray(y_true_watts, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred_watts, dtype=float), 0.0)
    saved: list[Path] = []
    if y_true_on is None:
        raise ValueError("Waveform plots require dataset CSV ON/OFF labels in y_true_on")

    source = str(state_label_source).lower().strip()
    shade_from_power = source == "threshold"
    thr = None
    if shade_from_power:
        if on_thresholds_watts is None:
            raise ValueError(
                "state_label_source=threshold requires on_thresholds_watts for true-ON shade"
            )
        thr = np.asarray(on_thresholds_watts, dtype=np.float32).reshape(-1)
        if thr.size == 1:
            thr = np.full(len(appliances), float(thr[0]), dtype=np.float32)
        if thr.size != len(appliances):
            raise ValueError(
                f"on_thresholds_watts length {thr.size} != num appliances {len(appliances)}"
            )

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
        true_on = y_true_on[:, idx]
        app_thr = float(thr[idx]) if thr is not None else None
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
                y_true_on=true_on,
                y_pred_on=pred_on,
                true_on_threshold_watts=app_thr,
                shade_true_on_from_power=shade_from_power,
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
                aggregate=aggregate,
                y_true_on=true_on,
                y_pred_on=pred_on,
                true_on_threshold_watts=app_thr,
                shade_true_on_from_power=shade_from_power,
                csv_timesteps=csv_timesteps,
                title=ctx_title,
                dpi=dpi,
            )
            saved.append(ctx_path)
    return saved


def build_val_test_comparison_frame(
    val_metrics: pd.DataFrame | str | Path,
    test_metrics: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """Per-appliance + overall val/test MAE with macro/micro F1 and gaps."""
    if not isinstance(val_metrics, pd.DataFrame):
        val_metrics = pd.read_csv(val_metrics)
    if not isinstance(test_metrics, pd.DataFrame):
        test_metrics = pd.read_csv(test_metrics)

    def _macro(row: pd.Series) -> float:
        if "macro_f1" in row.index and pd.notna(row["macro_f1"]):
            return float(row["macro_f1"])
        return float(row["f1"])

    def _micro(row: pd.Series) -> float | None:
        if "micro_f1" not in row.index or pd.isna(row["micro_f1"]):
            return None
        return float(row["micro_f1"])

    val_app = val_metrics.set_index("appliance")
    test_app = test_metrics.set_index("appliance")
    appliances = [a for a in val_app.index if a in test_app.index]
    rows = []
    for app in appliances:
        v = val_app.loc[app]
        t = test_app.loc[app]
        v_ma, t_ma = _macro(v), _macro(t)
        v_mi, t_mi = _micro(v), _micro(t)
        rows.append({
            "appliance": app,
            "val_MAE": float(v["mae"]),
            "test_MAE": float(t["mae"]),
            "MAE_gap": float(t["mae"] - v["mae"]),
            "val_SAE": float(v["sae"]),
            "test_SAE": float(t["sae"]),
            "SAE_gap": float(t["sae"] - v["sae"]),
            "val_maF1": v_ma,
            "test_maF1": t_ma,
            "maF1_gap": float(t_ma - v_ma),
            "val_miF1": v_mi,
            "test_miF1": t_mi,
            "miF1_gap": None if v_mi is None or t_mi is None else float(t_mi - v_mi),
            # Backward-compatible aliases used by older notes/code.
            "val_F1": v_ma,
            "test_F1": t_ma,
            "F1_gap": float(t_ma - v_ma),
        })
    return pd.DataFrame(rows)


def save_val_test_comparison_figure(
    val_metrics: pd.DataFrame | str | Path,
    test_metrics: pd.DataFrame | str | Path,
    output_path: str | Path,
    *,
    epoch: int | None = None,
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    """Compact val/test table: MAE + SAE + per-app F1; overall maF1/miF1 below."""
    compare = build_val_test_comparison_frame(val_metrics, test_metrics)
    output_path = _ensure_parent(output_path)
    if compare.empty:
        return output_path

    # Put overall last if present.
    if "overall" in set(compare["appliance"]):
        apps = [a for a in compare["appliance"] if a != "overall"] + ["overall"]
        compare = compare.set_index("appliance").loc[apps].reset_index()

    # Narrow table: MAE/SAE/per-app F1. Overall maF1/miF1 go in the footer.
    col_labels = [
        "appliance",
        "val_MAE",
        "test_MAE",
        "MAE_gap",
        "val_SAE",
        "test_SAE",
        "SAE_gap",
        "val_F1",
        "test_F1",
        "F1_gap",
    ]

    def _fmt_f1(x) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{float(x):.4f}"

    def _fmt_gap(x) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{float(x):+.4f}"

    cell_text = []
    for _, r in compare.iterrows():
        # Per-app F1; overall row uses macro F1 in the F1 columns.
        cell_text.append([
            str(r["appliance"]),
            f"{r['val_MAE']:.2f}",
            f"{r['test_MAE']:.2f}",
            f"{r['MAE_gap']:+.2f}",
            f"{r['val_SAE']:.2f}",
            f"{r['test_SAE']:.2f}",
            f"{r['SAE_gap']:+.2f}",
            _fmt_f1(r["val_maF1"]),
            _fmt_f1(r["test_maF1"]),
            _fmt_gap(r["maF1_gap"]),
        ])

    overall = compare[compare["appliance"] == "overall"]
    footer_lines: list[str] = []
    if not overall.empty:
        o = overall.iloc[0]
        footer_lines.append(
            f"maF1  val={_fmt_f1(o['val_maF1'])}  test={_fmt_f1(o['test_maF1'])}  "
            f"gap={_fmt_gap(o['maF1_gap'])}   "
            f"miF1  val={_fmt_f1(o['val_miF1'])}  test={_fmt_f1(o['test_miF1'])}  "
            f"gap={_fmt_gap(o['miF1_gap'])}"
        )
        mae_gap = float(o["MAE_gap"])
        f1_gap = float(o["maF1_gap"])
        if abs(mae_gap) < 5 and abs(f1_gap) < 0.05:
            transfer = "transfer: val≈test"
        elif mae_gap > 0 or f1_gap < 0:
            transfer = "transfer: test weaker (house gap)"
        else:
            transfer = "transfer: test better (check leakage)"
        footer_lines.append(transfer)

    n_rows = len(cell_text)  # data rows (excludes header)
    n_footer = len(footer_lines)
    # Compact table, but leave readable space for maF1/miF1 footer under it.
    row_inch = 0.22
    title_inch = 0.28
    footer_inch = 0.22 * n_footer + (0.10 if n_footer else 0.0)
    fig_h = title_inch + row_inch * (n_rows + 1) + footer_inch + 0.10
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.axis("off")
    if title is None:
        title = f"ep{epoch} val vs test" if epoch is not None else "val vs test"
    ax.set_title(title, fontsize=9, pad=1, loc="left")

    # Footer band under the table (enough to read maF1 / transfer lines).
    footer_frac = min(0.36, 0.10 * max(n_footer, 1) + 0.06) if n_footer else 0.0
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        bbox=[0.0, footer_frac, 1.0, 1.0 - footer_frac],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.0)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2f3e4e")
        table[0, j].set_text_props(color="white", weight="bold", fontsize=8)
    for i, app in enumerate(compare["appliance"], start=1):
        if str(app) == "overall":
            for j in range(len(col_labels)):
                table[i, j].set_facecolor("#e8eef5")
                table[i, j].set_text_props(weight="bold")
        elif i % 2 == 0:
            for j in range(len(col_labels)):
                table[i, j].set_facecolor("#f7f7f7")

    for gap_col, key in (
        ("MAE_gap", "MAE_gap"),
        ("SAE_gap", "SAE_gap"),
        ("F1_gap", "maF1_gap"),
    ):
        j = col_labels.index(gap_col)
        for i, (_, r) in enumerate(compare.iterrows(), start=1):
            val = r[key]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            # MAE/SAE: test>val is worse; F1: test<val is worse
            worse = val > 0 if gap_col in {"MAE_gap", "SAE_gap"} else val < 0
            better = val < 0 if gap_col in {"MAE_gap", "SAE_gap"} else val > 0
            if worse:
                table[i, j].set_text_props(color="#b00020")
            elif better:
                table[i, j].set_text_props(color="#1b7f3a")

    # Footer centered in the reserved band, with clear gap below the table.
    if footer_lines:
        # Top of footer band is footer_frac; leave a small gap under table edge.
        y_top = footer_frac - 0.02
        y_bot = 0.04
        span = max(y_top - y_bot, 0.06)
        # Evenly space lines from near-table down toward bottom.
        for i, line in enumerate(footer_lines):
            # i=0 is maF1 line (closer to table), i=1 is transfer note.
            frac = (i + 0.55) / max(n_footer, 1)
            y = y_top - frac * span
            ax.text(
                0.5,
                y,
                line,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7.5,
                family="monospace",
                clip_on=False,
            )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.04)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    csv_path = Path(output_path).with_suffix(".csv")
    compare.to_csv(csv_path, index=False)
    return Path(output_path)


def _list_metric_epoch_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    root = Path(run_dir) / "metrics_by_epoch"
    if not root.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for p in root.iterdir():
        if not p.is_dir() or not p.name.startswith("epoch_"):
            continue
        try:
            ep = int(p.name.split("_", 1)[1])
        except ValueError:
            continue
        out.append((ep, p))
    return sorted(out, key=lambda x: x[0])


def _list_waveform_epoch_dirs(run_dir: Path, split: str) -> list[tuple[int, Path]]:
    root = Path(run_dir) / "waveforms" / split
    if not root.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for p in root.iterdir():
        if not p.is_dir() or not p.name.startswith("epoch_"):
            continue
        try:
            ep = int(p.name.split("_", 1)[1])
        except ValueError:
            continue
        out.append((ep, p))
    return sorted(out, key=lambda x: x[0])


def _pick_period_waveform_png(
    app_dir: Path,
    *,
    period_index: int = 1,
    prefer_context: bool = False,
) -> Path | None:
    """Pick one ON-period PNG (default period 01, focused crop, not context)."""
    if not app_dir.is_dir():
        return None
    period_tag = f"_{int(period_index):02d}_"
    candidates = sorted(app_dir.glob(f"*{period_tag}*.png"))
    if not candidates:
        candidates = sorted(app_dir.glob("*.png"))
    if not candidates:
        return None

    def _is_context(path: Path) -> bool:
        return "_context" in path.stem

    focused = [p for p in candidates if not _is_context(p)]
    context = [p for p in candidates if _is_context(p)]
    pool = context if prefer_context and context else (focused or candidates)
    tagged = [p for p in pool if period_tag in p.name]
    return (tagged or pool)[0]


def _trim_white_border(img: np.ndarray, *, thresh: float = 0.97, pad: int = 4) -> np.ndarray:
    """Crop near-white margins so stacked collages do not waste vertical space."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        content = arr < thresh
    else:
        # Ignore alpha when present.
        rgb = arr[..., :3]
        content = np.any(rgb < thresh, axis=-1)
    rows = np.where(content.any(axis=1))[0]
    cols = np.where(content.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return arr
    r0 = max(int(rows[0]) - pad, 0)
    r1 = min(int(rows[-1]) + pad + 1, arr.shape[0])
    c0 = max(int(cols[0]) - pad, 0)
    c1 = min(int(cols[-1]) + pad + 1, arr.shape[1])
    return arr[r0:r1, c0:c1]


def _resize_width(img: np.ndarray, width: int) -> np.ndarray:
    """Nearest-neighbor width match for stacking differently sized PNGs."""
    h, w = img.shape[:2]
    if w == width:
        return img
    new_h = max(1, int(round(h * (width / float(w)))))
    # Map destination pixels back to source (fast, no PIL dependency).
    ys = (np.linspace(0, h - 1, new_h)).astype(np.int32)
    xs = (np.linspace(0, w - 1, width)).astype(np.int32)
    return img[ys][:, xs]


def _vstack_trimmed_images(
    panels: list[tuple[str, np.ndarray]],
    *,
    gap_px: int = 10,
    label_band_px: int = 28,
) -> np.ndarray:
    """Vertically pack labeled panels with a thin gap (no subplot whitespace)."""
    if not panels:
        raise ValueError("no panels to stack")
    width = max(img.shape[1] for _, img in panels)
    chunks: list[np.ndarray] = []
    for i, (label, img) in enumerate(panels):
        img = _resize_width(_trim_white_border(img), width)
        if img.ndim == 2:
            rgba = np.stack([img, img, img, np.ones_like(img)], axis=-1)
        elif img.shape[-1] == 3:
            rgba = np.concatenate([img, np.ones(img.shape[:2] + (1,), dtype=img.dtype)], axis=-1)
        else:
            rgba = img.astype(np.float32, copy=False)
            if rgba.max() > 1.5:
                rgba = rgba / 255.0

        if label_band_px > 0 and label:
            fig = plt.figure(figsize=(width / 100.0, label_band_px / 100.0), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.text(0.01, 0.5, label, va="center", ha="left", fontsize=10, fontweight="bold")
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.float32) / 255.0
            plt.close(fig)
            band = _resize_width(buf, width)
            if band.shape[0] != label_band_px:
                ys = (np.linspace(0, band.shape[0] - 1, label_band_px)).astype(np.int32)
                band = band[ys]
            chunks.append(band)
        chunks.append(rgba.astype(np.float32, copy=False))
        if i < len(panels) - 1 and gap_px > 0:
            chunks.append(np.full((gap_px, width, 4), 0.85, dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def _overall_from_epoch_comparison(ep_dir: Path) -> dict | None:
    """Load overall row from an epoch's val/test comparison CSV (if present)."""
    csv_path = ep_dir / "validation_test_comparison.csv"
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty or "appliance" not in df.columns:
        return None
    overall = df[df["appliance"].astype(str) == "overall"]
    if overall.empty:
        return None
    return overall.iloc[0].to_dict()


def _resolve_best_epoch_for_collage(
    run_dir: Path,
    epoch_dirs: list[tuple[int, Path]],
    *,
    best_epoch: int | None = None,
) -> tuple[int | None, dict | None, str]:
    """Pick best plotted epoch + overall metrics for the collage footer.

    Preference:
      1) Explicit ``best_epoch`` if that epoch has a comparison table
      2) ``training_time.json`` / ``run_manifest.json`` best_epoch
      3) Among plotted epochs: highest overall val_maF1 (alias val_F1)
    """
    by_ep = {ep: d for ep, d in epoch_dirs}

    def _stats(ep: int) -> dict | None:
        d = by_ep.get(ep)
        return _overall_from_epoch_comparison(d) if d is not None else None

    if best_epoch is not None and int(best_epoch) > 0:
        ep = int(best_epoch)
        st = _stats(ep)
        if st is not None:
            return ep, st, "checkpoint"
        # Checkpoint epoch may fall between plot_interval ticks — still report it.
        return ep, None, "checkpoint"

    for name in ("training_time.json", "run_manifest.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ep = meta.get("best_epoch")
        if ep is None:
            continue
        ep = int(ep)
        if ep > 0:
            st = _stats(ep)
            if st is not None:
                return ep, st, "checkpoint"
            return ep, None, "checkpoint"

    # Fallback: best among plotted epochs by overall validation macro-F1.
    best_ep: int | None = None
    best_st: dict | None = None
    best_f1 = float("-inf")
    for ep, ep_dir in epoch_dirs:
        st = _overall_from_epoch_comparison(ep_dir)
        if st is None:
            continue
        f1 = st.get("val_maF1", st.get("val_F1"))
        if f1 is None or (isinstance(f1, float) and np.isnan(f1)):
            continue
        f1 = float(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_ep = ep
            best_st = st
    return best_ep, best_st, "max val_F1"


def _format_best_epoch_footer(
    best_epoch: int,
    stats: dict | None,
    *,
    rule: str,
) -> str:
    """One-line footer: best epoch + key overall metrics when available."""

    def _f(key_a: str, key_b: str | None = None, *, digits: int = 2) -> str:
        if stats is None:
            return "—"
        val = stats.get(key_a)
        if val is None and key_b is not None:
            val = stats.get(key_b)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        if digits == 4:
            return f"{float(val):.4f}"
        return f"{float(val):.2f}"

    return (
        f"best epoch: {int(best_epoch)}  "
        f"val_MAE={_f('val_MAE')}  test_MAE={_f('test_MAE')}  "
        f"val_F1={_f('val_maF1', 'val_F1', digits=4)}  "
        f"test_F1={_f('test_maF1', 'test_F1', digits=4)}  "
        f"({rule})"
    )


def save_multi_epoch_metrics_collage(
    run_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    title: str | None = None,
    dpi: int = 600,
    best_epoch: int | None = None,
) -> Path | None:
    """Stack every epoch's val/test comparison PNG; footer shows best epoch."""
    from PIL import Image, ImageDraw, ImageFont

    run_dir = Path(run_dir)
    epoch_dirs = _list_metric_epoch_dirs(run_dir)
    panels: list[tuple[str, np.ndarray]] = []
    for ep, ep_dir in epoch_dirs:
        png = ep_dir / "validation_test_comparison.png"
        if not png.exists():
            continue
        panels.append((f"epoch {ep}", plt.imread(png)))
    if not panels:
        return None

    # Each panel already has a short "epN val vs test" title — no extra epoch banners.
    stacked = _vstack_trimmed_images(panels, gap_px=12, label_band_px=0)
    output_path = _ensure_parent(
        output_path
        if output_path is not None
        else run_dir / "comparisons" / "metrics_all_epochs.png"
    )
    # Optional tiny header; empty string skips it.
    if title is None:
        title = ""

    rgba = np.clip(stacked, 0.0, 1.0)
    if rgba.dtype != np.float32 and rgba.dtype != np.float64:
        rgba = rgba.astype(np.float32)
        if rgba.max() > 1.5:
            rgba = rgba / 255.0
    body = (rgba * 255.0).astype(np.uint8)
    if body.shape[-1] == 3:
        body = np.concatenate([body, np.full(body.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)

    width = int(body.shape[1])
    best_ep, best_stats, best_rule = _resolve_best_epoch_for_collage(
        run_dir, epoch_dirs, best_epoch=best_epoch
    )
    footer_text = (
        _format_best_epoch_footer(best_ep, best_stats, rule=best_rule)
        if best_ep is not None
        else ""
    )

    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, int(round(0.016 * width))))
    except OSError:
        font = ImageFont.load_default()

    title_h = 0
    title_band = None
    if title:
        title_h = max(36, int(round(0.028 * width)))
        title_band = Image.new("RGBA", (width, title_h), (255, 255, 255, 255))
        draw = ImageDraw.Draw(title_band)
        draw.text((8, title_h // 2), title, fill=(20, 20, 20, 255), font=font, anchor="lm")

    footer_h = 0
    footer_band = None
    if footer_text:
        footer_h = max(40, int(round(0.032 * width)))
        footer_band = Image.new("RGBA", (width, footer_h), (232, 238, 245, 255))
        draw = ImageDraw.Draw(footer_band)
        # Top rule so the best-epoch line is visually the "last line" of the figure.
        draw.line([(0, 0), (width, 0)], fill=(47, 62, 78, 255), width=2)
        draw.text(
            (width // 2, footer_h // 2 + 1),
            footer_text,
            fill=(20, 20, 20, 255),
            font=font,
            anchor="mm",
        )

    total_h = title_h + body.shape[0] + footer_h
    canvas = Image.new("RGBA", (width, total_h), (255, 255, 255, 255))
    y = 0
    if title_band is not None:
        canvas.paste(title_band, (0, y))
        y += title_h
    canvas.paste(Image.fromarray(body, mode="RGBA"), (0, y))
    y += body.shape[0]
    if footer_band is not None:
        canvas.paste(footer_band, (0, y))
    canvas.convert("RGB").save(output_path, format="PNG", dpi=(dpi, dpi), optimize=True)
    return Path(output_path)


def save_multi_epoch_waveform_collages(
    run_dir: str | Path,
    appliances: list[str],
    *,
    output_dir: str | Path | None = None,
    period_index: int = 1,
    prefer_context: bool = False,
    dpi: int = 600,
    title_prefix: str = "",
) -> list[Path]:
    """Two high-res PNGs per period: validation-only and test-only across epochs.

    Files::
        ALL_appliances_period01_by_epoch_validation.png
        ALL_appliances_period01_by_epoch_test.png

    Layout (each split separately)::
        epoch 50  | kettle | fridge | ... |
        epoch 100 | kettle | fridge | ... |
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir / "comparisons" / "waveforms_by_epoch"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not appliances:
        return []

    saved: list[Path] = []
    for split in ("validation", "test"):
        ep_map = {ep: p for ep, p in _list_waveform_epoch_dirs(run_dir, split)}
        epochs = sorted(ep_map)
        if not epochs:
            continue

        grid_rows: list[tuple[int, list[np.ndarray | None]]] = []
        for ep in epochs:
            imgs: list[np.ndarray | None] = []
            any_hit = False
            for app in appliances:
                png = _pick_period_waveform_png(
                    ep_map[ep] / app,
                    period_index=period_index,
                    prefer_context=prefer_context,
                )
                if png is not None:
                    imgs.append(_trim_white_border(plt.imread(png)))
                    any_hit = True
                else:
                    imgs.append(None)
            if any_hit:
                grid_rows.append((ep, imgs))
        if not grid_rows:
            continue

        n_rows = len(grid_rows)
        n_cols = len(appliances)
        cell_w, cell_h = 4.2, 2.7
        fig_w = min(2.4 + cell_w * n_cols, 36.0)
        fig_h = min(1.0 + cell_h * n_rows, 48.0)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_w, fig_h),
            squeeze=False,
            gridspec_kw={"wspace": 0.04, "hspace": 0.10},
        )
        for i, (ep, imgs) in enumerate(grid_rows):
            for j, (app, img) in enumerate(zip(appliances, imgs)):
                ax = axes[i, j]
                if img is None:
                    ax.set_facecolor("#eeeeee")
                    ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=12, color="#888888")
                else:
                    ax.imshow(img, interpolation="nearest", aspect="auto")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)
                    spine.set_color("#bbbbbb")
                if i == 0:
                    ax.set_title(app, fontsize=12, pad=3)
                if j == 0:
                    ax.set_ylabel(f"ep{ep}", fontsize=10, rotation=0, labelpad=36, va="center")

        fig.suptitle(
            f"{title_prefix}{split} — all appliances period {period_index:02d} by epoch".strip(),
            fontsize=14,
            y=0.995,
        )
        fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.01, wspace=0.04, hspace=0.10)
        out = output_dir / f"ALL_appliances_period{period_index:02d}_by_epoch_{split}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        saved.append(out)

    return saved


def plot_validation_metrics(
    history: pd.DataFrame | str | Path,
    output_path: str | Path,
    *,
    epoch_col: str = "epoch",
    title: str = "Validation Metrics",
    best_epoch: int | None = None,
    figsize: float = 5.5,
    dpi: int = 150,
) -> Path:
    """Per-epoch val F1 / Acc (left) and MAE in watts (right).

    Matches the console line: ON-F1, Acc, MAE=… W. Prefers ``val_mae_watts``;
    falls back to ``val_mae`` for older history.csv files.
    """
    if not isinstance(history, pd.DataFrame):
        history = pd.read_csv(history)
    if history.empty:
        return _ensure_parent(output_path)

    x = history[epoch_col] if epoch_col in history else np.arange(len(history))
    has_f1 = "val_f1" in history.columns and not history["val_f1"].isna().all()
    has_acc = "val_acc" in history.columns and not history["val_acc"].isna().all()
    mae_col = None
    for candidate in ("val_mae_watts", "val_mae"):
        if candidate in history.columns and not history[candidate].isna().all():
            mae_col = candidate
            break
    if not (has_f1 or has_acc or mae_col):
        return _ensure_parent(output_path)

    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    ax.set_box_aspect(1)
    handles: list = []
    labels: list[str] = []

    if has_f1:
        (ln,) = ax.plot(
            x,
            history["val_f1"],
            marker="o",
            markersize=3,
            linewidth=1.8,
            color="#1f77b4",
            label="val ON-F1",
        )
        handles.append(ln)
        labels.append("val ON-F1")
    if has_acc:
        (ln,) = ax.plot(
            x,
            history["val_acc"],
            marker="s",
            markersize=3,
            linewidth=1.6,
            color="#2ca02c",
            label="val Acc",
        )
        handles.append(ln)
        labels.append("val Acc")
    ax.set_ylabel("F1 / Acc")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Epoch")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    _set_epoch_axis(ax, x)

    if mae_col is not None:
        ax_mae = ax.twinx()
        mae_label = "val MAE (W)" if mae_col == "val_mae_watts" else "val MAE"
        (ln_mae,) = ax_mae.plot(
            x,
            history[mae_col],
            marker="^",
            markersize=3,
            linewidth=1.6,
            color="#d62728",
            label=mae_label,
        )
        ax_mae.set_ylabel("MAE (W)" if mae_col == "val_mae_watts" else "MAE")
        handles.append(ln_mae)
        labels.append(mae_label)

    if best_epoch is not None and best_epoch > 0:
        ln_best = ax.axvline(
            best_epoch,
            color="green",
            linestyle="--",
            linewidth=1.2,
            label=f"best epoch {best_epoch}",
        )
        handles.append(ln_best)
        labels.append(f"best epoch {best_epoch}")

    ax.legend(handles, labels, fontsize=8, loc="best")
    fig.tight_layout()
    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


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

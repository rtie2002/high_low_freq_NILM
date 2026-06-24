"""
Interactive multi-feature NILM visualizer.

Default view:
    appliance_power label + P_active

Purpose:
    Compare selected HF features against the appliance power label while
    highlighting ON regions using the on_off column.

Examples:
    python data_quality_checking/multi_feature_label_visualize.py
    python data_quality_checking/multi_feature_label_visualize.py --path feature_selection/dataset/on_only_wk30_wk31/kettle_house2_wk30_to_wk31_merged.csv
    python data_quality_checking/multi_feature_label_visualize.py --features P_active,I_rms,THDI,DWT_E0
    python data_quality_checking/multi_feature_label_visualize.py --scale zscore
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "feature_selection",
    "dataset",
    "on_only_wk30_wk31",
)
MAX_VISIBLE_FEATURES = 12
MAX_LEGEND_ITEMS = 12


def split_features(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def find_csv_files(data_dir: str) -> list[str]:
    if not os.path.exists(data_dir):
        return []
    return sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.lower().endswith(".csv")
    )


def choose_file(data_dir: str) -> str | None:
    files = find_csv_files(data_dir)
    if not files:
        print(f"No CSV files found in: {data_dir}")
        return None

    print(f"\nAvailable CSV files in: {data_dir}")
    for idx, path in enumerate(files):
        print(f" [{idx}] {os.path.basename(path)}")

    raw = input("\nEnter index, appliance keyword, or full CSV path: ").strip().strip('"')
    if os.path.exists(raw):
        return raw
    if raw.isdigit() and int(raw) < len(files):
        return files[int(raw)]
    if raw:
        matches = [path for path in files if raw.lower() in os.path.basename(path).lower()]
        if matches:
            return matches[0]
    return None


def detect_label_column(df: pd.DataFrame, file_path: str, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested label column not found: {requested}")
        return requested

    filename = os.path.basename(file_path).lower()
    power_cols = [col for col in df.columns if col.endswith("_power")]
    if not power_cols:
        raise ValueError("No appliance power label found. Use --label COLUMN.")

    for col in power_cols:
        appliance = col[: -len("_power")].lower()
        if appliance in filename:
            return col
    return power_cols[0]


def detect_multi_appliance_columns(df: pd.DataFrame) -> list[str]:
    apps = []
    for col in df.columns:
        if not col.endswith("_on"):
            continue
        app = col[: -len("_on")]
        if f"{app}_power" in df.columns:
            apps.append(app)
    return apps


def numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col == "on_off":
            continue
        if col.endswith("_on"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def clean_series(values: Iterable[float]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)
    return arr.interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)


def scale_series(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return values
    if mode == "zscore":
        std = float(np.std(values))
        return (values - float(np.mean(values))) / (std if std > 0 else 1.0)
    if mode == "minmax":
        vmin, vmax = float(np.min(values)), float(np.max(values))
        span = vmax - vmin
        return (values - vmin) / (span if span > 0 else 1.0)
    raise ValueError(f"Unknown scale mode: {mode}")


def on_segments(mask: np.ndarray) -> list[tuple[int, int, int]]:
    if mask is None:
        return []
    clean = np.asarray(mask).astype(float)
    clean = np.nan_to_num(clean, nan=0.0)
    clean = (clean > 0).astype(int)
    diff = np.diff(np.concatenate([[0], clean, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e), idx + 1) for idx, (s, e) in enumerate(zip(starts, ends))]


def default_feature_list(
    df: pd.DataFrame,
    label_col: str,
    requested: list[str],
    multi_apps: list[str] | None = None,
) -> list[str]:
    if requested:
        return requested
    features = []
    if multi_apps:
        if "aggregate" in df.columns:
            features.append("aggregate")
        if "P_active" in df.columns:
            features.append("P_active")
        features.extend(f"{app}_power" for app in multi_apps if f"{app}_power" in df.columns)
        return list(dict.fromkeys(features))

    features = [label_col]
    if "P_active" in df.columns and "P_active" != label_col:
        features.append("P_active")
    return features


def interactive_multi_appliance_viewer(
    file_path: str,
    df: pd.DataFrame,
    appliances: list[str],
    features: list[str] | None = None,
    scale: str = "none",
    view_span: int = 1024,
) -> None:
    """Professional stacked view for aligned multi-appliance CSV files."""
    total_points = len(df)
    aggregate_col = "aggregate" if "aggregate" in df.columns else None
    default_cols = ([aggregate_col] if aggregate_col else []) + [
        f"{app}_power" for app in appliances if f"{app}_power" in df.columns
    ]
    selected = features or default_cols
    selected = [col for col in selected if col in df.columns]
    if aggregate_col and aggregate_col not in selected:
        selected.insert(0, aggregate_col)

    raw_data = {
        col: clean_series(df[col])
        for col in numeric_columns(df)
        if col in df.columns
    }
    on_masks = {
        app: pd.to_numeric(df[f"{app}_on"], errors="coerce").fillna(0).to_numpy()
        for app in appliances
        if f"{app}_on" in df.columns
    }
    segments_by_app = {app: on_segments(mask) for app, mask in on_masks.items()}

    app_colors = dict(
        zip(
            appliances,
            plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(appliances)],
        )
    )
    if len(app_colors) < len(appliances):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        app_colors = {app: colors[idx % len(colors)] for idx, app in enumerate(appliances)}

    plot_apps = [app for app in appliances if f"{app}_power" in raw_data]
    n_rows = 1 + len(plot_apps)
    fig_height = min(12.0, max(8.5, 1.65 * n_rows))
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(15.5, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15] + [1.0] * len(plot_apps), "hspace": 0.10},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    plt.subplots_adjust(left=0.075, right=0.82, bottom=0.22, top=0.925)

    title = fig.suptitle("", fontsize=12.5, fontweight="bold")
    status = fig.text(0.5, 0.012, "", ha="center", va="bottom", fontsize=9.5, color="#146c2e")
    axis_by_name = {"aggregate": axes[0]}
    for idx, app in enumerate(plot_apps, start=1):
        axis_by_name[app] = axes[idx]

    state = {
        "start": 0,
        "span": min(max(100, view_span), total_points),
        "scale": scale,
        "show_on": True,
        "visible": {"aggregate": aggregate_col is not None, **{app: True for app in plot_apps}},
        "lines": [],
        "patches": [],
    }

    def visible_slice() -> tuple[int, int, np.ndarray]:
        start = int(state["start"])
        end = min(start + int(state["span"]), total_points)
        return start, end, np.arange(start, end)

    def scaled(col: str, start: int, end: int) -> np.ndarray:
        return scale_series(raw_data[col], state["scale"])[start:end]

    def clear_artists() -> None:
        for artist in state["lines"] + state["patches"]:
            artist.remove()
        state["lines"] = []
        state["patches"] = []

    def add_on_regions(ax, app: str, start: int, end: int) -> int:
        if not state["show_on"] or not state["visible"].get(app, True):
            return 0
        visible = [(s, e, n) for s, e, n in segments_by_app.get(app, []) if e > start and s < end]
        for seg_start, seg_end, _ in visible:
            patch = ax.axvspan(
                max(seg_start, start),
                min(seg_end, end),
                color=app_colors[app],
                alpha=0.12,
                linewidth=0,
                zorder=0,
            )
            state["patches"].append(patch)
        return len(visible)

    def set_axis_style(ax, ylabel: str) -> None:
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="x", alpha=0.22)
        ax.grid(True, axis="y", alpha=0.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def fit_axes() -> None:
        start, end, _ = visible_slice()
        for key, ax in axis_by_name.items():
            vals = []
            if key == "aggregate" and aggregate_col and state["visible"].get("aggregate", True):
                vals.append(scaled(aggregate_col, start, end))
            elif key != "aggregate" and state["visible"].get(key, True):
                vals.append(scaled(f"{key}_power", start, end))
            if not vals:
                ax.set_ylim(-1, 1)
                continue
            arr = np.concatenate(vals)
            ymin, ymax = float(np.nanmin(arr)), float(np.nanmax(arr))
            span = ymax - ymin if ymax > ymin else 1.0
            ax.set_ylim(ymin - span * 0.12, ymax + span * 0.18)

    def redraw(_=None) -> None:
        clear_artists()
        start, end, x = visible_slice()
        state["start"] = start
        title.set_text(
            f"{os.path.basename(file_path)} | multi-appliance | scale={state['scale']} | "
            f"{start:,}->{end:,}"
        )

        for ax in axes:
            ax.set_xlim(start, end)

        ax0 = axes[0]
        if aggregate_col and state["visible"].get("aggregate", True):
            (line,) = ax0.plot(
                x,
                scaled(aggregate_col, start, end),
                color="#333333",
                linewidth=1.8,
                label=aggregate_col,
                zorder=3,
            )
            state["lines"].append(line)
            ax0.legend(loc="upper right", fontsize=8, frameon=False)
        else:
            legend = ax0.get_legend()
            if legend:
                legend.remove()
        set_axis_style(ax0, "Aggregate W" if state["scale"] == "none" else "Aggregate")

        visible_summary = []
        for idx, app in enumerate(plot_apps, start=1):
            ax = axes[idx]
            shown = state["visible"].get(app, True)
            visible_events = add_on_regions(ax, app, start, end)
            visible_summary.append(f"{app}:{visible_events}/{len(segments_by_app.get(app, []))}")
            if shown:
                (line,) = ax.plot(
                    x,
                    scaled(f"{app}_power", start, end),
                    color=app_colors[app],
                    linewidth=1.65,
                    label=f"{app}_power",
                    zorder=3,
                )
                state["lines"].append(line)
                ax.legend(loc="upper right", fontsize=8, frameon=False)
            else:
                legend = ax.get_legend()
                if legend:
                    legend.remove()
            set_axis_style(ax, f"{app}\nW" if state["scale"] == "none" else app)

        axes[-1].set_xlabel("sample index")
        fit_axes()
        status.set_text("visible ON events: " + " | ".join(visible_summary))
        fig.canvas.draw_idle()

    def sync_from_sliders(_=None) -> None:
        state["start"] = int(pos_slider.val)
        state["span"] = int(span_slider.val)
        redraw()

    def move(delta: int) -> None:
        pos_slider.set_val(min(max(0, state["start"] + delta), max_start))

    def on_check(label: str) -> None:
        key = "aggregate" if label == "aggregate" else label
        state["visible"][key] = not state["visible"].get(key, True)
        redraw()

    def on_scale(label: str) -> None:
        state["scale"] = "none" if label == "raw" else label
        redraw()

    def toggle_on(_=None) -> None:
        state["show_on"] = not state["show_on"]
        on_button.label.set_text(f"ON shade: {'on' if state['show_on'] else 'off'}")
        redraw()

    def print_stats(_=None) -> None:
        start, end, _ = visible_slice()
        print("\n" + "=" * 88)
        print(f"MULTI-APPLIANCE WINDOW STATISTICS: rows {start:,} to {end:,}")
        for app in plot_apps:
            col = f"{app}_power"
            vals = raw_data[col][start:end]
            mask = on_masks.get(app, np.zeros(end - start))[start:end]
            print(
                f"{app:16s} mean={np.mean(vals):10.3f} min={np.min(vals):10.3f} "
                f"max={np.max(vals):10.3f} on_ratio={np.mean(mask > 0):8.4f}"
            )
        if aggregate_col:
            vals = raw_data[aggregate_col][start:end]
            print(f"{aggregate_col:16s} mean={np.mean(vals):10.3f} min={np.min(vals):10.3f} max={np.max(vals):10.3f}")
        print("=" * 88)

    control_y = 0.075
    ax_pos = plt.axes([0.09, control_y + 0.075, 0.46, 0.026])
    max_start = max(0, total_points - 1)
    pos_slider = Slider(ax_pos, "Start", 0, max_start, valinit=0, valstep=1, valfmt="%d")
    ax_span = plt.axes([0.09, control_y + 0.025, 0.46, 0.026])
    span_slider = Slider(
        ax_span,
        "Span",
        50,
        max(50, min(total_points, 50000)),
        valinit=state["span"],
        valstep=50,
        valfmt="%d",
    )
    pos_slider.valtext.set_visible(False)
    span_slider.valtext.set_visible(False)
    pos_slider.on_changed(sync_from_sliders)
    span_slider.on_changed(sync_from_sliders)

    ax_back = plt.axes([0.59, control_y + 0.075, 0.07, 0.035])
    ax_next = plt.axes([0.67, control_y + 0.075, 0.07, 0.035])
    ax_fit = plt.axes([0.59, control_y + 0.025, 0.07, 0.035])
    ax_stats = plt.axes([0.67, control_y + 0.025, 0.07, 0.035])
    ax_on = plt.axes([0.75, control_y + 0.025, 0.10, 0.035])
    back_button = Button(ax_back, "Back")
    next_button = Button(ax_next, "Next")
    fit_button = Button(ax_fit, "Fit")
    stats_button = Button(ax_stats, "Stats")
    on_button = Button(ax_on, "ON shade: on")
    back_button.on_clicked(lambda _: move(-state["span"] // 2))
    next_button.on_clicked(lambda _: move(state["span"] // 2))
    fit_button.on_clicked(lambda _: (fit_axes(), fig.canvas.draw_idle()))
    stats_button.on_clicked(print_stats)
    on_button.on_clicked(toggle_on)

    check_labels = (["aggregate"] if aggregate_col else []) + plot_apps
    check_status = [state["visible"].get(label, True) for label in check_labels]
    ax_checks = plt.axes([0.84, 0.47, 0.14, 0.34])
    checks = CheckButtons(ax_checks, check_labels, check_status)
    ax_checks.set_title("Show / hide", fontsize=9)
    checks.on_clicked(on_check)

    ax_scale = plt.axes([0.84, 0.27, 0.14, 0.13])
    scale_radio = RadioButtons(ax_scale, ["raw", "zscore", "minmax"], active=["none", "zscore", "minmax"].index(scale))
    ax_scale.set_title("Scale", fontsize=9)
    scale_radio.on_clicked(on_scale)

    # Keep widgets alive.
    state["widget_refs"] = [
        pos_slider,
        span_slider,
        back_button,
        next_button,
        fit_button,
        stats_button,
        on_button,
        checks,
        scale_radio,
    ]

    print(f"Rows        : {total_points:,}")
    print("Mode        : multi-appliance stacked")
    print(f"Appliances  : {', '.join(plot_apps)}")
    for app in plot_apps:
        on_rows = int(np.asarray(on_masks.get(app, [])).sum())
        print(f"ON segments : {app:<15} {len(segments_by_app.get(app, [])):>5} events | {on_rows:>8,} ON rows")

    redraw()
    plt.show()


def interactive_viewer(
    file_path: str,
    label_col: str | None = None,
    features: list[str] | None = None,
    scale: str = "none",
    view_span: int = 1024,
) -> None:
    print(f"\nLoading data: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        print("CSV is empty.")
        return

    multi_apps = detect_multi_appliance_columns(df)
    is_multi_appliance = len(multi_apps) >= 2
    if is_multi_appliance:
        interactive_multi_appliance_viewer(
            file_path=file_path,
            df=df,
            appliances=multi_apps,
            features=features or [],
            scale=scale,
            view_span=view_span,
        )
        return

    if is_multi_appliance and label_col is None and "aggregate" in df.columns:
        label_col = "aggregate"
    else:
        label_col = detect_label_column(df, file_path, label_col)
    numeric = numeric_columns(df)
    if label_col not in numeric:
        numeric.append(label_col)

    selected = default_feature_list(df, label_col, features or [], multi_apps if is_multi_appliance else None)
    missing = [col for col in selected if col not in df.columns]
    if missing:
        raise ValueError(f"Feature(s) not found: {missing}")

    if is_multi_appliance:
        on_masks = {app: df[f"{app}_on"].to_numpy() for app in multi_apps}
    else:
        on_masks = {label_col.replace("_power", ""): df["on_off"].to_numpy()} if "on_off" in df.columns else {}
    segments_by_app = {app: on_segments(mask) for app, mask in on_masks.items()}
    segments = [seg for app_segments in segments_by_app.values() for seg in app_segments]
    total_points = len(df)

    raw_data = {col: clean_series(df[col]) for col in numeric if col in df.columns}

    print(f"Rows        : {total_points:,}")
    print(f"Mode        : {'multi-appliance' if is_multi_appliance else 'single-appliance'}")
    print(f"Label       : {label_col}")
    if is_multi_appliance:
        print(f"Appliances  : {', '.join(multi_apps)}")
    print(f"Default view: {', '.join(selected)}")
    if is_multi_appliance:
        for app in multi_apps:
            on_rows = int(pd.to_numeric(df[f"{app}_on"], errors="coerce").fillna(0).gt(0).sum())
            print(f"ON segments : {app:<15} {len(segments_by_app[app]):>5} events | {on_rows:>8,} ON rows")
    else:
        print(f"ON segments : {len(segments)}")
    print("\nSelectable numeric columns:")
    print(", ".join(numeric))
    print("\nTip: type comma-separated feature names in the Features box, then click Apply.")

    state = {
        "start": 0,
        "span": min(max(100, view_span), total_points),
        "features": selected,
        "scale": scale,
        "show_on": True,
        "patches": [],
        "labels": [],
        "picker_refs": [],
        "widget_refs": [],
        "feature_box_summary": "",
        "hidden_features": set(),
    }
    event_colors = {
        app: plt.rcParams["axes.prop_cycle"].by_key()["color"][idx % len(plt.rcParams["axes.prop_cycle"].by_key()["color"])]
        for idx, app in enumerate(multi_apps if is_multi_appliance else list(on_masks.keys()))
    }

    fig, ax = plt.subplots(figsize=(14, 7.5))
    plt.subplots_adjust(left=0.07, right=0.78, bottom=0.27, top=0.90)
    status = fig.text(0.5, 0.015, "", ha="center", va="bottom", fontsize=10, color="darkgreen")

    lines: dict[str, plt.Line2D] = {}
    legend_map: dict[plt.Line2D, plt.Line2D] = {}

    def title_text(end_idx: int) -> str:
        basename = os.path.basename(file_path)
        return (
            f"{basename} | label={label_col} | scale={state['scale']} | "
            f"{state['start']:,}->{end_idx:,}"
        )

    def visible_slice() -> tuple[int, int, np.ndarray]:
        start = int(state["start"])
        end = min(start + int(state["span"]), total_points)
        return start, end, np.arange(start, end)

    def current_values(col: str, start: int, end: int) -> np.ndarray:
        return scale_series(raw_data[col], state["scale"])[start:end]

    def clear_on_regions() -> None:
        for patch in state["patches"]:
            patch.remove()
        for label in state["labels"]:
            label.remove()
        state["patches"] = []
        state["labels"] = []

    def redraw_on_regions(start: int, end: int) -> None:
        clear_on_regions()
        if not state["show_on"] or not segments:
            status.set_text("")
            return

        if is_multi_appliance:
            transform = blended_transform_factory(ax.transData, ax.transAxes)
            lane_height = min(0.045, 0.23 / max(1, len(multi_apps)))
            top = 0.985
            visible_total = 0
            summary = []
            for app_idx, app in enumerate(multi_apps):
                app_segments = segments_by_app.get(app, [])
                visible = [(s, e, n) for s, e, n in app_segments if e > start and s < end]
                visible_total += len(visible)
                summary.append(f"{app}:{len(visible)}/{len(app_segments)}")
                y0 = top - (app_idx + 1) * lane_height
                color = event_colors[app]
                for seg_start, seg_end, _number in visible:
                    x0 = max(seg_start, start)
                    x1 = min(seg_end, end)
                    width = max(1, x1 - x0)
                    patch = Rectangle(
                        (x0, y0),
                        width,
                        lane_height * 0.72,
                        transform=transform,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=0.5,
                        alpha=0.78,
                        zorder=5,
                        clip_on=False,
                    )
                    ax.add_patch(patch)
                    state["patches"].append(patch)
                text = ax.text(
                    1.005,
                    y0 + lane_height * 0.36,
                    app,
                    transform=ax.transAxes,
                    va="center",
                    fontsize=8,
                    color=color,
                    fontweight="bold",
                )
                state["labels"].append(text)
            status.set_text(
                f"Multi ON events visible: {visible_total} | "
                + " | ".join(summary[:5])
            )
            return

        y0, y1 = ax.get_ylim()
        label_y = y1 - (y1 - y0) * 0.04
        visible = [(s, e, n) for s, e, n in segments if e > start and s < end]
        for seg_start, seg_end, number in visible:
            patch = ax.axvspan(seg_start, seg_end, color="lightgreen", alpha=0.30, zorder=-10)
            state["patches"].append(patch)
            label_x = (max(seg_start, start) + min(seg_end, end)) / 2
            text = ax.text(
                label_x,
                label_y,
                str(number),
                ha="center",
                va="top",
                fontsize=8,
                color="darkgreen",
                fontweight="bold",
            )
            state["labels"].append(text)
        status.set_text(f"ON periods total: {len(segments)} | visible: {len(visible)}")

    def redraw(_=None) -> None:
        start, end, x = visible_slice()
        state["start"] = start

        for old_line in lines.values():
            old_line.remove()
        lines.clear()
        legend_map.clear()

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        plot_features = list(state["features"])
        if len(plot_features) > MAX_VISIBLE_FEATURES:
            plot_features = plot_features[:MAX_VISIBLE_FEATURES]

        for idx, col in enumerate(plot_features):
            if col not in raw_data:
                continue
            width = 2.2 if col == label_col else 1.4
            alpha = 0.95 if col == label_col else 0.78
            label = f"{col} (label)" if col == label_col else col
            (line,) = ax.plot(
                x,
                current_values(col, start, end),
                label=label,
                linewidth=width,
                alpha=alpha,
                color=colors[idx % len(colors)],
                picker=5,
            )
            line.set_visible(col not in state["hidden_features"])
            lines[col] = line

        ax.set_xlim(start, end)
        extra_title = ""
        if len(state["features"]) > len(plot_features):
            extra_title = f" | showing {len(plot_features)}/{len(state['features'])} selected"
        ax.set_title(title_text(end) + extra_title)
        ax.set_ylabel("Value" if state["scale"] == "none" else f"{state['scale']} value")
        ax.grid(True, alpha=0.25)
        legend_lines = list(lines.values())
        legend_labels = [line.get_label() for line in legend_lines]
        if len(legend_lines) > MAX_LEGEND_ITEMS:
            legend_lines = legend_lines[:MAX_LEGEND_ITEMS]
            legend_labels = legend_labels[:MAX_LEGEND_ITEMS]
            legend_title = f"First {MAX_LEGEND_ITEMS}/{len(lines)} shown"
        else:
            legend_title = "Click to hide/show"
        legend_font = 8 if len(legend_lines) <= 8 else 7
        legend_cols = 1 if len(legend_lines) <= 10 else 2

        legend = ax.legend(
            legend_lines,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=legend_font,
            title=legend_title,
            title_fontsize=legend_font,
            ncols=legend_cols,
            handlelength=1.2,
            labelspacing=0.35,
            borderpad=0.35,
        )
        if legend is not None:
            for legend_line, plot_line in zip(legend.get_lines(), legend_lines):
                legend_line.set_picker(8)
                legend_line.set_pickradius(8)
                legend_line.set_alpha(1.0 if plot_line.get_visible() else 0.22)
                legend_map[legend_line] = plot_line
        redraw_on_regions(start, end)
        fig.canvas.draw_idle()

    def auto_fit(_=None) -> None:
        start, end, _ = visible_slice()
        values = []
        for line in lines.values():
            if line.get_visible():
                y = line.get_ydata()
                if len(y) > 0:
                    values.append(np.asarray(y, dtype=float))
        if not values:
            return
        combined = np.concatenate(values)
        ymin, ymax = float(np.min(combined)), float(np.max(combined))
        span = ymax - ymin
        if span <= 0:
            span = 1.0
        ax.set_ylim(ymin - span * 0.10, ymax + span * 0.10)
        redraw_on_regions(start, end)
        fig.canvas.draw_idle()

    def apply_features(_=None) -> None:
        if feature_box.text == state.get("feature_box_summary"):
            print("[info] Feature list came from picker. Use Pick to edit it, or type feature names manually.")
            return
        requested = split_features(feature_box.text)
        valid = [col for col in requested if col in raw_data]
        invalid = [col for col in requested if col not in raw_data]
        if invalid:
            print(f"[warning] ignored unknown feature(s): {', '.join(invalid)}")
        if not valid:
            print("[warning] no valid features requested; keeping current view")
            return
        if label_col not in valid:
            valid.insert(0, label_col)
        state["features"] = valid
        state["hidden_features"] = {col for col in state["hidden_features"] if col in valid}
        redraw()
        auto_fit()

    def open_feature_picker(_=None) -> None:
        picker_cols = [col for col in numeric if col in raw_data and col != label_col]
        if not picker_cols:
            print("[warning] no selectable feature columns found")
            return

        selected_set = set(state["features"])
        n_cols = 3 if len(picker_cols) > 28 else 2
        chunk_size = int(np.ceil(len(picker_cols) / n_cols))

        picker_fig = plt.figure(figsize=(14, 10))
        picker_fig.suptitle(
            f"Select features to compare with {label_col}",
            fontsize=13,
            fontweight="bold",
        )

        check_groups: list[CheckButtons] = []
        for idx in range(n_cols):
            start = idx * chunk_size
            end = min(start + chunk_size, len(picker_cols))
            labels = picker_cols[start:end]
            if not labels:
                continue

            left = 0.05 + idx * (0.90 / n_cols)
            width = 0.82 / n_cols
            ax_check = picker_fig.add_axes([left, 0.16, width, 0.74])
            checks = CheckButtons(
                ax_check,
                labels,
                [label in selected_set for label in labels],
            )
            ax_check.set_title(f"Features {start + 1}-{end}", fontsize=10)
            check_groups.append(checks)

        def picked_features() -> list[str]:
            picked = [label_col]
            for checks in check_groups:
                statuses = checks.get_status()
                labels = [text.get_text() for text in checks.labels]
                picked.extend(label for label, enabled in zip(labels, statuses) if enabled)
            return picked

        def apply_picker(_event=None) -> None:
            picked = picked_features()
            state["features"] = picked
            state["hidden_features"] = {col for col in state["hidden_features"] if col in picked}
            summary = f"{len(picked)} selected via Pick"
            state["feature_box_summary"] = summary
            feature_box.set_val(summary)
            if len(picked) > MAX_VISIBLE_FEATURES:
                print(
                    f"[info] {len(picked)} features selected. "
                    f"Only first {MAX_VISIBLE_FEATURES} are displayed to keep the plot readable. "
                    "Use Pick to choose a smaller focused group."
                )
            redraw()
            auto_fit()
            picker_fig.canvas.draw_idle()

        def clear_picker(_event=None) -> None:
            for checks in check_groups:
                for idx, enabled in enumerate(checks.get_status()):
                    if enabled:
                        checks.set_active(idx)

        def select_all_picker(_event=None) -> None:
            for checks in check_groups:
                for idx, enabled in enumerate(checks.get_status()):
                    if not enabled:
                        checks.set_active(idx)

        ax_apply_picker = picker_fig.add_axes([0.18, 0.05, 0.14, 0.055])
        ax_clear_picker = picker_fig.add_axes([0.36, 0.05, 0.14, 0.055])
        ax_all_picker = picker_fig.add_axes([0.54, 0.05, 0.14, 0.055])
        ax_close_picker = picker_fig.add_axes([0.72, 0.05, 0.14, 0.055])

        btn_apply_picker = Button(ax_apply_picker, "Apply")
        btn_clear_picker = Button(ax_clear_picker, "Clear")
        btn_all_picker = Button(ax_all_picker, "Select all")
        btn_close_picker = Button(ax_close_picker, "Close")

        btn_apply_picker.on_clicked(apply_picker)
        btn_clear_picker.on_clicked(clear_picker)
        btn_all_picker.on_clicked(select_all_picker)
        btn_close_picker.on_clicked(lambda _event: plt.close(picker_fig))
        picker_fig.text(
            0.5,
            0.005,
            "Tip: keep the list small, then use zscore/minmax if feature scales differ.",
            ha="center",
            fontsize=9,
            color="dimgray",
        )
        # Keep widget objects alive; Matplotlib stores widget callbacks weakly.
        state["picker_refs"] = [
            picker_fig,
            check_groups,
            btn_apply_picker,
            btn_clear_picker,
            btn_all_picker,
            btn_close_picker,
        ]
        picker_fig.canvas.draw_idle()
        manager = getattr(picker_fig.canvas, "manager", None)
        if manager is not None:
            manager.show()
        plt.show(block=False)

    def toggle_on(_=None) -> None:
        state["show_on"] = not state["show_on"]
        on_button.label.set_text(f"ON shade: {'on' if state['show_on'] else 'off'}")
        redraw()

    def set_scale(mode: str) -> None:
        state["scale"] = mode
        redraw()
        auto_fit()

    def print_stats(_=None) -> None:
        start, end, _ = visible_slice()
        print("\n" + "=" * 88)
        print(f"VISIBLE WINDOW STATISTICS: rows {start:,} to {end:,}")
        for col in state["features"]:
            if col not in raw_data:
                continue
            vals = raw_data[col][start:end]
            print(
                f"{col:24} mean={np.mean(vals):12.4f} "
                f"min={np.min(vals):12.4f} max={np.max(vals):12.4f} "
                f"std={np.std(vals):12.4f}"
            )
        for app, mask in on_masks.items():
            visible_on = np.asarray(mask[start:end]).astype(float) > 0
            print(f"{app + ' on ratio':24} {visible_on.mean():12.4f}")
        print("=" * 88)

    ax_pos = plt.axes([0.09, 0.135, 0.48, 0.028])
    max_start = max(0, total_points - 1)
    pos_slider = Slider(ax_pos, "Start", 0, max_start, valinit=0, valstep=1, valfmt="%d")

    ax_span = plt.axes([0.09, 0.085, 0.48, 0.028])
    span_slider = Slider(
        ax_span,
        "Span",
        50,
        max(50, min(total_points, 50000)),
        valinit=state["span"],
        valstep=50,
        valfmt="%d",
    )

    ax_features = plt.axes([0.09, 0.195, 0.48, 0.035])
    feature_box = TextBox(ax_features, "Features", initial=",".join(state["features"]))

    ax_apply = plt.axes([0.595, 0.195, 0.075, 0.035])
    ax_pick = plt.axes([0.680, 0.195, 0.075, 0.035])
    ax_prev = plt.axes([0.595, 0.135, 0.075, 0.038])
    ax_next = plt.axes([0.680, 0.135, 0.075, 0.038])
    ax_fit = plt.axes([0.595, 0.085, 0.075, 0.038])
    ax_stats = plt.axes([0.680, 0.085, 0.075, 0.038])
    ax_on = plt.axes([0.595, 0.035, 0.160, 0.040])
    ax_raw = plt.axes([0.815, 0.145, 0.095, 0.035])
    ax_z = plt.axes([0.815, 0.100, 0.095, 0.035])
    ax_mm = plt.axes([0.815, 0.055, 0.095, 0.035])

    apply_button = Button(ax_apply, "Apply")
    pick_button = Button(ax_pick, "Pick")
    prev_button = Button(ax_prev, "Back")
    next_button = Button(ax_next, "Next")
    fit_button = Button(ax_fit, "Fit")
    stats_button = Button(ax_stats, "Stats")
    on_button = Button(ax_on, "ON shade: on")
    raw_button = Button(ax_raw, "raw")
    z_button = Button(ax_z, "zscore")
    mm_button = Button(ax_mm, "minmax")

    def sync_from_sliders(_=None) -> None:
        state["start"] = int(pos_slider.val)
        state["span"] = int(span_slider.val)
        redraw()

    pos_slider.on_changed(sync_from_sliders)
    span_slider.on_changed(sync_from_sliders)
    pos_slider.valtext.set_visible(False)
    span_slider.valtext.set_visible(False)
    apply_button.on_clicked(apply_features)
    pick_button.on_clicked(open_feature_picker)
    feature_box.on_submit(apply_features)
    prev_button.on_clicked(lambda _: pos_slider.set_val(max(0, state["start"] - state["span"] // 2)))
    next_button.on_clicked(lambda _: pos_slider.set_val(min(max_start, state["start"] + state["span"] // 2)))
    fit_button.on_clicked(auto_fit)
    stats_button.on_clicked(print_stats)
    on_button.on_clicked(toggle_on)
    raw_button.on_clicked(lambda _: set_scale("none"))
    z_button.on_clicked(lambda _: set_scale("zscore"))
    mm_button.on_clicked(lambda _: set_scale("minmax"))
    state["widget_refs"] = [
        pos_slider,
        span_slider,
        feature_box,
        apply_button,
        pick_button,
        prev_button,
        next_button,
        fit_button,
        stats_button,
        on_button,
        raw_button,
        z_button,
        mm_button,
    ]

    def on_pick(event) -> None:
        legend_line = event.artist
        if legend_line not in legend_map:
            return
        plot_line = legend_map[legend_line]
        visible = not plot_line.get_visible()
        plot_line.set_visible(visible)
        feature = next((col for col, line in lines.items() if line is plot_line), None)
        if feature is not None:
            if visible:
                state["hidden_features"].discard(feature)
            else:
                state["hidden_features"].add(feature)
        legend_line.set_alpha(1.0 if visible else 0.22)
        auto_fit()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)

    redraw()
    auto_fit()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-feature NILM label visualizer")
    parser.add_argument("--path", type=str, default=None, help="CSV file to visualize")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label", type=str, default=None, help="Label column, e.g. kettle_power")
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature list. Label is auto-added if omitted.",
    )
    parser.add_argument("--scale", choices=["none", "zscore", "minmax"], default="none")
    parser.add_argument("--view_span", type=int, default=1024)
    args = parser.parse_args()

    file_path = args.path or choose_file(args.data_dir)
    if not file_path or not os.path.exists(file_path):
        print(f"Error: file not found: {file_path}")
        return

    interactive_viewer(
        file_path=file_path,
        label_col=args.label,
        features=split_features(args.features),
        scale=args.scale,
        view_span=args.view_span,
    )


if __name__ == "__main__":
    main()

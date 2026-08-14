"""Interactive checkpoint prediction visualizer for MultiNILM-Fractional.

This script loads a trained checkpoint, automatically uses the experiment test
CSV, runs model prediction, then opens an interactive plot:

    aggregate
    true vs predicted appliance power
    true ON shading and predicted ON shading

Examples:
    python data_quality_checking/checkpoint_prediction_visualize.py

    python data_quality_checking/checkpoint_prediction_visualize.py ^
      --checkpoint "multi_appliances_NILM/runs/mixed_ukdale_refit_3w (domain adaptation)/multinilm_fractional/best.pt"

    python data_quality_checking/checkpoint_prediction_visualize.py --split validation
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NILM_ROOT = PROJECT_ROOT / "multi_appliances_NILM"
if str(NILM_ROOT) not in sys.path:
    sys.path.insert(0, str(NILM_ROOT))


def _select_interactive_backend() -> None:
    """Pick a GUI backend that is really importable in the active conda env."""
    candidates = [
        ("TkAgg", "matplotlib.backends.backend_tkagg"),
        ("WXAgg", "matplotlib.backends.backend_wxagg"),
        ("QtAgg", "matplotlib.backends.backend_qtagg"),
    ]
    for backend, module_name in candidates:
        try:
            importlib.import_module(module_name)
            matplotlib.use(backend, force=True)
            return
        except Exception:
            continue


_select_interactive_backend()

if "agg" in matplotlib.get_backend().lower():
    print(
        "[warning] Matplotlib is using a non-interactive backend. "
        "Install/enable Tk, wxPython, or PyQt/PySide to open the viewer window.",
        flush=True,
    )

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from adapters.config import (  # noqa: E402
    load_experiment,
    load_model_config,
    merge_configs,
    model_name_from_config,
    resolve_tensor_dtype,
)
from adapters.mat_nilm import MATNILMAdapter  # noqa: E402
from adapters.matuda import MATUDAAdapter  # noqa: E402
from adapters.multinilm import MultiNILMAdapter  # noqa: E402
from adapters.multinilm_fractional import MultiNILMFractionalAdapter  # noqa: E402
from adapters.multinilm_kle import MultiNILMKLEAdapter  # noqa: E402
from adapters.multinilm_no_distill import MultiNILMNoDistillAdapter  # noqa: E402
from adapters.transfer_multi_appliance import TransferMultiApplianceAdapter  # noqa: E402


MODELS = {
    "mat_nilm": MATNILMAdapter,
    "matuda": MATUDAAdapter,
    "multinilm": MultiNILMAdapter,
    "multinilm_fractional": MultiNILMFractionalAdapter,
    "multinilm_kle": MultiNILMKLEAdapter,
    "multinilm_no_distill": MultiNILMNoDistillAdapter,
    "transfer_multi_appliance": TransferMultiApplianceAdapter,
}


def get_adapter(model_name: str, merged_cfg: dict, data_root: str | None = None):
    if model_name not in MODELS:
        known = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown model {model_name!r}. Available: {known}")
    return MODELS[model_name](merged_cfg, data_root=data_root)


def _default_run_dir(experiment_id: str, model_name: str) -> Path:
    return NILM_ROOT / "runs" / experiment_id / model_name


DEFAULT_MODEL = "multinilm_fractional"
DEFAULT_EXPERIMENT = NILM_ROOT / "config" / "experiment_mixed_ukdale_refit_3w.yaml"
DEFAULT_MODEL_CONFIG = NILM_ROOT / "config" / "models" / "multinilm_fractional.yaml"
DEFAULT_CHECKPOINT = None
DEFAULT_SPLIT = "test"
DEFAULT_VIEW_SPAN = 4096
DEFAULT_MAX_BATCHES = None


def on_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    clean = np.asarray(mask).reshape(-1).astype(float)
    clean = np.nan_to_num(clean, nan=0.0)
    clean = (clean >= 0.5).astype(np.int8)
    diff = np.diff(np.concatenate([[0], clean, [0]]))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _split_key(split: str) -> str:
    return "validation" if split in {"val", "validation"} else split


def _load_readable_time(adapter, split: str, csv_timesteps: np.ndarray) -> np.ndarray | None:
    loader = adapter._data_loader()
    try:
        csv_path = loader._resolve_csv_path(_split_key(split))  # noqa: SLF001
        header = pd.read_csv(csv_path, nrows=0)
        if "readable_time" not in header.columns:
            return None
        time_col = pd.read_csv(csv_path, usecols=["readable_time"])
        return time_col.iloc[csv_timesteps]["readable_time"].to_numpy()
    except Exception:
        return None


def _safe_ylim(ax, arrays: list[np.ndarray]) -> None:
    vals = [np.asarray(a, dtype=float).reshape(-1) for a in arrays if len(a)]
    if not vals:
        ax.set_ylim(-1.0, 1.0)
        return
    arr = np.concatenate(vals)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        ax.set_ylim(-1.0, 1.0)
        return
    ymin, ymax = float(np.min(arr)), float(np.max(arr))
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax), 1.0)
    ax.set_ylim(ymin - 0.12 * span, ymax + 0.18 * span)


def choose_checkpoint(default_run_dir: Path) -> Path:
    """Ask user which best.pt to visualize when --checkpoint is omitted."""
    candidates = sorted(NILM_ROOT.glob("runs/**/best.pt"), key=lambda p: str(p).lower())
    if default_run_dir.joinpath("best.pt").exists():
        default_best = default_run_dir / "best.pt"
        candidates = [default_best] + [p for p in candidates if p != default_best]

    if not candidates:
        raw = input("No best.pt found under multi_appliances_NILM/runs. Enter checkpoint path: ").strip().strip('"')
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    print("\nAvailable best.pt checkpoints:")
    for idx, path in enumerate(candidates):
        try:
            rel = path.relative_to(NILM_ROOT)
        except ValueError:
            rel = path
        print(f" [{idx:02d}] {rel}")

    raw = input("\nEnter checkpoint index or full path: ").strip().strip('"')
    if raw.isdigit() and int(raw) < len(candidates):
        return candidates[int(raw)]

    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_prediction_bundle(
    *,
    model_name: str,
    experiment_path: Path,
    model_config_path: Path,
    checkpoint_path: Path | None,
    split: str,
    max_batches: int | None,
):
    experiment = load_experiment(experiment_path)
    model_cfg = load_model_config(model_config_path)
    if model_name_from_config(model_cfg) != model_name:
        raise ValueError(
            f"--model {model_name!r} does not match {model_config_path} "
            f"(model_name={model_name_from_config(model_cfg)!r})"
        )

    merged = merge_configs(experiment, model_cfg)
    data_root = merged.get("data_root")
    if data_root is not None:
        data_root = Path(data_root)
        if not data_root.is_absolute():
            data_root = NILM_ROOT / data_root

    adapter = get_adapter(model_name, merged, data_root=str(data_root) if data_root else None)
    run_dir = _default_run_dir(merged["experiment_id"], model_name)
    checkpoint = checkpoint_path or choose_checkpoint(run_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, tensor_dtype = resolve_tensor_dtype(adapter.model_cfg)
    model = adapter.build_model(device)
    if tensor_dtype == torch.float64:
        model = model.double()

    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()

    loader = adapter.build_dataloader(split)
    bundle = adapter.predict_dataloader(
        model,
        loader,
        device,
        max_batches=max_batches,
        split=split,
    )
    epoch = int(payload.get("epoch", -1)) if isinstance(payload, dict) else -1
    return adapter, bundle, checkpoint, epoch


def interactive_prediction_viewer(
    *,
    adapter,
    bundle,
    checkpoint: Path,
    checkpoint_epoch: int,
    split: str,
    view_span: int,
) -> None:
    loader = adapter._data_loader()
    appliances = list(bundle.appliances)
    n_points = int(len(bundle.y_pred_watts))
    if n_points == 0:
        print("No prediction points to visualize.")
        return

    csv_timesteps = (
        np.asarray(bundle.csv_timesteps, dtype=np.int64).reshape(-1)
        if bundle.csv_timesteps is not None
        else np.arange(n_points, dtype=np.int64)
    )[:n_points]

    aggregate = loader.mains_watts_at_timesteps(_split_key(split), csv_timesteps)
    true_watts = loader.appliance_watts_at_timesteps(_split_key(split), csv_timesteps)
    pred_watts = np.asarray(bundle.y_pred_watts, dtype=float)
    true_on = loader.csv_on_labels_at_timesteps(_split_key(split), csv_timesteps)
    pred_on = (
        np.asarray(bundle.y_pred_on, dtype=np.int32)
        if bundle.y_pred_on is not None
        else (pred_watts > 0).astype(np.int32)
    )
    readable_time = _load_readable_time(adapter, split, csv_timesteps)

    true_segments = {app: on_segments(true_on[:, i]) for i, app in enumerate(appliances)}
    pred_segments = {app: on_segments(pred_on[:, i]) for i, app in enumerate(appliances)}

    n_rows = 1 + len(appliances)
    fig_height = min(13.5, max(8.5, 1.75 * n_rows))
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(16, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15] + [1.0] * len(appliances), "hspace": 0.10},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    plt.subplots_adjust(left=0.075, right=0.84, bottom=0.20, top=0.91)

    palette = {
        "aggregate": "#263238",
        "sum_pred": "#7E57C2",
        "true_line": "#1F77B4",
        "pred_line": "#E67E22",
        "correct_on": "#A8E6B1",
        "wrong_on": "#F6B7B7",
        "status": "#355C7D",
    }

    state = {
        "start": 0,
        "span": min(max(100, int(view_span)), n_points),
        "scale": "raw",
        "show_f1_marks": True,
        "visible": {app: True for app in appliances},
        "lines": [],
        "patches": [],
        "widget_refs": [],
    }

    title = fig.suptitle("", fontsize=12.5, fontweight="bold")
    status = fig.text(
        0.5,
        0.012,
        "",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=palette["status"],
    )

    def visible_slice() -> tuple[int, int, np.ndarray]:
        start = int(state["start"])
        end = min(start + int(state["span"]), n_points)
        return start, end, np.arange(start, end)

    def transform_values(values: np.ndarray, start: int, end: int) -> np.ndarray:
        y = np.asarray(values[start:end], dtype=float)
        if state["scale"] == "raw":
            return y
        if state["scale"] == "zscore":
            std = float(np.std(y))
            return (y - float(np.mean(y))) / (std if std > 0 else 1.0)
        if state["scale"] == "minmax":
            lo, hi = float(np.min(y)), float(np.max(y))
            return (y - lo) / ((hi - lo) if hi > lo else 1.0)
        return y

    def clear_artists() -> None:
        for artist in state["lines"] + state["patches"]:
            artist.remove()
        state["lines"] = []
        state["patches"] = []

    def relayout_axes() -> list[tuple[int, str]]:
        visible_apps = [
            (app_i, app)
            for app_i, app in enumerate(appliances)
            if state["visible"].get(app, True)
        ]
        visible_rows = [(-1, "aggregate"), *visible_apps]
        left, right = 0.075, 0.84
        bottom, top = 0.20, 0.91
        gap = 0.012
        weights = [1.12] + [1.0] * len(visible_apps)
        height_total = top - bottom - gap * max(0, len(visible_rows) - 1)
        weight_total = sum(weights)

        y_top = top
        for row_idx, (item_idx, name) in enumerate(visible_rows):
            ax = axes[0] if name == "aggregate" else axes[item_idx + 1]
            height = height_total * weights[row_idx] / weight_total
            y0 = y_top - height
            ax.set_position([left, y0, right - left, height])
            ax.set_visible(True)
            y_top = y0 - gap

        visible_app_names = {app for _, app in visible_apps}
        for app_i, app in enumerate(appliances):
            if app not in visible_app_names:
                axes[app_i + 1].set_visible(False)
        return visible_rows

    def shade_mask(ax, mask: np.ndarray, start: int, color: str, alpha: float) -> int:
        count = 0
        for s, e in on_segments(mask):
            patch = ax.axvspan(
                start + s,
                start + e,
                color=color,
                alpha=alpha,
                lw=0,
                zorder=0,
            )
            state["patches"].append(patch)
            count += 1
        return count

    def shade_segments(ax, app: str, start: int, end: int) -> tuple[int, int]:
        if not state["show_f1_marks"]:
            return 0, 0
        app_i = appliances.index(app)
        t = true_on[start:end, app_i].astype(bool)
        p = pred_on[start:end, app_i].astype(bool)
        correct_on = t & p
        wrong_on = np.logical_xor(t, p)
        correct_count = shade_mask(ax, correct_on, start, palette["correct_on"], 0.30)
        wrong_count = shade_mask(ax, wrong_on, start, palette["wrong_on"], 0.32)
        return correct_count, wrong_count

    def redraw(_=None) -> None:
        clear_artists()
        visible_rows = relayout_axes()
        start, end, x = visible_slice()
        state["start"] = start
        if readable_time is not None and end > start:
            time_text = f"{readable_time[start]} -> {readable_time[end - 1]}"
        else:
            time_text = f"rows {start:,}->{end:,}"
        title.set_text(
            f"{adapter.name} {split} prediction | best epoch {checkpoint_epoch} | "
            f"{os.path.basename(str(checkpoint))} | {time_text}"
        )

        ax0 = axes[0]
        y_agg = transform_values(aggregate, start, end)
        y_sum = transform_values(pred_watts.sum(axis=1), start, end)
        line = ax0.plot(
            x,
            y_agg,
            color=palette["aggregate"],
            lw=1.8,
            label="aggregate",
        )[0]
        state["lines"].append(line)
        line = ax0.plot(
            x,
            y_sum,
            color=palette["sum_pred"],
            lw=1.35,
            alpha=0.90,
            linestyle="--",
            label="sum predicted",
        )[0]
        state["lines"].append(line)
        ax0.set_ylabel("Aggregate W" if state["scale"] == "raw" else "Aggregate")
        ax0.grid(True, alpha=0.22)
        ax0.legend(loc="upper right", fontsize=8, frameon=False)
        _safe_ylim(ax0, [y_agg, y_sum])

        summaries = []
        for app_i, app in enumerate(appliances):
            ax = axes[app_i + 1]
            shown = bool(state["visible"].get(app, True))
            if not shown:
                legend = ax.get_legend()
                if legend:
                    legend.remove()
                continue
            correct_count, wrong_count = shade_segments(ax, app, start, end)
            summaries.append(f"{app} correct={correct_count} wrong={wrong_count}")
            if shown:
                y_true = transform_values(true_watts[:, app_i], start, end)
                y_pred = transform_values(pred_watts[:, app_i], start, end)
                line = ax.plot(
                    x,
                    y_true,
                    color=palette["true_line"],
                    lw=1.55,
                    alpha=0.92,
                    label="true power",
                )[0]
                state["lines"].append(line)
                line = ax.plot(
                    x,
                    y_pred,
                    color=palette["pred_line"],
                    lw=1.45,
                    alpha=0.96,
                    linestyle="--",
                    label="predicted power",
                )[0]
                state["lines"].append(line)
                _safe_ylim(ax, [y_true, y_pred])
                ax.legend(loc="upper right", fontsize=8, frameon=False)
            ax.set_ylabel(f"{app}\nW" if state["scale"] == "raw" else app, fontsize=9)
            ax.grid(True, axis="x", alpha=0.22)
            ax.grid(True, axis="y", alpha=0.12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for ax in axes:
            ax.set_xlabel("")
        last_axis = axes[0]
        for item_idx, name in visible_rows:
            last_axis = axes[0] if name == "aggregate" else axes[item_idx + 1]
        last_axis.set_xlabel("prediction timeline row")
        for ax in axes:
            ax.set_xlim(start, end)
        status.set_text(
            "F1 ON marks: green=correct ON overlap, red=missed/false ON"
            + ((" | " + " | ".join(summaries)) if summaries else "")
        )
        fig.canvas.draw_idle()

    def sync_from_sliders(_=None) -> None:
        state["start"] = int(pos_slider.val)
        state["span"] = int(span_slider.val)
        redraw()

    def move(delta: int) -> None:
        pos_slider.set_val(min(max(0, state["start"] + delta), max_start))

    def on_check(label: str) -> None:
        state["visible"][label] = not state["visible"].get(label, True)
        redraw()

    def on_scale(label: str) -> None:
        state["scale"] = label
        redraw()

    def toggle_f1_marks(_=None) -> None:
        state["show_f1_marks"] = not state["show_f1_marks"]
        f1_button.label.set_text(f"F1 marks: {'on' if state['show_f1_marks'] else 'off'}")
        redraw()

    def print_stats(_=None) -> None:
        start, end, _ = visible_slice()
        print("\n" + "=" * 100)
        print(f"PREDICTION WINDOW: rows {start:,} to {end:,}")
        if readable_time is not None and end > start:
            print(f"Time: {readable_time[start]} -> {readable_time[end - 1]}")
        print(f"Aggregate mean={np.mean(aggregate[start:end]):.2f} W max={np.max(aggregate[start:end]):.2f} W")
        for app_i, app in enumerate(appliances):
            err = pred_watts[start:end, app_i] - true_watts[start:end, app_i]
            f1_true = true_on[start:end, app_i].astype(bool)
            f1_pred = pred_on[start:end, app_i].astype(bool)
            tp = int(np.logical_and(f1_true, f1_pred).sum())
            fp = int(np.logical_and(~f1_true, f1_pred).sum())
            fn = int(np.logical_and(f1_true, ~f1_pred).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            print(
                f"{app:16s} MAE={np.mean(np.abs(err)):8.2f} W "
                f"true_on={np.mean(f1_true):6.3f} pred_on={np.mean(f1_pred):6.3f} F1={f1:6.3f}"
            )
        print("=" * 100)

    control_y = 0.070
    ax_pos = plt.axes([0.09, control_y + 0.075, 0.46, 0.026])
    max_start = max(0, n_points - 1)
    pos_slider = Slider(ax_pos, "Start", 0, max_start, valinit=0, valstep=1, valfmt="%d")
    ax_span = plt.axes([0.09, control_y + 0.025, 0.46, 0.026])
    span_slider = Slider(
        ax_span,
        "Span",
        50,
        max(50, min(n_points, 100000)),
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
    ax_stats = plt.axes([0.75, control_y + 0.075, 0.07, 0.035])
    ax_f1 = plt.axes([0.59, control_y + 0.025, 0.16, 0.035])
    back_button = Button(ax_back, "Back")
    next_button = Button(ax_next, "Next")
    stats_button = Button(ax_stats, "Stats")
    f1_button = Button(ax_f1, "F1 marks: on")
    back_button.on_clicked(lambda _: move(-state["span"] // 2))
    next_button.on_clicked(lambda _: move(state["span"] // 2))
    stats_button.on_clicked(print_stats)
    f1_button.on_clicked(toggle_f1_marks)

    ax_checks = plt.axes([0.855, 0.46, 0.13, 0.34])
    checks = CheckButtons(ax_checks, appliances, [True] * len(appliances))
    ax_checks.set_title("Appliances", fontsize=9)
    checks.on_clicked(on_check)

    ax_scale = plt.axes([0.855, 0.25, 0.13, 0.14])
    scale_radio = RadioButtons(ax_scale, ["raw", "zscore", "minmax"], active=0)
    ax_scale.set_title("Scale", fontsize=9)
    scale_radio.on_clicked(on_scale)

    fig.legend(
        handles=[
            Patch(facecolor=palette["correct_on"], alpha=0.55, label="correct ON overlap"),
            Patch(facecolor=palette["wrong_on"], alpha=0.58, label="missed / false ON"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.94),
        fontsize=8,
        frameon=False,
    )

    state["widget_refs"] = [
        pos_slider,
        span_slider,
        back_button,
        next_button,
        stats_button,
        f1_button,
        checks,
        scale_radio,
    ]

    print(f"Checkpoint : {checkpoint}")
    print(f"Epoch      : {checkpoint_epoch}")
    print(f"Split      : {split}")
    print(f"Points     : {n_points:,}")
    print(f"Appliances : {', '.join(appliances)}")
    for i, app in enumerate(appliances):
        print(
            f"{app:16s} true_events={len(true_segments[app]):5d} "
            f"pred_events={len(pred_segments[app]):5d} "
            f"MAE={np.mean(np.abs(pred_watts[:, i] - true_watts[:, i])):8.2f} W"
        )

    redraw()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize checkpoint predictions on the NILM test CSV.")
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=["train", "validation", "test"], default=DEFAULT_SPLIT)
    parser.add_argument("--view-span", type=int, default=DEFAULT_VIEW_SPAN)
    parser.add_argument("--max-batches", type=int, default=DEFAULT_MAX_BATCHES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Loading {args.model} checkpoint prediction view: "
        f"experiment={args.experiment}, split={args.split}",
        flush=True,
    )
    adapter, bundle, checkpoint, epoch = load_prediction_bundle(
        model_name=args.model,
        experiment_path=args.experiment,
        model_config_path=args.model_config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        max_batches=args.max_batches,
    )
    interactive_prediction_viewer(
        adapter=adapter,
        bundle=bundle,
        checkpoint=checkpoint,
        checkpoint_epoch=epoch,
        split=args.split,
        view_span=args.view_span,
    )


if __name__ == "__main__":
    main()

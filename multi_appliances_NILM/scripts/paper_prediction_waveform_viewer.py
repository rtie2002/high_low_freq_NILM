#!/usr/bin/env python
"""Interactive paper-style waveform viewer for checkpoint predictions.

Shows aggregate/background power, real appliance power, and predicted appliance
power without ON/OFF shading. The viewer is designed for choosing fair windows
that include both active and inactive regions, then exporting high-DPI figures.

Example:
  python multi_appliances_NILM/scripts/paper_prediction_waveform_viewer.py ^
    --checkpoint multi_appliances_NILM/runs/EXP/MODEL/best.pt ^
    --experiment multi_appliances_NILM/config/experiment_ukdale.yaml ^
    --model-config multi_appliances_NILM/config/models/multinilm_fractional_relational.yaml ^
    --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd


def configure_interactive_backend() -> None:
    for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
        try:
            matplotlib.use(backend, force=True)
            return
        except Exception:
            continue
    print(
        "Warning: no interactive Matplotlib backend found. "
        "Install tkinter/Qt or use this script on a machine with GUI support.",
        flush=True,
    )


configure_interactive_backend()

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
PROJECT_DIR = SCRIPT_DIR.parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.common import PredictionBundle
from adapters.config import load_experiment, load_model_config, merge_configs, model_name_from_config
from adapters.mat_nilm import MATNILMAdapter
from adapters.matuda import MATUDAAdapter
from adapters.multinilm import MultiNILMAdapter
from adapters.multinilm_fractional import MultiNILMFractionalAdapter
from adapters.multinilm_fractional_cascade import MultiNILMFractionalCascadeAdapter
from adapters.multinilm_fractional_residual import MultiNILMFractionalResidualAdapter
from adapters.multinilm_kle import MultiNILMKLEAdapter
from adapters.multinilm_no_distill import MultiNILMNoDistillAdapter
from adapters.transfer_multi_appliance import TransferMultiApplianceAdapter


MODELS = {
    "mat_nilm": MATNILMAdapter,
    "matuda": MATUDAAdapter,
    "multinilm": MultiNILMAdapter,
    "multinilm_fractional_cascade": MultiNILMFractionalCascadeAdapter,
    "multinilm_fractional": MultiNILMFractionalAdapter,
    "multinilm_fractional_residual": MultiNILMFractionalResidualAdapter,
    "multinilm_kle": MultiNILMKLEAdapter,
    "multinilm_no_distill": MultiNILMNoDistillAdapter,
    "transfer_multi_appliance": TransferMultiApplianceAdapter,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Journal-style NILM prediction waveform viewer.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint .pt to evaluate/plot. If omitted, the script asks you to paste one.",
    )
    parser.add_argument("--experiment", type=Path, required=True, help="Experiment dataset YAML.")
    parser.add_argument("--model-config", type=Path, required=True, help="Model YAML used by the checkpoint.")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--data-path", type=Path, default=None, help="Optional override for experiment data_root.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory. Default: checkpoint parent.")
    parser.add_argument("--predictions", type=Path, default=None, help="Existing *_predictions.npz to load.")
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Do not run checkpoint inference if predictions are missing.",
    )
    parser.add_argument("--start", type=int, default=0, help="Initial prediction timeline index.")
    parser.add_argument("--span", type=int, default=1200, help="Initial visible samples.")
    parser.add_argument("--dpi", type=int, default=600, help="Export DPI.")
    parser.add_argument("--fig-width", type=float, default=7.2, help="Export figure width in inches.")
    parser.add_argument("--fig-height", type=float, default=3.2, help="Export figure height in inches.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Export folder. Default: run_dir/paper_waveforms.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (PROJECT_DIR / path).resolve()


def prompt_checkpoint(path: Path | None) -> Path:
    if path is not None and str(path).strip() and "YOUR_CHECKPOINT" not in str(path):
        resolved = resolve_path(path)
        if resolved.is_file():
            return resolved
        print(f"Checkpoint not found: {resolved}", flush=True)

    while True:
        raw = input("Paste checkpoint path (.pt): ").strip().strip('"').strip("'")
        if not raw:
            print("Please paste a checkpoint path.", flush=True)
            continue
        resolved = resolve_path(Path(raw))
        if resolved.is_file():
            return resolved
        print(f"File not found: {resolved}", flush=True)


def default_run_dir(checkpoint: Path) -> Path:
    return checkpoint.resolve().parent


def build_adapter(
    experiment_path: Path,
    model_config_path: Path,
    data_path: Path | None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    experiment = load_experiment(experiment_path)
    model_cfg = load_model_config(model_config_path)
    model_name = model_name_from_config(model_cfg)
    if model_name not in MODELS:
        raise ValueError(f"Unsupported model_name={model_name}. Available: {sorted(MODELS)}")

    merged = merge_configs(experiment, model_cfg)
    data_root = data_path or merged.get("data_root")
    if data_root is not None:
        data_root = Path(data_root)
        if not data_root.is_absolute():
            data_root = ROOT / data_root
    adapter = MODELS[model_name](merged, data_root=str(data_root) if data_root else None)
    return adapter, experiment, model_cfg


def prediction_path(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.predictions:
        return resolve_path(args.predictions)
    return run_dir / f"{args.split}_predictions.npz"


def load_or_create_bundle(
    args: argparse.Namespace,
    adapter: Any,
    run_dir: Path,
    checkpoint: Path,
) -> PredictionBundle:
    pred_path = prediction_path(args, run_dir)
    if pred_path.is_file():
        print(f"Loading predictions: {pred_path}", flush=True)
        return PredictionBundle.load(pred_path)
    if args.no_evaluate:
        raise FileNotFoundError(f"Missing predictions and --no-evaluate is set: {pred_path}")
    print(f"Predictions not found. Running checkpoint inference for split={args.split}...", flush=True)
    from runner import evaluate_model

    pred_path = evaluate_model(
        adapter,
        checkpoint=checkpoint,
        run_dir=run_dir,
        split=args.split,
        show_cost_summary=False,
    )
    return PredictionBundle.load(pred_path)


def load_time_axis(adapter: Any, bundle: PredictionBundle, split: str) -> np.ndarray | None:
    csv_timesteps = bundle.csv_timesteps
    if csv_timesteps is None:
        return None
    loader = adapter._data_loader()
    csv_path = loader._resolve_csv_path(split)
    if not csv_path.is_file():
        return None
    try:
        time_col = pd.read_csv(csv_path, usecols=["readable_time"])
    except Exception:
        return None
    times = pd.to_datetime(time_col["readable_time"], errors="coerce")
    indices = np.asarray(csv_timesteps, dtype=np.int64).reshape(-1)
    if indices.size == 0 or int(indices.max()) >= len(times):
        return None
    return times.iloc[indices].to_numpy()


def bundle_aggregate_watts_local(
    data_loader: Any,
    split: str,
    *,
    n_points: int,
    csv_timesteps: np.ndarray | None,
) -> np.ndarray | None:
    if csv_timesteps is None:
        return None
    ts = np.asarray(csv_timesteps, dtype=np.int64).reshape(-1)
    if len(ts) < int(n_points):
        return None
    try:
        return data_loader.mains_watts_at_timesteps(split, ts[: int(n_points)])
    except Exception:
        return None


def bundle_csv_appliance_watts_local(
    data_loader: Any,
    split: str,
    *,
    n_points: int,
    csv_timesteps: np.ndarray | None,
) -> np.ndarray | None:
    if csv_timesteps is None:
        return None
    ts = np.asarray(csv_timesteps, dtype=np.int64).reshape(-1)
    if len(ts) < int(n_points):
        return None
    try:
        return data_loader.appliance_watts_at_timesteps(split, ts[: int(n_points)])
    except Exception:
        return None


def aligned_arrays(adapter: Any, bundle: PredictionBundle, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    n = len(bundle.y_pred_watts)
    aggregate = bundle_aggregate_watts_local(
        adapter._data_loader(),
        split,
        n_points=n,
        csv_timesteps=bundle.csv_timesteps,
    )
    true_watts = bundle_csv_appliance_watts_local(
        adapter._data_loader(),
        split,
        n_points=n,
        csv_timesteps=bundle.csv_timesteps,
    )
    if true_watts is None:
        true_watts = bundle.y_true_watts
    return (
        np.asarray(aggregate, dtype=float) if aggregate is not None else np.zeros(n, dtype=float),
        np.asarray(true_watts, dtype=float),
        np.maximum(np.asarray(bundle.y_pred_watts, dtype=float), 0.0),
        load_time_axis(adapter, bundle, split),
    )


def style_axes(ax: plt.Axes, ax_bg: plt.Axes | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.75)
    ax.tick_params(axis="both", labelsize=9)
    if ax_bg is not None:
        ax_bg.spines["top"].set_visible(False)
        ax_bg.spines["left"].set_visible(False)
        ax_bg.tick_params(axis="y", labelsize=8, colors="#6f6f6f")


def find_balanced_windows(true_power: np.ndarray, span: int, limit: int = 20) -> list[int]:
    """Suggest starts containing both active and inactive samples."""
    y = np.asarray(true_power, dtype=float).reshape(-1)
    n = len(y)
    if n <= span:
        return [0]
    active = y > max(5.0, 0.05 * float(np.nanmax(y) if np.nanmax(y) > 0 else 1.0))
    candidates: list[tuple[float, int]] = []
    step = max(1, span // 4)
    for start in range(0, n - span + 1, step):
        view = active[start : start + span]
        frac = float(view.mean())
        if 0.02 <= frac <= 0.80:
            score = abs(frac - 0.25)
            candidates.append((score, start))
    candidates.sort(key=lambda item: item[0])
    starts = [start for _, start in candidates[:limit]]
    return starts or list(range(0, min(n - span + 1, step * limit), step))


def draw_waveform(
    *,
    output_path: Path | None,
    appliances: list[str],
    app_idx: int,
    start: int,
    span: int,
    aggregate: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_axis: np.ndarray | None,
    dpi: int,
    fig_width: float,
    fig_height: float,
    title_prefix: str,
    show: bool,
) -> tuple[plt.Figure, plt.Axes]:
    n = len(y_pred)
    start = max(0, min(int(start), max(0, n - 1)))
    end = max(start + 1, min(n, start + int(span)))
    sl = slice(start, end)

    if time_axis is not None:
        x = time_axis[sl]
        xlabel = "Time"
    elif len(aggregate) == n:
        x = np.arange(start, end)
        xlabel = "Prediction timeline index"
    else:
        x = np.arange(start, end)
        xlabel = "Prediction timeline index"

    app = appliances[app_idx]
    real = y_true[sl, app_idx]
    pred = y_pred[sl, app_idx]
    bg = aggregate[sl] if len(aggregate) >= end else None

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax_bg = ax.twinx() if bg is not None else None

    if bg is not None and ax_bg is not None:
        ax_bg.plot(
            x,
            bg,
            color="#7f7f7f",
            linewidth=1.0,
            alpha=0.42,
            linestyle="-",
            label="Aggregate/background power",
            zorder=1,
        )
        ax_bg.set_ylabel("Aggregate power (W)", fontsize=9, color="#6f6f6f")

    ax.plot(x, real, color="#1f4e79", linewidth=1.7, label="Real appliance power", zorder=3)
    ax.plot(x, pred, color="#b22222", linewidth=1.55, linestyle="--", label="Predicted appliance power", zorder=4)
    ax.set_ylabel(f"{app} power (W)", fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(f"{title_prefix}{app}: real vs predicted power", fontsize=11, pad=8)
    style_axes(ax, ax_bg)

    ymax = max(1.0, float(np.nanmax(real)), float(np.nanmax(pred)))
    ymin = min(0.0, float(np.nanmin(real)), float(np.nanmin(pred)))
    pad = max(1.0, 0.12 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)

    lines, labels = ax.get_legend_handles_labels()
    if ax_bg is not None:
        bg_lines, bg_labels = ax_bg.get_legend_handles_labels()
        lines += bg_lines
        labels += bg_labels
    ax.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(pad=0.8)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {output_path}", flush=True)
    if not show:
        plt.close(fig)
    return fig, ax


def interactive_viewer(
    *,
    bundle: PredictionBundle,
    aggregate: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_axis: np.ndarray | None,
    out_dir: Path,
    initial_start: int,
    initial_span: int,
    dpi: int,
    fig_width: float,
    fig_height: float,
) -> None:
    appliances = bundle.appliances
    n = len(y_pred)
    state = {
        "app_idx": 0,
        "start": max(0, min(initial_start, max(0, n - 1))),
        "span": max(10, min(initial_span, n)),
        "suggestions": find_balanced_windows(y_true[:, 0], max(10, min(initial_span, n))),
        "suggest_idx": 0,
    }

    fig = plt.figure(figsize=(10.8, 6.2))
    ax = fig.add_axes([0.08, 0.24, 0.68, 0.62])
    ax_bg = ax.twinx()

    ax_app = fig.add_axes([0.80, 0.45, 0.17, 0.38])
    radios = RadioButtons(ax_app, appliances, active=0)

    ax_start = fig.add_axes([0.08, 0.145, 0.68, 0.035])
    ax_span = fig.add_axes([0.08, 0.095, 0.68, 0.035])
    start_slider = Slider(ax_start, "Start", 0, max(0, n - 1), valinit=state["start"], valstep=1)
    span_slider = Slider(ax_span, "Span", 10, max(10, n), valinit=state["span"], valstep=1)
    start_slider.valtext.set_visible(False)
    span_slider.valtext.set_visible(False)

    ax_start_box = fig.add_axes([0.80, 0.35, 0.075, 0.04])
    ax_span_box = fig.add_axes([0.895, 0.35, 0.075, 0.04])
    start_box = TextBox(ax_start_box, "Start", initial=str(state["start"]))
    span_box = TextBox(ax_span_box, "Span", initial=str(state["span"]))

    ax_prev = fig.add_axes([0.80, 0.26, 0.075, 0.045])
    ax_next = fig.add_axes([0.895, 0.26, 0.075, 0.045])
    ax_suggest = fig.add_axes([0.80, 0.19, 0.17, 0.045])
    ax_save = fig.add_axes([0.80, 0.11, 0.17, 0.052])
    prev_btn = Button(ax_prev, "Back")
    next_btn = Button(ax_next, "Next")
    suggest_btn = Button(ax_suggest, "Fair window")
    save_btn = Button(ax_save, "Save high-DPI")

    def clear_axes() -> None:
        ax.clear()
        ax_bg.clear()

    def redraw(_=None) -> None:
        state["start"] = int(start_slider.val)
        state["span"] = int(span_slider.val)
        start_box.set_val(str(state["start"]))
        span_box.set_val(str(state["span"]))
        clear_axes()
        start = state["start"]
        span = state["span"]
        end = max(start + 1, min(n, start + span))
        sl = slice(start, end)
        app_idx = state["app_idx"]
        app = appliances[app_idx]
        x = time_axis[sl] if time_axis is not None else np.arange(start, end)
        real = y_true[sl, app_idx]
        pred = y_pred[sl, app_idx]
        bg = aggregate[sl] if len(aggregate) >= end else None

        if bg is not None:
            ax_bg.plot(x, bg, color="#7f7f7f", linewidth=1.0, alpha=0.42, label="Aggregate/background power")
            ax_bg.set_ylabel("Aggregate power (W)", fontsize=9, color="#6f6f6f")
        ax.plot(x, real, color="#1f4e79", linewidth=1.7, label="Real appliance power")
        ax.plot(x, pred, color="#b22222", linewidth=1.55, linestyle="--", label="Predicted appliance power")
        ax.set_title(f"{bundle.model_name} {bundle.split} | {app} | samples {start}:{end}", fontsize=11)
        ax.set_ylabel(f"{app} power (W)", fontsize=10)
        ax.set_xlabel("Time" if time_axis is not None else "Prediction timeline index", fontsize=10)
        style_axes(ax, ax_bg)
        ymax = max(1.0, float(np.nanmax(real)), float(np.nanmax(pred)))
        ymin = min(0.0, float(np.nanmin(real)), float(np.nanmin(pred)))
        pad = max(1.0, 0.12 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)
        lines, labels = ax.get_legend_handles_labels()
        bg_lines, bg_labels = ax_bg.get_legend_handles_labels()
        ax.legend(lines + bg_lines, labels + bg_labels, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=9)
        fig.canvas.draw_idle()

    def set_app(label: str) -> None:
        state["app_idx"] = appliances.index(label)
        state["suggestions"] = find_balanced_windows(y_true[:, state["app_idx"]], int(span_slider.val))
        state["suggest_idx"] = 0
        redraw()

    def submit_start(text: str) -> None:
        try:
            start_slider.set_val(max(0, min(int(float(text)), max(0, n - 1))))
        except ValueError:
            redraw()

    def submit_span(text: str) -> None:
        try:
            span_slider.set_val(max(10, min(int(float(text)), n)))
        except ValueError:
            redraw()

    def move(delta: int) -> None:
        start_slider.set_val(max(0, min(max(0, n - 1), int(start_slider.val) + delta)))

    def fair_window(_=None) -> None:
        suggestions = state["suggestions"]
        if not suggestions:
            suggestions = find_balanced_windows(y_true[:, state["app_idx"]], int(span_slider.val))
            state["suggestions"] = suggestions
        idx = state["suggest_idx"] % len(suggestions)
        state["suggest_idx"] += 1
        start_slider.set_val(int(suggestions[idx]))

    def save_current(_=None) -> None:
        app = appliances[state["app_idx"]]
        path = out_dir / f"{bundle.split}_{app}_{state['start']:07d}_{state['start'] + state['span']:07d}.png"
        draw_waveform(
            output_path=path,
            appliances=appliances,
            app_idx=state["app_idx"],
            start=state["start"],
            span=state["span"],
            aggregate=aggregate,
            y_true=y_true,
            y_pred=y_pred,
            time_axis=time_axis,
            dpi=dpi,
            fig_width=fig_width,
            fig_height=fig_height,
            title_prefix=f"{bundle.model_name} {bundle.split} | ",
            show=False,
        )

    radios.on_clicked(set_app)
    start_slider.on_changed(redraw)
    span_slider.on_changed(redraw)
    start_box.on_submit(submit_start)
    span_box.on_submit(submit_span)
    prev_btn.on_clicked(lambda _: move(-int(span_slider.val // 2)))
    next_btn.on_clicked(lambda _: move(int(span_slider.val // 2)))
    suggest_btn.on_clicked(fair_window)
    save_btn.on_clicked(save_current)

    state["widget_refs"] = [
        radios,
        start_slider,
        span_slider,
        start_box,
        span_box,
        prev_btn,
        next_btn,
        suggest_btn,
        save_btn,
    ]
    redraw()
    plt.show()


def main() -> None:
    args = parse_args()
    checkpoint = prompt_checkpoint(args.checkpoint)
    experiment = resolve_path(args.experiment)
    model_config = resolve_path(args.model_config)
    data_path = resolve_path(args.data_path) if args.data_path else None
    run_dir = resolve_path(args.run_dir) if args.run_dir else default_run_dir(checkpoint)
    out_dir = resolve_path(args.out_dir) if args.out_dir else run_dir / "paper_waveforms"

    adapter, _, _ = build_adapter(experiment, model_config, data_path)
    bundle = load_or_create_bundle(args, adapter, run_dir, checkpoint)
    aggregate, y_true, y_pred, time_axis = aligned_arrays(adapter, bundle, args.split)

    print("Viewer ready.", flush=True)
    print(f"checkpoint : {checkpoint}", flush=True)
    print(f"split      : {args.split}", flush=True)
    print(f"samples    : {len(y_pred):,}", flush=True)
    print(f"appliances : {', '.join(bundle.appliances)}", flush=True)
    print(f"exports    : {out_dir}", flush=True)

    interactive_viewer(
        bundle=bundle,
        aggregate=aggregate,
        y_true=y_true,
        y_pred=y_pred,
        time_axis=time_axis,
        out_dir=out_dir,
        initial_start=args.start,
        initial_span=args.span,
        dpi=args.dpi,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
    )


if __name__ == "__main__":
    main()

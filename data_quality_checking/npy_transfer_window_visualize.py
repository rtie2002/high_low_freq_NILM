"""
Visualize transfer_learning_multi-appliance .npy window datasets.

The downloaded author data is already windowed:
    shape = (num_windows, 480, 6)

Channel order used by the author code:
    0 aggregate
    1 kettle
    2 microwave
    3 fridge
    4 dishwasher
    5 washingmachine

The .npy files store normalized power only. They do not store explicit
ON/OFF labels. This viewer reconstructs ON/OFF labels with the same
threshold idea used by the author pipeline:
    normalized_power > (on_threshold_watts - mean) / std

Example:
    python data_quality_checking/npy_transfer_window_visualize.py

    python data_quality_checking/npy_transfer_window_visualize.py ^
      --file "NILM_model/baseline/transfer_learning_multi-appliance/dataset_management/redd/total/test_set.npy" ^
      --dataset redd
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPY = (
    PROJECT_ROOT
    / "NILM_model"
    / "baseline"
    / "transfer_learning_multi-appliance"
    / "dataset_management"
    / "redd"
    / "total"
    / "test_set.npy"
)
DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "NILM_model"
    / "baseline"
    / "transfer_learning_multi-appliance"
    / "dataset_management"
)

CHANNELS = ["aggregate", "kettle", "microwave", "fridge", "dishwasher", "washingmachine"]
APPLIANCES = CHANNELS[1:]


@dataclass(frozen=True)
class ChannelParam:
    mean: float
    std: float
    threshold: float | None = None
    note: str = ""


PARAMS: dict[str, dict[str, ChannelParam]] = {
    "redd": {
        "aggregate": ChannelParam(mean=300.0, std=550.0),
        "kettle": ChannelParam(mean=0.0, std=1.0, threshold=0.0, note="dummy/skipped in REDD paper tables"),
        "microwave": ChannelParam(mean=150.0, std=500.0, threshold=50.0),
        "fridge": ChannelParam(mean=70.0, std=100.0, threshold=50.0),
        "dishwasher": ChannelParam(mean=400.0, std=450.0, threshold=50.0),
        "washingmachine": ChannelParam(mean=1000.0, std=1500.0, threshold=20.0),
    },
    "ukdale": {
        "aggregate": ChannelParam(mean=400.0, std=500.0),
        "kettle": ChannelParam(mean=100.0, std=500.0, threshold=40.0),
        "microwave": ChannelParam(mean=60.0, std=300.0, threshold=100.0),
        "fridge": ChannelParam(mean=50.0, std=50.0, threshold=50.0),
        "dishwasher": ChannelParam(mean=700.0, std=1000.0, threshold=30.0),
        "washingmachine": ChannelParam(mean=400.0, std=700.0, threshold=30.0),
    },
    "refit": {
        "aggregate": ChannelParam(mean=500.0, std=800.0),
        "kettle": ChannelParam(mean=50.0, std=80.0, threshold=20.0),
        "microwave": ChannelParam(mean=500.0, std=800.0, threshold=500.0),
        "fridge": ChannelParam(mean=350.0, std=700.0, threshold=19.0),
        "dishwasher": ChannelParam(mean=100.0, std=400.0, threshold=20.0),
        "washingmachine": ChannelParam(mean=100.0, std=500.0, threshold=20.0),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect transfer-learning NILM .npy windows.")
    parser.add_argument("--file", type=Path, default=None, help="Path to train/val/test .npy file.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder searched when --file is not provided.",
    )
    parser.add_argument("--dataset", choices=["auto", "redd", "ukdale", "refit"], default="auto")
    parser.add_argument("--window", type=int, default=0, help="Initial stored window index, used if --start is not set.")
    parser.add_argument("--start", type=int, default=None, help="Initial virtual sample index after concatenating windows.")
    parser.add_argument("--span", type=int, default=2400, help="Number of virtual timesteps shown at once.")
    parser.add_argument("--save", type=Path, default=None, help="Save current window figure to PNG and exit.")
    parser.add_argument("--summary-only", action="store_true", help="Print shape/channel summary only.")
    parser.add_argument(
        "--scale",
        choices=["raw", "normalized", "zscore", "minmax"],
        default="raw",
        help="Initial plotting scale. raw means denormalized watts.",
    )
    return parser.parse_args()


def find_npy_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(path for path in data_dir.rglob("*.npy") if path.is_file())


def choose_npy_file(data_dir: Path) -> Path:
    files = find_npy_files(data_dir)
    print(f"\nAvailable .npy files in: {data_dir}")
    if not files:
        print("  No .npy files found.")
        print(f"  Press Enter to use default: {DEFAULT_NPY}")
    else:
        for idx, path in enumerate(files):
            try:
                shown = path.relative_to(PROJECT_ROOT)
            except ValueError:
                shown = path
            print(f" [{idx:02d}] {shown}")
        print(f"\nPress Enter to use default: {DEFAULT_NPY}")

    raw = input("Enter index or full .npy path: ").strip().strip('"')
    if not raw:
        return DEFAULT_NPY
    candidate = Path(raw)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute() and (PROJECT_ROOT / candidate).exists():
        return PROJECT_ROOT / candidate
    if raw.isdigit() and int(raw) < len(files):
        return files[int(raw)]
    raise FileNotFoundError(f"Could not find selected .npy file: {raw}")


def infer_dataset(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    lowered = str(path).lower()
    for name in ("redd", "ukdale", "refit"):
        if name in lowered:
            return name
    return "redd"


def open_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing .npy file: {path}")
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3 or arr.shape[2] != 6:
        raise ValueError(f"Expected shape (num_windows, 480, 6), got {arr.shape}")
    return arr


def denormalize(values: np.ndarray, name: str, dataset: str) -> np.ndarray:
    param = PARAMS[dataset][name]
    return values * param.std + param.mean


def scale_series(values: np.ndarray, mode: str) -> np.ndarray:
    if mode in ("raw", "normalized"):
        return values
    if mode == "zscore":
        std = float(np.nanstd(values))
        return (values - float(np.nanmean(values))) / (std if std > 0 else 1.0)
    if mode == "minmax":
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        span = vmax - vmin
        return (values - vmin) / (span if span > 0 else 1.0)
    raise ValueError(f"Unknown scale mode: {mode}")


def normalized_threshold(name: str, dataset: str) -> float:
    param = PARAMS[dataset][name]
    if param.threshold is None:
        raise ValueError(f"{name} has no ON threshold")
    return (param.threshold - param.mean) / param.std


def on_status_from_normalized(values: np.ndarray, name: str, dataset: str) -> np.ndarray:
    return values > normalized_threshold(name, dataset)


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, active in enumerate(mask.astype(bool)):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def summarize(arr: np.ndarray, path: Path, dataset: str) -> None:
    print("\nTransfer-learning .npy dataset summary")
    print("=" * 78)
    print(f"file      : {path}")
    print(f"dataset   : {dataset}")
    print(f"shape     : {arr.shape}  (windows, samples, channels)")
    print(f"dtype     : {arr.dtype}")
    print(f"channels  : {', '.join(CHANNELS)}")
    print("\nImportant: this .npy stores normalized power windows, not explicit labels.")
    print("ON/OFF labels below are reconstructed from author threshold parameters.\n")
    if arr.shape[0] > 1:
        agg_boundary_jump = np.abs(arr[1:, 0, 0] - arr[:-1, -1, 0])
        agg_std = PARAMS[dataset]["aggregate"].std
        print("Stored-window order check")
        print(
            f"  aggregate boundary jump: median={np.median(agg_boundary_jump) * agg_std:.2f} W, "
            f"p90={np.percentile(agg_boundary_jump, 90) * agg_std:.2f} W, "
            f"max={np.max(agg_boundary_jump) * agg_std:.2f} W"
        )
        print("  Large jumps mean adjacent stored windows are not chronological neighbors.\n")

    header = (
        f"{'channel':<16} {'mean':>9} {'std':>9} {'thr_W':>9} "
        f"{'thr_norm':>10} {'active_win':>11} {'on_%':>9} {'min_W':>10} {'max_W':>10}  note"
    )
    print(header)
    print("-" * len(header))

    for idx, name in enumerate(CHANNELS):
        param = PARAMS[dataset][name]
        raw = np.asarray(arr[:, :, idx])
        watts = denormalize(raw, name, dataset)
        if name == "aggregate":
            print(
                f"{name:<16} {param.mean:9.2f} {param.std:9.2f} {'-':>9} "
                f"{'-':>10} {'-':>11} {'-':>9} {watts.min():10.2f} {watts.max():10.2f}"
            )
            continue

        status = on_status_from_normalized(raw, name, dataset)
        active_windows = int(np.any(status, axis=1).sum())
        on_percent = float(status.mean() * 100.0)
        print(
            f"{name:<16} {param.mean:9.2f} {param.std:9.2f} {param.threshold:9.2f} "
            f"{normalized_threshold(name, dataset):10.4f} {active_windows:11d} "
            f"{on_percent:8.3f}% {watts.min():10.2f} {watts.max():10.2f}  {param.note}"
        )
    print()


class NpyWindowViewer:
    def __init__(
        self,
        arr: np.ndarray,
        path: Path,
        dataset: str,
        initial_window: int,
        scale: str,
        initial_start: int | None = None,
        initial_span: int = 2400,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

        self.arr = arr
        self.path = path
        self.dataset = dataset
        self.scale = scale
        self.window_len = int(arr.shape[1])
        self.virtual_len = int(arr.shape[0] * arr.shape[1])
        self.start = int(initial_start if initial_start is not None else initial_window * self.window_len)
        self.start = int(np.clip(self.start, 0, max(0, self.virtual_len - 1)))
        self.span = int(np.clip(initial_span, 50, max(50, self.virtual_len)))
        self.show_aggregate = True
        self.show_on = True
        self.visible = {name: True for name in CHANNELS}
        self.plt = plt

        self.fig, self.axes = plt.subplots(
            len(CHANNELS),
            1,
            figsize=(15.5, 10.5),
            sharex=True,
            constrained_layout=False,
        )
        self.fig.subplots_adjust(left=0.075, right=0.82, top=0.90, bottom=0.18, hspace=0.16)

        start_ax = self.fig.add_axes([0.09, 0.115, 0.52, 0.026])
        self.start_slider = Slider(
            start_ax,
            "Start",
            0,
            max(0, self.virtual_len - 1),
            valinit=self.start,
            valstep=1,
            valfmt="%0.0f",
        )
        self.start_slider.on_changed(self._on_start_slider)
        self.start_slider.valtext.set_visible(False)

        span_ax = self.fig.add_axes([0.09, 0.065, 0.52, 0.026])
        self.span_slider = Slider(
            span_ax,
            "Span",
            50,
            max(50, min(self.virtual_len, 50000)),
            valinit=min(self.span, max(50, min(self.virtual_len, 50000))),
            valstep=50,
            valfmt="%0.0f",
        )
        self.span_slider.on_changed(self._on_span_slider)
        self.span_slider.valtext.set_visible(False)

        prev_ax = self.fig.add_axes([0.64, 0.105, 0.07, 0.035])
        next_ax = self.fig.add_axes([0.72, 0.105, 0.07, 0.035])
        stats_ax = self.fig.add_axes([0.64, 0.055, 0.07, 0.035])
        on_ax = self.fig.add_axes([0.72, 0.055, 0.10, 0.035])
        self.prev_button = Button(prev_ax, "Back")
        self.next_button = Button(next_ax, "Next")
        self.stats_button = Button(stats_ax, "Stats")
        self.on_button = Button(on_ax, "ON shade: on")
        self.prev_button.on_clicked(lambda _event: self.move(-max(1, self.span // 2)))
        self.next_button.on_clicked(lambda _event: self.move(max(1, self.span // 2)))
        self.stats_button.on_clicked(lambda _event: self.print_stats())
        self.on_button.on_clicked(lambda _event: self.toggle_on())

        check_ax = self.fig.add_axes([0.845, 0.50, 0.14, 0.34])
        self.check = CheckButtons(check_ax, CHANNELS, [self.visible[name] for name in CHANNELS])
        check_ax.set_title("Show / hide", fontsize=9)
        self.check.on_clicked(self._on_check)

        scale_ax = self.fig.add_axes([0.845, 0.27, 0.14, 0.16])
        scale_labels = ["raw", "normalized", "zscore", "minmax"]
        self.scale_radio = RadioButtons(scale_ax, scale_labels, active=scale_labels.index(self.scale))
        scale_ax.set_title("Scale", fontsize=9)
        self.scale_radio.on_clicked(self._on_scale)

        self.status = self.fig.text(0.5, 0.018, "", ha="center", va="bottom", fontsize=9.5, color="#146c2e")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.update()

    def set_start(self, value: int) -> None:
        self.start = int(np.clip(value, 0, max(0, self.virtual_len - 1)))
        self.start_slider.set_val(self.start)

    def move(self, delta: int) -> None:
        self.set_start(self.start + delta)

    def _on_start_slider(self, value: float) -> None:
        self.start = int(value)
        self.update()

    def _on_span_slider(self, value: float) -> None:
        self.span = int(value)
        self.update()

    def _on_check(self, _label: str) -> None:
        states = self.check.get_status()
        for name, visible in zip(CHANNELS, states):
            self.visible[name] = bool(visible)
        self.show_aggregate = self.visible.get("aggregate", True)
        self.update()

    def _on_scale(self, label: str) -> None:
        self.scale = label
        self.update()

    def toggle_on(self) -> None:
        self.show_on = not self.show_on
        self.on_button.label.set_text(f"ON shade: {'on' if self.show_on else 'off'}")
        self.update()

    def _on_key(self, event) -> None:
        if event.key == "right":
            self.move(max(1, self.span // 10))
        elif event.key == "left":
            self.move(-max(1, self.span // 10))
        elif event.key == "pagedown":
            self.move(max(1, self.span))
        elif event.key == "pageup":
            self.move(-max(1, self.span))
        elif event.key == "home":
            self.set_start(0)
        elif event.key == "end":
            self.set_start(max(0, self.virtual_len - self.span))

    def visible_slice(self) -> tuple[int, int, np.ndarray, np.ndarray]:
        start = int(np.clip(self.start, 0, max(0, self.virtual_len - 1)))
        end = min(start + int(self.span), self.virtual_len)
        first_window = start // self.window_len
        last_window = max(first_window, (end - 1) // self.window_len)
        offset = start - first_window * self.window_len
        needed = end - start
        block = np.asarray(self.arr[first_window : last_window + 1]).reshape(-1, self.arr.shape[2])
        visible = block[offset : offset + needed]
        x = np.arange(start, end)
        return start, end, x, visible

    def _series(self, window: np.ndarray, channel_idx: int, name: str) -> np.ndarray:
        values = np.asarray(window[:, channel_idx], dtype=float)
        if self.scale == "normalized":
            base = values
        else:
            base = denormalize(values, name, self.dataset)
        return scale_series(base, self.scale)

    def _threshold_for_plot(self, name: str) -> float | None:
        if self.scale == "normalized":
            return normalized_threshold(name, self.dataset)
        if self.scale == "raw":
            return PARAMS[self.dataset][name].threshold
        return None

    def print_stats(self) -> None:
        start, end, _, window = self.visible_slice()
        print("\n" + "=" * 88)
        print(f"NPY VISIBLE STATISTICS: virtual samples {start:,} to {end:,}")
        print(f"Source windows covered: {start // self.window_len:,} to {(end - 1) // self.window_len:,}")
        for idx, name in enumerate(CHANNELS):
            raw = np.asarray(window[:, idx], dtype=float)
            watts = denormalize(raw, name, self.dataset)
            if name == "aggregate":
                print(f"{name:16s} mean={np.mean(watts):10.3f} min={np.min(watts):10.3f} max={np.max(watts):10.3f}")
                continue
            status = on_status_from_normalized(raw, name, self.dataset)
            print(
                f"{name:16s} mean={np.mean(watts):10.3f} min={np.min(watts):10.3f} "
                f"max={np.max(watts):10.3f} on_ratio={np.mean(status):8.4f} "
                f"on_samples={int(status.sum()):4d}/{len(status)}"
            )
        print("=" * 88)

    def update(self) -> None:
        start_sample, end_sample, x, window = self.visible_slice()
        first_window = start_sample // self.window_len
        last_window = max(first_window, (end_sample - 1) // self.window_len)
        aggregate = self._series(window, 0, "aggregate")

        title_unit = {
            "raw": "denormalized watts",
            "normalized": "stored normalized values",
            "zscore": "window z-score",
            "minmax": "window min-max",
        }[self.scale]
        self.fig.suptitle(
            f"{self.path.name} | {self.dataset.upper()} | shape={self.arr.shape} | "
            f"samples {start_sample:,}->{end_sample:,} | source windows {first_window:,}->{last_window:,} | {title_unit}\n"
            "Viewing stored-window order only; author preprocessing may shuffle windows, so this is not guaranteed chronological.",
            fontsize=12,
        )

        for ax in self.axes:
            ax.clear()
            ax.set_xlim(start_sample, end_sample)

        agg_ax = self.axes[0]
        if self.visible.get("aggregate", True):
            agg_ax.plot(x, aggregate, color="#333333", linewidth=1.8, label="aggregate")
            agg_ax.legend(loc="upper right", fontsize=8, frameon=False)
        agg_ax.set_ylabel("Aggregate W" if self.scale == "raw" else "aggregate")
        agg_ax.grid(True, alpha=0.25)
        agg_ax.spines["top"].set_visible(False)
        agg_ax.spines["right"].set_visible(False)

        on_summary = []
        for plot_idx, name in enumerate(APPLIANCES, start=1):
            ax = self.axes[plot_idx]
            channel_idx = CHANNELS.index(name)
            raw = np.asarray(window[:, channel_idx], dtype=float)
            power = self._series(window, channel_idx, name)
            status = on_status_from_normalized(raw, name, self.dataset)
            on_summary.append(f"{name}:{len(contiguous_runs(status))} events/{int(status.sum())} samples")

            if self.show_on and self.visible.get(name, True):
                for run_start, run_end in contiguous_runs(status):
                    ax.axvspan(
                        start_sample + run_start,
                        start_sample + run_end - 1,
                        color="#8fd19e",
                        alpha=0.28,
                        linewidth=0,
                        zorder=0,
                    )

            if self.show_aggregate and self.visible.get("aggregate", True) and self.visible.get(name, True):
                agg_scaled = aggregate
                if self.scale == "raw":
                    app_max = max(float(np.nanpercentile(power, 99.5)), PARAMS[self.dataset][name].threshold or 0.0, 1.0)
                    agg_max = max(float(np.nanpercentile(np.abs(aggregate), 99.5)), 1.0)
                    agg_scaled = aggregate / agg_max * app_max
                ax.plot(x, agg_scaled, color="#c0c0c0", linewidth=1.0, alpha=0.65, label="aggregate scaled")

            if self.visible.get(name, True):
                ax.plot(x, power, color="#1f77b4", linewidth=1.65, label=f"{name} power")

            param = PARAMS[self.dataset][name]
            thr = self._threshold_for_plot(name)
            if thr is not None and self.visible.get(name, True):
                ax.axhline(thr, color="#2ca02c", linewidth=1.0, linestyle="--", alpha=0.75, label="ON threshold")

            note = f" ({param.note})" if param.note else ""
            ax.set_ylabel(f"{name}\nW" if self.scale == "raw" else name.replace("washingmachine", "washing\nmachine"))
            ax.set_title(f"{name}{note} | ON samples in view: {int(status.sum())}/{len(status)}", loc="left", fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if self.visible.get(name, True):
                ax.legend(loc="upper right", fontsize=8, frameon=False)

            finite = power[np.isfinite(power)]
            if finite.size and self.visible.get(name, True):
                low = min(float(np.nanmin(finite)), float(thr) if thr is not None else 0.0)
                high = max(float(np.nanpercentile(finite, 99.5)), float(thr) if thr is not None else 0.0, 1.0)
                pad = max((high - low) * 0.15, 1.0)
                ax.set_ylim(low - pad, high + pad)
            else:
                ax.set_ylim(-1, 1)

        self.axes[-1].set_xlabel("virtual sample index after concatenating stored windows")
        self.status.set_text("visible ON in current view: " + " | ".join(on_summary))
        self.fig.canvas.draw_idle()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.update()
        self.fig.savefig(path, dpi=180, bbox_inches="tight")

    def show(self) -> None:
        self.plt.show()


def main() -> None:
    args = parse_args()
    if args.save:
        import matplotlib

        matplotlib.use("Agg")

    path = (args.file if args.file is not None else choose_npy_file(args.data_dir.resolve())).resolve()
    dataset = infer_dataset(path, args.dataset)
    arr = open_array(path)

    summarize(arr, path, dataset)
    if args.summary_only:
        return

    viewer = NpyWindowViewer(
        arr=arr,
        path=path,
        dataset=dataset,
        initial_window=args.window,
        scale=args.scale,
        initial_start=args.start,
        initial_span=args.span,
    )
    if args.save:
        viewer.save(args.save.resolve())
        print(f"Saved figure: {args.save.resolve()}")
    else:
        viewer.show()


if __name__ == "__main__":
    main()

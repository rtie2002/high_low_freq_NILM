"""Dynamic conv activation / feature-map plots for any model with Conv1d layers.

Figure: input + ground truth (top-left), per-appliance CNN activation heatmaps (3x2 grid).
Layers are discovered automatically from the model graph; no hard-coded architecture.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from adapters.dataloader import WindowDataset, _output_row_offset, _output_slice, _split_key
from evaluation.plots import (
    FULL_CYCLE_APPLIANCES,
    _find_on_events,
    _pick_random_on_events,
    _window_for_on_event,
)


@dataclass
class LayerSpec:
    label: str
    module: nn.Module


@dataclass
class FeatureMapConfig:
    enabled: bool = False
    max_examples: int = 3
    cmap: str = "magma"
    dpi: int = 200
    figsize: tuple[float, float] = (10.0, 9.0)
    display_activation: str = "relu"
    min_on_duration: int = 10

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> FeatureMapConfig:
        raw = raw or {}
        fig = raw.get("figsize", (10.0, 9.0))
        if isinstance(fig, (list, tuple)) and len(fig) == 2:
            figsize = (float(fig[0]), float(fig[1]))
        else:
            figsize = (10.0, 9.0)
        return cls(
            enabled=bool(raw.get("enabled", False)),
            max_examples=max(1, int(raw.get("max_examples", 3))),
            cmap=str(raw.get("cmap", "magma")),
            dpi=int(raw.get("dpi", 200)),
            figsize=figsize,
            display_activation=str(raw.get("display_activation", "relu")).lower(),
            min_on_duration=max(1, int(raw.get("min_on_duration", 10))),
        )


def _is_conv1d(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv1d)


def _meaningful_convs(module: nn.Module) -> list[nn.Conv1d]:
    out: list[nn.Conv1d] = []
    for child in module.modules():
        if isinstance(child, nn.Conv1d):
            out.append(child)
    return out


def _pick_best_conv(convs: list[nn.Conv1d]) -> nn.Conv1d | None:
    if not convs:
        return None
    wide = [c for c in convs if int(c.kernel_size[0]) > 1]
    if wide:
        return wide[-1]
    multi_channel = [c for c in convs if int(c.out_channels) > 1]
    if multi_channel:
        return multi_channel[-1]
    return convs[-1]


def discover_feature_layers(model: nn.Module, appliances: list[str]) -> list[LayerSpec]:
    """Find per-appliance head conv layers to plot."""
    layers: list[LayerSpec] = []
    seen: set[int] = set()

    def add(label: str, module: nn.Module) -> None:
        key = id(module)
        if key in seen:
            return
        seen.add(key)
        layers.append(LayerSpec(label=label, module=module))

    for list_name in ("appliance_heads", "heads"):
        heads = getattr(model, list_name, None)
        if not isinstance(heads, nn.ModuleList):
            continue
        for idx, head in enumerate(heads):
            conv = _pick_best_conv(_meaningful_convs(head))
            if conv is None:
                continue
            app = appliances[idx] if idx < len(appliances) else f"head_{idx}"
            add(f"CNN {app}", conv)

    if not layers:
        for name, module in model.named_modules():
            if _is_conv1d(module) and int(module.kernel_size[0]) > 1:
                add(name or "conv", module)

    if not layers:
        for name, module in model.named_modules():
            if _is_conv1d(module):
                add(name or "conv", module)

    return layers


@dataclass
class ActivationCapture:
    layers: list[LayerSpec]
    store: dict[str, torch.Tensor] = field(default_factory=dict)
    _handles: list[Any] = field(default_factory=list)

    def _make_hook(self, key: str):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                self.store[key] = output.detach()
            elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                self.store[key] = output[0].detach()

        return hook

    def register(self) -> None:
        self.clear()
        for spec in self.layers:
            self._handles.append(spec.module.register_forward_hook(self._make_hook(spec.label)))

    def clear(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.store.clear()

    def remove(self) -> None:
        """Unregister hooks but keep captured tensors for post-processing."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@contextmanager
def capture_activations(layers: list[LayerSpec]) -> Iterator[ActivationCapture]:
    cap = ActivationCapture(layers=layers)
    cap.register()
    try:
        yield cap
    finally:
        cap.remove()


def _activation_to_map(tensor: torch.Tensor) -> np.ndarray:
    """Return (channels, time) numpy array from conv output."""
    arr = tensor.float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[0] if arr.shape[0] <= 4 else arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected conv activation (C, T), got shape {arr.shape}")
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr


def _prepare_activation_map(arr: np.ndarray, cfg: FeatureMapConfig) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if cfg.display_activation == "relu":
        out = np.maximum(out, 0.0)
    return out


def _display_maps(
    layer_maps: list[tuple[str, np.ndarray]],
    cfg: FeatureMapConfig,
) -> tuple[list[tuple[str, np.ndarray]], float, float]:
    """Return display arrays plus shared vmin/vmax for imshow."""
    prepared = [(label, _prepare_activation_map(arr, cfg)) for label, arr in layer_maps]
    if not prepared:
        return prepared, 0.0, 1.0

    global_max = max(float(np.max(arr)) for _, arr in prepared)
    return prepared, 0.0, max(global_max, 1e-12)


_HEATMAP_GRID_SLOTS = [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]


@dataclass(frozen=True)
class OnPeriodPlotSettings:
    margin_min: int = 40
    margin_frac: float = 0.08
    period_samples: int | None = None
    full_cycle_appliances: frozenset[str] = FULL_CYCLE_APPLIANCES
    dynamic_figsize: bool = True

    @classmethod
    def from_plot_cfg(cls, plot_cfg: dict[str, Any] | None) -> OnPeriodPlotSettings:
        plot_cfg = plot_cfg or {}
        raw = plot_cfg.get("on_period_samples", 0)
        period_samples = None if raw is None or int(raw) <= 0 else int(raw)
        full_cycle = plot_cfg.get("full_cycle_appliances")
        return cls(
            margin_min=int(plot_cfg.get("on_period_margin_min", 40)),
            margin_frac=float(plot_cfg.get("on_period_margin_frac", 0.08)),
            period_samples=period_samples,
            full_cycle_appliances=frozenset(full_cycle or FULL_CYCLE_APPLIANCES),
            dynamic_figsize=bool(plot_cfg.get("waveform_dynamic_figsize", True)),
        )


def _output_csv_on_labels(
    data_loader,
    split: str,
    dataset: WindowDataset,
    index: int,
) -> np.ndarray:
    """Dataset CSV *_on labels on the same timeline as model outputs."""
    _, _, z_csv = data_loader.get_splits()[_split_key(split)]
    start = int(dataset.indices[index])
    if dataset.target_mode == "full_input":
        sl = slice(start, start + dataset.seq_len)
    else:
        sl = _output_slice(start, dataset.seq_len, dataset.windowing)
    return z_csv[sl].astype(np.float64)


def _crop_to_on_period(
    series_len: int,
    event_start: int,
    event_end: int,
    *,
    settings: OnPeriodPlotSettings,
    appliance: str,
) -> tuple[int, int]:
    period_cap = None if appliance in settings.full_cycle_appliances else settings.period_samples
    return _window_for_on_event(
        event_start,
        event_end,
        series_len,
        margin_min=settings.margin_min,
        margin_frac=settings.margin_frac,
        max_samples=period_cap,
    )


def _slice_on_period(
    arr: np.ndarray,
    start: int,
    end: int,
    *,
    channel_first: bool = False,
) -> np.ndarray:
    if channel_first:
        return arr[:, start:end]
    return arr[start:end]


def _figsize_for_period(
    cfg: FeatureMapConfig,
    n_samples: int,
    settings: OnPeriodPlotSettings,
) -> tuple[float, float]:
    fig_w, fig_h = cfg.figsize
    if settings.dynamic_figsize and n_samples > 200:
        fig_w = min(fig_w * 2.5, fig_w * (n_samples / 200) ** 0.45)
    return fig_w, fig_h


def _output_csv_range(dataset: WindowDataset, index: int) -> tuple[int, int]:
    """CSV row range [start, end) covered by one model output window."""
    offset = _output_row_offset(dataset.windowing, dataset.seq_len)
    start = int(dataset.indices[index]) + offset
    out_len = int(dataset.windowing.get("output_window_length", 1))
    if dataset.target_mode == "full_input":
        out_len = dataset.seq_len
        start = int(dataset.indices[index])
    return start, start + out_len


def _event_margin(ev_start: int, ev_end: int, settings: OnPeriodPlotSettings) -> int:
    event_len = max(1, ev_end - ev_start + 1)
    return max(settings.margin_min, int(settings.margin_frac * event_len))


def _event_fits_window(ev_start: int, ev_end: int, win_start: int, win_end: int) -> bool:
    return win_start <= ev_start and win_end > ev_end


def _find_best_window_for_event(
    dataset: WindowDataset,
    ev_start: int,
    ev_end: int,
    *,
    settings: OnPeriodPlotSettings,
    require_full_event: bool,
) -> tuple[int, int, int] | None:
    """Return (dataset_index, local_event_start, local_event_end) for a CSV ON event."""
    margin = _event_margin(ev_start, ev_end, settings)
    best: tuple[float, int, int, int] | None = None

    for index in range(len(dataset)):
        win_start, win_end = _output_csv_range(dataset, index)
        win_len = win_end - win_start
        if require_full_event and not _event_fits_window(ev_start, ev_end, win_start, win_end):
            continue

        local_start = max(ev_start, win_start) - win_start
        local_end = min(ev_end, win_end - 1) - win_start
        if local_end < local_start:
            continue

        left_room = local_start
        right_room = win_len - 1 - local_end
        score = float(min(left_room, right_room))
        if left_room >= margin and right_room >= margin:
            score += 10_000.0
        if require_full_event:
            score += float(ev_end - ev_start + 1)

        if best is None or score > best[0]:
            best = (score, index, int(local_start), int(local_end))

    if best is None:
        return None
    _, index, local_start, local_end = best
    return index, local_start, local_end


def _raw_series_for_csv_slice(
    data_loader,
    split: str,
    csv_start: int,
    csv_end: int,
    appliance_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_raw, y_raw, _ = data_loader.get_splits()[_split_key(split)]
    csv_end = min(int(csv_end), len(x_raw))
    csv_start = max(0, int(csv_start))
    agg_w = x_raw[csv_start:csv_end].astype(np.float64)
    gt_w = data_loader.norm.denorm(y_raw[csv_start:csv_end])
    if gt_w.ndim == 2:
        gt_w = gt_w[:, appliance_idx]
    return agg_w, np.asarray(gt_w, dtype=np.float64).reshape(-1)


def _split_long_event(
    ev_start: int,
    ev_end: int,
    *,
    max_len: int,
    min_duration: int,
) -> list[tuple[int, int]]:
    """Break an ON run into segments short enough to fit one model window."""
    if ev_end < ev_start:
        return []
    if ev_end - ev_start + 1 <= max_len:
        return [(ev_start, ev_end)]

    chunks: list[tuple[int, int]] = []
    start = ev_start
    while start <= ev_end:
        end = min(ev_end, start + max_len - 1)
        if end - start + 1 >= min_duration:
            chunks.append((start, end))
        if end >= ev_end:
            break
        start = end + 1
    return chunks


def find_on_period_examples(
    data_loader,
    split: str,
    *,
    appliance: str,
    appliance_idx: int,
    max_examples: int,
    min_on_duration: int,
    settings: OnPeriodPlotSettings,
    seed: int = 0,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]]:
    """Pick CSV-labelled ON periods that fit fully inside one model window."""
    dataset = data_loader._make_window_dataset(split)
    _, y_raw, z_csv = data_loader.get_splits()[_split_key(split)]
    on_flat = z_csv[:, appliance_idx].astype(np.float64)
    power_flat = data_loader.norm.denorm(y_raw)[:, appliance_idx]

    min_dur = max(min_on_duration, 60 if appliance in settings.full_cycle_appliances else min_on_duration)
    prefer_longest = appliance in settings.full_cycle_appliances
    rng = np.random.default_rng(seed)

    flat_events = _pick_random_on_events(
        on_flat,
        power_flat,
        n_periods=max(max_examples * 8, max_examples),
        rng=rng,
        min_duration=min_dur,
        prefer_longest=prefer_longest,
    )

    out_len = int(dataset.windowing.get("output_window_length", 256))
    max_event_len = max(out_len - 2 * settings.margin_min, min_dur)
    expanded_events: list[tuple[int, int]] = []
    for ev_start, ev_end in flat_events:
        expanded_events.extend(
            _split_long_event(
                ev_start,
                ev_end,
                max_len=max_event_len,
                min_duration=min_dur,
            )
        )

    examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    for ev_start, ev_end in expanded_events:
        if len(examples) >= max_examples:
            break

        placement = _find_best_window_for_event(
            dataset,
            ev_start,
            ev_end,
            settings=settings,
            require_full_event=not prefer_longest,
        )
        if placement is None and prefer_longest:
            placement = _find_best_window_for_event(
                dataset,
                ev_start,
                ev_end,
                settings=settings,
                require_full_event=False,
            )
        if placement is None:
            continue

        index, local_start, local_end = placement
        win_start, win_end = _output_csv_range(dataset, index)
        win_len = win_end - win_start
        crop_start, crop_end = _crop_to_on_period(
            win_len,
            local_start,
            local_end,
            settings=settings,
            appliance=appliance,
        )

        key = (index, crop_start, crop_end)
        if key in seen:
            continue
        seen.add(key)

        x, y, z = dataset[index]
        examples.append(
            (
                x.unsqueeze(0),
                y.unsqueeze(0),
                z.unsqueeze(0),
                index,
                int(crop_start),
                int(crop_end),
            )
        )

    if examples:
        return examples

    # Fallback: first window with any CSV-labelled ON in its output range.
    for index in range(len(dataset)):
        csv_on = _output_csv_on_labels(data_loader, split, dataset, index)
        on = csv_on if csv_on.ndim == 1 else csv_on[:, appliance_idx]
        events = _find_on_events(on, min_duration=min_dur)
        if not events:
            continue
        local_start, local_end = max(events, key=lambda t: t[1] - t[0])
        win_start, win_end = _output_csv_range(dataset, index)
        crop_start, crop_end = _crop_to_on_period(
            win_end - win_start,
            local_start,
            local_end,
            settings=settings,
            appliance=appliance,
        )
        x, y, z = dataset[index]
        return [(x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0), index, crop_start, crop_end)]

    x, y, z = dataset[0]
    n = len(y) if y.dim() == 1 else y.shape[0]
    return [(x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0), 0, 0, max(0, n - 1))]


def _plot_input_panel(ax: plt.Axes, aggregate_w: np.ndarray, gt_w: np.ndarray) -> None:
    t = np.arange(len(aggregate_w))
    ax.plot(t, aggregate_w, label="Input window", color="#1f77b4", linewidth=1.0)
    ax.plot(t, gt_w, label="Ground truth", color="#ff7f0e", linewidth=1.0)
    ax.set_ylabel("Power [W]")
    ax.set_xlim(0, max(1, len(t) - 1))
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("samples")


def _plot_activation_panel(
    ax: plt.Axes,
    display: np.ndarray,
    label: str,
    *,
    cfg: FeatureMapConfig,
    vmin: float,
    vmax: float,
) -> Any:
    im = ax.imshow(
        display,
        aspect="auto",
        origin="lower",
        cmap=cfg.cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        extent=(0, display.shape[1], 0, display.shape[0]),
    )
    ax.set_ylabel("Channels")
    ax.set_xlabel("samples")
    ax.set_title(label)
    return im


def plot_feature_map_figure(
    *,
    aggregate_w: np.ndarray,
    gt_w: np.ndarray,
    layer_maps: list[tuple[str, np.ndarray]],
    appliance: str,
    cfg: FeatureMapConfig,
) -> plt.Figure:
    display_maps, vmin, vmax = _display_maps(layer_maps, cfg)

    fig, axes = plt.subplots(3, 2, figsize=cfg.figsize, squeeze=True)
    _plot_input_panel(axes[0, 0], aggregate_w, gt_w)
    axes[0, 0].set_title(
        f"Latent features comparison on aggregate containing {appliance} footprint"
    )

    last_im = None
    for slot, (label, display) in zip(_HEATMAP_GRID_SLOTS, display_maps):
        row, col = slot
        last_im = _plot_activation_panel(
            axes[row, col],
            display,
            label,
            cfg=cfg,
            vmin=vmin,
            vmax=vmax,
        )

    if last_im is not None:
        fig.subplots_adjust(bottom=0.08, hspace=0.35, wspace=0.25)
        cbar = fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.035,
            pad=0.06,
        )
        cbar.set_label("Activation")
    return fig


def save_feature_maps(
    adapter,
    model: nn.Module,
    loader: DataLoader,
    output_dir: str | Path,
    *,
    split: str,
    device: torch.device,
    appliances: list[str] | None = None,
    cfg: FeatureMapConfig | None = None,
    max_batches: int | None = None,
) -> list[Path]:
    """Run one forward pass per ON window and save activation figures."""
    plot_cfg = adapter.model_cfg.get("training", {}).get("plots", {})
    cfg = cfg or FeatureMapConfig.from_dict(plot_cfg.get("feature_maps"))
    if not cfg.enabled:
        return []

    on_period = OnPeriodPlotSettings.from_plot_cfg(plot_cfg)
    appliances = appliances or adapter.cfg["appliances"]
    layers = discover_feature_layers(model, appliances)
    if not layers:
        print(f"Feature maps: no Conv1d layers found for {adapter.name}", flush=True)
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_loader = adapter._data_loader()
    model.eval()
    saved: list[Path] = []
    seed = int(adapter.cfg.get("seed", 0))

    def _collect_layer_maps(cap: ActivationCapture) -> list[tuple[str, np.ndarray]]:
        layer_maps: list[tuple[str, np.ndarray]] = []
        for spec in layers:
            if spec.label not in cap.store:
                continue
            try:
                arr = _activation_to_map(cap.store[spec.label])
            except ValueError:
                continue
            layer_maps.append((spec.label, arr))
        return layer_maps

    for app_idx, app in enumerate(appliances):
        examples = find_on_period_examples(
            data_loader,
            split,
            appliance=app,
            appliance_idx=app_idx,
            max_examples=cfg.max_examples,
            min_on_duration=cfg.min_on_duration,
            settings=on_period,
            seed=seed + app_idx * 997 + (0 if split == "validation" else 1) * 17,
        )
        app_dir = output_dir / app
        app_dir.mkdir(parents=True, exist_ok=True)

        for ex_i, (x, y, z, dataset_idx, crop_start, crop_end) in enumerate(examples):
            x_dev = x.to(device)
            with capture_activations(layers) as cap, torch.no_grad():
                model(x_dev)
                layer_maps = _collect_layer_maps(cap)

            if not layer_maps:
                continue

            dataset = data_loader._make_window_dataset(split)
            win_start, _ = _output_csv_range(dataset, dataset_idx)
            agg_w, gt_w = _raw_series_for_csv_slice(
                data_loader,
                split,
                win_start + crop_start,
                win_start + crop_end,
                app_idx,
            )
            layer_maps = [
                (label, _slice_on_period(arr, crop_start, crop_end, channel_first=True))
                for label, arr in layer_maps
            ]

            fig_cfg = replace(
                cfg,
                figsize=_figsize_for_period(cfg, len(agg_w), on_period),
            )
            fig = plot_feature_map_figure(
                aggregate_w=agg_w,
                gt_w=gt_w,
                layer_maps=layer_maps,
                appliance=app,
                cfg=fig_cfg,
            )
            out_path = app_dir / f"feature_map_{ex_i:02d}.png"
            fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(out_path)

    if saved:
        layer_list = ", ".join(spec.label for spec in layers)
        print(
            f"Saved {len(saved)} feature-map PNG(s) under {output_dir}/ "
            f"(layers: {layer_list})",
            flush=True,
        )
    else:
        print(
            f"Feature maps: enabled but no PNGs saved under {output_dir}/ "
            f"(check ON windows or conv hooks for {adapter.name})",
            flush=True,
        )
    return saved

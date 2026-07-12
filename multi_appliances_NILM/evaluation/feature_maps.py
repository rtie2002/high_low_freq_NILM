"""Dynamic conv activation / feature-map plots for any model with Conv1d layers.

ON-period selection reuses evaluation.plots (same CSV labels and crop as waveforms).
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

from adapters.dataloader import WindowDataset, _output_row_offset, _split_key
from evaluation.plots import (
    OnPeriodSelection,
    dataset_on_labels_for_bundle,
    select_appliance_on_periods,
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
        )


def _is_conv1d(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv1d)


def _meaningful_convs(module: nn.Module) -> list[nn.Conv1d]:
    return [child for child in module.modules() if isinstance(child, nn.Conv1d)]


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
    arr = tensor.float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected conv activation (C, T), got shape {arr.shape}")
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr


def _output_csv_range(dataset: WindowDataset, index: int) -> tuple[int, int]:
    offset = _output_row_offset(dataset.windowing, dataset.seq_len)
    start = int(dataset.indices[index]) + offset
    out_len = int(dataset.windowing.get("output_window_length", 1))
    if dataset.target_mode == "full_input":
        start = int(dataset.indices[index])
        out_len = dataset.seq_len
    return start, start + out_len


def _bundle_aggregate(adapter, split: str, bundle) -> np.ndarray | None:
    try:
        n_points = len(bundle.y_true_watts)
        data_loader = adapter._data_loader()
        key = "validation" if split == "validation" else "test"
        if bundle.csv_timesteps is not None and len(bundle.csv_timesteps) >= n_points:
            raw_x, _, _ = data_loader.get_raw_csv_arrays(key)
            return raw_x[bundle.csv_timesteps[:n_points]].astype(np.float64)
        x, _, _ = data_loader.get_splits()[key]
        windowing = data_loader.model_cfg["windowing"]
        seq_len = int(windowing["input_window_length"])
        if windowing.get("force_even_input_length", False) and seq_len % 2 != 0:
            seq_len += 1
        offset = seq_len - 1
        end = min(offset + n_points, len(x))
        if end <= offset:
            return None
        return x[offset:end].astype(np.float64)
    except Exception:
        return None


def _activations_for_period(
    *,
    data_loader,
    split: str,
    csv_timesteps: np.ndarray | None,
    period: OnPeriodSelection,
    model: nn.Module,
    layers: list[LayerSpec],
    device: torch.device,
) -> list[tuple[str, np.ndarray]]:
    """Collect conv activations on the same flat timeline slice as waveform plots."""
    dataset = data_loader._make_window_dataset(split)
    n = period.crop_end - period.crop_start
    if n <= 0:
        return []

    if csv_timesteps is not None and len(csv_timesteps) >= period.crop_end:
        csv_slice = np.asarray(csv_timesteps[period.crop_start : period.crop_end], dtype=np.int64)
    else:
        csv_slice = np.arange(period.crop_start, period.crop_end, dtype=np.int64)

    maps: dict[str, np.ndarray] = {}
    counts = np.zeros(n, dtype=np.float64)
    csv_min = int(csv_slice.min())
    csv_max = int(csv_slice.max())

    for index in range(len(dataset)):
        win_start, win_end = _output_csv_range(dataset, index)
        if win_end <= csv_min or win_start > csv_max:
            continue

        x, _, _ = dataset[index]
        with capture_activations(layers) as cap, torch.no_grad():
            model(x.unsqueeze(0).to(device))

        for spec in layers:
            if spec.label not in cap.store:
                continue
            arr = _activation_to_map(cap.store[spec.label])
            if spec.label not in maps:
                maps[spec.label] = np.zeros((arr.shape[0], n), dtype=np.float64)

            for flat_i, csv_row in enumerate(csv_slice):
                csv_row = int(csv_row)
                if not (win_start <= csv_row < win_end):
                    continue
                local = csv_row - win_start
                if 0 <= local < arr.shape[1]:
                    maps[spec.label][:, flat_i] += arr[:, local]
                    counts[flat_i] += 1.0

    counts = np.maximum(counts, 1.0)
    return [(label, arr / counts) for label, arr in maps.items()]


def _prepare_activation_map(arr: np.ndarray, cfg: FeatureMapConfig) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if cfg.display_activation == "relu":
        out = np.maximum(out, 0.0)
    return out


def _display_maps(
    layer_maps: list[tuple[str, np.ndarray]],
    cfg: FeatureMapConfig,
) -> tuple[list[tuple[str, np.ndarray]], float, float]:
    prepared = [(label, _prepare_activation_map(arr, cfg)) for label, arr in layer_maps]
    if not prepared:
        return prepared, 0.0, 1.0
    global_max = max(float(np.max(arr)) for _, arr in prepared)
    return prepared, 0.0, max(global_max, 1e-12)


_HEATMAP_GRID_SLOTS = [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]


def _figsize_for_period(cfg: FeatureMapConfig, n_samples: int, *, dynamic: bool) -> tuple[float, float]:
    fig_w, fig_h = cfg.figsize
    if dynamic and n_samples > 200:
        fig_w = min(fig_w * 2.5, fig_w * (n_samples / 200) ** 0.45)
    return fig_w, fig_h


def _plot_input_panel(ax: plt.Axes, aggregate_w: np.ndarray, gt_w: np.ndarray) -> None:
    t = np.arange(len(aggregate_w))
    ax.plot(t, aggregate_w, label="Input window", color="#1f77b4", linewidth=1.0)
    ax.plot(t, gt_w, label="Ground truth", color="#ff7f0e", linewidth=1.0)
    ax.set_ylabel("Power [W]")
    ax.set_xlim(0, max(1, len(t) - 1))
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("samples")


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
        im = axes[row, col].imshow(
            display,
            aspect="auto",
            origin="lower",
            cmap=cfg.cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            extent=(0, display.shape[1], 0, display.shape[0]),
        )
        axes[row, col].set_ylabel("Channels")
        axes[row, col].set_xlabel("samples")
        axes[row, col].set_title(label)
        last_im = im

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
    plot_cfg = adapter.model_cfg.get("training", {}).get("plots", {})
    cfg = cfg or FeatureMapConfig.from_dict(plot_cfg.get("feature_maps"))
    if not cfg.enabled:
        return []

    appliances = appliances or adapter.cfg["appliances"]
    layers = discover_feature_layers(model, appliances)
    if not layers:
        print(f"Feature maps: no Conv1d layers found for {adapter.name}", flush=True)
        return []

    model.eval()
    bundle = adapter.predict_dataloader(
        model,
        loader,
        device,
        split=split,
        max_batches=max_batches,
    )
    aggregate = _bundle_aggregate(adapter, split, bundle)
    y_true_watts = np.asarray(bundle.y_true_watts, dtype=np.float64)
    y_true_on = dataset_on_labels_for_bundle(
        adapter._data_loader(),
        split,
        len(y_true_watts),
        bundle.csv_timesteps,
    )

    raw_period = plot_cfg.get("on_period_samples", 0)
    period_samples = None if raw_period is None or int(raw_period) <= 0 else int(raw_period)
    rng = np.random.default_rng(int(adapter.cfg.get("seed", 0)))
    selections = select_appliance_on_periods(
        appliances,
        y_true_watts,
        y_true_on,
        n_periods=cfg.max_examples,
        period_samples=period_samples,
        full_cycle_appliances=plot_cfg.get("full_cycle_appliances"),
        margin_min=int(plot_cfg.get("on_period_margin_min", 40)),
        margin_frac=float(plot_cfg.get("on_period_margin_frac", 0.08)),
        min_on_duration=10,
        rng=rng,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_loader = adapter._data_loader()
    dynamic_fig = bool(plot_cfg.get("waveform_dynamic_figsize", True))
    saved: list[Path] = []

    for app_idx, app in enumerate(appliances):
        app_dir = output_dir / app
        app_dir.mkdir(parents=True, exist_ok=True)

        for ex_i, period in enumerate(selections.get(app, [])):
            sl = slice(period.crop_start, period.crop_end)
            gt_w = y_true_watts[sl, app_idx]
            agg_w = aggregate[sl] if aggregate is not None and len(aggregate) >= period.crop_end else gt_w

            layer_maps = _activations_for_period(
                data_loader=data_loader,
                split=split,
                csv_timesteps=bundle.csv_timesteps,
                period=period,
                model=model,
                layers=layers,
                device=device,
            )
            if not layer_maps:
                continue

            fig_cfg = replace(
                cfg,
                figsize=_figsize_for_period(cfg, period.crop_end - period.crop_start, dynamic=dynamic_fig),
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

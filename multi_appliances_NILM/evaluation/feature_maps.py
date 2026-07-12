"""Dynamic conv activation / feature-map plots for any model with Conv1d layers.

Figure: input + ground truth (top-left), per-appliance CNN activation heatmaps (3x2 grid).
Layers are discovered automatically from the model graph; no hard-coded architecture.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from adapters.dataloader import _output_row_offset
from evaluation.plots import _find_on_events


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


def _align_aggregate_to_targets(
    aggregate_w: np.ndarray,
    target_len: int,
    windowing: dict[str, Any],
) -> np.ndarray:
    """Crop/pad aggregate watts to the same timeline as model output targets."""
    agg = aggregate_w.reshape(-1)
    if len(agg) == target_len:
        return agg
    seq_len = len(agg)
    offset = _output_row_offset(windowing, seq_len)
    end = min(seq_len, offset + target_len)
    if end <= offset:
        return agg[:target_len]
    out = agg[offset:end]
    if len(out) < target_len:
        pad = target_len - len(out)
        out = np.pad(out, (0, pad), mode="constant")
    return out[:target_len]


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


def _window_on_score(z: np.ndarray, min_duration: int) -> float:
    events = _find_on_events(z, min_duration=min_duration)
    if not events:
        return float(z.mean())
    return float(max(e - s + 1 for s, e in events))


def find_on_windows(
    loader: DataLoader,
    *,
    appliance_idx: int,
    max_examples: int,
    min_on_duration: int,
    max_batches: int = 64,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """Return [(x, y, z, batch_index), ...] windows with clear ON periods."""
    candidates: list[tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, int]] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x, y, z = batch
        for i in range(len(x)):
            z_i = z[i, :, appliance_idx].numpy() if z.dim() == 3 else z[i, appliance_idx].numpy()
            score = _window_on_score(z_i, min_on_duration)
            if score <= 0:
                continue
            candidates.append((score, x[i : i + 1], y[i : i + 1], z[i : i + 1], batch_idx * len(x) + i))

    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        x, y, z = next(iter(loader))
        return [(x[:1], y[:1], z[:1], 0)]

    out = []
    for score, x, y, z, idx in candidates[:max_examples]:
        out.append((x, y, z, idx))
    return out


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

    def _denorm_window(
        x: torch.Tensor,
        y: torch.Tensor,
        app_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        norm = data_loader.norm
        agg = x[0, :, 0].numpy().reshape(-1)
        if norm.input_mean is not None and norm.input_std is not None:
            agg_w = np.maximum(agg * float(norm.input_std) + float(norm.input_mean), 0.0)
        else:
            agg_w = agg
        if y.dim() == 3:
            gt = y[0, :, app_idx].numpy().reshape(-1)
        else:
            gt = y[0, app_idx].numpy().reshape(-1)
        if norm.target_mean is not None and norm.target_std is not None:
            gt_w = np.maximum(
                gt * float(norm.target_std[app_idx]) + float(norm.target_mean[app_idx]),
                0.0,
            )
        else:
            gt_w = data_loader.denorm_to_watts(gt)
        windowing = adapter.model_cfg.get("windowing", {})
        agg_w = _align_aggregate_to_targets(agg_w, len(gt_w), windowing)
        return agg_w, gt_w

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
        windows = find_on_windows(
            loader,
            appliance_idx=app_idx,
            max_examples=cfg.max_examples,
            min_on_duration=cfg.min_on_duration,
            max_batches=64 if max_batches is None else max_batches,
        )
        app_dir = output_dir / app
        app_dir.mkdir(parents=True, exist_ok=True)

        for ex_i, (x, y, z, _win_idx) in enumerate(windows):
            x_dev = x.to(device)
            with capture_activations(layers) as cap, torch.no_grad():
                model(x_dev)
                layer_maps = _collect_layer_maps(cap)

            if not layer_maps:
                continue

            agg_w, gt_w = _denorm_window(x, y, app_idx)

            fig = plot_feature_map_figure(
                aggregate_w=agg_w,
                gt_w=gt_w,
                layer_maps=layer_maps,
                appliance=app,
                cfg=cfg,
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

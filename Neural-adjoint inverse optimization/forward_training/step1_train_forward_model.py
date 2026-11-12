"""Train the learned appliance waveform model used before inverse optimisation.

Stage 1 uses labelled appliance powers only.  For every window the model sees
``p[b, i, t]`` in a common scaled-watt coordinate and reconstructs the same five
traces.  The loss is

    L = (1 / (B * 5 * T)) sum_{b,i,t} (p_hat[b,i,t] - p[b,i,t])**2.

The aggregate returned by :class:`ApplianceConditionedCNN` is a fixed sum of the
five reconstructed targets.  It is intentionally not used as this stage's loss:
a learned aggregate predictor could minimise its error by learning addition
without learning usable appliance waveform constraints.

At Stage 2, freeze this checkpoint's decoder and optimise five latent code
trajectories against an aggregate-only meter reading.  Neither residual power
nor ON/OFF labels are inputs to this Stage 1 model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Allow direct execution from forward_training/ using sibling project modules.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forward_model.cnn import ApplianceConditionedCNN, build_forward_model
from utils.config import load_experiment, load_yaml
from utils.data_loader import NILMDataLoader


@dataclass(frozen=True)
class RawPowerWindows(Dataset[Tensor]):
    """Contiguous raw appliance-power windows in the common watt coordinate.

    Each returned item has shape ``[5, T]`` and contains watts / power_scale.
    Starts are built separately for every sequence ID, so a window never crosses
    a house boundary or a sequence boundary recorded in preprocessing.
    """

    scaled_powers: Tensor
    starts: Tensor
    length: int

    @classmethod
    def from_split(
        cls,
        *,
        powers_watts: np.ndarray,
        segment_ids: np.ndarray,
        length: int,
        stride: int,
        power_scale_watts: float,
    ) -> "RawPowerWindows":
        powers = np.asarray(powers_watts, dtype=np.float32)
        segments = np.asarray(segment_ids, dtype=np.int64)
        if powers.ndim != 2 or powers.shape[1] != len(ApplianceConditionedCNN.APPLIANCES):
            raise ValueError("Expected raw appliance powers with shape [rows, 5].")
        if len(segments) != len(powers):
            raise ValueError("segment_ids and appliance powers must have equal length.")
        if length <= 0 or stride <= 0 or power_scale_watts <= 0:
            raise ValueError("window length, stride, and power_scale_watts must be positive.")
        if np.any(~np.isfinite(powers)):
            raise ValueError("Appliance-power CSV contains non-finite values.")
        if float(powers.min(initial=0.0)) < 0.0:
            raise ValueError("Appliance-power CSV contains negative watts; audit preprocessing first.")

        boundaries = np.flatnonzero(np.r_[True, segments[1:] != segments[:-1], True])
        starts = [
            np.arange(start, end - length + 1, stride, dtype=np.int64)
            for start, end in zip(boundaries[:-1], boundaries[1:])
            if end - start >= length
        ]
        all_starts = np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)
        if len(all_starts) == 0:
            raise ValueError("No valid contiguous windows for this split and window length.")

        # Transpose once.  Dataset access then produces [5, T], model input is [B, 5, T].
        scaled = torch.from_numpy(np.ascontiguousarray(powers.T / power_scale_watts))
        return cls(scaled, torch.from_numpy(all_starts), int(length))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> Tensor:
        start = int(self.starts[index])
        return self.scaled_powers[:, start : start + self.length]


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _resolve_data_root(experiment_path: Path, experiment: dict[str, Any]) -> Path:
    """Accept both the old project's root-relative and config-relative data paths."""
    configured = Path(experiment["data_root"])
    if configured.is_absolute():
        return configured
    candidates = (experiment_path.parent / configured, experiment_path.parent.parent / configured)
    return next((path for path in candidates if path.exists()), candidates[-1])


def _set_seed(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return device


def _build_loader(
    dataset: Dataset[Tensor],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def _balanced_on_off_mse(prediction: Tensor, target: Tensor, thresholds: Tensor) -> Tensor:
    """Average source-wise ON and OFF squared errors with equal weight.

    This is used only with training submeter labels.  For source i, the loss is
    ``.5 mean(error**2 | p_i >= threshold_i) + .5 mean(error**2 | p_i < threshold_i)``
    whenever both sets occur in the mini-batch.  If a set is absent, the
    available set is used so the loss remains finite for rare appliances.
    """
    squared_error = (prediction - target).square()
    on_mask = target >= thresholds[None, :, None]
    source_losses = []
    for appliance in range(target.shape[1]):
        appliance_error = squared_error[:, appliance]
        appliance_on = on_mask[:, appliance]
        state_losses = []
        if bool(appliance_on.any()):
            state_losses.append(appliance_error[appliance_on].mean())
        if bool((~appliance_on).any()):
            state_losses.append(appliance_error[~appliance_on].mean())
        source_losses.append(torch.stack(state_losses).mean())
    return torch.stack(source_losses).mean()


def _training_code_center(
    model: torch.nn.Module,
    dataset: Dataset[Tensor],
    *,
    device: torch.device,
    maximum_windows: int,
) -> Tensor:
    """Build a detached training-only initial code for unrolled inversion.

    Stage 2 starts from a mean code estimated from training appliance windows.
    F1 mirrors that valid inference-time strategy rather than starting the
    unrolled solve from the target window's encoder code.  The centre is
    recomputed at each epoch because the encoder changes during training, but
    it is intentionally detached from the F1 computational graph.
    """
    picks = np.linspace(0, len(dataset) - 1, min(maximum_windows, len(dataset)), dtype=int)
    previous_mode = model.training
    model.eval()
    with torch.no_grad():
        windows = torch.stack([dataset[int(index)] for index in picks]).to(device)
        center = model.encode(windows).mean(dim=(0, 3), keepdim=True)
    model.train(previous_mode)
    return center.detach()


def _unrolled_inverse_recovery(
    model: torch.nn.Module,
    target: Tensor,
    *,
    initial_center: Tensor,
    steps: int,
    step_size: float,
    gradient_normalization: str,
    gradient_norm_epsilon: float,
    inverse_loss_name: str,
    on_thresholds_scaled: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Differentiate K latent gradient-descent steps through the frozen-style decoder.

    ``target`` is [B,5,T] in scaled watts.  During training the controlled
    measurement is exactly ``meter = sum_i target_i``.  The latent update is

        z_(k+1) = z_k - alpha * d/dz mean((sum_i G_theta(z_i,i) - meter)^2).

    ``create_graph=True`` retains the dependence of this gradient on decoder
    parameters theta.  Thus inverse attribution loss backpropagates through
    the K optimisation updates instead of treating the solve as detached data.

    ``rms`` normalisation leaves the negative-gradient *direction* unchanged
    but divides each window's code gradient by its RMS magnitude.  This makes
    alpha an update size in latent-code units, rather than a quantity coupled
    to the changing Softplus slope of the decoder.  It prevents a fixed alpha
    that is inert at initialisation but divergent after one outer update.
    """
    batch, _, length = target.shape
    latent_length = math.ceil(length / model.DOWNSAMPLE)
    if initial_center.shape[-1] not in {1, latent_length}:
        raise ValueError("Training code centre and target window have incompatible latent lengths.")
    # The Stage-2 train-mean initializer averages temporal code positions and
    # tiles the resulting [1,5,Z,1] centre across the requested latent length.
    # Use exactly that convention here, rather than target-derived encoder codes.
    codes = initial_center.expand(batch, -1, -1, latent_length).clone().requires_grad_(True)
    meter = target.sum(dim=1)
    meter_initial = None
    for _ in range(steps):
        power = model.decode(codes, output_length=length)
        meter_loss = (power.sum(dim=1) - meter).square().mean()
        if meter_initial is None:
            meter_initial = meter_loss
        gradient = torch.autograd.grad(meter_loss, codes, create_graph=True)[0]
        if gradient_normalization == "rms":
            gradient_rms = gradient.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
            gradient = gradient / gradient_rms.clamp_min(gradient_norm_epsilon)
        elif gradient_normalization != "none":
            raise ValueError("inverse_aware.gradient_normalization must be 'rms' or 'none'.")
        codes = codes - step_size * gradient
    recovered = model.decode(codes, output_length=length)
    meter_final = (recovered.sum(dim=1) - meter).square().mean()
    if inverse_loss_name == "mse":
        inverse_loss = (recovered - target).square().mean()
    elif inverse_loss_name == "event_balanced_mse":
        inverse_loss = _balanced_on_off_mse(recovered, target, on_thresholds_scaled)
    else:
        raise ValueError("inverse_loss.name must be mse or event_balanced_mse.")
    return recovered, inverse_loss, meter_initial, meter_final


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Tensor],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float | None,
    loss_name: str,
    on_thresholds_scaled: Tensor | None,
    inverse_aware: dict[str, Any] | None = None,
    inverse_initial_center: Tensor | None = None,
    max_batches: int | None = None,
) -> tuple[float, float, float, dict[str, float]]:
    """Return raw MSE, MAE, optimized objective and inverse diagnostics.

    ``event_balanced_mse`` is used only while training.  Let ``m[b,i,t]`` be
    whether target power exceeds the fixed training threshold for appliance i.
    For each appliance, the objective is

        .5 mean_{m=1}(error**2) + .5 mean_{m=0}(error**2).

    Thus abundant OFF samples cannot dominate the training risk.  Empty ON or
    OFF groups in a mini-batch are omitted for that appliance; validation and
    checkpoint selection always use ordinary unweighted MSE.
    """
    training = optimizer is not None
    model.train(training)
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    value_count = 0
    objective_sum = 0.0
    objective_batches = 0
    inverse_loss_sum = 0.0
    meter_initial_sum = 0.0
    meter_final_sum = 0.0
    inverse_batches = 0
    inverse_decoder_gradient_norm = float("nan")

    for batch_index, appliance_power in enumerate(loader):
        # x[b,i,t] = p_watts[b,i,t] / S. Each trace is its own target.
        target = appliance_power.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            # z_i = E_psi(x_i), x_hat_i = G_theta(z_i, i).
            reconstructed, _ = model(target)
            squared_error = (reconstructed - target).square()
            if training and loss_name == "event_balanced_mse":
                if on_thresholds_scaled is None:
                    raise ValueError("event_balanced_mse requires training thresholds.")
                reconstruction_loss = _balanced_on_off_mse(reconstructed, target, on_thresholds_scaled)
            else:
                # L_mse = mean_{b,i,t} (p_hat[b,i,t] - p[b,i,t])^2.
                reconstruction_loss = squared_error.mean()
            inverse_loss = None
            if training and inverse_aware is not None:
                if inverse_initial_center is None or on_thresholds_scaled is None:
                    raise ValueError("Inverse-aware training requires a training code centre and thresholds.")
                _, inverse_loss, meter_initial, meter_final = _unrolled_inverse_recovery(
                    model, target, initial_center=inverse_initial_center,
                    steps=int(inverse_aware["unroll_steps"]),
                    step_size=float(inverse_aware["step_size"]),
                    gradient_normalization=str(inverse_aware["gradient_normalization"]),
                    gradient_norm_epsilon=float(inverse_aware["gradient_norm_epsilon"]),
                    inverse_loss_name=str(inverse_aware["loss"]),
                    on_thresholds_scaled=on_thresholds_scaled,
                )
                # This is the F1 objective: ordinary waveform reconstruction
                # plus attribution error after a differentiable inverse solve.
                loss = reconstruction_loss + float(inverse_aware["weight"]) * inverse_loss
                inverse_loss_sum += float(inverse_loss.detach())
                meter_initial_sum += float(meter_initial.detach())
                meter_final_sum += float(meter_final.detach())
                inverse_batches += 1
                if batch_index == 0:
                    decoder_parameters = [
                        parameter for name, parameter in model.named_parameters() if name.startswith("decoder")
                    ]
                    gradients = torch.autograd.grad(inverse_loss, decoder_parameters, retain_graph=True)
                    inverse_decoder_gradient_norm = float(torch.sqrt(sum(
                        gradient.detach().square().sum() for gradient in gradients if gradient is not None
                    )))
                    if not math.isfinite(inverse_decoder_gradient_norm) or inverse_decoder_gradient_norm == 0.0:
                        raise RuntimeError("Inverse loss did not produce a finite, nonzero decoder gradient.")
            else:
                loss = reconstruction_loss
            if training:
                # Stage 1 updates encoder and decoder weights (psi, theta).
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

        error = reconstructed.detach() - target
        squared_error_sum += float(error.square().sum().item())
        absolute_error_sum += float(error.abs().sum().item())
        value_count += error.numel()
        objective_sum += float(loss.detach())
        objective_batches += 1
        if max_batches is not None and batch_index + 1 >= max_batches:
            break

    diagnostics = {
        "inverse_loss": inverse_loss_sum / inverse_batches if inverse_batches else float("nan"),
        "meter_loss_before": meter_initial_sum / inverse_batches if inverse_batches else float("nan"),
        "meter_loss_after": meter_final_sum / inverse_batches if inverse_batches else float("nan"),
        "inverse_decoder_gradient_norm": inverse_decoder_gradient_norm,
    }
    return squared_error_sum / value_count, absolute_error_sum / value_count, objective_sum / objective_batches, diagnostics


def _save_curve(history: list[dict[str, float]], destination: Path) -> None:
    """Save a compact validation curve without adding a trainer framework."""
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [row["train_mse"] for row in history], label="train MSE")
    plt.plot(epochs, [row["val_mse"] for row in history], label="validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled power MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()


def _copy_config(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)


def train(config_path: Path, *, run_dir: Path | None, overwrite: bool, smoke: bool) -> Path:
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent
    experiment_path = _resolve_path(config["experiment"], config_dir)
    experiment = load_experiment(experiment_path)
    # The preprocessed CSV already records sequence boundaries. Stage 1
    # windows by CSV rows and uses those IDs instead of inferring time gaps.
    csv_cfg = dict(experiment["csv"])
    csv_cfg.pop("sample_seconds", None)
    experiment = {**experiment, "csv": csv_cfg}
    run_cfg = config["run"]
    data_settings = config["data"]
    model_cfg = config["model"]
    optimizer_cfg = config["optimizer"]
    checkpoint_cfg = config["checkpoint"]
    if model_cfg["name"] not in {"appliance_conditioned_cnn", "appliance_specific_cnn"}:
        raise ValueError("Unsupported forward-model name.")
    if config["loss"]["name"] not in {"mse", "event_balanced_mse", "inverse_aware_mse"}:
        raise ValueError("loss.name must be mse, event_balanced_mse, or inverse_aware_mse.")
    if optimizer_cfg["name"].lower() != "adam":
        raise ValueError("Stage 1 currently supports optimizer.name=adam only.")
    if checkpoint_cfg["monitor"] != "validation_mse" or checkpoint_cfg["mode"] != "min":
        raise ValueError("Stage 1 checkpoint selection is validation_mse with mode=min.")

    seed = int(run_cfg["seed"])
    deterministic = bool(run_cfg["deterministic"])
    _set_seed(seed, deterministic=deterministic)
    device = _resolve_device(str(run_cfg["device"]))

    run_root = _resolve_path(run_cfg["root"], config_dir)
    output_dir = (run_dir or run_root / str(run_cfg["name"])).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Run directory already exists: {output_dir}. Use --overwrite or --run-dir.")
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_config(config_path, output_dir / "train_forward.yaml")

    training_cfg = config["training"]
    length = int(data_settings["window_length"])
    stride = int(data_settings["window_stride"])
    batch_size = int(training_cfg["batch_size"])
    power_scale_watts = float(data_settings["power_scale_watts"])
    data_cfg = {
        "windowing": {
            "input_window_length": length,
            "output_window_length": length,
            "input_stride": stride,
            "eval_stride": stride,
            "training_targets": "full_input",
        },
        "training": {"batch_size": batch_size, "num_workers": int(training_cfg.get("num_workers", 0))},
        "data": {"state_label_source": "csv"},
    }
    source_loader = NILMDataLoader(experiment, data_cfg, _resolve_data_root(experiment_path, experiment))
    train_split = source_loader.get_split("train")
    validation_split = source_loader.get_split("validation")
    expected_order = list(ApplianceConditionedCNN.APPLIANCES)
    if source_loader.appliances != expected_order:
        raise ValueError(f"Expected appliance order {expected_order}, got {source_loader.appliances}.")

    train_windows = RawPowerWindows.from_split(
        powers_watts=train_split.targets,
        segment_ids=train_split.segment_ids,
        length=length,
        stride=stride,
        power_scale_watts=power_scale_watts,
    )
    validation_windows = RawPowerWindows.from_split(
        powers_watts=validation_split.targets,
        segment_ids=validation_split.segment_ids,
        length=length,
        stride=stride,
        power_scale_watts=power_scale_watts,
    )
    num_workers = int(training_cfg.get("num_workers", 0))
    train_loader = _build_loader(
        train_windows,
        batch_size=batch_size,
        shuffle=bool(training_cfg["shuffle_train"]),
        seed=seed,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = _build_loader(
        validation_windows,
        batch_size=batch_size,
        shuffle=False,
        seed=seed + 1,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_forward_model(model_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_cfg["learning_rate"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    thresholds_watts = np.asarray(
        [experiment["evaluation"]["on_thresholds_watts"][app] for app in expected_order], dtype=np.float32,
    )
    on_thresholds_scaled = torch.as_tensor(thresholds_watts / power_scale_watts, device=device)
    inverse_aware = None
    if config["loss"]["name"] == "inverse_aware_mse":
        inverse_aware = dict(config.get("inverse_aware", {}))
        required = {
            "unroll_steps", "step_size", "weight", "loss", "center_windows",
            "gradient_normalization", "gradient_norm_epsilon",
        }
        missing = required.difference(inverse_aware)
        if missing:
            raise ValueError(f"inverse_aware is missing {sorted(missing)}.")
        if int(inverse_aware["unroll_steps"]) < 1 or int(inverse_aware["center_windows"]) < 1:
            raise ValueError("inverse_aware unroll_steps and center_windows must be positive.")
        if float(inverse_aware["step_size"]) <= 0 or float(inverse_aware["weight"]) < 0:
            raise ValueError("inverse_aware step_size must be positive and weight nonnegative.")
        if str(inverse_aware["gradient_normalization"]) not in {"rms", "none"}:
            raise ValueError("inverse_aware.gradient_normalization must be 'rms' or 'none'.")
        if float(inverse_aware["gradient_norm_epsilon"]) <= 0:
            raise ValueError("inverse_aware.gradient_norm_epsilon must be positive.")

    max_epochs = 1 if smoke else int(training_cfg["epochs"])
    max_batches = 2 if smoke else None
    patience = int(training_cfg.get("early_stopping_patience", max_epochs))
    min_epochs = int(training_cfg.get("min_epochs", 1))
    clip = training_cfg.get("gradient_clip_norm")
    clip = float(clip) if clip is not None else None
    history: list[dict[str, float]] = []
    best_mse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        inverse_center = None
        if inverse_aware is not None:
            inverse_center = _training_code_center(
                model, train_windows, device=device, maximum_windows=int(inverse_aware["center_windows"]),
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        train_mse, train_mae, train_objective, train_inverse = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=clip,
            loss_name=str(config["loss"]["name"]),
            on_thresholds_scaled=on_thresholds_scaled,
            inverse_aware=inverse_aware,
            inverse_initial_center=inverse_center,
            max_batches=max_batches,
        )
        with torch.no_grad():
            val_mse, val_mae, _, _ = _run_epoch(
                model,
                validation_loader,
                device=device,
                optimizer=None,
                gradient_clip_norm=None,
                loss_name="mse",
                on_thresholds_scaled=None,
                max_batches=max_batches,
            )
        row = {
            "epoch": float(epoch),
            "train_mse": train_mse,
            "train_objective": train_objective,
            "train_mae_scaled": train_mae,
            "val_mse": val_mse,
            "val_mae_scaled": val_mae,
            "val_mae_watts": val_mae * power_scale_watts,
            "train_inverse_loss": train_inverse["inverse_loss"],
            "train_meter_loss_before": train_inverse["meter_loss_before"],
            "train_meter_loss_after": train_inverse["meter_loss_after"],
            "inverse_decoder_gradient_norm": train_inverse["inverse_decoder_gradient_norm"],
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if inverse_aware is not None and device.type == "cuda" else float("nan")
            ),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | train_mse={train_mse:.6g} | "
            f"val_mse={val_mse:.6g} | val_mae={row['val_mae_watts']:.2f} W"
        )
        if val_mse < best_mse:
            best_mse = val_mse
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": dict(model_cfg),
                    "appliances": expected_order,
                    "power_scale_watts": power_scale_watts,
                    "window_length": length,
                    "best_epoch": best_epoch,
                    "validation_mse": best_mse,
                },
                output_dir / str(checkpoint_cfg["filename"]),
            )
        else:
            stale_epochs += 1
        if epoch >= min_epochs and stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}; best validation MSE was epoch {best_epoch}.")
            break

    elapsed_seconds = time.perf_counter() - started
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    _save_curve(history, output_dir / "loss_curve.png")
    manifest = {
        "stage": "forward_model_training",
        "status": "smoke" if smoke else "completed",
        "seed": seed,
        "device": str(device),
        "deterministic": deterministic,
        "experiment_path": str(experiment_path),
        "data_root": str(_resolve_data_root(experiment_path, experiment)),
        "appliances": expected_order,
        "residual_input": False,
        "on_off_input": False,
        "training_loss": str(config["loss"]["name"]),
        "inverse_aware": inverse_aware,
        "event_balance_thresholds_watts": dict(zip(expected_order, thresholds_watts.tolist())),
        "power_scale_watts": power_scale_watts,
        "window_length": length,
        "window_stride": stride,
        "train_windows": len(train_windows),
        "validation_windows": len(validation_windows),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_mse": best_mse,
        "runtime_seconds": elapsed_seconds,
        "checkpoint": str(output_dir / str(checkpoint_cfg["filename"])),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"Saved best checkpoint: {output_dir / str(checkpoint_cfg['filename'])}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage 1 appliance waveform CNN.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("train_forward.yaml").resolve(),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run one epoch with two batches per split.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments.config, run_dir=arguments.run_dir, overwrite=arguments.overwrite, smoke=arguments.smoke)

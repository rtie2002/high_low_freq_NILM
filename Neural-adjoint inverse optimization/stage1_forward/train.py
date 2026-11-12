"""Stage 1: load data, build the CNN, train, validate, and save the best checkpoint."""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from stage1_forward.data import RawPowerWindows
from stage1_forward.losses import event_balanced_mse, inverse_recovery_loss, reconstruction_mse
from stage1_forward.visualize import save_loss_curve
from models.cnn import ApplianceConditionedCNN, build_forward_model
from utils.config import load_experiment, load_yaml, resolve_data_root
from utils.data_loader import NILMDataLoader


def train(
    config_path: Path,
    *,
    run_dir: Path | None = None,
    smoke: bool = False,
    overwrite: bool = False,
) -> Path:
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    if "model_config" in config:
        config["model"] = load_yaml(config_path.parent / config["model_config"])["model"]
    config_dir = config_path.parent
    experiment_path = config_path
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

    seed = int(run_cfg["seed"])
    deterministic = bool(run_cfg["deterministic"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    device_name = str(run_cfg["device"])
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    run_root = (config_dir / run_cfg["root"]).resolve()
    output_dir = (run_dir or run_root / str(run_cfg["name"])).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Run directory already contains files: {output_dir}. "
            "Choose a new --run-dir or pass --overwrite intentionally."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
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
    source_loader = NILMDataLoader(experiment, data_cfg, resolve_data_root(experiment_path, experiment))
    train_split = source_loader.get_split("train")
    validation_split = source_loader.get_split("validation")
    expected_order = list(ApplianceConditionedCNN.APPLIANCES)

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
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_windows, shuffle=bool(training_cfg["shuffle_train"]),
        generator=torch.Generator().manual_seed(seed), **loader_options,
    )
    validation_loader = DataLoader(
        validation_windows, shuffle=False,
        generator=torch.Generator().manual_seed(seed + 1), **loader_options,
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

    max_epochs = 1 if smoke else int(training_cfg["epochs"])
    max_batches = 2 if smoke else None
    patience = int(training_cfg.get("early_stopping_patience", max_epochs))
    min_epochs = int(training_cfg.get("min_epochs", 1))
    history: list[dict[str, float]] = []
    best_mse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        inverse_center = None
        if inverse_aware is not None:
            inverse_center = training_code_center(
                model, train_windows, device=device, maximum_windows=int(inverse_aware["center_windows"]),
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        train_mse, train_mae, train_objective, train_inverse = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            config=config,
            on_thresholds_scaled=on_thresholds_scaled,
            inverse_initial_center=inverse_center,
            max_batches=max_batches,
        )
        with torch.no_grad():
            val_mse, val_mae, _, _ = run_epoch(
                model,
                validation_loader,
                device=device,
                optimizer=None,
                config=config,
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
    save_loss_curve(history, output_dir / "loss_curve.png")
    manifest = {
        "stage": "forward_model_training",
        "config_path": str(config_path),
        "configuration": config,
        "status": "smoke" if smoke else "completed",
        "seed": seed,
        "device": str(device),
        "deterministic": deterministic,
        "experiment_path": str(experiment_path),
        "data_root": str(resolve_data_root(experiment_path, experiment)),
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


# One epoch: reconstruction -> optional inverse-aware loss -> parameter update.
def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Tensor],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    on_thresholds_scaled: Tensor | None = None,
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
    loss_name = config["loss"]["name"]
    gradient_clip_norm = config["training"].get("gradient_clip_norm")
    inverse_aware = config.get("inverse_aware") if loss_name == "inverse_aware_mse" else None
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
                reconstruction_loss = event_balanced_mse(reconstructed, target, on_thresholds_scaled)
            else:
                # L_mse = mean_{b,i,t} (p_hat[b,i,t] - p[b,i,t])^2.
                reconstruction_loss = reconstruction_mse(reconstructed, target)
            inverse_loss = None
            if training and inverse_aware is not None:
                _, inverse_loss, meter_initial, meter_final = unrolled_inverse_recovery(
                    model, target, initial_center=inverse_initial_center,
                    settings=inverse_aware,
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


def training_code_center(
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


def unrolled_inverse_recovery(
    model: torch.nn.Module,
    target: Tensor,
    *,
    initial_center: Tensor,
    settings: dict[str, Any],
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
    steps = int(settings["unroll_steps"])
    step_size = float(settings["step_size"])
    gradient_normalization = settings["gradient_normalization"]
    gradient_norm_epsilon = float(settings["gradient_norm_epsilon"])
    inverse_loss_name = settings["loss"]
    batch, _, length = target.shape
    latent_length = math.ceil(length / model.DOWNSAMPLE)
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
        codes = codes - step_size * gradient
    recovered = model.decode(codes, output_length=length)
    meter_final = (recovered.sum(dim=1) - meter).square().mean()
    inverse_loss = inverse_recovery_loss(
        recovered, target, name=inverse_loss_name, thresholds=on_thresholds_scaled,
    )
    return recovered, inverse_loss, meter_initial, meter_final


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the appliance waveform CNN.")
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "configs/stage1.yaml")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="One epoch, two batches per split.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in --run-dir.")
    args = parser.parse_args()
    train(args.config, run_dir=args.run_dir, smoke=args.smoke, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

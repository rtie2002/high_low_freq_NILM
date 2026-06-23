from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_nilm_metrics
from .plots import plot_prediction_waveforms, plot_training_history


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _config_to_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    raise TypeError("config must be a dict or an object with __dict__")


def _model_prediction(outputs: dict[str, torch.Tensor], scale: float) -> tuple[np.ndarray, np.ndarray]:
    pred_watts = outputs["gated_power"].detach().cpu().numpy() * scale
    pred_on = outputs["on_prob"].detach().cpu().numpy()
    return pred_watts.reshape(-1), pred_on.reshape(-1)


@torch.no_grad()
def evaluate_nilm_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    *,
    device: torch.device,
    scale: float,
    sae_period: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    pred_watts, true_watts, pred_on, true_on = [], [], [], []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        on = batch["on"].to(device, non_blocking=True)
        outputs = model(x)
        loss_parts = criterion(outputs, y, on)
        losses.append(float(loss_parts["loss"].item()))

        watts, on_prob = _model_prediction(outputs, scale)
        pred_watts.append(watts)
        true_watts.append(batch["y_watts"].numpy().reshape(-1))
        pred_on.append(on_prob)
        true_on.append(batch["on"].numpy().reshape(-1))

    metrics = compute_nilm_metrics(
        np.concatenate(true_watts),
        np.concatenate(pred_watts),
        np.concatenate(true_on),
        np.concatenate(pred_on),
        sae_period=sae_period,
    )
    return float(np.mean(losses)), metrics


def train_nilm_model(
    *,
    model_name: str,
    appliance: str,
    model: torch.nn.Module,
    criterion,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: Any,
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    cfg = _config_to_dict(config)
    scale = float(cfg["scale"])
    epochs = int(cfg["epochs"])
    patience = int(cfg["patience"])
    sae_period = int(cfg["sae_period"])

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / f"best_{appliance}.pt"
    history_path = run_dir / f"history_{appliance}.csv"

    best_val = float("inf")
    best_epoch = -1
    stale_epochs = 0

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "val_mae", "val_sae", "val_f1"],
        )
        writer.writeheader()

        for epoch in range(epochs):
            model.train()
            train_losses: list[float] = []
            progress = tqdm(train_loader, desc=f"{appliance} epoch {epoch + 1}/{epochs}")
            for batch in progress:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                on = batch["on"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(x)
                loss_parts = criterion(outputs, y, on)
                loss_parts["loss"].backward()
                optimizer.step()

                loss_value = float(loss_parts["loss"].item())
                train_losses.append(loss_value)
                progress.set_postfix(loss=f"{loss_value:.4f}")

            val_loss, val_metrics = evaluate_nilm_model(
                model,
                val_loader,
                criterion,
                device=device,
                scale=scale,
                sae_period=sae_period,
            )
            train_loss = float(np.mean(train_losses))
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mae": val_metrics["mae"],
                    "val_sae": val_metrics["sae"],
                    "val_f1": val_metrics["f1"],
                }
            )
            handle.flush()
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                f"val_mae={val_metrics['mae']:.3f} val_sae={val_metrics['sae']:.3f} "
                f"val_f1={val_metrics['f1']:.3f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "model_name": model_name,
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "appliance": appliance,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val,
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                    break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, val_metrics = evaluate_nilm_model(
        model,
        val_loader,
        criterion,
        device=device,
        scale=scale,
        sae_period=sae_period,
    )
    _, test_metrics = evaluate_nilm_model(
        model,
        test_loader,
        criterion,
        device=device,
        scale=scale,
        sae_period=sae_period,
    )
    metrics = {
        "model": model_name,
        "appliance": appliance,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "validation": val_metrics,
        "test": test_metrics,
    }
    metrics_path = run_dir / f"metrics_{appliance}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    history_plot_path = run_dir / f"history_{appliance}.png"
    plot_training_history(history_path, history_plot_path, title=f"{model_name} {appliance} Training")

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved training plot: {history_plot_path}")
    print(json.dumps(metrics, indent=2))
    return metrics


@torch.no_grad()
def run_nilm_inference(
    *,
    model_name: str,
    appliance: str,
    model: torch.nn.Module,
    loader: DataLoader,
    config: Any,
    output_dir: Path,
    split: str,
    device: torch.device,
    plot_samples: int = 2000,
) -> dict[str, float]:
    cfg = _config_to_dict(config)
    scale = float(cfg["scale"])
    sae_period = int(cfg["sae_period"])

    model.eval()
    aggregate_watts, pred_watts, true_watts, pred_on, true_on = [], [], [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        outputs = model(x)
        watts, on_prob = _model_prediction(outputs, scale)
        aggregate_watts.append(batch["aggregate_watts"].numpy().reshape(-1))
        pred_watts.append(watts)
        true_watts.append(batch["y_watts"].numpy().reshape(-1))
        pred_on.append(on_prob)
        true_on.append(batch["on"].numpy().reshape(-1))

    arrays = {
        "aggregate_watts": np.concatenate(aggregate_watts),
        "y_true_watts": np.concatenate(true_watts),
        "y_pred_watts": np.concatenate(pred_watts),
        "y_true_on": np.concatenate(true_on),
        "y_pred_on_prob": np.concatenate(pred_on),
    }
    metrics = compute_nilm_metrics(
        arrays["y_true_watts"],
        arrays["y_pred_watts"],
        arrays["y_true_on"],
        arrays["y_pred_on_prob"],
        sae_period=sae_period,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{split}_{appliance}"
    np.savez_compressed(output_dir / f"{stem}_predictions.npz", **arrays)

    prediction_frame = pd.DataFrame(
        {
            "sample_index": np.arange(len(arrays["y_true_watts"])),
            "aggregate": arrays["aggregate_watts"],
            f"{appliance}_power": arrays["y_true_watts"],
            f"pred_{appliance}_power": arrays["y_pred_watts"],
            f"{appliance}_on": arrays["y_true_on"],
            f"pred_{appliance}_on_prob": arrays["y_pred_on_prob"],
        }
    )
    prediction_csv = output_dir / f"{stem}_predictions.csv"
    prediction_frame.to_csv(prediction_csv, index=False)

    waveform_path = output_dir / f"{stem}_waveforms.png"
    plot_prediction_waveforms(
        prediction_frame,
        waveform_path,
        aggregate_col="aggregate",
        true_pred_pairs={appliance: (f"{appliance}_power", f"pred_{appliance}_power")},
        samples=plot_samples,
        title=f"{model_name} {appliance} {split} Predictions",
    )
    metrics_path = output_dir / f"{stem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved predictions: {prediction_csv}")
    print(f"Saved waveform plot: {waveform_path}")
    print(f"Saved metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))
    return metrics

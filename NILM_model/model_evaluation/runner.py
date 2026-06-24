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
from .plots import plot_loss_details, plot_prediction_waveforms, plot_training_history


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
    pred_watts = np.maximum(pred_watts, 0.0)
    pred_on = outputs["on_prob"].detach().cpu().numpy()
    return pred_watts, pred_on


def _target_names(config: Any, appliance: str) -> list[str]:
    cfg = _config_to_dict(config)
    names = cfg.get("target_appliances") or []
    if names:
        return list(names)
    return [appliance]


def _average_metrics(y_true, y_pred, true_on, pred_on, sae_period: int) -> dict[str, float]:
    return compute_nilm_metrics(
        np.asarray(y_true).reshape(-1),
        np.asarray(y_pred).reshape(-1),
        np.asarray(true_on).reshape(-1),
        np.asarray(pred_on).reshape(-1),
        sae_period=sae_period,
    )


def _per_appliance_metrics(y_true, y_pred, true_on, pred_on, names: list[str], sae_period: int) -> dict[str, dict[str, float]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_on = np.asarray(true_on)
    pred_on = np.asarray(pred_on)
    if y_true.ndim == 2:
        y_true = y_true[:, None, :]
        y_pred = y_pred[:, None, :]
        true_on = true_on[:, None, :]
        pred_on = pred_on[:, None, :]
    metrics = {}
    for idx, name in enumerate(names):
        metrics[name] = compute_nilm_metrics(
            y_true[:, idx, :].reshape(-1),
            y_pred[:, idx, :].reshape(-1),
            true_on[:, idx, :].reshape(-1),
            pred_on[:, idx, :].reshape(-1),
            sae_period=sae_period,
        )
    return metrics


def _loss_detail_from_parts(loss_parts: dict[str, torch.Tensor], target_names: list[str]) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "loss": float(loss_parts["loss"].detach().cpu().item()),
        "output_loss": float(loss_parts["output_loss"].detach().cpu().item()),
        "on_loss": float(loss_parts["on_loss"].detach().cpu().item()),
    }
    if "output_loss_per_appliance" in loss_parts:
        output = loss_parts["output_loss_per_appliance"].detach().cpu().numpy().reshape(-1)
        on = loss_parts["on_loss_per_appliance"].detach().cpu().numpy().reshape(-1)
        total = loss_parts["loss_per_appliance"].detach().cpu().numpy().reshape(-1)
        names = target_names if len(target_names) == len(output) else [f"appliance_{idx}" for idx in range(len(output))]
        detail["per_appliance"] = {
            name: {
                "loss": float(total[idx]),
                "output_loss": float(output[idx]),
                "on_loss": float(on[idx]),
            }
            for idx, name in enumerate(names)
        }
    else:
        detail["per_appliance"] = {}
    return detail


def _mean_loss_details(details: list[dict[str, Any]], target_names: list[str]) -> dict[str, Any]:
    if not details:
        return {"loss": float("nan"), "output_loss": float("nan"), "on_loss": float("nan"), "per_appliance": {}}
    mean_detail = {
        "loss": float(np.mean([item["loss"] for item in details])),
        "output_loss": float(np.mean([item["output_loss"] for item in details])),
        "on_loss": float(np.mean([item["on_loss"] for item in details])),
        "per_appliance": {},
    }
    for name in target_names:
        if name not in details[0].get("per_appliance", {}):
            continue
        mean_detail["per_appliance"][name] = {
            "loss": float(np.mean([item["per_appliance"][name]["loss"] for item in details])),
            "output_loss": float(np.mean([item["per_appliance"][name]["output_loss"] for item in details])),
            "on_loss": float(np.mean([item["per_appliance"][name]["on_loss"] for item in details])),
        }
    return mean_detail


def _loss_detail_row(epoch: int, prefix: str, detail: dict[str, Any]) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "epoch": epoch,
        f"{prefix}_loss": detail["loss"],
        f"{prefix}_output_loss": detail["output_loss"],
        f"{prefix}_on_loss": detail["on_loss"],
    }
    for name, values in detail.get("per_appliance", {}).items():
        row[f"{prefix}_{name}_loss"] = values["loss"]
        row[f"{prefix}_{name}_output_loss"] = values["output_loss"]
        row[f"{prefix}_{name}_on_loss"] = values["on_loss"]
    return row


def _format_loss_table(train_detail: dict[str, Any], val_detail: dict[str, Any]) -> str:
    lines = [
        "loss breakdown:",
        "  item             train_reg  train_cls  train_total    val_reg    val_cls  val_total",
        "  overall          "
        f"{train_detail['output_loss']:9.5f}  {train_detail['on_loss']:9.5f}  {train_detail['loss']:11.5f}  "
        f"{val_detail['output_loss']:9.5f}  {val_detail['on_loss']:9.5f}  {val_detail['loss']:9.5f}",
    ]
    train_apps = train_detail.get("per_appliance", {})
    val_apps = val_detail.get("per_appliance", {})
    for name in train_apps:
        train_values = train_apps[name]
        val_values = val_apps.get(name, {"output_loss": float("nan"), "on_loss": float("nan"), "loss": float("nan")})
        lines.append(
            f"  {name:<15} "
            f"{train_values['output_loss']:9.5f}  {train_values['on_loss']:9.5f}  {train_values['loss']:11.5f}  "
            f"{val_values['output_loss']:9.5f}  {val_values['on_loss']:9.5f}  {val_values['loss']:9.5f}"
        )
    return "\n".join(lines)


@torch.no_grad()
def _save_live_waveform(
    *,
    model_name: str,
    appliance: str,
    model: torch.nn.Module,
    loader: DataLoader,
    config: Any,
    output_path: Path,
    device: torch.device,
    split: str = "validation",
    plot_samples: int = 2000,
) -> None:
    cfg = _config_to_dict(config)
    scale = float(cfg["scale"])
    target_names = _target_names(config, appliance)

    model.eval()
    aggregate_watts, pred_watts, true_watts, pred_on, true_on = [], [], [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        outputs = model(x)
        watts, on_prob = _model_prediction(outputs, scale)
        aggregate_watts.append(batch["aggregate_watts"].numpy())
        pred_watts.append(watts)
        true_watts.append(batch["y_watts"].numpy())
        pred_on.append(on_prob)
        true_on.append(batch["on"].numpy())

    aggregate_flat = np.concatenate(aggregate_watts).reshape(-1)
    true_watts_arr = np.concatenate(true_watts, axis=0)
    pred_watts_arr = np.concatenate(pred_watts, axis=0)
    true_on_arr = np.concatenate(true_on, axis=0)
    pred_on_arr = np.concatenate(pred_on, axis=0)

    if true_watts_arr.ndim == 2:
        true_watts_arr = true_watts_arr[:, None, :]
        pred_watts_arr = pred_watts_arr[:, None, :]
        true_on_arr = true_on_arr[:, None, :]
        pred_on_arr = pred_on_arr[:, None, :]

    prediction_data: dict[str, Any] = {
        "sample_index": np.arange(len(aggregate_flat)),
        "aggregate": aggregate_flat,
    }
    true_pred_pairs = {}
    for idx, name in enumerate(target_names):
        prediction_data[f"{name}_power"] = true_watts_arr[:, idx, :].reshape(-1)
        prediction_data[f"pred_{name}_power"] = pred_watts_arr[:, idx, :].reshape(-1)
        prediction_data[f"{name}_on"] = true_on_arr[:, idx, :].reshape(-1)
        prediction_data[f"pred_{name}_on_prob"] = pred_on_arr[:, idx, :].reshape(-1)
        true_pred_pairs[name] = (f"{name}_power", f"pred_{name}_power")

    plot_prediction_waveforms(
        pd.DataFrame(prediction_data),
        output_path,
        aggregate_col="aggregate",
        true_pred_pairs=true_pred_pairs,
        samples=plot_samples,
        title=f"{model_name} {appliance} {split} Live Waveform",
    )


@torch.no_grad()
def evaluate_nilm_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    *,
    device: torch.device,
    scale: float,
    sae_period: int,
    target_names: list[str] | None = None,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    loss_details: list[dict[str, Any]] = []
    pred_watts, true_watts, pred_on, true_on = [], [], [], []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        on = batch["on"].to(device, non_blocking=True)
        outputs = model(x)
        loss_parts = criterion(outputs, y, on)
        losses.append(float(loss_parts["loss"].item()))
        loss_details.append(_loss_detail_from_parts(loss_parts, target_names or []))

        watts, on_prob = _model_prediction(outputs, scale)
        pred_watts.append(watts)
        true_watts.append(batch["y_watts"].numpy())
        pred_on.append(on_prob)
        true_on.append(batch["on"].numpy())

    true_watts_arr = np.concatenate(true_watts, axis=0)
    pred_watts_arr = np.concatenate(pred_watts, axis=0)
    true_on_arr = np.concatenate(true_on, axis=0)
    pred_on_arr = np.concatenate(pred_on, axis=0)
    metrics = _average_metrics(
        true_watts_arr,
        pred_watts_arr,
        true_on_arr,
        pred_on_arr,
        sae_period,
    )
    if target_names:
        metrics["per_appliance"] = _per_appliance_metrics(
            true_watts_arr,
            pred_watts_arr,
            true_on_arr,
            pred_on_arr,
            target_names,
            sae_period,
        )
        metrics["loss_detail"] = _mean_loss_details(loss_details, target_names)
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
    target_names = _target_names(config, appliance)

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / f"best_{appliance}.pt"
    history_path = run_dir / f"history_{appliance}.csv"
    loss_detail_path = run_dir / f"loss_detail_{appliance}.csv"
    live_history_path = run_dir / f"live_history_{appliance}.png"
    live_loss_detail_path = run_dir / f"live_loss_detail_{appliance}.png"
    live_waveform_path = run_dir / f"live_waveform_{appliance}.png"

    best_val = float("inf")
    best_epoch = -1
    stale_epochs = 0

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        detail_handle = loss_detail_path.open("w", newline="", encoding="utf-8")
        detail_writer = None
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "val_mae", "val_sae", "val_f1"],
        )
        writer.writeheader()

        for epoch in range(epochs):
            model.train()
            train_losses: list[float] = []
            train_loss_details: list[dict[str, Any]] = []
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
                train_loss_details.append(_loss_detail_from_parts(loss_parts, target_names))
                progress.set_postfix(loss=f"{loss_value:.4f}")

            val_loss, val_metrics = evaluate_nilm_model(
                model,
                val_loader,
                criterion,
                device=device,
                scale=scale,
                sae_period=sae_period,
                target_names=target_names,
            )
            train_loss = float(np.mean(train_losses))
            train_loss_detail = _mean_loss_details(train_loss_details, target_names)
            val_loss_detail = val_metrics.get("loss_detail", {})
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mae": val_metrics["mae"],
                    "val_sae": val_metrics["sae"],
                    "val_f1": val_metrics["f1"],
                }
            )
            detail_row = {
                **_loss_detail_row(epoch + 1, "train", train_loss_detail),
                **{key: value for key, value in _loss_detail_row(epoch + 1, "val", val_loss_detail).items() if key != "epoch"},
            }
            if detail_writer is None:
                detail_writer = csv.DictWriter(detail_handle, fieldnames=list(detail_row))
                detail_writer.writeheader()
            detail_writer.writerow(detail_row)
            handle.flush()
            detail_handle.flush()
            print(
                f"epoch={epoch + 1} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                f"val_mae={val_metrics['mae']:.3f} val_sae={val_metrics['sae']:.3f} "
                f"val_f1={val_metrics['f1']:.3f}"
            )
            print(_format_loss_table(train_loss_detail, val_loss_detail))
            plot_training_history(
                history_path,
                live_history_path,
                title=f"{model_name} {appliance} Live Training",
            )
            plot_loss_details(
                loss_detail_path,
                live_loss_detail_path,
                title=f"{model_name} {appliance} Live Detailed Loss",
            )
            _save_live_waveform(
                model_name=model_name,
                appliance=appliance,
                model=model,
                loader=val_loader,
                config=config,
                output_path=live_waveform_path,
                device=device,
                split="validation",
            )
            print(f"Live PNGs: {live_history_path}, {live_loss_detail_path}, {live_waveform_path}")

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                stale_epochs = 0
                torch.save(
                    {
                        "model_name": model_name,
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "appliance": appliance,
                        "target_appliances": target_names,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val,
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"Early stopping at epoch {epoch + 1}; best epoch was {best_epoch}.")
                    break
        detail_handle.close()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, val_metrics = evaluate_nilm_model(
        model,
        val_loader,
        criterion,
        device=device,
        scale=scale,
        sae_period=sae_period,
        target_names=target_names,
    )
    _, test_metrics = evaluate_nilm_model(
        model,
        test_loader,
        criterion,
        device=device,
        scale=scale,
        sae_period=sae_period,
        target_names=target_names,
    )
    metrics = {
        "model": model_name,
        "appliance": appliance,
        "target_appliances": target_names,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "validation": val_metrics,
        "test": test_metrics,
    }
    metrics_path = run_dir / f"metrics_{appliance}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    history_plot_path = run_dir / f"history_{appliance}.png"
    loss_detail_plot_path = run_dir / f"loss_detail_{appliance}.png"
    plot_training_history(history_path, history_plot_path, title=f"{model_name} {appliance} Training")
    plot_loss_details(
        loss_detail_path,
        loss_detail_plot_path,
        title=f"{model_name} {appliance} Detailed Loss",
    )

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved training plot: {history_plot_path}")
    print(f"Saved detailed losses: {loss_detail_path}")
    print(f"Saved detailed loss plot: {loss_detail_plot_path}")
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
    target_names = _target_names(config, appliance)

    model.eval()
    aggregate_watts, pred_watts, true_watts, pred_on, true_on = [], [], [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        outputs = model(x)
        watts, on_prob = _model_prediction(outputs, scale)
        aggregate_watts.append(batch["aggregate_watts"].numpy())
        pred_watts.append(watts)
        true_watts.append(batch["y_watts"].numpy())
        pred_on.append(on_prob)
        true_on.append(batch["on"].numpy())

    arrays = {
        "aggregate_watts": np.concatenate(aggregate_watts),
        "y_true_watts": np.concatenate(true_watts, axis=0),
        "y_pred_watts": np.concatenate(pred_watts, axis=0),
        "y_true_on": np.concatenate(true_on, axis=0),
        "y_pred_on_prob": np.concatenate(pred_on, axis=0),
    }
    metrics = _average_metrics(
        arrays["y_true_watts"],
        arrays["y_pred_watts"],
        arrays["y_true_on"],
        arrays["y_pred_on_prob"],
        sae_period,
    )
    per_appliance = _per_appliance_metrics(
        arrays["y_true_watts"],
        arrays["y_pred_watts"],
        arrays["y_true_on"],
        arrays["y_pred_on_prob"],
        target_names,
        sae_period,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{split}_{appliance}"
    np.savez_compressed(output_dir / f"{stem}_predictions.npz", **arrays)

    aggregate_flat = arrays["aggregate_watts"].reshape(-1)
    prediction_data = {
        "sample_index": np.arange(len(aggregate_flat)),
        "aggregate": aggregate_flat,
    }
    true_watts = arrays["y_true_watts"]
    pred_watts_arr = arrays["y_pred_watts"]
    true_on_arr = arrays["y_true_on"]
    pred_on_arr = arrays["y_pred_on_prob"]
    if true_watts.ndim == 2:
        true_watts = true_watts[:, None, :]
        pred_watts_arr = pred_watts_arr[:, None, :]
        true_on_arr = true_on_arr[:, None, :]
        pred_on_arr = pred_on_arr[:, None, :]
    true_pred_pairs = {}
    for idx, name in enumerate(target_names):
        prediction_data[f"{name}_power"] = true_watts[:, idx, :].reshape(-1)
        prediction_data[f"pred_{name}_power"] = pred_watts_arr[:, idx, :].reshape(-1)
        prediction_data[f"{name}_on"] = true_on_arr[:, idx, :].reshape(-1)
        prediction_data[f"pred_{name}_on_prob"] = pred_on_arr[:, idx, :].reshape(-1)
        true_pred_pairs[name] = (f"{name}_power", f"pred_{name}_power")
    prediction_frame = pd.DataFrame(prediction_data)
    prediction_csv = output_dir / f"{stem}_predictions.csv"
    prediction_frame.to_csv(prediction_csv, index=False)

    waveform_path = output_dir / f"{stem}_waveforms.png"
    plot_prediction_waveforms(
        prediction_frame,
        waveform_path,
        aggregate_col="aggregate",
        true_pred_pairs=true_pred_pairs,
        samples=plot_samples,
        title=f"{model_name} {appliance} {split} Predictions",
    )
    metrics_path = output_dir / f"{stem}_metrics.json"
    metrics_payload = {"average": metrics, "per_appliance": per_appliance}
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Saved predictions: {prediction_csv}")
    print(f"Saved waveform plot: {waveform_path}")
    print(f"Saved metrics: {metrics_path}")
    return metrics_payload

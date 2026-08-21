"""Shared training/evaluation/inference loops for multi-appliance NILM.

This file is the common experiment engine used by every model adapter.

High-level flow:

    train_model(...)
        1. Read training settings from model config
        2. Choose CPU/GPU and optional mixed precision settings
        3. Build model, optional init checkpoint, loss, optimizer
        4. Build train/validation/test dataloaders
        5. Print a summary of the data pipeline
        6. Run the epoch loop
        7. Save best checkpoint, history, and live plots

    evaluate_model(...)
        1. Build model and load checkpoint
        2. Build the requested dataloader
        3. Run adapter.predict_dataloader(...)
        4. Save raw predictions
        5. Compute metrics
        6. Save waveform plots

Model-specific math stays outside this file:

    adapters/multinilm.py
    adapters/mat_nilm.py

Those adapters define what happens for one batch. This runner decides when a
batch is training, validation, or final inference.
"""

from __future__ import annotations

import itertools
import json
import shutil
import time
from pathlib import Path

from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from adapters.common import StepOutput
from adapters.config import appliance_list, resolve_lr_scheduler_settings, resolve_tensor_dtype
from adapters.dataloader import (
    NILMDataLoader,
    _resolve_input_length,
    _target_mode,
    get_normalization_cfg,
    get_state_label_source,
    resolve_mains_column,
    resolve_state_thresholds_watts,
)
from evaluation.live_monitor import LiveTrainingMonitor
from evaluation.feature_maps import FeatureMapConfig, save_feature_maps
from evaluation.metrics import _macro_mae_norm, evaluate_bundle
from evaluation.power_postprocess import apply_power_postprocess_pair, resolve_power_postprocess
from evaluation.state_postprocess import maybe_calibrate_and_apply
from evaluation.plots import (
    bundle_aggregate_watts,
    bundle_csv_appliance_watts,
    dataset_on_labels_for_bundle,
    save_appliance_on_waveforms,
)
from evaluation.run_summary import (
    build_hardware_info,
    checkpoint_size_mb,
    count_model_parameters,
    format_parameter_count,
    print_evaluation_report,
    save_run_manifest,
)


def _reset_dir(path: Path) -> Path:
    """Create an empty directory, avoiding Windows rmtree→recreate races.

    ``shutil.rmtree`` can finish deleting asynchronously; recreating files under
    the same path immediately may raise FileNotFoundError on the next open/save.
    Rename-away then recreate is safer on NTFS.
    """
    path = Path(path)
    if path.exists():
        trash = path.with_name(f"{path.name}.__old__")
        if trash.exists():
            shutil.rmtree(trash, ignore_errors=True)
        try:
            path.replace(trash)
        except OSError:
            shutil.rmtree(path, ignore_errors=True)
        else:
            shutil.rmtree(trash, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int) -> None:

    """Make training more repeatable.

    Called once near the start of train_model(...).  This controls Python,
    NumPy, and PyTorch random number generators.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _data_preprocess_note(
    model_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
) -> list[str]:
    data_cfg = model_cfg.get("data", {})
    lines = [f"mains column: {resolve_mains_column(experiment_cfg, model_cfg)}"]
    if norm := get_normalization_cfg(experiment_cfg):
        lines.append("preprocess: dataset z-score (experiment yaml)")
        agg = norm.get("aggregate", {})
        if "mean" in agg and "std" in agg:
            lines.append(f"aggregate stats: mean={agg['mean']}, std={agg['std']}")
    elif scale := data_cfg.get("power_scale"):
        lines.append(f"preprocess: divide power/mains by {scale}")
    else:
        lines.append("preprocess: none (use CSV values as loaded)")
    if str(data_cfg.get("state_label_source", "auto")).lower() == "threshold":
        per_app = resolve_state_thresholds_watts(experiment_cfg, appliance_list(experiment_cfg, model_cfg))
        summary = ", ".join(f"{app}>{int(val)}W" for app, val in zip(appliance_list(experiment_cfg, model_cfg), per_app))
        lines.append(f"state labels: per-appliance thresholds from experiment yaml ({summary})")
    elif str(data_cfg.get("state_label_source", "auto")).lower() == "csv":
        lines.append("state labels: dataset CSV *_on columns")
    return lines


def _summary_line(label: str, value: str, *, width: int = 14) -> None:
    print(f"  {label:<{width}} {value}", flush=True)


def _display_csv_path(csv_path: str) -> str:
    """Show dataset-relative path when possible, else parent/filename."""
    p = Path(csv_path)
    parts = p.parts
    if "datasets" in parts:
        idx = parts.index("datasets")
        return str(Path(*parts[idx:]))
    if len(parts) >= 2:
        return f"{p.parent.name}/{p.name}"
    return p.name


def _print_training_data_summary(
    *,
    experiment_id: str,
    model_name: str,
    appliances: list[str],
    model_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    data_loader: NILMDataLoader,
    batch_size: int,
    epochs: int,
    device: str,
) -> None:
    w = model_cfg["windowing"]
    train_cfg = model_cfg.get("training", {})
    width = 78
    bar = "=" * width
    rule = "-" * width

    in_len = _resolve_input_length(w)
    out_len = int(w.get("output_window_length", 1))
    train_stride = int(w["input_stride"])
    eval_stride = int(w.get("eval_stride", train_stride))
    alignment = w.get("output_alignment", "end")

    print(bar, flush=True)
    print("TRAINING SETUP", flush=True)
    print(rule, flush=True)
    _summary_line("Experiment", experiment_id)
    _summary_line("Model", model_name)
    _summary_line("Device", device)
    _summary_line("Appliances", f"{', '.join(appliances)} ({len(appliances)})")
    print(flush=True)
    _summary_line("Window", f"{in_len} in -> {out_len} out ({alignment})")
    _summary_line("Stride", f"train {train_stride}  |  eval {eval_stride}")
    _summary_line("Batch", f"{batch_size} x {epochs} epochs")
    _summary_line(
        "Optimizer",
        f"Adam lr={train_cfg.get('learning_rate')} wd={train_cfg.get('weight_decay', 0)}",
    )
    sched_cfg = resolve_lr_scheduler_settings(train_cfg)
    if sched_cfg["enabled"]:
        sched_text = (
            f"{sched_cfg['type']} (monitor={sched_cfg['monitor']}, "
            f"patience={sched_cfg['patience']})"
        )
    elif train_cfg.get("lr_scheduler"):
        sched_text = (
            f"off — preset {sched_cfg['preset_type']} "
            f"(monitor={sched_cfg['monitor']}, patience={sched_cfg['patience']})"
        )
    else:
        sched_text = "off"
    _summary_line("Scheduler", sched_text)
    early_stop_text = str(train_cfg.get("early_stop_patience", 0))
    if int(train_cfg.get("early_stop_min_epochs", 0)) > 0:
        early_stop_text += f" after epoch {train_cfg.get('early_stop_min_epochs')}"
    _summary_line("Early stop", early_stop_text)
    _summary_line("Train shuffle", str(train_cfg.get("train_shuffle", True)))
    _summary_line("Tensor dtype", str(train_cfg.get("tensor_dtype", "float32")))

    ckpt = train_cfg.get("checkpoint_monitor")
    if ckpt:
        ckpt_text = str(ckpt)
        if str(ckpt).lower() in {"val_mae_minus_f1", "mae_minus_f1"}:
            space = str(train_cfg.get("checkpoint_mae_space", "normalized")).lower()
            w = _checkpoint_mae_weight(train_cfg)
            ckpt_text += f"  ({w:g}×{space} MAE - F1)"
        _summary_line("Checkpoint", ckpt_text)

    data_notes = _data_preprocess_note(model_cfg, experiment_cfg)
    if data_notes:
        print(flush=True)
        for note in data_notes:
            if note.startswith("mains column:"):
                continue
            if note.startswith("preprocess:"):
                _summary_line("Preprocess", note.split(":", 1)[1].strip())
            elif note.startswith("state labels:"):
                text = note.split(":", 1)[1].strip()
                if len(text) > 58:
                    text = text[:55] + "..."
                _summary_line("State labels", text)
            else:
                _summary_line("Data", note)

    print(flush=True)
    print("DATA SPLITS", flush=True)
    print(rule, flush=True)

    split_infos = {
        split: data_loader.describe_split(split, batch_size=batch_size)
        for split in ("train", "validation", "test")
    }

    print("  CSV files", flush=True)
    for split in ("train", "validation", "test"):
        info = split_infos[split]
        _summary_line(split, _display_csv_path(info["csv_path"]), width=12)
    print(flush=True)

    headers = ("Split", "Timesteps", "Windows", "Batches", "Stride", "Target")
    rows: list[tuple[str, ...]] = []
    for split in ("train", "validation", "test"):
        info = split_infos[split]
        rows.append(
            (
                split,
                f"{info['timesteps']:,}",
                f"{info['windows']:,}",
                str(info["batches"]),
                str(info["stride"]),
                str(info["target_mode"]),
            )
        )

    col_widths = [
        max(len(headers[0]), max(len(r[0]) for r in rows)),
        max(len(headers[1]), max(len(r[1]) for r in rows)),
        max(len(headers[2]), max(len(r[2]) for r in rows)),
        max(len(headers[3]), max(len(r[3]) for r in rows)),
        max(len(headers[4]), max(len(r[4]) for r in rows)),
        max(len(headers[5]), max(len(r[5]) for r in rows)),
    ]

    def _row(cells: tuple[str, ...]) -> str:
        return "  " + "  ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    print(_row(headers), flush=True)
    print("  " + "  ".join("-" * col_widths[i] for i in range(len(headers))), flush=True)
    for row in rows:
        print(_row(row), flush=True)

    print(bar, flush=True)
    print(flush=True)


def _format_duration(seconds: float) -> str:
    """Format seconds as a compact human-readable duration."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_epoch_summary(
    *,
    epoch_no: int,
    epochs: int,
    train_logs: dict[str, float],
    val_logs: dict[str, float] | None = None,
    val_loss: float,
    val_f1: float,
    val_acc: float,
    val_mae: float,
    ckpt_score: float,
    ckpt_detail: str,
    train_time_sec: float,
    val_time_sec: float,
    improved: bool,
    da_active: bool = False,
    lambda_domain: float = 0.0,
    domain_method: str = "coral",
    domain_mu: float = 0.4,
    domain_mix: str = "convex",
) -> str:
    """Professional multi-line epoch report.

    Sections
    --------
    1) Train objective — scalars that enter backprop (L_total)
    2) Train breakdown — raw components vs terms after balance / lambda
    3) Validation — same L_NILM balance as train, no domain; plus F1/Acc/MAE
    4) Checkpoint / timing
    """
    header = f"Epoch {epoch_no:>3}/{epochs}"
    if improved:
        header = f"{header}  * best"

    def _parts(logs: dict[str, float]) -> tuple[float, float, float, float]:
        power = float(logs.get("loss_power", float("nan")))
        state_raw = float(logs.get("loss_state", float("nan")))
        state_term = float(logs.get("loss_state_term", state_raw))
        nilm = power + state_term
        if power != power:
            nilm = float("nan")
        return power, state_raw, state_term, nilm

    l_total = float(train_logs.get("loss", float("nan")))
    l_power, l_state_raw, l_state_term, l_nilm = _parts(train_logs)
    l_dom_raw = float(train_logs.get("loss_domain", 0.0))
    # Scaled domain term (domain_scale=equal) else same as raw.
    l_dom_scaled = float(train_logs.get("loss_domain_term", l_dom_raw))
    mix = str(domain_mix or "convex").lower()
    if da_active:
        if mix == "additive":
            l_nilm_term = l_nilm
            l_dom_term = float(lambda_domain) * l_dom_scaled
            total_formula = "L_NILM + lambda*domain_term"
            nilm_arrow = f"raw={l_nilm:.4f}   -> {l_nilm_term:.4f}"
            dom_arrow = (
                f"raw={l_dom_raw:.4f}   scaled={l_dom_scaled:.4f}   "
                f"-> {l_dom_term:.4f}   (lambda={lambda_domain:g} * scaled"
            )
        else:
            # Lin convex: (1-λ) L_NILM + λ · domain_term
            l_nilm_term = (1.0 - float(lambda_domain)) * l_nilm
            l_dom_term = float(lambda_domain) * l_dom_scaled
            total_formula = "(1-lambda)*L_NILM + lambda*domain_term"
            nilm_arrow = (
                f"raw={l_nilm:.4f}   -> {l_nilm_term:.4f}   "
                f"((1-lambda)={1.0 - float(lambda_domain):g} * raw)"
            )
            dom_arrow = (
                f"raw={l_dom_raw:.4f}   scaled={l_dom_scaled:.4f}   "
                f"-> {l_dom_term:.4f}   (lambda={lambda_domain:g} * scaled"
            )
    else:
        l_nilm_term = l_nilm
        l_dom_term = 0.0
        total_formula = "L_NILM"
        nilm_arrow = f"{l_nilm:.4f}   = power + state_term"
        dom_arrow = ""

    val_logs = val_logs or {}
    val_power, val_state_raw, val_state_term, val_nilm = _parts(val_logs)
    if val_nilm != val_nilm:
        val_nilm = float(val_loss)

    lines = [
        header,
        "  -- train objective (used for backprop) --",
        f"  L_total     {l_total:.4f}   = {total_formula}",
        f"  L_NILM      {nilm_arrow}",
    ]

    lines.append("  -- train components (raw -> into L) --")
    if "loss_power" in train_logs:
        lines.append(
            f"  power       raw={l_power:.4f}   -> {l_power:.4f}   (level MSE)"
        )
    if "loss_state" in train_logs:
        lines.append(
            f"  state       raw={l_state_raw:.4f}   -> {l_state_term:.4f}   (BCE, balanced)"
        )

    if da_active and "loss_domain" in train_logs:
        method = str(domain_method).lower()
        if method == "both":
            method_label = f"MMD+CORAL mu={domain_mu:g}"
        elif method == "mmd":
            method_label = "MMD"
        else:
            method_label = "CORAL"
        lines.append(f"  domain      {dom_arrow}, {method_label})")
    else:
        lines.append("  domain      (off)")

    lines.append("  -- validation (same L_NILM scale, no DA) --")
    lines.append(f"  L_NILM      {val_nilm:.4f}   = power + state_term")
    if val_power == val_power:
        lines.append(
            f"  power       raw={val_power:.4f}   -> {val_power:.4f}   (level MSE)"
        )
    if val_state_raw == val_state_raw:
        lines.append(
            f"  state       raw={val_state_raw:.4f}   -> {val_state_term:.4f}   (BCE, balanced)"
        )
    lines.append(
        f"  metrics     F1={val_f1:.4f}   Acc={val_acc:.4f}   MAE={val_mae:.2f} W"
    )

    epoch_time_sec = train_time_sec + val_time_sec
    ckpt_bits = f"score={ckpt_score:.4f}"
    if ckpt_detail:
        ckpt_bits = f"{ckpt_bits}  ({ckpt_detail})"
    lines.extend(
        [
            "  -- checkpoint / time --",
            f"  {ckpt_bits}",
            f"  time={_format_duration(epoch_time_sec)}"
            f"  (train {_format_duration(train_time_sec)},"
            f" val {_format_duration(val_time_sec)})",
        ]
    )
    return "\n".join(lines)


def _aggregate_logs(log_keys: list[str], n_batches: int, totals: dict[str, float]) -> dict[str, float]:
    """Convert accumulated batch logs into epoch-average logs."""
    return {k: totals.get(k, 0.0) / max(n_batches, 1) for k in log_keys}


def _state_f1_logs(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute validation ON/OFF metrics from collected state predictions.

    Each adapter returns pred_state / true_state (or threshold-derived labels)
    in StepOutput.aux. Reported metrics:

        val_f1     : macro mean F1 over appliances (ON-class focus)
        val_maf1   : same as val_f1 (compat)
        val_mif1   : micro F1 pooled over all appliance×timestep decisions
        val_acc    : macro mean accuracy over appliances
        val_macc   : same as val_acc (compat)
        val_miacc  : micro accuracy pooled over all decisions
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    f1_scores = []
    acc_scores = []
    total_tp = total_fp = total_fn = total_tn = 0

    for app_i in range(y_true.shape[1]):
        yt = y_true[:, app_i]
        yp = y_pred[:, app_i]
        tp = int(np.logical_and(yt, yp).sum())
        fp = int(np.logical_and(~yt, yp).sum())
        fn = int(np.logical_and(yt, ~yp).sum())
        tn = int(np.logical_and(~yt, ~yp).sum())
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        f1_scores.append(2 * tp / max(2 * tp + fp + fn, 1))
        acc_scores.append((tp + tn) / max(tp + tn + fp + fn, 1))

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    micro_f1 = float(2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1))
    macro_acc = float(np.mean(acc_scores)) if acc_scores else 0.0
    micro_acc = float((total_tp + total_tn) / max(total_tp + total_tn + total_fp + total_fn, 1))
    return {
        "val_f1": macro_f1,
        "val_maf1": macro_f1,
        "val_mif1": micro_f1,
        "val_acc": macro_acc,
        "val_macc": macro_acc,
        "val_miacc": micro_acc,
    }


def _flat_batch_array(value) -> np.ndarray:
    """Flatten one batch of power/state windows to (N, A)."""
    arr = value.detach().float().cpu().numpy()
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr.astype(np.float64)


def _batch_state_arrays(adapter, aux: dict) -> tuple[np.ndarray, np.ndarray]:
    """ON/OFF labels for one validation batch."""
    z_pred = _flat_batch_array(aux["pred_state"]).astype(np.int32)
    source = get_state_label_source(adapter.model_cfg)
    if source == "threshold":
        threshold = _state_eval_thresholds(adapter.model_cfg, adapter.experiment, adapter.cfg["appliances"])
        if threshold is None:
            raise ValueError("threshold state_label_source requires ON thresholds")
        y_true = _flat_batch_array(aux["true_power"])
        y_true_w = adapter._data_loader().denorm_to_watts(y_true)
        z_true = (y_true_w > np.asarray(threshold, dtype=np.float32)).astype(np.int32)
        return z_true, z_pred

    z_true = _flat_batch_array(aux["true_state"]).astype(np.int32)
    return z_true, z_pred


def _validation_batch_metrics(adapter, aux: dict) -> dict[str, float]:
    """Baseline-style validation metrics: compute per batch, mean later at epoch end."""
    y_pred = _flat_batch_array(aux["pred_power"])
    y_true = _flat_batch_array(aux["true_power"])
    mae_norm = _macro_mae_norm(y_true, y_pred)

    loader = adapter._data_loader()
    y_pred_w = loader.denorm_to_watts(y_pred)
    y_true_w = loader.denorm_to_watts(y_true)
    mae_watts = _macro_mae_norm(y_true_w, y_pred_w)

    z_true, z_pred = _batch_state_arrays(adapter, aux)
    state_logs = _state_f1_logs(z_true, z_pred)
    return {
        "mae_norm": mae_norm,
        "mae_watts_epoch": mae_watts,
        **state_logs,
    }


def _mean_batch_validation_logs(batch_logs: list[dict[str, float]]) -> dict[str, float]:
    """Average per-batch validation metrics (transfer-learning baseline aggregation)."""
    if not batch_logs:
        return {}
    keys = batch_logs[0].keys()
    return {key: float(np.mean([row[key] for row in batch_logs])) for key in keys}


def _epoch_state_arrays(adapter, aux_batches: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Legacy epoch-pooled ON/OFF arrays (used by offline tools if needed)."""
    """Build validation ON/OFF arrays from the source selected in model yaml."""
    source = get_state_label_source(adapter.model_cfg)
    if source == "threshold":
        threshold = _state_eval_thresholds(adapter.model_cfg, adapter.experiment, adapter.cfg["appliances"])
        if threshold is None:
            raise ValueError("threshold state_label_source requires ON thresholds")
        loader = adapter._data_loader()
        y_true = np.concatenate(aux_batches["true_power"], axis=0)
        if y_true.ndim > 2:
            y_true = y_true.reshape(-1, y_true.shape[-1])
        y_true = loader.denorm_to_watts(y_true)
        threshold = np.asarray(threshold, dtype=np.float32)
        z_true = (y_true > threshold).astype(np.int32)
        z_pred = np.concatenate(aux_batches["pred_state"], axis=0)
        if z_pred.ndim > 2:
            z_pred = z_pred.reshape(-1, z_pred.shape[-1])
        return z_true, z_pred.astype(np.int32)

    z_pred = np.concatenate(aux_batches["pred_state"], axis=0)
    z_true = np.concatenate(aux_batches["true_state"], axis=0)
    if z_pred.ndim > 2:
        z_pred = z_pred.reshape(-1, z_pred.shape[-1])
        z_true = z_true.reshape(-1, z_true.shape[-1])
    return z_true.astype(np.int32), z_pred.astype(np.int32)


def _epoch_power_mae_logs(
    adapter,
    aux_batches: dict[str, list[np.ndarray]],
) -> dict[str, float]:
    """Epoch-level MAE in normalized and watt spaces (macro over appliances).

    Normalized MAE matches the transfer-learning baseline scale (~0.05-1.0) so it
    can be balanced against val_f1 when selecting best.pt.
    """
    y_pred = np.concatenate(aux_batches["pred_power"], axis=0).astype(np.float64)
    y_true = np.concatenate(aux_batches["true_power"], axis=0).astype(np.float64)
    if y_pred.ndim > 2:
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.reshape(-1, y_true.shape[-1])

    norm_per_app = [
        float(np.mean(np.abs(y_pred[:, app_i] - y_true[:, app_i])))
        for app_i in range(y_true.shape[1])
    ]
    mae_norm = float(np.mean(norm_per_app)) if norm_per_app else float("inf")

    loader = adapter._data_loader()
    y_pred_w = loader.denorm_to_watts(y_pred)
    y_true_w = loader.denorm_to_watts(y_true)
    watts_per_app = [
        float(np.mean(np.abs(y_pred_w[:, app_i] - y_true_w[:, app_i])))
        for app_i in range(y_true.shape[1])
    ]
    mae_watts_epoch = float(np.mean(watts_per_app)) if watts_per_app else float("inf")

    return {
        "mae_norm": mae_norm,
        "mae_watts_epoch": mae_watts_epoch,
    }


def _checkpoint_mae_for_score(monitor_key: str, train_cfg: dict, logs: dict[str, float]) -> float:
    """Pick MAE space used by checkpoint/scheduler composite scores."""
    if monitor_key == "val_mae_minus_f1":
        space = str(train_cfg.get("checkpoint_mae_space", "normalized")).lower()
        if space == "watts":
            return float(logs.get("mae_watts_epoch", logs.get("mae", float("inf"))))
        return float(logs.get("mae_norm", float("inf")))
    return float(logs.get("mae", float("inf")))


def _checkpoint_mae_weight(train_cfg: dict) -> float:
    """Scale MAE before subtracting F1 so both terms can trade off.

    Raw ``mae_norm − F1`` is dominated by F1 because mae_norm≈0.1 while F1≈0.7.
    Use ``score = mae_weight * mae − F1`` (lower better).

    Rule of thumb: ``mae_weight ≈ typical_F1 / typical_MAE_norm`` (e.g. 0.7/0.1 → 7)
    so both terms sit near the same magnitude. Epoch-to-epoch: a +0.01 MAE and
    −0.01 F1 still trade 1:1 only when weight=1; raise weight if you want MAE
    changes to matter more relative to small F1 wiggles.
    """
    return float(train_cfg.get("checkpoint_mae_weight", 7.0))


def _mae_minus_f1_score(mae: float, f1: float, *, mae_weight: float) -> float:
    return float(mae_weight) * float(mae) - float(f1)


def _batch_to_device(
    batch,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
):
    """Move every tensor inside a batch to GPU/CPU.

    Dataloader batches are usually:

        (x, y, z)

    but this helper also supports nested tuples/lists/dicts, so future models
    can use richer batch formats without changing runner.py.

    When ``dtype`` is set, floating-point tensors are cast (e.g. float64 training).
    Integer tensors such as state labels are left unchanged.
    """
    if isinstance(batch, (tuple, list)):
        return type(batch)(_batch_to_device(item, device, dtype=dtype) for item in batch)
    if isinstance(batch, dict):
        return {key: _batch_to_device(value, device, dtype=dtype) for key, value in batch.items()}
    if isinstance(batch, torch.Tensor):
        out = batch.to(device, non_blocking=True)
        if dtype is not None and out.is_floating_point():
            out = out.to(dtype=dtype)
        return out
    return batch


def _resolve_checkpoint_monitor(train_cfg: dict) -> tuple[str, str, float]:
    """Decide which validation metric selects best.pt.

    Config examples:

        checkpoint_monitor: val_mae
        checkpoint_monitor: val_f1
        checkpoint_monitor: val_mae_minus_f1   # balanced: w*MAE - F1
        checkpoint_mae_space: normalized       # normalized | watts
        checkpoint_mae_weight: 7.0             # ≈ typical_F1 / typical_MAE_norm

    Return:

        metric_key   -> which key to read from validation logs
        mode         -> "min" for loss/MAE, "max" for F1
        initial_best -> starting best score
    """
    monitor = str(train_cfg.get("checkpoint_monitor", "val_loss")).lower()
    aliases = {
        "val_f1": "val_f1",
        "val_maf1": "val_f1",
        "val_loss": "loss",
        "val_mae": "mae",
        "val_mae_minus_f1": "val_mae_minus_f1",
        "mae_minus_f1": "val_mae_minus_f1",
    }
    key = aliases.get(monitor, monitor)
    if key == "val_f1":
        return key, "max", float("-inf")
    return key, "min", float("inf")


def _epoch_score(monitor_key: str, logs: dict[str, float], train_cfg: dict | None = None) -> float:
    """Read the checkpoint metric from one epoch's validation logs."""
    train_cfg = train_cfg or {}
    if monitor_key == "val_mae_minus_f1":
        mae = _checkpoint_mae_for_score(monitor_key, train_cfg, logs)
        f1 = float(logs.get("val_f1", 0.0))
        return _mae_minus_f1_score(mae, f1, mae_weight=_checkpoint_mae_weight(train_cfg))
    if monitor_key in logs:
        return float(logs[monitor_key])
    if monitor_key == "val_f1":
        return float(logs.get("val_f1", 0.0))
    if monitor_key == "mae":
        return float(logs.get("mae", float("inf")))
    return float(logs.get("loss", float("inf")))


def _is_better(score: float, best: float, mode: str) -> bool:
    """Return True when current validation score beats the previous best."""
    return score > best if mode == "max" else score < best


def _load_init_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> dict[str, int]:
    """Load a source-domain checkpoint, skipping tensors whose shapes changed."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint}")

    payload = torch.load(checkpoint, map_location=device)
    source_state = payload.get("model_state_dict", payload)
    target_state = model.state_dict()

    loadable = {}
    skipped = {}
    for name, tensor in source_state.items():
        if name not in target_state:
            skipped[name] = "missing_in_target"
            continue
        if tuple(tensor.shape) != tuple(target_state[name].shape):
            skipped[name] = f"shape {tuple(tensor.shape)} -> {tuple(target_state[name].shape)}"
            continue
        loadable[name] = tensor

    missing, unexpected = model.load_state_dict(loadable, strict=False)
    print(
        f"Initialized from source checkpoint: {checkpoint}\n"
        f"  loaded tensors : {len(loadable)}\n"
        f"  skipped tensors: {len(skipped)}\n"
        f"  missing tensors: {len(missing)}\n"
        f"  unexpected     : {len(unexpected)}",
        flush=True,
    )
    if skipped:
        preview = list(skipped.items())[:12]
        print("  skipped preview:", flush=True)
        for name, reason in preview:
            print(f"    {name}: {reason}", flush=True)
        if len(skipped) > len(preview):
            print(f"    ... {len(skipped) - len(preview)} more", flush=True)
    return {"loaded": len(loadable), "skipped": len(skipped), "missing": len(missing)}


def _resolve_amp_dtype(train_cfg: dict) -> torch.dtype:
    """Read AMP dtype from config. AMP is optional and controlled by YAML."""
    name = str(train_cfg.get("amp_dtype", "bf16")).lower()
    return torch.bfloat16 if name == "bf16" else torch.float16


def _configure_cuda(train_cfg: dict) -> None:
    """Apply optional CUDA speed settings from training config."""
    if not torch.cuda.is_available():
        return
    if bool(train_cfg.get("cudnn_benchmark", True)):
        torch.backends.cudnn.benchmark = True
    if bool(train_cfg.get("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _resolve_domain_adaptation(adapter) -> tuple[bool, str]:
    """Return (active, target_split) for Lin-style unlabeled target-domain DA.

    Active only when both:
      domain_adaptation.enabled: true
      loss.lambda_domain > 0

    Target split defaults to ``test`` (e.g. UK-DALE H2 aggregates). Only the
    aggregate ``x`` is used; appliance labels from that split are ignored.
    """
    da_cfg = adapter.model_cfg.get("domain_adaptation") or {}
    enabled = bool(da_cfg.get("enabled", False))
    target_split = str(da_cfg.get("target_split", "test"))
    lambda_domain = float(adapter.model_cfg.get("loss", {}).get("lambda_domain", 0.0))

    if enabled and lambda_domain == 0.0:
        print(
            "WARNING: domain_adaptation.enabled=true but loss.lambda_domain=0; "
            "DA dual-loader path is disabled. Set lambda_domain > 0 to enable.",
            flush=True,
        )
        return False, target_split
    if (not enabled) and lambda_domain != 0.0:
        print(
            "WARNING: loss.lambda_domain>0 but domain_adaptation.enabled=false; "
            "DA dual-loader path is disabled. Set enabled: true to enable.",
            flush=True,
        )
        return False, target_split
    return enabled and lambda_domain != 0.0, target_split


def _run_epoch(
    adapter,
    model: torch.nn.Module,
    loss_fn,
    loader,
    *,
    device: torch.device,
    train: bool,
    optimizer=None,
    grad_clip: float = 0.0,
    log_keys: list[str] | None = None,
    collect_states: bool = False,
    desc: str | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    scaler: torch.cuda.amp.GradScaler | None = None,
    target_loader=None,
) -> dict[str, float]:
    """Run one epoch for either training or validation.

    This is the shared inner loop used by every model.

    Training call:

        _run_epoch(..., train=True, optimizer=optim)

        - model.train()
        - gradients enabled
        - adapter.step(...) computes forward pass and loss
        - loss.backward()
        - optimizer.step()

    Optional domain adaptation (training only):

        _run_epoch(..., train=True, target_loader=target_loader)

        - each source batch is paired with one target aggregate batch
        - target_loader cycles if shorter/longer than the source loader
        - adapter.step(..., target_batch=...) adds CORAL/MMD when supported

    Validation call:

        _run_epoch(..., train=False, collect_states=True)

        - model.eval()
        - gradients disabled
        - adapter.step(...) computes forward pass and loss
        - no backward
        - no optimizer update
        - collects predicted/true ON/OFF states for F1
        - target_loader is ignored (no DA on val)

    The adapter.step(...) name is intentionally neutral because one batch has
    the same forward/loss logic in both training and validation. The runner is
    the part that decides whether that batch updates model weights.
    """
    # These are the values averaged and returned at the end of the epoch.
    log_keys = log_keys or ["loss", "loss_state", "loss_power", "mae"]
    totals = {k: 0.0 for k in log_keys}
    n_batches = 0

    # During validation we collect state/power outputs so F1 can follow the
    # source selected in model yaml: CSV labels or thresholded watts.
    aux_batches: dict[str, list[np.ndarray]] = {
        "pred_state": [],
        "true_state": [],
        "pred_power": [],
        "true_power": [],
    }
    batch_val_metrics: list[dict[str, float]] = []

    # Important difference between training and validation:
    #   train mode enables dropout/BatchNorm update behavior.
    #   eval mode freezes dropout behavior and uses BatchNorm running stats.
    if train:
        model.train()
    else:
        model.eval()

    try:
        n_total = len(loader)
    except TypeError:
        n_total = None

    phase = desc or ("train" if train else "val")
    epoch_started = time.perf_counter()

    # Unlabeled target aggregates for domain alignment (train only).
    use_target = train and target_loader is not None
    target_iter = itertools.cycle(target_loader) if use_target else None

    # Important difference between training and validation:
    #   torch.enable_grad() allows backward() during training.
    #   torch.no_grad() saves memory/time during validation.
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(
            loader,
            total=n_total,
            desc=phase,
            leave=False,
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for batch in pbar:
            # Move x/y/z tensors to the same device (and dtype when model is float64).
            model_dtype = next(model.parameters()).dtype
            cast_dtype = model_dtype if model_dtype == torch.float64 else None
            batch = _batch_to_device(batch, device, dtype=cast_dtype)

            target_batch = None
            if target_iter is not None:
                target_batch = _batch_to_device(next(target_iter), device, dtype=cast_dtype)

            # Clear previous gradients before computing the next training batch.
            if train:
                optimizer.zero_grad(set_to_none=True)

            # Optional mixed precision. If use_amp is False, this block behaves
            # like normal full-precision PyTorch.
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp and device.type == "cuda",
            ):
                # Model-specific one-batch logic. target_batch is only passed
                # when DA is active so other adapters stay unchanged.
                if target_batch is not None:
                    step: StepOutput = adapter.step(
                        model, loss_fn, batch, target_batch=target_batch
                    )
                else:
                    step = adapter.step(model, loss_fn, batch)

            # This block is the only place where model weights are updated.
            # It is skipped completely during validation.
            if train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(step.loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    step.loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            # Accumulate every log value returned by the adapter.
            # Example logs:
            #   loss, loss_power, loss_state, mae
            for k in step.logs:
                if k not in totals:
                    totals[k] = 0.0
                    log_keys.append(k)
            for k in log_keys:
                totals[k] += step.logs.get(k, 0.0)

            # Validation F1/MAE: baseline averages per-batch metrics, not one epoch pool.
            if collect_states and step.aux:
                for key in aux_batches:
                    if key in step.aux:
                        aux_batches[key].append(step.aux[key].detach().float().cpu().numpy())
                batch_val_metrics.append(_validation_batch_metrics(adapter, step.aux))

            n_batches += 1
            if n_batches % 20 == 0 or n_batches == n_total:
                postfix = {"loss": f"{step.logs.get('loss', 0.0):.4f}"}
                if "loss_state" in step.logs:
                    postfix["state"] = f"{step.logs['loss_state']:.4f}"
                if "loss_power" in step.logs:
                    postfix["power"] = f"{step.logs['loss_power']:.4f}"
                if "loss_domain" in step.logs and use_target:
                    postfix["loss_domain"] = f"{step.logs['loss_domain']:.4f}"
                    if "loss_domain_term" in step.logs:
                        postfix["dom_term"] = f"{step.logs['loss_domain_term']:.4f}"
                pbar.set_postfix(**postfix)

    # Convert accumulated batch values to one average value per epoch.
    logs = _aggregate_logs(log_keys, n_batches, totals)

    # Validation metrics must be computed over the whole epoch, not by averaging
    # per-batch F1. F1 is nonlinear in TP/FP/FN, and batch averaging can pick the
    # wrong checkpoint for rare appliances.
    if collect_states and aux_batches["pred_state"]:
        z_true_epoch, z_pred_epoch = _epoch_state_arrays(adapter, aux_batches)
        logs.update(_state_f1_logs(z_true_epoch, z_pred_epoch))
        logs.update(_epoch_power_mae_logs(adapter, aux_batches))
        if monitor_key := str(adapter.model_cfg.get("training", {}).get("checkpoint_monitor", "")).lower():
            if monitor_key in {"val_mae_minus_f1", "mae_minus_f1"}:
                train_cfg = adapter.model_cfg.get("training", {})
                mae = _checkpoint_mae_for_score("val_mae_minus_f1", train_cfg, logs)
                f1 = float(logs.get("val_f1", 0.0))
                logs["val_mae_minus_f1"] = _mae_minus_f1_score(
                    mae, f1, mae_weight=_checkpoint_mae_weight(train_cfg)
                )
    logs["elapsed_sec"] = time.perf_counter() - epoch_started
    return logs


def _setup_device_amp_scaler(
    train_cfg: dict,
    *,
    tensor_dtype: torch.dtype = torch.float32,
) -> tuple[torch.device, bool, torch.dtype, torch.cuda.amp.GradScaler]:
    """Resolve device and optional automatic mixed precision settings.

    Step by step:
        1. Pick CUDA when available, otherwise CPU.
        2. Apply optional CUDA speed flags from config.
        3. Decide whether AMP should be enabled.
        4. Resolve AMP dtype (bf16 or fp16).
        5. Build the GradScaler used only for fp16 training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_cuda(train_cfg)
    use_amp = (
        bool(train_cfg.get("use_amp", False))
        and device.type == "cuda"
        and tensor_dtype != torch.float64
    )
    amp_dtype = _resolve_amp_dtype(train_cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    return device, use_amp, amp_dtype, scaler


def _resolve_seed(adapter, train_cfg: dict, seed: int | None) -> int:
    """Resolve the final training seed.

    Priority:
        1. Explicit --seed from the command line
        2. model training seed (model-specific baseline parity)
        3. experiment.yaml seed
    """
    if seed is None:
        seed = train_cfg.get("seed") or adapter.cfg.get("seed")
    if seed is None:
        raise ValueError("Set seed in experiment yaml, model training config, or pass --seed")
    return int(seed)


def _build_model(
    adapter,
    device: torch.device,
    train_cfg: dict,
    init_checkpoint: Path | None,
    *,
    tensor_dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    """Build the model and apply optional initialization steps.

    Step by step:
        1. adapter.build_model(...) creates the architecture on the target device.
        2. If transfer-learning initialization is configured, load compatible
           tensors from that checkpoint.
        3. If torch.compile is enabled, wrap the model after loading weights.
    """
    model = adapter.build_model(device)
    configured_init = train_cfg.get("init_checkpoint") or train_cfg.get("pretrained_checkpoint")
    resolved_init = init_checkpoint or (Path(configured_init) if configured_init else None)
    if resolved_init is not None:
        _load_init_checkpoint(model, resolved_init, device)
    if tensor_dtype == torch.float64:
        model = model.double()
    if bool(train_cfg.get("torch_compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model


def _build_loss_and_optimizer(
    adapter,
    model: torch.nn.Module,
    device: torch.device,
    *,
    tensor_dtype: torch.dtype = torch.float32,
):
    """Build the loss object, move it to device if needed, and create optimizer."""
    loss_fn = adapter.build_loss()
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)
        if tensor_dtype == torch.float64:
            loss_fn = loss_fn.double()
    optim, sched = adapter.configure_optimizer(model)
    return loss_fn, optim, sched


def _build_loaders(adapter):
    """Build the three standard dataloaders used by the pipeline.

    Notes:
        - train_loader is used for weight updates
        - val_loader is used for checkpoint selection
        - test_loader is loaded here so live monitoring can also draw test plots
    """
    train_loader = adapter.build_dataloader("train")
    val_loader = adapter.build_dataloader("validation")
    test_loader = adapter.build_dataloader("test")
    return train_loader, val_loader, test_loader


def _compute_scheduler_key(train_cfg: dict, monitor_key: str) -> str:
    """Map scheduler monitor aliases onto actual validation log keys."""
    sched_cfg = resolve_lr_scheduler_settings(train_cfg)
    scheduler_raw = sched_cfg["monitor"]
    scheduler_aliases = {
        "val_f1": "val_f1",
        "val_maf1": "val_f1",
        "val_loss": "loss",
        "val_mae": "mae",
        "val_mae_minus_f1": "val_mae_minus_f1",
        "mae_minus_f1": "val_mae_minus_f1",
    }
    return scheduler_aliases.get(scheduler_raw, scheduler_raw)


def _state_eval_thresholds(model_cfg: dict, experiment_cfg: dict, appliances: list[str]) -> np.ndarray | None:
    """Per-appliance ON thresholds from experiment yaml when threshold mode is enabled."""
    if get_state_label_source(model_cfg) != "threshold":
        return None
    return resolve_state_thresholds_watts(experiment_cfg, appliances)


def _save_latest_waveforms(
    *,
    monitor: LiveTrainingMonitor,
    adapter,
    model: torch.nn.Module,
    val_loader,
    test_loader,
    device: torch.device,
    epoch_no: int,
    best_epoch: int,
) -> None:
    """Save the current epoch's loss plots and latest waveform examples."""
    monitor.save_loss_plots(epoch=epoch_no, best_epoch=best_epoch or None)
    monitor.save_waveforms(
        adapter,
        model,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epoch=epoch_no,
    )


def _save_best_waveforms(
    *,
    monitor: LiveTrainingMonitor,
    adapter,
    model: torch.nn.Module,
    val_loader,
    test_loader,
    device: torch.device,
    best_epoch_no: int,
) -> None:
    """Save waveform examples under the 'best' folder for the current best epoch."""
    monitor.save_waveforms(
        adapter,
        model,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epoch=best_epoch_no,
        include_best=True,
    )


def train_model(
    adapter,
    run_dir: Path,
    *,
    epochs: int | None = None,
    seed: int | None = None,
    init_checkpoint: Path | None = None,
) -> Path:
    """Train one model using the shared pipeline and return best.pt.

    This function is model-agnostic. It does not know the details of MATNILM,
    MultiNILM, or any future model. The adapter supplies model-specific pieces:

        adapter.build_model(...)
        adapter.build_loss()
        adapter.build_dataloader(...)
        adapter.step(...)
        adapter.configure_optimizer(...)

    Training/validation happens inside this function:

        train_logs = _run_epoch(..., train=True)
        val_logs   = _run_epoch(..., train=False)
    """
    # Step 1:
    # Read training settings from the selected model config.
    train_cfg = adapter.model_cfg["training"]
    epochs = int(train_cfg["epochs"]) if epochs is None else int(epochs)
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0. Use a one-batch smoke test for pipeline checks.")

    _, tensor_dtype = resolve_tensor_dtype(adapter.model_cfg)

    # Step 2:
    # Choose device and optional AMP behavior.
    device, use_amp, amp_dtype, scaler = _setup_device_amp_scaler(train_cfg, tensor_dtype=tensor_dtype)

    # Step 3:
    # Build model, optional init checkpoint, loss, and optimizer.
    model = _build_model(
        adapter, device, train_cfg, init_checkpoint, tensor_dtype=tensor_dtype
    )
    print("\nModel architecture:", flush=True)
    print(model, flush=True)
    param_stats = count_model_parameters(model)
    print(
        f"Parameters: {format_parameter_count(param_stats['parameters_total'])} "
        f"({param_stats['parameters_trainable']:,} trainable)",
        flush=True,
    )
    loss_fn, optim, sched = _build_loss_and_optimizer(
        adapter, model, device, tensor_dtype=tensor_dtype
    )

    print(f"Device: {device}", flush=True)
    print(f"Tensor dtype: {tensor_dtype}", flush=True)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        amp_label = str(train_cfg.get("amp_dtype", "bf16")) if use_amp else "off"
        workers = int(train_cfg.get("num_workers", 0))
        tqdm.write(f"GPU: {name} | AMP: {amp_label} | DataLoader workers: {workers}")

    # Step 4:
    # Build the standard train/validation/test dataloaders.
    print("Loading CSV splits into memory (train, val, test)...", flush=True)
    train_loader, val_loader, test_loader = _build_loaders(adapter)

    # Optional unlabeled target-domain loader for CORAL/MMD (Lin-style).
    da_active, da_target_split = _resolve_domain_adaptation(adapter)
    target_loader = None
    da_lambda = float(adapter.model_cfg.get("loss", {}).get("lambda_domain", 0.0))
    da_method = str(adapter.model_cfg.get("loss", {}).get("domain_method", "coral"))
    da_mu = float(adapter.model_cfg.get("loss", {}).get("domain_mu", 0.4))
    if da_active:
        if da_target_split == "train":
            target_loader = train_loader
        elif da_target_split in {"validation", "val"}:
            target_loader = val_loader
        elif da_target_split == "test":
            target_loader = test_loader
        else:
            target_loader = adapter.build_dataloader(da_target_split)
        method_note = da_method
        if da_method.lower() == "both":
            method_note = f"both (Eq.12: mu={da_mu:g}·MMD² + (1-mu)·CORAL)"
        print(
            "------------------------------------------------------------------------------\n"
            "DOMAIN ADAPTATION\n"
            "------------------------------------------------------------------------------\n"
            f"  Status         ON\n"
            f"  Target split   {da_target_split}  (aggregates only; labels unused)\n"
            f"  Method         {method_note}\n"
            f"  lambda_domain  {da_lambda:g}\n"
            f"  domain_mix     {str(adapter.model_cfg.get('loss', {}).get('domain_mix', 'convex'))}\n"
            f"  Feature layers {adapter.model_cfg.get('architecture', {}).get('domain_feature_layers', ['aligned'])}",
            flush=True,
        )
    else:
        print(
            "------------------------------------------------------------------------------\n"
            "DOMAIN ADAPTATION\n"
            "------------------------------------------------------------------------------\n"
            "  Status         OFF  (supervised source only)",
            flush=True,
        )

    # Step 5:
    # Resolve the final seed and print the data pipeline summary.
    seed_int = _resolve_seed(adapter, train_cfg, seed)
    data_loader = adapter._data_loader()
    _print_training_data_summary(
        experiment_id=adapter.experiment["experiment_id"],
        model_name=adapter.name,
        appliances=adapter.cfg["appliances"],
        model_cfg=adapter.model_cfg,
        experiment_cfg=adapter.experiment,
        data_loader=data_loader,
        batch_size=int(train_loader.batch_size),
        epochs=int(epochs),
        device=str(device),
    )

    plot_cfg = train_cfg.get("plots", {})
    seed_everything(seed_int)

    # Step 6:
    # Prepare output paths, checkpoint selection rules, and live monitor.
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"

    param_stats = count_model_parameters(model)
    save_run_manifest(
        run_dir / "run_manifest.json",
        {
            "experiment_id": adapter.experiment["experiment_id"],
            "model_name": adapter.name,
            "seed": int(seed_int),
            "batch_size": int(train_loader.batch_size),
            "epochs_configured": int(epochs),
            "checkpoint_monitor": str(train_cfg.get("checkpoint_monitor", "val_loss")),
            "checkpoint_mae_space": str(train_cfg.get("checkpoint_mae_space", "normalized")),
            "windowing": adapter.model_cfg.get("windowing", {}),
            "appliances": adapter.cfg["appliances"],
            "domain_adaptation": {
                "enabled": bool(da_active),
                "target_split": da_target_split if da_active else None,
                "lambda_domain": float(
                    adapter.model_cfg.get("loss", {}).get("lambda_domain", 0.0)
                ),
                "domain_method": str(
                    adapter.model_cfg.get("loss", {}).get("domain_method", "coral")
                ),
            },
            **param_stats,
            **build_hardware_info(device),
        },
    )
    try:
        import yaml

        with open(run_dir / "config_merged.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(adapter.cfg, f, sort_keys=False)
    except Exception:
        pass

    # Decide which validation metric saves best.pt.
    # Example:
    #   checkpoint_monitor: val_mae -> lower is better
    #   checkpoint_monitor: val_f1  -> higher is better
    monitor_key, monitor_mode, best_score = _resolve_checkpoint_monitor(train_cfg)

    # Scheduler can monitor a different validation metric if configured.
    scheduler_key = _compute_scheduler_key(train_cfg, monitor_key)
    best_epoch = 0
    history = []
    appliances = adapter.cfg["appliances"]
    monitor = LiveTrainingMonitor(
        run_dir,
        model_name=adapter.name,
        appliances=appliances,
        plot_cfg=plot_cfg,
        seed=int(seed_int),
    )
    grad_clip = float(train_cfg.get("gradient_clip", 0.0))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 0))
    early_stop_min_epochs = int(train_cfg.get("early_stop_min_epochs", 0))
    epochs_without_improvement = 0
    training_started = time.perf_counter()

    try:
        # Step 7:
        # Main epoch loop. Each epoch does:
        #   7a. training pass
        #   7b. validation pass
        #   7c. scheduler update
        #   7d. history/log update
        #   7e. checkpoint comparison
        #   7f. live plots
        #   7g. best checkpoint save
        #   7h. optional early stopping
        # Hur-style 2-stage: source-only warmup, then DA (+ optional PL).
        da_warmup_epochs = int(
            adapter.model_cfg.get("loss", {}).get("da_warmup_epochs", 0)
        )
        da_ramp_epochs = int(
            adapter.model_cfg.get("loss", {}).get("da_ramp_epochs", 0)
        )
        pl_weight_cfg = float(adapter.model_cfg.get("loss", {}).get("pl_weight", 0.0))
        if da_warmup_epochs > 0 and da_active:
            print(
                f"DA schedule: warmup {da_warmup_epochs} ep (lambda=0, pl=0); "
                f"then ramp {da_ramp_epochs} ep to lambda={da_lambda:g}, "
                f"pl={pl_weight_cfg:g} (AHDA-style soft start).",
                flush=True,
            )

        # Freeze DA if source-val state BCE rises after DA starts (negative transfer).
        da_freeze = False
        best_val_state_raw = float("inf")
        val_state_bad_epochs = 0

        for epoch in range(epochs):
            epoch_no = epoch + 1
            epoch_tag = f"Epoch {epoch_no}/{epochs}"

            if hasattr(loss_fn, "lambda_domain"):
                if da_freeze:
                    loss_fn.lambda_domain = 0.0
                    if hasattr(loss_fn, "pl_weight"):
                        loss_fn.pl_weight = 0.0
                elif da_warmup_epochs > 0 and epoch < da_warmup_epochs:
                    loss_fn.lambda_domain = 0.0
                    if hasattr(loss_fn, "pl_weight"):
                        loss_fn.pl_weight = 0.0
                else:
                    # Soft ramp after warmup (linear); ramp=0 → hard switch (legacy).
                    if da_ramp_epochs > 0:
                        t = min(
                            1.0,
                            float(epoch - da_warmup_epochs + 1) / float(da_ramp_epochs),
                        )
                        loss_fn.lambda_domain = float(da_lambda) * t
                        if hasattr(loss_fn, "pl_weight"):
                            loss_fn.pl_weight = float(pl_weight_cfg) * t
                    else:
                        loss_fn.lambda_domain = float(da_lambda)
                        if hasattr(loss_fn, "pl_weight"):
                            loss_fn.pl_weight = float(pl_weight_cfg)

            # 6a. Training epoch
            # This is where model weights are updated.
            # When DA is on, each source batch is paired with a target aggregate.
            train_logs = _run_epoch(
                adapter,
                model,
                loss_fn,
                train_loader,
                device=device,
                train=True,
                optimizer=optim,
                grad_clip=grad_clip,
                desc=f"{epoch_tag} | train",
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
                target_loader=target_loader,
            )

            # 6b. Validation epoch
            # This is where checkpoint metrics are measured. No weights change.
            val_logs = _run_epoch(
                adapter,
                model,
                loss_fn,
                val_loader,
                device=device,
                train=False,
                collect_states=True,
                desc=f"{epoch_tag} | val",
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            val_loss = float(val_logs["loss"])
            val_f1 = float(val_logs.get("val_f1", 0.0))
            val_acc = float(val_logs.get("val_acc", 0.0))
            val_mae = float(val_logs.get("mae", 0.0))
            train_time_sec = float(train_logs.get("elapsed_sec", 0.0))
            val_time_sec = float(val_logs.get("elapsed_sec", 0.0))
            epoch_time_sec = train_time_sec + val_time_sec
            cumulative_time_sec = time.perf_counter() - training_started

            # Optional: freeze DA if source-val raw BCE keeps rising (negative transfer).
            # Default patience=0 → never freeze (Lin-style: keep λ fixed all epochs).
            if (
                da_active
                and (not da_freeze)
                and epoch >= da_warmup_epochs
                and "loss_state" in val_logs
            ):
                vs = float(val_logs["loss_state"])
                if vs < best_val_state_raw - 1e-4:
                    best_val_state_raw = vs
                    val_state_bad_epochs = 0
                else:
                    val_state_bad_epochs += 1
                patience = int(
                    adapter.model_cfg.get("loss", {}).get("da_freeze_patience", 0)
                )
                if patience > 0 and val_state_bad_epochs >= patience:
                    da_freeze = True
                    tqdm.write(
                        f"  DA FREEZE: val state BCE rising for {patience} epochs "
                        f"(now {vs:.4f}); lambda/pl set to 0 for remaining epochs."
                    )

            # 6c. Optional scheduler step after validation.
            sched_metric = _epoch_score(scheduler_key, val_logs, train_cfg)
            if sched is not None:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(sched_metric)
                else:
                    sched.step()

            # 6d. Save training history for later plotting/inspection.
            history.append(
                {
                    "epoch": epoch_no,
                    **{f"train_{k}": v for k, v in train_logs.items() if k != "elapsed_sec"},
                    **{
                        f"val_{k}": v
                        for k, v in val_logs.items()
                        if k.startswith("temporal_long_gate_")
                    },
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "val_mae_norm": float(val_logs.get("mae_norm", 0.0)),
                    "val_mae_watts": float(val_logs.get("mae_watts_epoch", val_mae)),
                    "val_acc": val_acc,
                    "val_maf1": float(val_logs.get("val_maf1", val_f1)),
                    "val_mif1": float(val_logs.get("val_mif1", 0.0)),
                    "val_miacc": float(val_logs.get("val_miacc", val_acc)),
                    "train_time_sec": train_time_sec,
                    "val_time_sec": val_time_sec,
                    "epoch_time_sec": epoch_time_sec,
                    "cumulative_time_sec": cumulative_time_sec,
                }
            )
            monitor.append_epoch(epoch=epoch_no, train_logs=train_logs, val_logs=val_logs)

            # 6e. Check if this epoch should replace best.pt.
            improved = False
            ckpt_score = _epoch_score(monitor_key, val_logs, train_cfg)
            if _is_better(ckpt_score, best_score, monitor_mode):
                improved = True

            ckpt_detail = ""
            if monitor_key == "val_mae_minus_f1":
                mae_ckpt = _checkpoint_mae_for_score(monitor_key, train_cfg, val_logs)
                space = str(train_cfg.get("checkpoint_mae_space", "normalized")).lower()
                w = _checkpoint_mae_weight(train_cfg)
                ckpt_detail = (
                    f"{w:g}×{space} mae - f1, mae={mae_ckpt:.4f}, f1={val_f1:.4f}"
                )

            # Prefer live lambda logged by the adapter (warmup-safe).
            live_lambda = float(
                train_logs.get(
                    "lambda_domain",
                    getattr(loss_fn, "lambda_domain", da_lambda),
                )
            )
            tqdm.write(
                _format_epoch_summary(
                    epoch_no=epoch_no,
                    epochs=epochs,
                    train_logs=train_logs,
                    val_logs=val_logs,
                    val_loss=val_loss,
                    val_f1=val_f1,
                    val_acc=val_acc,
                    val_mae=val_mae,
                    ckpt_score=ckpt_score,
                    ckpt_detail=ckpt_detail,
                    train_time_sec=train_time_sec,
                    val_time_sec=val_time_sec,
                    improved=improved,
                    da_active=da_active and live_lambda != 0.0,
                    lambda_domain=live_lambda,
                    domain_method=da_method,
                    domain_mu=da_mu,
                    domain_mix=str(
                        adapter.model_cfg.get("loss", {}).get("domain_mix", "convex")
                    ),
                )
            )

            # 6f. Save live loss/waveform plots at configured intervals.
            if monitor.should_plot(epoch_no):
                _save_latest_waveforms(
                    monitor=monitor,
                    adapter=adapter,
                    model=model,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    epoch_no=epoch_no,
                    best_epoch=best_epoch,
                )
                tqdm.write(
                    f"  {epoch_tag} | saved waveforms -> "
                    f".../waveforms/{{validation,test}}/epoch_{epoch_no:04d}/ (+ latest/)"
                )
                tqdm.write(
                    f"  {epoch_tag} | saved metrics tables -> "
                    f".../metrics_by_epoch/epoch_{epoch_no:04d}/ (+ metrics_history.csv)"
                )
                tqdm.write(
                    f"  {epoch_tag} | saved val/test table figure -> "
                    f".../metrics_by_epoch/epoch_{epoch_no:04d}/validation_test_comparison.png"
                )
                tqdm.write(
                    f"  {epoch_tag} | saved one-picture comparisons -> "
                    f".../comparisons/metrics_all_epochs.png + "
                    f".../waveforms_by_epoch/ALL_appliances_period{{01..N}}_by_epoch_{{validation,test}}.png"
                )
                if FeatureMapConfig.from_dict(plot_cfg.get("feature_maps")).enabled:
                    tqdm.write(
                        f"  {epoch_tag} | saved feature maps -> "
                        f".../feature_maps/{{validation,test}}/epoch_{epoch_no:04d}/ (+ latest/)"
                    )

            # 6g. Save best checkpoint when validation metric improves.
            if improved:
                best_score = ckpt_score
                best_epoch = epoch_no
                epochs_without_improvement = 0
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch_no}, best_path)
                # Waveform PNG export is expensive (full val+test infer + 300dpi).
                # Only refresh waveforms/best on plot_interval; best.pt still updates every improve.
                if monitor.should_plot(epoch_no):
                    _save_best_waveforms(
                        monitor=monitor,
                        adapter=adapter,
                        model=model,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        device=device,
                        best_epoch_no=best_epoch,
                    )
                    tqdm.write(f"  {epoch_tag} | saved best waveforms -> .../waveforms/{{validation,test}}/best/")
                    if FeatureMapConfig.from_dict(plot_cfg.get("feature_maps")).enabled:
                        tqdm.write(
                            f"  {epoch_tag} | saved best feature maps -> .../feature_maps/{{validation,test}}/best/"
                        )
            else:
                epochs_without_improvement += 1

            # 6h. Optional early stopping.
            can_early_stop = early_stop_min_epochs <= 0 or epoch_no >= early_stop_min_epochs
            if (
                can_early_stop
                and early_stop_patience > 0
                and epochs_without_improvement >= early_stop_patience
            ):
                monitor_label = str(train_cfg.get("checkpoint_monitor", monitor_key))
                tqdm.write(
                    f"{epoch_tag} | early stop — no {monitor_label} improvement "
                    f"for {early_stop_patience} epochs (best epoch {best_epoch})"
                )
                break

        # Step 8:
        # Save the final history file and close out the live monitor state.
        total_training_sec = time.perf_counter() - training_started
        epochs_completed = len(history)
        last_epoch = int(history[-1]["epoch"]) if history else 0
        # Early stop / natural end may land between plot_interval epochs.
        # Force one final latest + best waveform refresh so plots are not stale.
        if plot_cfg.get("enabled") is not False and history:
            need_final_plots = not monitor.should_plot(last_epoch)
            if need_final_plots:
                _save_latest_waveforms(
                    monitor=monitor,
                    adapter=adapter,
                    model=model,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    epoch_no=last_epoch,
                    best_epoch=best_epoch,
                )
            if best_epoch > 0 and best_path.is_file():
                # Reload best weights so waveforms/best match best.pt (not last epoch).
                ckpt = torch.load(best_path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"], strict=True)
                _save_best_waveforms(
                    monitor=monitor,
                    adapter=adapter,
                    model=model,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    best_epoch_no=best_epoch,
                )
            if need_final_plots or best_epoch > 0:
                tqdm.write(
                    f"Final plot update at epoch {last_epoch} "
                    f"(best epoch {best_epoch}; between plot_interval / end)"
                )
        timing_summary = {
            "total_seconds": total_training_sec,
            "total_formatted": _format_duration(total_training_sec),
            "epochs_completed": epochs_completed,
            "best_epoch": best_epoch,
            "best_score": float(best_score) if best_epoch > 0 else None,
            "checkpoint_monitor": monitor_key,
            "avg_epoch_seconds": total_training_sec / max(epochs_completed, 1),
            "avg_epoch_formatted": _format_duration(total_training_sec / max(epochs_completed, 1)),
            **param_stats,
            "checkpoint_file": best_path.name,
            "checkpoint_size_mb": checkpoint_size_mb(best_path),
        }
        with open(run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with open(run_dir / "training_time.json", "w", encoding="utf-8") as f:
            json.dump(timing_summary, f, indent=2)
        save_run_manifest(run_dir / "run_manifest.json", timing_summary)
        ckpt_mb = timing_summary["checkpoint_size_mb"]
        ckpt_note = f"checkpoint {ckpt_mb:.2f} MB" if ckpt_mb is not None else "checkpoint n/a"
        tqdm.write(
            f"Training finished in {timing_summary['total_formatted']} "
            f"({epochs_completed} epochs, best epoch {best_epoch}, "
            f"avg {_format_duration(timing_summary['avg_epoch_seconds'])}/epoch) | "
            f"params {format_parameter_count(param_stats['parameters_total'])} | "
            f"{ckpt_note}",
        )
        monitor.finalize(best_epoch=best_epoch, last_epoch=last_epoch)
    finally:
        monitor.close()

    return best_path


def evaluate_model(
    adapter,
    checkpoint: Path,
    run_dir: Path,
    split: str = "test",
    *,
    show_cost_summary: bool = True,
) -> Path:
    """Run inference from a saved checkpoint and save results.

    Step by step:
        1. Build the model on CPU/GPU
        2. Load checkpoint weights
        3. Build the requested dataloader
        4. Ask the adapter to generate a PredictionBundle
        5. Save the raw bundle arrays
        6. Compute and save metrics
        7. Save waveform plots for inspection
    """
    # Step 1-2:
    # Build the architecture and load the saved weights.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, tensor_dtype = resolve_tensor_dtype(adapter.model_cfg)
    model = adapter.build_model(device)
    if tensor_dtype == torch.float64:
        model = model.double()
    ckpt = torch.load(checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(
            f"Checkpoint load (strict=False): missing={list(missing)} unexpected={list(unexpected)}",
            flush=True,
        )
    model.eval()

    # Step 3-4:
    # Build the requested split loader and run model inference through the adapter.
    loader = adapter.build_dataloader(split)
    bundle = adapter.predict_dataloader(model, loader, device, split=split)

    bundle, state_calibration = maybe_calibrate_and_apply(
        bundle,
        adapter.model_cfg,
        run_dir,
        split,
    )
    if state_calibration is not None:
        thresholds = state_calibration.get("thresholds", {})
        summary = ", ".join(
            f"{app}={float(thresholds.get(app, 0.5)):.2f}" for app in bundle.appliances
        )
        print(f"State calibration applied ({split}): {summary}", flush=True)

    # Step 5:
    # Save final predictions so later code can reload them without rerunning the model.
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / f"{split}_predictions.npz"
    bundle.save(pred_path)

    # Step 6:
    # Compute the standard metrics from the shared PredictionBundle format.
    power_postprocess = resolve_power_postprocess(
        adapter.experiment,
        bundle.appliances,
        adapter.model_cfg,
    )
    metrics = evaluate_bundle(
        bundle,
        sae_period=int(adapter.experiment["evaluation"].get("sae_period", 1200)),
        on_threshold_watts=_state_eval_thresholds(adapter.model_cfg, adapter.experiment, bundle.appliances),
        state_label_source=get_state_label_source(adapter.model_cfg),
        power_postprocess=power_postprocess,
    )
    metrics_path = run_dir / f"{split}_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    # Also archive under metrics_by_epoch for comparison with mid-training tables.
    ckpt_epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
    if ckpt_epoch > 0:
        archive_dir = run_dir / "metrics_by_epoch" / f"evaluate_epoch_{ckpt_epoch:04d}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(archive_dir / f"{split}_metrics.csv", index=False)

    # Step 7:
    # Waveform plots always use dataset CSV *_on labels for true ON periods.
    # Training/F1 above may still follow data.state_label_source in model yaml.
    plot_cfg = adapter.model_cfg.get("training", {}).get("plots", {})
    # Keep training-time epoch_* waveform history; only refresh the evaluate/ slot.
    # On Windows, rmtree(dir) then immediate recreate can race (delayed deletes) and
    # cause FileNotFoundError on savefig — wipe children or replace via rename.
    waveform_dir = run_dir / "waveforms" / split / "evaluate"
    _reset_dir(waveform_dir)

    raw_period = plot_cfg.get("on_period_samples", 0)
    period_samples = None if raw_period is None or int(raw_period) <= 0 else int(raw_period)
    waveform_true_on = dataset_on_labels_for_bundle(
        adapter._data_loader(),
        split,
        len(bundle.y_true_watts),
        bundle.csv_timesteps,
    )

    y_true_watts, y_pred_watts = apply_power_postprocess_pair(
        bundle.y_true_watts,
        bundle.y_pred_watts,
        power_postprocess,
    )
    aggregate = bundle_aggregate_watts(
        adapter._data_loader(),
        split,
        n_points=len(y_true_watts),
        csv_timesteps=bundle.csv_timesteps,
    )
    y_true_plot = bundle_csv_appliance_watts(
        adapter._data_loader(),
        split,
        n_points=len(y_true_watts),
        csv_timesteps=bundle.csv_timesteps,
    )
    if y_true_plot is None:
        y_true_plot = y_true_watts
    # Match training label source strictly (csv → CSV *_on shade; never power>thr).
    state_src = get_state_label_source(adapter.model_cfg)
    on_thresholds = _state_eval_thresholds(adapter.model_cfg, adapter.experiment, bundle.appliances)
    saved = save_appliance_on_waveforms(
        waveform_dir,
        appliances=bundle.appliances,
        y_true_watts=y_true_plot,
        y_pred_watts=y_pred_watts,
        y_true_on=waveform_true_on,
        y_pred_on=bundle.y_pred_on,
        on_thresholds_watts=on_thresholds,
        state_label_source=state_src,
        aggregate=aggregate,
        csv_timesteps=bundle.csv_timesteps,
        n_periods=int(plot_cfg.get("plot_on_periods", 5)),
        period_samples=period_samples,
        full_cycle_appliances=plot_cfg.get("full_cycle_appliances"),
        margin_min=int(plot_cfg.get("on_period_margin_min", 40)),
        margin_frac=float(plot_cfg.get("on_period_margin_frac", 0.08)),
        figsize=float(plot_cfg.get("waveform_figsize", 5.5)),
        dynamic_figsize=bool(plot_cfg.get("waveform_dynamic_figsize", True)),
        dpi=int(plot_cfg.get("waveform_dpi", 300)),
        context_scale=float(plot_cfg.get("waveform_context_scale", 10)),
        rng=np.random.default_rng(int(adapter.cfg.get("seed", 0))),
        file_prefix="on",
        title_prefix=f"{adapter.name} {split} — ",
    )

    print_evaluation_report(metrics, run_dir, split=split, show_cost_summary=show_cost_summary)
    print(f"Saved {len(saved)} waveform PNGs under {waveform_dir}/<appliance>/")

    feature_cfg = FeatureMapConfig.from_dict(plot_cfg.get("feature_maps"))
    if feature_cfg.enabled:
        feature_dir = run_dir / "feature_maps" / split
        save_feature_maps(
            adapter,
            model,
            loader,
            feature_dir,
            split=split,
            device=device,
            cfg=feature_cfg,
        )

    return pred_path

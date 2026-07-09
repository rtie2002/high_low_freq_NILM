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

import json
import shutil
from pathlib import Path

from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from adapters.common import StepOutput
from adapters.dataloader import (
    NILMDataLoader,
    _resolve_input_length,
    _target_mode,
    get_normalization_cfg,
    get_state_label_source,
    get_state_threshold,
    resolve_mains_column,
)
from evaluation.live_monitor import LiveTrainingMonitor
from evaluation.metrics import evaluate_bundle
from evaluation.plots import save_appliance_on_waveforms


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
    if thr := data_cfg.get("state_threshold_watts"):
        source = str(data_cfg.get("state_label_source", "auto")).lower()
        if source == "threshold":
            lines.append(f"state labels: rebuilt from power > {thr} W")
        elif source == "csv":
            lines.append(f"state labels: CSV *_on columns (threshold {thr} W ignored)")
        else:
            lines.append(f"state labels: auto (threshold {thr} W if requested by model)")
    return lines


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
    thin = "-" * width

    print(bar, flush=True)
    print(f"EXPERIMENT: {experiment_id}  |  MODEL: {model_name}  |  DEVICE: {device}", flush=True)
    print(bar, flush=True)
    print(f"Appliances ({len(appliances)}): {', '.join(appliances)}", flush=True)
    print(flush=True)
    print("Windowing", flush=True)
    print(f"  input length (effective): {_resolve_input_length(w)}", flush=True)
    print(f"  output length:            {int(w.get('output_window_length', 1))}", flush=True)
    print(f"  output alignment:         {w.get('output_alignment', 'end')}", flush=True)
    print(f"  train stride:             {int(w['input_stride'])}", flush=True)
    print(f"  eval stride:              {int(w.get('eval_stride', w['input_stride']))}", flush=True)
    print(f"  train target mode:        {_target_mode(w, 'train')}", flush=True)
    print(f"  eval target mode:         {_target_mode(w, 'validation')}", flush=True)
    print(f"  batch size:               {batch_size}", flush=True)
    print(f"  epochs:                   {epochs}", flush=True)
    if train_cfg.get("use_amp"):
        print(f"  mixed precision:          {train_cfg.get('amp_dtype', 'bf16')}", flush=True)
    print(f"  dataloader workers:       {int(train_cfg.get('num_workers', 0))}", flush=True)
    if train_cfg.get("checkpoint_monitor"):
        print(f"  checkpoint monitor:         {train_cfg['checkpoint_monitor']}", flush=True)
    print(flush=True)
    print("Data", flush=True)
    for line in _data_preprocess_note(model_cfg, experiment_cfg):
        print(f"  {line}", flush=True)

    for split in ("train", "validation", "test"):
        info = data_loader.describe_split(split, batch_size=batch_size)
        print(flush=True)
        print(thin, flush=True)
        print(f"SPLIT: {split.upper()}", flush=True)
        print(f"  csv file:      {info['csv_path']}", flush=True)
        print(f"  timesteps:     {info['timesteps']:,}  (rows after dropna)", flush=True)
        print(f"  input length:  {info['input_length']}", flush=True)
        print(f"  output length: {info['output_length']}", flush=True)
        print(f"  stride:        {info['stride']}", flush=True)
        print(f"  target mode:   {info['target_mode']}", flush=True)
        print(f"  windows:       {info['windows']:,}", flush=True)
        print(f"  batches:       {info['batches']:,}  (@ batch_size={batch_size})", flush=True)
        if split == "train":
            print(f"  used in:       training ({info['batches']:,} batches/epoch)", flush=True)
        elif split == "validation":
            print("  used in:       validation + checkpoint selection", flush=True)
        else:
            print("  used in:       final test evaluation (after training)", flush=True)

    print(flush=True)
    print(bar, flush=True)
    print(flush=True)


def _aggregate_logs(log_keys: list[str], n_batches: int, totals: dict[str, float]) -> dict[str, float]:
    """Convert accumulated batch logs into epoch-average logs."""
    return {k: totals.get(k, 0.0) / max(n_batches, 1) for k in log_keys}


def _state_f1_logs(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute validation F1 from collected ON/OFF predictions.

    This helper is generic for MATNILM, MultiNILM, and future models.
    Each adapter only needs to return these two tensors in StepOutput.aux:

        pred_state -> predicted ON/OFF labels
        true_state -> true ON/OFF labels

    The runner then reports:

        val_f1   : mean F1 over appliances
        val_maf1 : same value, kept for compatibility
        val_mif1 : micro F1 over all appliance/timestep decisions
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    scores = []
    total_tp = total_fp = total_fn = 0

    for app_i in range(y_true.shape[1]):
        yt = y_true[:, app_i]
        yp = y_pred[:, app_i]
        tp = int(np.logical_and(yt, yp).sum())
        fp = int(np.logical_and(~yt, yp).sum())
        fn = int(np.logical_and(yt, ~yp).sum())
        total_tp += tp
        total_fp += fp
        total_fn += fn
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))

    macro_f1 = float(np.mean(scores)) if scores else 0.0
    micro_f1 = float(2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1))
    return {"val_f1": macro_f1, "val_maf1": macro_f1, "val_mif1": micro_f1}


def _epoch_state_arrays(adapter, aux_batches: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Build validation ON/OFF arrays from the source selected in model yaml."""
    source = get_state_label_source(adapter.model_cfg)
    if source == "threshold":
        threshold = _state_eval_thresholds(adapter.model_cfg, adapter.experiment, adapter.cfg["appliances"])
        if threshold is None:
            raise ValueError("threshold state_label_source requires data.state_threshold_watts")
        loader = adapter._data_loader()
        y_pred = np.concatenate(aux_batches["pred_power"], axis=0)
        y_true = np.concatenate(aux_batches["true_power"], axis=0)
        if y_pred.ndim > 2:
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = loader.denorm_to_watts(y_pred)
        y_true = loader.denorm_to_watts(y_true)
        threshold = np.asarray(threshold, dtype=np.float32)
        return (y_true > threshold).astype(np.int32), (y_pred > threshold).astype(np.int32)

    z_pred = np.concatenate(aux_batches["pred_state"], axis=0)
    z_true = np.concatenate(aux_batches["true_state"], axis=0)
    if z_pred.ndim > 2:
        z_pred = z_pred.reshape(-1, z_pred.shape[-1])
        z_true = z_true.reshape(-1, z_true.shape[-1])
    return z_true.astype(np.int32), z_pred.astype(np.int32)


def _batch_to_device(batch, device: torch.device):
    """Move every tensor inside a batch to GPU/CPU.

    Dataloader batches are usually:

        (x, y, z)

    but this helper also supports nested tuples/lists/dicts, so future models
    can use richer batch formats without changing runner.py.
    """
    if isinstance(batch, (tuple, list)):
        return type(batch)(_batch_to_device(item, device) for item in batch)
    if isinstance(batch, dict):
        return {key: _batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch


def _resolve_checkpoint_monitor(train_cfg: dict) -> tuple[str, str, float]:
    """Decide which validation metric selects best.pt.

    Config example:

        checkpoint_monitor: val_mae

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
    }
    key = aliases.get(monitor, monitor)
    if key == "val_f1":
        return key, "max", float("-inf")
    return key, "min", float("inf")


def _epoch_score(monitor_key: str, logs: dict[str, float]) -> float:
    """Read the checkpoint metric from one epoch's validation logs."""
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

    Validation call:

        _run_epoch(..., train=False, collect_states=True)

        - model.eval()
        - gradients disabled
        - adapter.step(...) computes forward pass and loss
        - no backward
        - no optimizer update
        - collects predicted/true ON/OFF states for F1

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
            # Move x/y/z tensors to the same device as the model.
            batch = _batch_to_device(batch, device)

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
                # Model-specific one-batch logic.
                step: StepOutput = adapter.step(model, loss_fn, batch)

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

            # Validation F1 is computed at epoch end using the configured state source.
            if collect_states and step.aux:
                for key in aux_batches:
                    if key in step.aux:
                        aux_batches[key].append(step.aux[key].detach().cpu().numpy())

            n_batches += 1
            if n_batches % 20 == 0 or n_batches == n_total:
                pbar.set_postfix(loss=f"{step.logs.get('loss', 0.0):.4f}")

    # Convert accumulated batch values to one average value per epoch.
    logs = _aggregate_logs(log_keys, n_batches, totals)

    # Add val_f1/val_maf1/val_mif1 for validation checkpointing and monitoring.
    if collect_states and aux_batches["true_state"]:
        z_true, z_pred = _epoch_state_arrays(adapter, aux_batches)
        logs.update(_state_f1_logs(z_true, z_pred))
    return logs


def _setup_device_amp_scaler(train_cfg: dict) -> tuple[torch.device, bool, torch.dtype, torch.cuda.amp.GradScaler]:
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
    use_amp = bool(train_cfg.get("use_amp", False)) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(train_cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    return device, use_amp, amp_dtype, scaler


def _resolve_seed(adapter, train_cfg: dict, seed: int | None) -> int:
    """Resolve the final training seed.

    Priority:
        1. Explicit --seed from the command line
        2. experiment.yaml seed
        3. model training seed
    """
    if seed is None:
        seed = adapter.cfg.get("seed") or train_cfg.get("seed")
    if seed is None:
        raise ValueError("Set seed in experiment yaml, model training config, or pass --seed")
    return int(seed)


def _build_model(adapter, device: torch.device, train_cfg: dict, init_checkpoint: Path | None) -> torch.nn.Module:
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
    if bool(train_cfg.get("torch_compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model


def _build_loss_and_optimizer(adapter, model: torch.nn.Module, device: torch.device):
    """Build the loss object, move it to device if needed, and create optimizer."""
    loss_fn = adapter.build_loss()
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)
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
    scheduler_raw = str(train_cfg.get("scheduler_monitor", monitor_key)).lower()
    scheduler_aliases = {
        "val_f1": "val_f1",
        "val_maf1": "val_f1",
        "val_loss": "loss",
        "val_mae": "mae",
    }
    return scheduler_aliases.get(scheduler_raw, scheduler_raw)


def _evaluation_on_thresholds(experiment_cfg: dict, appliances: list[str]) -> float | np.ndarray:
    """Return one global ON threshold or one threshold per appliance."""
    evaluation = experiment_cfg.get("evaluation", {})
    if per_app := evaluation.get("on_thresholds_watts"):
        return np.asarray([float(per_app[app]) for app in appliances], dtype=np.float32)
    return float(evaluation.get("on_threshold_watts", 15.0))


def _state_eval_thresholds(model_cfg: dict, experiment_cfg: dict, appliances: list[str]) -> float | np.ndarray | None:
    """Choose evaluation thresholds from the model yaml when threshold mode is enabled."""
    source = get_state_label_source(model_cfg)
    if source != "threshold":
        return None
    thr = get_state_threshold(model_cfg)
    if thr is not None:
        return float(thr)
    return _evaluation_on_thresholds(experiment_cfg, appliances)


def _waveform_dataset_on_labels(adapter, split: str, n_points: int) -> np.ndarray:
    """Dataset CSV *_on labels for waveform plots only.

    Waveforms always use the labels stored in the CSV files, even when training
    and F1 metrics follow data.state_label_source=threshold in model yaml.
    """
    return adapter._data_loader().window_flattened_csv_states(split, n_points)


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

    # Step 2:
    # Choose device and optional AMP behavior.
    device, use_amp, amp_dtype, scaler = _setup_device_amp_scaler(train_cfg)

    # Step 3:
    # Build model, optional init checkpoint, loss, and optimizer.
    model = _build_model(adapter, device, train_cfg, init_checkpoint)
    loss_fn, optim, sched = _build_loss_and_optimizer(adapter, model, device)

    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        amp_label = str(train_cfg.get("amp_dtype", "bf16")) if use_amp else "off"
        workers = int(train_cfg.get("num_workers", 0))
        tqdm.write(f"GPU: {name} | AMP: {amp_label} | DataLoader workers: {workers}")

    # Step 4:
    # Build the standard train/validation/test dataloaders.
    print("Loading CSV splits into memory (train, val, test)...", flush=True)
    train_loader, val_loader, test_loader = _build_loaders(adapter)

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
    epochs_without_improvement = 0

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
        for epoch in range(epochs):
            epoch_no = epoch + 1
            epoch_tag = f"Epoch {epoch_no}/{epochs}"

            # 6a. Training epoch
            # This is where model weights are updated.
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
            val_mae = float(val_logs.get("mae", 0.0))

            # 6c. Optional scheduler step after validation.
            sched_metric = _epoch_score(scheduler_key, val_logs)
            if sched is not None:
                sched.step(sched_metric)

            # 6d. Save training history for later plotting/inspection.
            history.append(
                {
                    "epoch": epoch_no,
                    **{f"train_{k}": v for k, v in train_logs.items()},
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "val_maf1": float(val_logs.get("val_maf1", val_f1)),
                    "val_mif1": float(val_logs.get("val_mif1", 0.0)),
                }
            )
            monitor.append_epoch(epoch=epoch_no, train_logs=train_logs, val_logs=val_logs)

            # 6e. Check if this epoch should replace best.pt.
            improved = False
            ckpt_score = _epoch_score(monitor_key, val_logs)
            if _is_better(ckpt_score, best_score, monitor_mode):
                improved = True

            tqdm.write(
                f"{epoch_tag} | train_loss={train_logs['loss']:.4f} | "
                f"val_loss={val_loss:.4f} | val_f1={val_f1:.4f} | "
                f"val_mae={val_mae:.4f}"
                + (" | new best" if improved else "")
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
                tqdm.write(f"  {epoch_tag} | saved latest waveforms -> .../waveforms/{{validation,test}}/latest/")

            # 6g. Save best checkpoint when validation metric improves.
            if improved:
                best_score = ckpt_score
                best_epoch = epoch_no
                epochs_without_improvement = 0
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch_no}, best_path)
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
            else:
                epochs_without_improvement += 1

            # 6h. Optional early stopping.
            if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                monitor_label = str(train_cfg.get("checkpoint_monitor", monitor_key))
                tqdm.write(
                    f"{epoch_tag} | early stop â€” no {monitor_label} improvement "
                    f"for {early_stop_patience} epochs (best epoch {best_epoch})"
                )
                break

        # Step 8:
        # Save the final history file and close out the live monitor state.
        with open(run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        monitor.finalize(best_epoch=best_epoch)
    finally:
        monitor.close()

    return best_path


def evaluate_model(adapter, checkpoint: Path, run_dir: Path, split: str = "test") -> Path:
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
    model = adapter.build_model(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Step 3-4:
    # Build the requested split loader and run model inference through the adapter.
    loader = adapter.build_dataloader(split)
    bundle = adapter.predict_dataloader(model, loader, device, split=split)

    # Step 5:
    # Save the raw predictions so later code can reload them without rerunning the model.
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / f"{split}_predictions.npz"
    bundle.save(pred_path)

    # Step 6:
    # Compute the standard metrics from the shared PredictionBundle format.
    metrics = evaluate_bundle(
        bundle,
        sae_period=int(adapter.experiment["evaluation"].get("sae_period", 1200)),
        on_threshold_watts=_state_eval_thresholds(adapter.model_cfg, adapter.experiment, bundle.appliances),
        state_label_source=get_state_label_source(adapter.model_cfg),
    )
    metrics_path = run_dir / f"{split}_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    # Step 7:
    # Waveform plots always use dataset CSV *_on labels for true ON periods.
    # Training/F1 above may still follow data.state_label_source in model yaml.
    plot_cfg = adapter.model_cfg.get("training", {}).get("plots", {})
    waveform_dir = run_dir / "waveforms" / split
    if waveform_dir.exists():
        shutil.rmtree(waveform_dir)

    raw_period = plot_cfg.get("on_period_samples", 0)
    period_samples = None if raw_period is None or int(raw_period) <= 0 else int(raw_period)
    waveform_true_on = _waveform_dataset_on_labels(adapter, split, len(bundle.y_true_watts))

    saved = save_appliance_on_waveforms(
        waveform_dir,
        appliances=bundle.appliances,
        y_true_watts=bundle.y_true_watts,
        y_pred_watts=bundle.y_pred_watts,
        y_true_on=waveform_true_on,
        y_pred_on=bundle.y_pred_on,
        csv_timesteps=bundle.csv_timesteps,
        n_periods=int(plot_cfg.get("plot_on_periods", 5)),
        period_samples=period_samples,
        full_cycle_appliances=plot_cfg.get("full_cycle_appliances"),
        margin_min=int(plot_cfg.get("on_period_margin_min", 40)),
        margin_frac=float(plot_cfg.get("on_period_margin_frac", 0.08)),
        figsize=float(plot_cfg.get("waveform_figsize", 5.5)),
        dynamic_figsize=bool(plot_cfg.get("waveform_dynamic_figsize", True)),
        dpi=int(plot_cfg.get("waveform_dpi", 300)),
        rng=np.random.default_rng(int(adapter.cfg.get("seed", 0))),
        file_prefix="on",
        title_prefix=f"{adapter.name} {split} â€” ",
    )

    per_app = metrics[metrics["appliance"] != "overall"]
    overall = metrics[metrics["appliance"] == "overall"]
    print(per_app[["appliance", "mae", "sae", "f1"]].to_string(index=False))
    if not overall.empty:
        row = overall.iloc[0]
        print(
            f"overall  mae={row['mae']:.4f}  sae={row['sae']:.4f}  "
            f"f1={row['f1']:.4f}  micro_f1={row['micro_f1']:.4f}"
        )
    print(f"Saved {len(saved)} waveform PNGs under {waveform_dir}/<appliance>/")
    return pred_path

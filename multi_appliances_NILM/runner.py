"""Train and evaluate loops — called from main.py."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from adapters.dataloader import print_training_data_summary
from adapters.types import StepOutput
from evaluation.live_monitor import LiveTrainingMonitor
from evaluation.metrics import evaluate_bundle
from evaluation.plots import save_appliance_on_waveforms


def seed_everything(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _aggregate_logs(log_keys: list[str], n_batches: int, totals: dict[str, float]) -> dict[str, float]:
    return {k: totals.get(k, 0.0) / max(n_batches, 1) for k in log_keys}


def _batch_to_device(batch, device: torch.device):
    if isinstance(batch, (tuple, list)):
        return type(batch)(_batch_to_device(item, device) for item in batch)
    if isinstance(batch, dict):
        return {key: _batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch


def _resolve_checkpoint_monitor(train_cfg: dict) -> tuple[str, str, float]:
    """Return (metric_key, mode, initial_best)."""
    monitor = str(train_cfg.get("checkpoint_monitor", "val_loss")).lower()
    aliases = {
        "val_f1": "val_f1",
        "val_maf1": "val_f1",
        "val_loss": "val_loss",
    }
    key = aliases.get(monitor, monitor)
    if key == "val_f1":
        return key, "max", float("-inf")
    return key, "min", float("inf")


def _is_better(score: float, best: float, mode: str) -> bool:
    return score > best if mode == "max" else score < best


def _resolve_amp_dtype(train_cfg: dict) -> torch.dtype:
    name = str(train_cfg.get("amp_dtype", "bf16")).lower()
    return torch.bfloat16 if name == "bf16" else torch.float16


def _configure_cuda(train_cfg: dict) -> None:
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
    log_keys = log_keys or ["loss", "loss_state", "loss_power", "mae"]
    totals = {k: 0.0 for k in log_keys}
    n_batches = 0
    state_preds: list[np.ndarray] = []
    state_trues: list[np.ndarray] = []

    if train:
        model.train()
    else:
        model.eval()

    try:
        n_total = len(loader)
    except TypeError:
        n_total = None

    phase = desc or ("train" if train else "val")
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
            batch = _batch_to_device(batch, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp and device.type == "cuda",
            ):
                step: StepOutput = adapter.training_step(model, loss_fn, batch)
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
            for k in log_keys:
                totals[k] += step.logs.get(k, 0.0)
            if collect_states and step.aux:
                state_preds.append(step.aux["pred_state"].detach())
                state_trues.append(step.aux["true_state"].detach())
            n_batches += 1
            if n_batches % 20 == 0 or n_batches == n_total:
                pbar.set_postfix(loss=f"{step.logs.get('loss', 0.0):.4f}")

    logs = _aggregate_logs(log_keys, n_batches, totals)
    if collect_states and state_preds:
        from adapters.unet_metrics import compute_unet_state_f1

        z_pred = torch.cat(state_preds, dim=0).cpu().numpy()
        z_true = torch.cat(state_trues, dim=0).cpu().numpy()
        logs.update(compute_unet_state_f1(z_true, z_pred))
    return logs


def train_model(
    adapter,
    run_dir: Path,
    *,
    epochs: int | None = None,
    seed: int | None = None,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = adapter.model_cfg["training"]
    _configure_cuda(train_cfg)
    use_amp = bool(train_cfg.get("use_amp", False)) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(train_cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    model = adapter.build_model(device)
    if bool(train_cfg.get("torch_compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    loss_fn = adapter.build_loss()
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)
    optim, sched = adapter.configure_optimizer(model)

    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        amp_label = str(train_cfg.get("amp_dtype", "bf16")) if use_amp else "off"
        workers = int(train_cfg.get("num_workers", 0))
        tqdm.write(f"GPU: {name} | AMP: {amp_label} | DataLoader workers: {workers}")
    print("Loading CSV splits into memory (train, val, test)...", flush=True)
    train_loader = adapter.build_dataloader("train")
    val_loader = adapter.build_dataloader("validation")
    test_loader = adapter.build_dataloader("test")

    train_cfg = adapter.model_cfg["training"]
    epochs = epochs or int(train_cfg["epochs"])
    if seed is None:
        seed = adapter.cfg.get("seed") or train_cfg.get("seed")
    if seed is None:
        raise ValueError("Set seed in experiment yaml, model training config, or pass --seed")

    data_loader = adapter._data_loader()
    print_training_data_summary(
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
    seed_everything(int(seed))

    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    monitor_key, monitor_mode, best_score = _resolve_checkpoint_monitor(train_cfg)
    scheduler_key = str(train_cfg.get("scheduler_monitor", monitor_key)).lower()
    if scheduler_key in {"val_f1", "val_maf1"}:
        scheduler_key = "val_f1"
    best_epoch = 0
    history = []
    appliances = adapter.cfg["appliances"]
    monitor = LiveTrainingMonitor(
        run_dir,
        model_name=adapter.name,
        appliances=appliances,
        plot_cfg=plot_cfg,
        seed=int(seed),
    )
    grad_clip = float(train_cfg.get("gradient_clip", 0.0))

    try:
        for epoch in range(epochs):
            epoch_no = epoch + 1
            epoch_tag = f"Epoch {epoch_no}/{epochs}"
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
            val_logs = _run_epoch(
                adapter,
                model,
                loss_fn,
                val_loader,
                device=device,
                train=False,
                collect_states=(monitor_key == "val_f1" or scheduler_key == "val_f1"),
                desc=f"{epoch_tag} | val",
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            val_loss = float(val_logs["loss"])
            val_f1 = float(val_logs.get("val_f1", 0.0))

            sched_metric = val_f1 if scheduler_key == "val_f1" else val_loss
            if sched is not None:
                sched.step(sched_metric)

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

            improved = False
            ckpt_score = val_f1 if monitor_key == "val_f1" else val_loss
            if _is_better(ckpt_score, best_score, monitor_mode):
                improved = True

            tqdm.write(
                f"{epoch_tag} | train_loss={train_logs['loss']:.4f} | "
                f"val_loss={val_loss:.4f} | val_f1={val_f1:.4f} | "
                f"val_mae={val_logs.get('mae', 0.0):.4f}"
                + (" | new best" if improved else "")
            )

            if monitor.should_plot(epoch_no):
                monitor.save_loss_plots(epoch=epoch_no, best_epoch=best_epoch or None)
                monitor.save_waveforms(
                    adapter,
                    model,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    epoch=epoch_no,
                )
                tqdm.write(f"  {epoch_tag} | saved latest waveforms -> .../waveforms/{{validation,test}}/latest/")

            if improved:
                best_score = ckpt_score
                best_epoch = epoch_no
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch_no}, best_path)
                if monitor.should_plot(epoch_no):
                    monitor.save_waveforms(
                        adapter,
                        model,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        device=device,
                        epoch=best_epoch,
                        include_best=True,
                    )
                    tqdm.write(f"  {epoch_tag} | saved best waveforms -> .../waveforms/{{validation,test}}/best/")

        with open(run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        monitor.finalize(best_epoch=best_epoch)
    finally:
        monitor.close()

    return best_path


def evaluate_model(adapter, checkpoint: Path, run_dir: Path, split: str = "test") -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = adapter.build_model(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    loader = adapter.build_dataloader(split)
    bundle = adapter.predict_dataloader(model, loader, device, split=split)

    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / f"{split}_predictions.npz"
    bundle.save(pred_path)

    metrics = evaluate_bundle(
        bundle,
        sae_period=int(adapter.experiment["evaluation"].get("sae_period", 1200)),
        on_threshold_watts=float(adapter.experiment["evaluation"].get("on_threshold_watts", 15.0)),
    )
    metrics_path = run_dir / f"{split}_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    plot_cfg = adapter.model_cfg.get("training", {}).get("plots", {})
    waveform_dir = run_dir / "waveforms" / split
    if waveform_dir.exists():
        shutil.rmtree(waveform_dir)

    raw_period = plot_cfg.get("on_period_samples", 0)
    period_samples = None if raw_period is None or int(raw_period) <= 0 else int(raw_period)

    saved = save_appliance_on_waveforms(
        waveform_dir,
        appliances=bundle.appliances,
        y_true_watts=bundle.y_true_watts,
        y_pred_watts=bundle.y_pred_watts,
        y_true_on=bundle.y_true_on,
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
        title_prefix=f"{adapter.name} {split} — ",
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

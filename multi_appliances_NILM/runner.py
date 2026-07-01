"""Train and evaluate loops — called from main.py."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

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
) -> dict[str, float]:
    log_keys = log_keys or ["loss", "loss_state", "loss_power", "mae"]
    totals = {k: 0.0 for k in log_keys}
    n_batches = 0

    if train:
        model.train()
    else:
        model.eval()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = _batch_to_device(batch, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            step: StepOutput = adapter.training_step(model, loss_fn, batch)
            if train:
                step.loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            for k in log_keys:
                totals[k] += step.logs.get(k, 0.0)
            n_batches += 1

    return _aggregate_logs(log_keys, n_batches, totals)


def train_model(
    adapter,
    run_dir: Path,
    *,
    epochs: int | None = None,
    seed: int | None = None,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = adapter.build_model(device)
    loss_fn = adapter.build_loss()
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)
    optim, sched = adapter.configure_optimizer(model)

    train_loader = adapter.build_dataloader("train")
    val_loader = adapter.build_dataloader("validation")
    test_loader = adapter.build_dataloader("test")
    train_cfg = adapter.model_cfg["training"]
    plot_cfg = train_cfg.get("plots", {})
    epochs = epochs or int(train_cfg["epochs"])
    if seed is None:
        seed = adapter.cfg.get("seed") or train_cfg.get("seed")
    if seed is None:
        raise ValueError("Set seed in experiment yaml, model training config, or pass --seed")
    seed_everything(int(seed))

    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    best_score = float("inf")
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
            train_logs = _run_epoch(
                adapter,
                model,
                loss_fn,
                train_loader,
                device=device,
                train=True,
                optimizer=optim,
                grad_clip=grad_clip,
            )
            val_logs = _run_epoch(adapter, model, loss_fn, val_loader, device=device, train=False)
            val_loss = float(val_logs["loss"])

            if sched is not None:
                sched.step(val_loss)

            history.append(
                {"epoch": epoch, **{f"train_{k}": v for k, v in train_logs.items()}, "val_loss": val_loss}
            )
            monitor.append_epoch(epoch=epoch_no, train_logs=train_logs, val_logs=val_logs)
            print(
                f"epoch {epoch:03d}  train_loss={train_logs['loss']:.4f}  "
                f"val_loss={val_loss:.4f}  val_mae={val_logs.get('mae', 0.0):.4f}"
            )

            if monitor.should_plot(epoch_no):
                monitor.save_loss_plots(epoch=epoch_no, best_epoch=best_epoch or None)
                saved = monitor.save_waveforms(
                    adapter,
                    model,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    epoch=epoch_no,
                )
                print(
                    f"  updated live plots: {monitor.live_history_png.name}, "
                    f"{monitor.live_loss_png.name}, {len(saved)} waveform PNGs in {monitor.waveforms_dir}"
                )

            if val_loss < best_score:
                best_score = val_loss
                best_epoch = epoch_no
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, best_path)
                if monitor.should_plot(epoch_no):
                    best_saved = monitor.save_waveforms(
                        adapter,
                        model,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        device=device,
                        epoch=best_epoch,
                        include_best=True,
                    )
                    print(f"  updated best waveforms: {len(best_saved)} PNGs under {monitor.waveforms_dir}/**/best/")

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
    bundle = adapter.predict_dataloader(model, loader, device)

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

    saved = save_appliance_on_waveforms(
        waveform_dir,
        appliances=bundle.appliances,
        y_true_watts=bundle.y_true_watts,
        y_pred_watts=bundle.y_pred_watts,
        y_true_on=bundle.y_true_on,
        y_pred_on=bundle.y_pred_on,
        n_periods=int(plot_cfg.get("plot_on_periods", 5)),
        period_samples=int(plot_cfg.get("on_period_samples", 400)),
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

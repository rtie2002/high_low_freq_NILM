"""
Train MATUDA (seq2point, FC-layer UDA, optional EGC-DA).

Remote example:
  C:\\Users\\PC\\anaconda3\\envs\\nilm\\python.exe scripts\\train_matuda.py ^
    --config configs\\matuda_m0_egc.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import DEFAULT_APPLIANCES, make_loaders  # noqa: E402
from src.matuda_loss import MATUDACriterion  # noqa: E402
from src.matuda_model import MATUDANet, count_parameters  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, loader, device, norm, appliances) -> dict:
    model.eval()
    k = len(appliances)
    sum_ae = np.zeros(k, dtype=np.float64)
    sum_se = np.zeros(k, dtype=np.float64)
    sum_true = np.zeros(k, dtype=np.float64)
    sum_pred = np.zeros(k, dtype=np.float64)
    n = 0
    tp = np.zeros(k, dtype=np.float64)
    fp = np.zeros(k, dtype=np.float64)
    fn = np.zeros(k, dtype=np.float64)

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].numpy()
        z = batch["z"].numpy()
        out = model(x)
        y_hat = out["powers"].cpu().numpy()
        z_hat = (torch.sigmoid(out["state_logits"]).cpu().numpy() >= 0.5).astype(
            np.float32
        )
        y_w = norm.denorm_power(y)
        y_hat_w = norm.denorm_power(y_hat)
        err = y_hat_w - y_w
        sum_ae += np.abs(err).sum(axis=0)
        sum_se += (err**2).sum(axis=0)
        sum_true += y_w.sum(axis=0)
        sum_pred += y_hat_w.sum(axis=0)
        n += y_w.shape[0]
        tp += ((z_hat == 1) & (z == 1)).sum(axis=0)
        fp += ((z_hat == 1) & (z == 0)).sum(axis=0)
        fn += ((z_hat == 0) & (z == 1)).sum(axis=0)

    mae = sum_ae / max(n, 1)
    rmse = np.sqrt(sum_se / max(n, 1))
    sae = np.abs(sum_pred - sum_true) / np.maximum(np.abs(sum_true), 1e-8)
    prec = tp / np.maximum(tp + fp, 1e-8)
    rec = tp / np.maximum(tp + fn, 1e-8)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-8)
    return {
        "mae_mean": float(mae.mean()),
        "rmse_mean": float(rmse.mean()),
        "sae_mean": float(sae.mean()),
        "f1_macro": float(f1.mean()),
        "precision_macro": float(prec.mean()),
        "recall_macro": float(rec.mean()),
        "mae_per_app": {a: float(v) for a, v in zip(appliances, mae)},
        "sae_per_app": {a: float(v) for a, v in zip(appliances, sae)},
        "f1_per_app": {a: float(v) for a, v in zip(appliances, f1)},
        "precision_per_app": {a: float(v) for a, v in zip(appliances, prec)},
        "recall_per_app": {a: float(v) for a, v in zip(appliances, rec)},
        "n": int(n),
    }


def train(cfg: dict) -> None:
    set_seed(int(cfg.get("seed", 2026)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.get("output_dir", ROOT / "results" / cfg["experiment_id"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    appliances = list(cfg.get("appliances", DEFAULT_APPLIANCES))
    loaders = make_loaders(
        Path(cfg["data_root"]),
        appliances=appliances,
        seq_len=int(cfg.get("seq_len", 599)),
        stride_train=int(cfg.get("stride_train", 30)),
        stride_eval=int(cfg.get("stride_eval", 60)),
        batch_size=int(cfg.get("batch_size", 64)),
        num_workers=int(cfg.get("num_workers", 0)),
        target_adapt_frac=float(cfg.get("target_adapt_frac", 0.7)),
    )
    norm = loaders["norm"]
    pos_weight = loaders["pos_weight"].to(device)
    print(f"ON rates (source): {loaders['on_rates']}", flush=True)
    print(f"pos_weight (cap=50): {pos_weight.tolist()}", flush=True)
    print(f"target_split: {loaders['target_split']}", flush=True)
    print(f"label_thresholds_W: {loaders['label_thresholds_watts']}", flush=True)
    print(f"use_gate={bool(cfg.get('use_gate', True))}", flush=True)

    model = MATUDANet(
        num_appliances=len(appliances),
        seq_len=int(cfg.get("seq_len", 599)),
        conv_channels=int(cfg.get("conv_channels", 96)),
        tcn_blocks=int(cfg.get("tcn_blocks", 8)),
        fc_dims=tuple(cfg.get("fc_dims", [512, 256, 128])),
        dropout=float(cfg.get("dropout", 0.15)),
        use_gate=bool(cfg.get("use_gate", True)),
    ).to(device)

    da = cfg.get("domain_adaptation", {})
    lambda_domain = float(da.get("lambda_domain", 0.0))
    if not da.get("enabled", False):
        lambda_domain = 0.0
        da_mode = "none"
    else:
        da_mode = str(da.get("mode", "global"))  # global | egc

    criterion = MATUDACriterion(
        lambda_domain=lambda_domain,
        mu_mmd=float(da.get("mu_mmd", 0.4)),
        power_weight=float(cfg.get("power_weight", 2.0)),
        state_weight=float(cfg.get("state_weight", 1.0)),
        pos_weight=pos_weight,
        da_mode=da_mode,
        domain_mix=str(da.get("domain_mix", "convex")),
        domain_scale=str(da.get("domain_scale", "equal")),
        conditional_weight=float(da.get("conditional_weight", 0.5)),
    )
    opt = AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("epochs", 80))
    steps_per_epoch = int(cfg.get("steps_per_epoch", 600))
    sched = CosineAnnealingLR(opt, T_max=max(epochs, 1))
    warmup_epochs = int(da.get("warmup_epochs", 10 if lambda_domain > 0 else 0))

    print(
        f"[{cfg['experiment_id']}] device={device} params={count_parameters(model):,} "
        f"da={da_mode} lambda={lambda_domain} warmup={warmup_epochs}",
        flush=True,
    )
    print(
        f"seq2point T={cfg.get('seq_len', 599)}  "
        f"source={len(loaders['source'].dataset)} target={len(loaders['target'].dataset)}",
        flush=True,
    )

    tgt_iter = _cycle(loaders["target"])
    best_val = float("inf")
    best_payload = None
    history = []

    for epoch in range(1, epochs + 1):
        if warmup_epochs > 0 and lambda_domain > 0:
            lam_epoch = lambda_domain * min(1.0, epoch / float(warmup_epochs))
        else:
            lam_epoch = lambda_domain

        model.train()
        t0 = time.time()
        loss_sum = sup_sum = dom_sum = 0.0
        n_steps = 0
        for step, batch_s in enumerate(loaders["source"]):
            if step >= steps_per_epoch:
                break
            xs = batch_s["x"].to(device)
            ys = batch_s["y"].to(device)
            zs = batch_s["z"].to(device)
            out_s = model(xs)

            if lam_epoch > 0 and da_mode != "none":
                xt = next(tgt_iter)["x"].to(device)
                out_t = model(xt)
                losses = criterion(out_s, out_t, ys, zs, lambda_override=lam_epoch)
            else:
                losses = criterion(out_s, None, ys, zs, lambda_override=0.0)

            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            loss_sum += float(losses["loss"])
            sup_sum += float(losses["loss_sup"])
            dom_sum += float(losses["loss_domain"])
            n_steps += 1

        sched.step()
        val_m = evaluate(model, loaders["val"], device, norm, appliances)
        test_m = evaluate(model, loaders["test"], device, norm, appliances)
        row = {
            "epoch": epoch,
            "lambda": lam_epoch,
            "loss": loss_sum / max(n_steps, 1),
            "loss_sup": sup_sum / max(n_steps, 1),
            "loss_domain": dom_sum / max(n_steps, 1),
            "sec": time.time() - t0,
            "val_mae": val_m["mae_mean"],
            "val_f1": val_m["f1_macro"],
            "test_mae": test_m["mae_mean"],
            "test_f1": test_m["f1_macro"],
            "test_sae": test_m["sae_mean"],
            "test_f1_per_app": test_m["f1_per_app"],
            "test_mae_per_app": test_m["mae_per_app"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} lam={lam_epoch:.2f} loss={row['loss']:.4f} "
            f"sup={row['loss_sup']:.4f} dom={row['loss_domain']:.4f}  "
            f"valMAE={row['val_mae']:.1f} valF1={row['val_f1']:.3f}  "
            f"H2MAE={row['test_mae']:.1f} H2F1={row['test_f1']:.3f}  "
            f"({row['sec']:.1f}s)",
            flush=True,
        )

        if val_m["mae_mean"] < best_val:
            best_val = val_m["mae_mean"]
            best_payload = {
                "model": model.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "val": val_m,
                "test": test_m,
                "appliances": appliances,
            }
            torch.save(best_payload, out_dir / "best.pt")

        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    summary = {
        "experiment_id": cfg["experiment_id"],
        "seed": int(cfg.get("seed", 2026)),
        "da_mode": da_mode,
        "lambda_domain": lambda_domain,
        "target_split": loaders["target_split"],
        "best_epoch": best_payload["epoch"] if best_payload else None,
        "best_val": best_payload["val"] if best_payload else None,
        "best_test_at_val_select": best_payload["test"] if best_payload else None,
        "last": history[-1] if history else None,
        "params": count_parameters(model),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"done -> {out_dir / 'best.pt'}", flush=True)
    print(json.dumps(summary["best_test_at_val_select"], indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.steps_per_epoch is not None:
        cfg["steps_per_epoch"] = args.steps_per_epoch
    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg["experiment_id"] = f"{cfg['experiment_id']}_seed{args.seed}"
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    elif args.seed is not None:
        base = cfg.get("output_dir", str(ROOT / "results" / cfg["experiment_id"]))
        # If seed was appended to experiment_id, keep output next to family folder.
        cfg["output_dir"] = str(Path(base).parent / cfg["experiment_id"])
    train(cfg)


if __name__ == "__main__":
    main()

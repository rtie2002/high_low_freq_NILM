"""
Generate paper figures: training curves + House-2 power prediction plots.

Uses evaluation.plots (adapted from multi_appliances_NILM/evaluation/plots.py).

  C:\\Users\\PC\\anaconda3\\envs\\nilm\\python.exe scripts\\plot_paper_figures.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.plots import (  # noqa: E402
    plot_appliance_grid,
    plot_matuda_training_history,
    plot_methods_f1_mae_comparison,
    save_appliance_on_waveforms,
)
from src.data import DEFAULT_APPLIANCES, make_loaders  # noqa: E402
from src.matuda_model import MATUDANet  # noqa: E402


@torch.no_grad()
def collect_predictions(model, loader, device, norm):
    model.eval()
    ys, yhs, zs, zhs, aggs = [], [], [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].numpy()
        z = batch["z"].numpy()
        out = model(x)
        y_hat = out["powers"].cpu().numpy()
        z_hat = (torch.sigmoid(out["state_logits"]).cpu().numpy() >= 0.5).astype(np.float32)
        # denorm
        ys.append(norm.denorm_power(y))
        yhs.append(norm.denorm_power(y_hat))
        zs.append(z)
        zhs.append(z_hat)
        # approximate aggregate at window center from normalized input
        c = x.shape[-1] // 2
        agg = x[:, 0, c].cpu().numpy() * (norm.agg_std + 1e-8) + norm.agg_mean
        aggs.append(agg)
    return (
        np.concatenate(ys, axis=0),
        np.concatenate(yhs, axis=0),
        np.concatenate(zs, axis=0),
        np.concatenate(zhs, axis=0),
        np.concatenate(aggs, axis=0),
    )


def load_model(ckpt_path: Path, device: torch.device) -> tuple[MATUDANet, dict, list[str]]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload.get("cfg", {})
    appliances = list(payload.get("appliances", cfg.get("appliances", DEFAULT_APPLIANCES)))
    model = MATUDANet(
        num_appliances=len(appliances),
        seq_len=int(cfg.get("seq_len", 599)),
        conv_channels=int(cfg.get("conv_channels", 96)),
        tcn_blocks=int(cfg.get("tcn_blocks", 8)),
        fc_dims=tuple(cfg.get("fc_dims", [512, 256, 128])),
        dropout=float(cfg.get("dropout", 0.15)),
        use_gate=bool(cfg.get("use_gate", True)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg, appliances


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=str,
        default=str(ROOT / "results"),
        help="Folder containing experiment subdirs with history.json / best.pt",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "paper" / "figures"),
    )
    ap.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="UK-DALE processed root (defaults to config data_root)",
    )
    ap.add_argument(
        "--methods",
        nargs="+",
        default=[
            "matuda_s2_b0_source_only",
            "matuda_s2_b1_fc_uda",
            "matuda_s2_m0_egc",
        ],
    )
    ap.add_argument("--labels", nargs="+", default=["Source-Only", "Global FC-UDA", "MATUDA"])
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Training curves from history.json (no GPU needed) ---
    histories = {}
    for method, label in zip(args.methods, args.labels):
        hist_path = results_root / method / "history.json"
        if not hist_path.exists():
            print(f"skip history: {hist_path}", flush=True)
            continue
        with open(hist_path, "r", encoding="utf-8") as f:
            hist = json.load(f)
        histories[label] = hist
        summary_path = results_root / method / "summary.json"
        best_epoch = None
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                best_epoch = json.load(f).get("best_epoch")
        path = plot_matuda_training_history(
            hist,
            out_dir / f"train_curves_{method}.png",
            title=f"{label} training dynamics",
            best_epoch=best_epoch,
        )
        print(f"wrote {path}", flush=True)

    if len(histories) >= 2:
        path = plot_methods_f1_mae_comparison(
            histories,
            out_dir / "train_compare_h2_f1_mae.png",
            title="House-2 metrics during training (same backbone)",
        )
        print(f"wrote {path}", flush=True)

    # --- Waveforms from best.pt (needs data + GPU/CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prefer MATUDA checkpoint for main prediction figure.
    preferred = [
        "matuda_s2_m0_egc",
        "matuda_s2_b0_source_only",
        "matuda_s2_b1_fc_uda",
    ]
    ckpt_method = next((m for m in preferred if (results_root / m / "best.pt").exists()), None)
    if ckpt_method is None:
        print("No best.pt found; loss curves only.", flush=True)
        return

    ckpt = results_root / ckpt_method / "best.pt"
    model, cfg, appliances = load_model(ckpt, device)
    data_root = Path(args.data_root or cfg.get("data_root", ""))
    if not data_root.exists():
        print(f"data_root missing: {data_root}", flush=True)
        return

    # Old Stage-2 checkpoints evaluated on full H2 (no chrono split in cfg).
    frac = cfg.get("target_adapt_frac", None)
    loaders = make_loaders(
        data_root,
        appliances=appliances,
        seq_len=int(cfg.get("seq_len", 599)),
        stride_train=int(cfg.get("stride_train", 30)),
        stride_eval=int(cfg.get("stride_eval", 60)),
        batch_size=64,
        num_workers=0,
        target_adapt_frac=None if frac is None else float(frac),
    )

    y_true, y_pred, z_true, z_pred, agg = collect_predictions(
        model, loaders["test"], device, loaders["norm"]
    )
    np.savez_compressed(
        out_dir / f"preds_{ckpt_method}.npz",
        y_true=y_true,
        y_pred=y_pred,
        z_true=z_true,
        z_pred=z_pred,
        aggregate=agg,
        appliances=np.asarray(appliances),
    )

    grid = plot_appliance_grid(
        appliances=appliances,
        y_true=y_true,
        y_pred=y_pred,
        output_path=out_dir / "h2_power_grid_matuda.png",
        title=f"House-2 power predictions ({ckpt_method})",
    )
    print(f"wrote {grid}", flush=True)

    waves = save_appliance_on_waveforms(
        out_dir / "waveforms_matuda",
        appliances=appliances,
        y_true_watts=y_true,
        y_pred_watts=y_pred,
        y_true_on=z_true,
        y_pred_on=z_pred,
        aggregate=agg,
        n_periods=2,
        title_prefix="MATUDA H2 ",
    )
    print(f"wrote {len(waves)} waveform plots -> {out_dir / 'waveforms_matuda'}", flush=True)

    # Also Source-Only waveforms if available (shows collapse).
    so = results_root / "matuda_s2_b0_source_only" / "best.pt"
    if so.exists():
        model_so, cfg_so, apps_so = load_model(so, device)
        y_t, y_p, z_t, z_p, agg_so = collect_predictions(
            model_so, loaders["test"], device, loaders["norm"]
        )
        plot_appliance_grid(
            appliances=apps_so,
            y_true=y_t,
            y_pred=y_p,
            output_path=out_dir / "h2_power_grid_source_only.png",
            title="House-2 power predictions (Source-Only)",
        )
        save_appliance_on_waveforms(
            out_dir / "waveforms_source_only",
            appliances=apps_so,
            y_true_watts=y_t,
            y_pred_watts=y_p,
            y_true_on=z_t,
            y_pred_on=z_p,
            aggregate=agg_so,
            n_periods=1,
            title_prefix="Source-Only H2 ",
        )
        print("wrote Source-Only comparison grids/waveforms", flush=True)

    print(f"FIGURES_DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

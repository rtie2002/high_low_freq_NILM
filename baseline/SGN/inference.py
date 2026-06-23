import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sgn.config import SGNConfig, default_data_dir
from sgn.data import REDDSGNWindowDataset
from sgn.metrics import compute_metrics
from sgn.model import SGN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained SGN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=default_data_dir())
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--appliance", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("predictions"))
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_stride", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = SGNConfig(**checkpoint["config"])
    appliance = args.appliance or checkpoint["appliance"]
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.eval_stride is not None:
        cfg.eval_stride = args.eval_stride

    model = SGN(
        cfg.input_length,
        cfg.output_length,
        cfg.input_channels,
        cfg.hidden_fc,
        cfg.dropout,
        gate_mode=cfg.gate_mode,
        standby_power=cfg.standby_power,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = REDDSGNWindowDataset(args.data_dir, args.split, appliance, cfg, stride=cfg.eval_stride)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    pred_watts, true_watts, pred_on, true_on = [], [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        outputs = model(x)
        pred_watts.append((outputs["gated_power"].cpu().numpy() * cfg.scale).reshape(-1))
        true_watts.append(batch["y_watts"].numpy().reshape(-1))
        pred_on.append(outputs["on_prob"].cpu().numpy().reshape(-1))
        true_on.append(batch["on"].numpy().reshape(-1))

    arrays = {
        "y_true_watts": np.concatenate(true_watts),
        "y_pred_watts": np.concatenate(pred_watts),
        "y_true_on": np.concatenate(true_on),
        "y_pred_on_prob": np.concatenate(pred_on),
    }
    metrics = compute_metrics(
        arrays["y_true_watts"],
        arrays["y_pred_watts"],
        arrays["y_true_on"],
        arrays["y_pred_on_prob"],
        sae_period=cfg.sae_period,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_{appliance}"
    np.savez_compressed(args.output_dir / f"{stem}_predictions.npz", **arrays)
    (args.output_dir / f"{stem}_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

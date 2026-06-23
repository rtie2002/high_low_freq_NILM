import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sgn.config import (
    ALL_APPLIANCES,
    APPLIANCES,
    CSV_APPLIANCES,
    SGNConfig,
    aggregate_std_scale,
    csv_training_stats,
    default_csv_config_path,
    default_data_dir,
    load_csv_config,
)
from sgn.data import CSVSGNWindowDataset, REDDSGNWindowDataset
from sgn.losses import SGNLoss
from sgn.metrics import compute_metrics
from sgn.model import SGN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SGN baseline on processed REDD data.")
    parser.add_argument("--data_source", choices=["redd_pkl", "csv"], default="redd_pkl")
    parser.add_argument("--data_dir", type=Path, default=default_data_dir())
    parser.add_argument("--csv_config", type=Path, default=default_csv_config_path())
    parser.add_argument("--run_dir", type=Path, default=Path("runs") / "sgn_redd")
    parser.add_argument(
        "--preset",
        choices=["matnilm", "sgn_paper", "custom"],
        default="sgn_paper",
        help="sgn_paper follows SGN paper hyperparameters; matnilm matches released MATNILM code defaults.",
    )
    parser.add_argument("--appliance", choices=["all", *ALL_APPLIANCES], default="all")
    parser.add_argument("--input_length", type=int, default=None)
    parser.add_argument("--output_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden_fc", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train_stride", type=int, default=1)
    parser.add_argument("--eval_stride", type=int, default=64)
    parser.add_argument(
        "--scale_mode",
        choices=["aggregate_std", "fixed_612"],
        default=None,
        help="SGN paper uses aggregate_std. MATNILM uses fixed_612.",
    )
    parser.add_argument("--gate_mode", choices=["soft", "hard"], default="soft")
    parser.add_argument("--standby_power", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_config(args: argparse.Namespace, csv_cfg: dict | None = None) -> SGNConfig:
    if args.preset == "sgn_paper":
        input_length = 864
        output_length = 64
        batch_size = 16
        learning_rate = 1.0e-4
        scale_mode = "aggregate_std"
    else:
        input_length = 864
        output_length = 864
        batch_size = 32
        learning_rate = 1.0e-3
        scale_mode = "fixed_612"

    input_length = args.input_length if args.input_length is not None else input_length
    output_length = args.output_length if args.output_length is not None else output_length
    batch_size = args.batch_size if args.batch_size is not None else batch_size
    learning_rate = args.lr if args.lr is not None else learning_rate
    scale_mode = args.scale_mode if args.scale_mode is not None else scale_mode
    feature_columns = ["aggregate"]
    feature_mean: list[float] = []
    feature_scale: list[float] = []
    input_channels = 1
    if args.data_source == "csv":
        if csv_cfg is None:
            raise ValueError("csv_cfg is required when data_source='csv'")
        feature_columns = list(csv_cfg["feature_columns"])
        scale, feature_mean, feature_scale = csv_training_stats(csv_cfg, scale_mode)
        input_channels = len(feature_columns)
    else:
        scale = aggregate_std_scale(args.data_dir) if scale_mode == "aggregate_std" else 612.0

    epochs = 2 if args.debug else args.epochs
    train_stride = max(args.train_stride, output_length) if args.debug else args.train_stride
    return SGNConfig(
        input_length=input_length,
        output_length=output_length,
        input_channels=input_channels,
        scale=scale,
        scale_mode=scale_mode,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        hidden_fc=args.hidden_fc,
        dropout=args.dropout,
        train_stride=train_stride,
        eval_stride=args.eval_stride,
        seed=args.seed,
        gate_mode=args.gate_mode,
        standby_power=args.standby_power,
    )


def make_loader(
    data_dir: Path,
    csv_cfg: dict | None,
    data_source: str,
    split: str,
    appliance: str,
    cfg: SGNConfig,
    stride: int,
    shuffle: bool,
) -> DataLoader:
    if data_source == "csv":
        if csv_cfg is None:
            raise ValueError("csv_cfg is required when data_source='csv'")
        dataset = CSVSGNWindowDataset(csv_cfg, split, appliance, cfg, stride=stride)
    else:
        dataset = REDDSGNWindowDataset(data_dir, split, appliance, cfg, stride=stride)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(
    model: SGN,
    loader: DataLoader,
    criterion: SGNLoss,
    device: torch.device,
    cfg: SGNConfig,
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

        pred_watts.append((outputs["gated_power"].detach().cpu().numpy() * cfg.scale).reshape(-1))
        true_watts.append(batch["y_watts"].numpy().reshape(-1))
        pred_on.append(outputs["on_prob"].detach().cpu().numpy().reshape(-1))
        true_on.append(batch["on"].numpy().reshape(-1))

    metrics = compute_metrics(
        np.concatenate(true_watts),
        np.concatenate(pred_watts),
        np.concatenate(true_on),
        np.concatenate(pred_on),
        sae_period=cfg.sae_period,
    )
    return float(np.mean(losses)), metrics


def train_one(
    appliance: str,
    args: argparse.Namespace,
    cfg: SGNConfig,
    device: torch.device,
    csv_cfg: dict | None,
) -> None:
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n== Training SGN for {appliance} on {device} ==")
    print(f"Data source: {args.data_source}")
    print(f"Data: {csv_cfg['csv_file'] if csv_cfg else args.data_dir}")
    print(f"Features: {cfg.feature_columns}")

    train_loader = make_loader(args.data_dir, csv_cfg, args.data_source, "train", appliance, cfg, cfg.train_stride, shuffle=True)
    val_loader = make_loader(args.data_dir, csv_cfg, args.data_source, "val", appliance, cfg, cfg.eval_stride, shuffle=False)
    test_loader = make_loader(args.data_dir, csv_cfg, args.data_source, "test", appliance, cfg, cfg.eval_stride, shuffle=False)

    model = SGN(
        cfg.input_length,
        cfg.output_length,
        cfg.input_channels,
        cfg.hidden_fc,
        cfg.dropout,
        gate_mode=cfg.gate_mode,
        standby_power=cfg.standby_power,
    ).to(device)
    criterion = SGNLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val = float("inf")
    best_epoch = -1
    stale_epochs = 0
    checkpoint_path = run_dir / f"best_{appliance}.pt"
    history_path = run_dir / f"history_{appliance}.csv"

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_mae",
                "val_sae",
                "val_f1",
            ],
        )
        writer.writeheader()

        for epoch in range(cfg.epochs):
            model.train()
            train_losses: list[float] = []
            progress = tqdm(train_loader, desc=f"{appliance} epoch {epoch + 1}/{cfg.epochs}")
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

            val_loss, val_metrics = evaluate(model, val_loader, criterion, device, cfg)
            train_loss = float(np.mean(train_losses))
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_metrics["mae"],
                "val_sae": val_metrics["sae"],
                "val_f1": val_metrics["f1"],
            }
            writer.writerow(row)
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
                        "model_state_dict": model.state_dict(),
                        "config": cfg.__dict__,
                        "appliance": appliance,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val,
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= cfg.patience:
                    print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                    break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, val_metrics = evaluate(model, val_loader, criterion, device, cfg)
    _, test_metrics = evaluate(model, test_loader, criterion, device, cfg)
    metrics = {
        "appliance": appliance,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "validation": val_metrics,
        "test": test_metrics,
    }
    metrics_path = run_dir / f"metrics_{appliance}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    args = parse_args()
    csv_cfg = load_csv_config(args.csv_config) if args.data_source == "csv" else None
    cfg = make_config(args, csv_cfg)
    seed_everything(cfg.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.data_source == "csv":
        available = sorted((csv_cfg or {}).get("appliances", CSV_APPLIANCES))
    else:
        available = sorted(APPLIANCES)
    if args.appliance == "all":
        appliances = available
    else:
        if args.appliance not in available:
            raise ValueError(
                f"Appliance '{args.appliance}' is not available for {args.data_source}. Choices: {available}"
            )
        appliances = [args.appliance]
    for appliance in appliances:
        train_one(appliance, args, cfg, device, csv_cfg)


if __name__ == "__main__":
    main()

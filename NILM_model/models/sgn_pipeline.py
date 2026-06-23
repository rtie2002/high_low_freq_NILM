from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

NILM_MODEL_DIR = Path(__file__).resolve().parents[1]
SGN_DIR = NILM_MODEL_DIR / "baseline" / "SGN"
if str(NILM_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(NILM_MODEL_DIR))
if str(SGN_DIR) not in sys.path:
    sys.path.insert(0, str(SGN_DIR))

from model_evaluation.runner import make_dataloader, run_nilm_inference, seed_everything, train_nilm_model
from sgn.config import (
    ALL_APPLIANCES,
    APPLIANCES,
    CSV_APPLIANCES,
    SGNConfig,
    aggregate_std_scale,
    csv_training_stats,
    default_csv_config_path,
    default_data_dir,
    default_model_config_path,
    load_csv_config,
    load_model_config,
)
from sgn.data import CSVSGNWindowDataset, REDDSGNWindowDataset
from sgn.losses import SGNLoss
from sgn.model import SGN


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SGN through the shared NILM pipeline.")
    parser.add_argument("--data_source", choices=["redd_pkl", "csv"], default="redd_pkl")
    parser.add_argument("--data_dir", type=Path, default=default_data_dir())
    parser.add_argument("--csv_config", type=Path, default=default_csv_config_path())
    parser.add_argument("--model_config", type=Path, default=default_model_config_path())
    parser.add_argument("--run_dir", type=Path, default=Path("runs") / "sgn_redd")
    parser.add_argument(
        "--preset",
        choices=["matnilm", "sgn_paper", "custom"],
        default=None,
        help="sgn_paper follows SGN paper hyperparameters; matnilm matches released MATNILM defaults.",
    )
    parser.add_argument("--appliance", choices=["all", *ALL_APPLIANCES], default="all")
    parser.add_argument("--input_length", type=int, default=None)
    parser.add_argument("--output_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--hidden_fc", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--train_stride", type=int, default=None)
    parser.add_argument("--eval_stride", type=int, default=None)
    parser.add_argument("--scale_mode", choices=["aggregate_std", "fixed_612"], default=None)
    parser.add_argument("--gate_mode", choices=["soft", "hard"], default=None)
    parser.add_argument("--standby_power", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def parse_inference_args(argv: list[str] | None = None, *, allow_unknown: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SGN inference through the shared NILM pipeline.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_source", choices=["redd_pkl", "csv"], default="redd_pkl")
    parser.add_argument("--data_dir", type=Path, default=default_data_dir())
    parser.add_argument("--csv_config", type=Path, default=default_csv_config_path())
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--appliance", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("predictions"))
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_stride", type=int, default=None)
    parser.add_argument("--plot_samples", type=int, default=2000)
    if allow_unknown:
        args, _ = parser.parse_known_args(argv)
        return args
    return parser.parse_args(argv)


def make_config(args: argparse.Namespace, model_cfg: dict, csv_cfg: dict | None = None) -> SGNConfig:
    preset = args.preset or model_cfg.get("preset", "sgn_paper")
    if preset == "sgn_paper":
        defaults = {
            "input_length": 864,
            "output_length": 64,
            "batch_size": 16,
            "learning_rate": 1.0e-4,
            "scale_mode": "aggregate_std",
        }
    else:
        defaults = {
            "input_length": 864,
            "output_length": 864,
            "batch_size": 32,
            "learning_rate": 1.0e-3,
            "scale_mode": "fixed_612",
        }
    if args.preset is None or model_cfg.get("preset", preset) == preset:
        defaults.update(model_cfg)

    def choose(name: str, arg_name: str | None = None):
        value = getattr(args, arg_name or name)
        return value if value is not None else defaults[name]

    input_length = int(choose("input_length"))
    output_length = int(choose("output_length"))
    batch_size = int(choose("batch_size"))
    learning_rate = float(choose("learning_rate", "lr"))
    scale_mode = str(choose("scale_mode"))
    patience = int(choose("patience"))
    num_workers = int(choose("num_workers"))
    hidden_fc = int(choose("hidden_fc"))
    dropout = float(choose("dropout"))
    eval_stride = int(choose("eval_stride"))
    seed = int(choose("seed"))
    gate_mode = str(choose("gate_mode"))
    standby_power = bool(defaults.get("standby_power", False) or args.standby_power)
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

    epochs = 2 if args.debug else int(choose("epochs"))
    configured_train_stride = int(choose("train_stride"))
    train_stride = max(configured_train_stride, output_length) if args.debug else configured_train_stride
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
        patience=patience,
        num_workers=num_workers,
        hidden_fc=hidden_fc,
        dropout=dropout,
        train_stride=train_stride,
        eval_stride=eval_stride,
        seed=seed,
        gate_mode=gate_mode,
        standby_power=standby_power,
    )


def make_dataset(
    data_dir: Path,
    csv_cfg: dict | None,
    data_source: str,
    split: str,
    appliance: str,
    cfg: SGNConfig,
    stride: int,
):
    if data_source == "csv":
        if csv_cfg is None:
            raise ValueError("csv_cfg is required when data_source='csv'")
        return CSVSGNWindowDataset(csv_cfg, split, appliance, cfg, stride=stride)
    return REDDSGNWindowDataset(data_dir, split, appliance, cfg, stride=stride)


def build_model(cfg: SGNConfig, device: torch.device) -> SGN:
    return SGN(
        cfg.input_length,
        cfg.output_length,
        cfg.input_channels,
        cfg.hidden_fc,
        cfg.dropout,
        gate_mode=cfg.gate_mode,
        standby_power=cfg.standby_power,
    ).to(device)


def train_one(
    appliance: str,
    args: argparse.Namespace,
    cfg: SGNConfig,
    device: torch.device,
    csv_cfg: dict | None,
) -> Path:
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n== Training SGN for {appliance} on {device} ==")
    print(f"Data source: {args.data_source}")
    print(f"Data: {csv_cfg['csv_file'] if csv_cfg else args.data_dir}")
    print(f"Features: {cfg.feature_columns}")

    train_loader = make_dataloader(
        make_dataset(args.data_dir, csv_cfg, args.data_source, "train", appliance, cfg, cfg.train_stride),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = make_dataloader(
        make_dataset(args.data_dir, csv_cfg, args.data_source, "val", appliance, cfg, cfg.eval_stride),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = make_dataloader(
        make_dataset(args.data_dir, csv_cfg, args.data_source, "test", appliance, cfg, cfg.eval_stride),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = build_model(cfg, device)
    criterion = SGNLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_nilm_model(
        model_name="SGN",
        appliance=appliance,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=cfg,
        run_dir=run_dir,
        device=device,
    )
    return run_dir / f"best_{appliance}.pt"


def train_main(argv: list[str] | None = None) -> list[Path]:
    args = parse_train_args(argv)
    model_cfg = load_model_config(args.model_config)
    csv_cfg = load_csv_config(args.csv_config) if args.data_source == "csv" else None
    cfg = make_config(args, model_cfg, csv_cfg)
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
            raise ValueError(f"Appliance '{args.appliance}' is not available for {args.data_source}. Choices: {available}")
        appliances = [args.appliance]
    return [train_one(appliance, args, cfg, device, csv_cfg) for appliance in appliances]


@torch.no_grad()
def inference_main(argv: list[str] | None = None, *, allow_unknown: bool = False) -> None:
    args = parse_inference_args(argv, allow_unknown=allow_unknown)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = SGNConfig(**checkpoint["config"])
    appliance = args.appliance or checkpoint["appliance"]
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.eval_stride is not None:
        cfg.eval_stride = args.eval_stride

    csv_cfg = load_csv_config(args.csv_config) if args.data_source == "csv" else None
    model = build_model(cfg, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.data_source == "csv":
        if appliance not in (csv_cfg or {}).get("appliances", CSV_APPLIANCES):
            raise ValueError(f"Appliance '{appliance}' is not available in CSV config {args.csv_config}")
        dataset = CSVSGNWindowDataset(csv_cfg, args.split, appliance, cfg, stride=cfg.eval_stride)
    else:
        dataset = REDDSGNWindowDataset(args.data_dir, args.split, appliance, cfg, stride=cfg.eval_stride)
    loader = make_dataloader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    run_nilm_inference(
        model_name="SGN",
        appliance=appliance,
        model=model,
        loader=loader,
        config=cfg,
        output_dir=args.output_dir,
        split=args.split,
        device=device,
        plot_samples=args.plot_samples,
    )

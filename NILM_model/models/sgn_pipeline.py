from __future__ import annotations

import argparse
import copy
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
    csv_appliance_on_stats,
    csv_training_stats,
    default_csv_config_path,
    default_data_dir,
    default_model_config_path,
    describe_csv_sources,
    describe_csv_split_label,
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
    parser.add_argument(
        "--appliance",
        choices=["all", *ALL_APPLIANCES],
        default=None,
        help="Target appliance set. Defaults to default_appliance in the model config, which is 'all'.",
    )
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
    parser.add_argument(
        "--plot_mode",
        choices=["live", "end", "interval"],
        default=None,
        help="live=every epoch PNGs; end=plot only after training; interval=every plot_interval epochs.",
    )
    parser.add_argument(
        "--run_all_epochs",
        action="store_true",
        help="Disable early stopping; run the full epoch count before plotting.",
    )
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
    defaults.update(model_cfg)

    if args.data_source == "csv" and csv_cfg is not None:
        sampling_seconds = int(csv_cfg.get("sampling_seconds", 0) or 0)
        if sampling_seconds == 6:
            if args.input_length is None:
                defaults["input_length"] = 432
            if args.output_length is None:
                defaults["output_length"] = 32
            if args.eval_stride is None:
                defaults["eval_stride"] = 32
            if "sae_period" not in defaults:
                defaults["sae_period"] = 3600 // sampling_seconds
            defaults["val_split_label"] = describe_csv_split_label(csv_cfg, "val")
            defaults["test_split_label"] = describe_csv_split_label(csv_cfg, "test")

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
    if sys.platform == "win32" and num_workers > 0:
        print(
            f"Windows: num_workers={num_workers} may hang at 'val 0%' between epochs. "
            "Recommend num_workers=0 in config. persistent_workers is enabled if you keep workers > 0."
        )
    hidden_fc = int(choose("hidden_fc"))
    dropout = float(choose("dropout"))
    eval_stride = int(choose("eval_stride"))
    seed = int(choose("seed"))
    gate_mode = str(choose("gate_mode"))
    standby_power = bool(defaults.get("standby_power", False) or args.standby_power)
    weight_decay = float(defaults.get("weight_decay", 0.0))
    early_stop_metric = str(defaults.get("early_stop_metric", "output_loss"))
    label_smoothing = float(defaults.get("label_smoothing", 0.0))
    reg_on_weight = float(defaults.get("reg_on_weight", 0.0))
    gated_on_weight = float(defaults.get("gated_on_weight", 0.0))
    on_confidence_weight = float(defaults.get("on_confidence_weight", 0.0))
    on_smooth_weight = float(defaults.get("on_smooth_weight", 0.0))
    bce_pos_weight = float(defaults.get("bce_pos_weight", 1.0))
    oversample_on = bool(defaults.get("oversample_on", False))
    oversample_max_weight = float(defaults.get("oversample_max_weight", 15.0))
    grad_clip_norm = float(defaults.get("grad_clip_norm", 1.0))
    lr_schedule = str(defaults.get("lr_schedule", "none"))
    lr_warmup_epochs = int(defaults.get("lr_warmup_epochs", 0))
    lr_scheduler_patience = int(defaults.get("lr_scheduler_patience", 0))
    lr_scheduler_factor = float(defaults.get("lr_scheduler_factor", 0.5))
    lr_min = float(defaults.get("lr_min", 1e-6))
    min_epochs = int(defaults.get("min_epochs", 5))
    sae_period = int(defaults.get("sae_period", 1200))
    val_split_label = str(defaults.get("val_split_label", "validation"))
    test_split_label = str(defaults.get("test_split_label", "test"))
    plot_mode = str(defaults.get("plot_mode", "live"))
    if args.plot_mode is not None:
        plot_mode = str(args.plot_mode)
    plot_interval = int(defaults.get("plot_interval", 1))
    run_all_epochs = bool(defaults.get("run_all_epochs", False) or args.run_all_epochs)
    feature_columns = ["aggregate"]
    feature_mean: list[float] = []
    feature_scale: list[float] = []
    input_channels = 1
    target_choice = args.appliance or model_cfg.get("default_appliance", "all")
    args.appliance = target_choice
    if args.data_source == "csv":
        available_appliances = sorted((csv_cfg or {}).get("appliances", CSV_APPLIANCES))
    else:
        available_appliances = sorted(APPLIANCES)
    target_appliances = available_appliances if target_choice == "all" else [target_choice]
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
        target_appliances=target_appliances,
        num_appliances=len(target_appliances),
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
        weight_decay=weight_decay,
        early_stop_metric=early_stop_metric,
        label_smoothing=label_smoothing,
        reg_on_weight=reg_on_weight,
        gated_on_weight=gated_on_weight,
        on_confidence_weight=on_confidence_weight,
        on_smooth_weight=on_smooth_weight,
        bce_pos_weight=bce_pos_weight,
        oversample_on=oversample_on,
        oversample_max_weight=oversample_max_weight,
        grad_clip_norm=grad_clip_norm,
        lr_schedule=lr_schedule,
        lr_warmup_epochs=lr_warmup_epochs,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_min=lr_min,
        min_epochs=min_epochs,
        sae_period=sae_period,
        val_split_label=val_split_label,
        test_split_label=test_split_label,
        plot_mode=plot_mode,
        plot_interval=plot_interval,
        run_all_epochs=run_all_epochs,
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
        num_appliances=cfg.num_appliances,
        gate_mode=cfg.gate_mode,
        standby_power=cfg.standby_power,
    ).to(device)


def describe_dataset(name: str, dataset) -> None:
    sample = dataset[0]
    print(f"{name} windows: {len(dataset)}")
    print(f"{name} x shape: {tuple(sample['x'].shape)}")
    print(f"{name} y power shape: {tuple(sample['y_watts'].shape)}")
    print(f"{name} y on/off shape: {tuple(sample['on'].shape)}")
    print(f"{name} aggregate shape: {tuple(sample['aggregate_watts'].shape)}")
    if hasattr(dataset, "features"):
        print(f"{name} source rows: {len(dataset.features)}")
    elif hasattr(dataset, "sequences"):
        lengths = [len(seq) for seq in dataset.sequences]
        print(f"{name} segments: {len(lengths)}")
        print(f"{name} segment lengths: {lengths}")


def describe_experiment_mapping(
    *,
    data_source: str,
    appliance: str,
    cfg: SGNConfig,
    csv_cfg: dict | None,
    data_dir: Path,
) -> None:
    print("\n== Experiment data mapping ==")
    if data_source == "csv":
        if csv_cfg is None:
            raise ValueError("csv_cfg is required when data_source='csv'")
        appliance_cfg = csv_cfg["appliances"][appliance]
        split_mode = csv_cfg.get("split_mode", "temporal")
        if split_mode == "holdout":
            print(f"Train CSV: {csv_cfg['train_csv_file']}")
            print(f"Test CSV: {csv_cfg['test_csv_file']}")
            val_mode = csv_cfg.get("val_mode", "temporal")
            if val_mode == "by_house":
                print(
                    f"Val mode: by_house (train houses {csv_cfg.get('train_house_ids', [1])}, "
                    f"val houses {csv_cfg.get('val_house_ids', [5])})"
                )
            elif val_mode == "by_house_tail":
                print(
                    f"Val mode: by_house_tail (val houses {csv_cfg.get('val_house_ids', [5])}, "
                    f"last {csv_cfg.get('val_last_days', 7)} days)"
                )
            elif val_mode == "separate_files":
                print(f"Val CSV: {csv_cfg['val_csv_file']}")
                print("Val mode: separate_files (pre-built validating CSV)")
            else:
                print(f"Val mode: temporal split on train CSV ({csv_cfg.get('split_ratios')})")
            print(f"Split mode: holdout (paper-style house split)")
        else:
            print(f"CSV file: {csv_cfg['csv_file']}")
            print(f"Split mode: temporal")
        print(f"Time column: {csv_cfg.get('time_column', 'not used')}")
        print(f"Sampling interval: {csv_cfg.get('sampling_seconds', 'unknown')} seconds")
        print(f"X input columns: {cfg.feature_columns}")
        print(f"X scaling: X / train_aggregate_std")
        print(f"Y power column: {appliance_cfg['power']}")
        print(f"Y scaling: Y_watts / train_aggregate_std")
        print(f"Y on/off column: {appliance_cfg['on']}")
        print("Y on/off scaling: unchanged 0/1 label from CSV")
        print(f"Prediction inverse scaling: predicted_normalized_power * train_aggregate_std")
        print(f"train_aggregate_std: {cfg.scale:.6f}")
        return

    print(f"Processed REDD data directory: {data_dir}")
    print("X input column: main aggregate power")
    print("X scaling: main / train_aggregate_std")
    print(f"Y appliance target: {appliance}")
    print("Y scaling: appliance_watts / train_aggregate_std")
    print(f"Y on/off label: 1 if appliance_watts > {cfg.on_threshold_watts:g} W else 0")
    print("Prediction inverse scaling: predicted_normalized_power * train_aggregate_std")
    print(f"train_aggregate_std: {cfg.scale:.6f}")


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
    print(f"Data: {describe_csv_sources(csv_cfg) if csv_cfg else args.data_dir}")
    print(f"Features: {cfg.feature_columns}")
    print(f"Target appliances: {cfg.target_appliances}")
    describe_experiment_mapping(
        data_source=args.data_source,
        appliance=appliance,
        cfg=cfg,
        csv_cfg=csv_cfg,
        data_dir=args.data_dir,
    )

    train_dataset = make_dataset(args.data_dir, csv_cfg, args.data_source, "train", appliance, cfg, cfg.train_stride)
    val_dataset = make_dataset(args.data_dir, csv_cfg, args.data_source, "val", appliance, cfg, cfg.eval_stride)
    test_dataset = make_dataset(args.data_dir, csv_cfg, args.data_source, "test", appliance, cfg, cfg.eval_stride)

    print("\n== Data summary ==")
    print(f"input_length: {cfg.input_length}")
    print(f"output_length: {cfg.output_length}")
    print(f"input_channels: {cfg.input_channels}")
    print(f"num_appliances: {cfg.num_appliances}")
    print(f"batch_size: {cfg.batch_size}")
    print(f"train_stride: {cfg.train_stride}")
    print(f"eval_stride: {cfg.eval_stride}")
    describe_dataset("train", train_dataset)
    describe_dataset("validation", val_dataset)
    describe_dataset("test", test_dataset)
    if csv_cfg is not None:
        on_stats = csv_appliance_on_stats(csv_cfg, appliance)
        if on_stats:
            mean_w = on_stats["mean_on_watts"]
            p95_w = on_stats["p95_on_watts"]
            print(
                f"Train CSV {appliance} ON power: mean={mean_w:.0f}W, p95={p95_w:.0f}W "
                f"→ normalized target @ scale={cfg.scale:.1f}: "
                f"mean={mean_w / cfg.scale:.2f}, p95={p95_w / cfg.scale:.2f}"
            )
            print(
                "Regression must learn normalized output ≈ p95 value on ON windows "
                "(if val raw/true ≪ 1.0 on H2 but train raw/true ≈ 1.0 → cross-house issue)."
            )

    train_loader = make_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        oversample_on=cfg.oversample_on,
        oversample_max_weight=cfg.oversample_max_weight,
    )
    val_loader = make_dataloader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = make_dataloader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = build_model(cfg, device)
    print("\n== Model architecture ==")
    print(model)
    criterion = SGNLoss(
        label_smoothing=cfg.label_smoothing,
        reg_on_weight=cfg.reg_on_weight,
        gated_on_weight=cfg.gated_on_weight,
        on_confidence_weight=cfg.on_confidence_weight,
        on_smooth_weight=cfg.on_smooth_weight,
        bce_pos_weight=cfg.bce_pos_weight,
    )
    if cfg.reg_on_weight > 0.0:
        print(f"ON-only regression loss: reg_on_weight={cfg.reg_on_weight}")
    if cfg.gated_on_weight > 0.0:
        print(f"ON-only gated output loss: gated_on_weight={cfg.gated_on_weight}")
    if cfg.on_confidence_weight > 0.0:
        print(f"ON confidence loss: on_confidence_weight={cfg.on_confidence_weight}")
    if cfg.on_smooth_weight > 0.0:
        print(f"ON temporal smoothness: on_smooth_weight={cfg.on_smooth_weight}")
    if cfg.bce_pos_weight > 1.0:
        print(f"BCE pos_weight: {cfg.bce_pos_weight} (ON samples weighted {cfg.bce_pos_weight}x)")
    print(f"Plot mode: {cfg.plot_mode}" + (f" (interval={cfg.plot_interval})" if cfg.plot_mode == "interval" else ""))
    if cfg.run_all_epochs:
        print(f"run_all_epochs=True: training all {cfg.epochs} epochs (no early stop).")
    if cfg.lr_schedule != "none":
        print(
            f"LR schedule: {cfg.lr_schedule} "
            f"(peak={cfg.learning_rate}, min={cfg.lr_min}, warmup={cfg.lr_warmup_epochs})"
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

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

    if args.data_source == "csv":
        available = sorted((csv_cfg or {}).get("appliances", CSV_APPLIANCES))
    else:
        available = sorted(APPLIANCES)

    target_choice = args.appliance or model_cfg.get("default_appliance", "all")
    if target_choice != "all" and target_choice not in available:
        raise ValueError(f"Appliance '{target_choice}' is not available for {args.data_source}. Choices: {available}")
    target_appliances = available if target_choice == "all" else [target_choice]

    checkpoints: list[Path] = []
    for appliance in target_appliances:
        appliance_args = copy.copy(args)
        appliance_args.appliance = appliance
        cfg = make_config(appliance_args, model_cfg, csv_cfg)
        seed_everything(cfg.seed)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoints.append(train_one(appliance, appliance_args, cfg, device, csv_cfg))
    return checkpoints


@torch.no_grad()
def inference_main(argv: list[str] | None = None, *, allow_unknown: bool = False) -> dict:
    args = parse_inference_args(argv, allow_unknown=allow_unknown)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = SGNConfig(**checkpoint["config"])
    appliance = args.appliance or checkpoint["appliance"]
    if appliance == "all" and cfg.target_appliances:
        dataset_appliance: str | list[str] = cfg.target_appliances
    else:
        dataset_appliance = appliance
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.eval_stride is not None:
        cfg.eval_stride = args.eval_stride

    csv_cfg = load_csv_config(args.csv_config) if args.data_source == "csv" else None
    model = build_model(cfg, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.data_source == "csv":
        available = (csv_cfg or {}).get("appliances", CSV_APPLIANCES)
        unknown = sorted(set(dataset_appliance if isinstance(dataset_appliance, list) else [dataset_appliance]) - set(available))
        if unknown:
            raise ValueError(f"Appliance '{appliance}' is not available in CSV config {args.csv_config}")
        dataset = CSVSGNWindowDataset(csv_cfg, args.split, dataset_appliance, cfg, stride=cfg.eval_stride)
    else:
        dataset = REDDSGNWindowDataset(args.data_dir, args.split, dataset_appliance, cfg, stride=cfg.eval_stride)
    loader = make_dataloader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    return run_nilm_inference(
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

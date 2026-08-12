#!/usr/bin/env python
"""CLI entry: pick dataset + model via YAML, run one shared train/eval pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.config import load_experiment, load_model_config, merge_configs, model_name_from_config
from adapters.mat_nilm import MATNILMAdapter
from adapters.matuda import MATUDAAdapter
from adapters.multinilm import MultiNILMAdapter
from adapters.multinilm_fractional import MultiNILMFractionalAdapter
from adapters.multinilm_kle import MultiNILMKLEAdapter
from adapters.multinilm_no_distill import MultiNILMNoDistillAdapter
from adapters.transfer_multi_appliance import TransferMultiApplianceAdapter
from evaluation.compare import compare_experiment
from evaluation.run_summary import print_run_cost_summary, print_val_test_comparison
from runner import evaluate_model, train_model

# Register new models here: name -> adapter class
MODELS = {
    "mat_nilm": MATNILMAdapter,
    "matuda": MATUDAAdapter,
    "multinilm": MultiNILMAdapter,
    "multinilm_fractional": MultiNILMFractionalAdapter,
    "multinilm_kle": MultiNILMKLEAdapter,
    "multinilm_no_distill": MultiNILMNoDistillAdapter,
    "transfer_multi_appliance": TransferMultiApplianceAdapter,
}

# Default run settings for "click Run on main.py".
# Edit these once, then run this file directly without command-line arguments.
DEFAULT_MODE = "train_evaluate"
DEFAULT_MODEL = "multinilm"
DEFAULT_EXPERIMENT = ROOT / "config" / "experiment_ukdale.yaml"
DEFAULT_MODEL_CONFIG: Path | None = None
DEFAULT_DATA_PATH: Path | None = None
DEFAULT_CHECKPOINT: Path | None = None
DEFAULT_INIT_CHECKPOINT: Path | None = None
DEFAULT_RUN_DIR: Path | None = None
DEFAULT_EPOCHS: int | None = None
DEFAULT_SEED: int | None = None


def get_adapter(model_name: str, merged_cfg: dict, data_root: str | None = None):
    if model_name not in MODELS:
        known = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown model '{model_name}'. Available: {known}")
    return MODELS[model_name](merged_cfg, data_root=data_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-appliance NILM: one pipeline, switch dataset/model via YAML.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "train_evaluate", "compare"],
        default=DEFAULT_MODE,
        help="train | evaluate | train_evaluate | compare",
    )
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=DEFAULT_EXPERIMENT,
        help="Dataset config (UK-DALE default). Try config/experiment_redd.yaml or experiment_refit.yaml",
    )
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG, help="Override config/models/<model>.yaml")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Override experiment data_root")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT, help="Fine-tune from another checkpoint")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _default_model_config(model_name: str) -> Path:
    return ROOT / "config" / "models" / f"{model_name}.yaml"


def _default_run_dir(experiment_id: str, model_name: str) -> Path:
    return ROOT / "runs" / experiment_id / model_name


def main() -> None:
    args = parse_args()
    print(
        f"Using mode={args.mode}, model={args.model}, "
        f"experiment={args.experiment}, model_config={args.model_config or 'auto'}",
        flush=True,
    )
    experiment = load_experiment(args.experiment)
    model_cfg_path = args.model_config or _default_model_config(args.model)
    model_cfg = load_model_config(model_cfg_path)

    if args.mode == "compare":
        # Run folders are named by model experiment_id (not dataset_id).
        eid = merge_configs(experiment, model_cfg)["experiment_id"]
        print(compare_experiment(ROOT / "runs", eid))
        return

    if model_name_from_config(model_cfg) != args.model:
        raise ValueError(
            f"--model {args.model!r} does not match {model_cfg_path} "
            f"(model_name: {model_name_from_config(model_cfg)!r})"
        )

    merged = merge_configs(experiment, model_cfg)
    data_root = args.data_path or merged.get("data_root")
    if data_root is not None:
        data_root = Path(data_root)
        if not data_root.is_absolute():
            data_root = ROOT / data_root

    adapter = get_adapter(args.model, merged, data_root=str(data_root) if data_root else None)
    run_dir = args.run_dir or _default_run_dir(merged["experiment_id"], args.model)

    if args.mode in ("train", "train_evaluate"):
        ckpt = train_model(
            adapter,
            run_dir,
            epochs=args.epochs,
            seed=args.seed,
            init_checkpoint=args.init_checkpoint,
        )
        print(f"Saved checkpoint: {ckpt}")

    if args.mode in ("evaluate", "train_evaluate"):
        ckpt = args.checkpoint or (run_dir / "best.pt")
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        for split in ("validation", "test"):
            print(f"\n{split.capitalize()} evaluation ({ckpt.name}):", flush=True)
            pred_path = evaluate_model(
                adapter,
                ckpt,
                run_dir,
                split=split,
                show_cost_summary=False,
            )
            print(f"Saved predictions: {pred_path}")
        print_val_test_comparison(run_dir)
        print_run_cost_summary(run_dir)


if __name__ == "__main__":
    main()

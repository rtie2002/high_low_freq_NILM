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
from adapters.multinilm import MultiNILMAdapter
from evaluation.compare import compare_experiment
from runner import evaluate_model, train_model

# Register new models here: name -> adapter class
MODELS = {
    "mat_nilm": MATNILMAdapter,
    "multinilm": MultiNILMAdapter,
}


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
        required=True,
        help="train | evaluate | train_evaluate | compare",
    )
    parser.add_argument("--model", choices=sorted(MODELS), default=None)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=ROOT / "config/experiment.yaml",
        help="Dataset config (UK-DALE default). Try config/experiment_redd.yaml or experiment_refit.yaml",
    )
    parser.add_argument("--model-config", type=Path, default=None, help="Override config/models/<model>.yaml")
    parser.add_argument("--data-path", type=Path, default=None, help="Override experiment data_root")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Fine-tune from another checkpoint")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _default_model_config(model_name: str) -> Path:
    return ROOT / "config" / "models" / f"{model_name}.yaml"


def _default_run_dir(experiment_id: str, model_name: str) -> Path:
    return ROOT / "runs" / experiment_id / model_name


def main() -> None:
    args = parse_args()
    if args.mode == "compare":
        experiment = load_experiment(args.experiment)
        print(compare_experiment(ROOT / "runs", experiment["experiment_id"]))
        return

    if args.model is None:
        raise SystemExit("--model is required for train / evaluate / train_evaluate")

    experiment = load_experiment(args.experiment)
    model_cfg_path = args.model_config or _default_model_config(args.model)
    model_cfg = load_model_config(model_cfg_path)
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
    run_dir = args.run_dir or _default_run_dir(experiment["experiment_id"], args.model)

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
        print(f"\nTest evaluation ({ckpt.name}):", flush=True)
        pred_path = evaluate_model(adapter, ckpt, run_dir, split="test")
        print(f"Saved predictions: {pred_path}")


if __name__ == "__main__":
    main()

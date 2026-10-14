#!/usr/bin/env python
"""Run the controlled multi-seed experiments for paper Tables V and VII.

Table V compares MART-NILM with UNet-NILM and MAT-Conv on the same five-week
house split. Table VII changes one MART-NILM architecture component at a time
and evaluates validation houses only. Test houses are never opened by the
architecture-ablation stage.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_experiment, load_model_config, merge_configs
from data.dataloader import resolve_test_scenarios
from main import get_adapter
from runner import evaluate_model, train_model


DEFAULT_EXPERIMENT = (
    ROOT / "config" / "experiment_mixed_ukdale_refit_5w_all_house_split.yaml"
)
MODEL_CONFIGS = {
    "unetnilm": ROOT / "config" / "models" / "unetnilm.yaml",
    "mat_nilm": ROOT / "config" / "models" / "mat_nilm.yaml",
    "mart_nilm": ROOT / "config" / "models" / "mart_nilm.yaml",
}
SEEDS = (2026, 2027, 2028)

ARCHITECTURE_VARIANTS = (
    "complete",
    "single_scale_stem",
    "tcn_without_dilation",
    "no_channel_attention",
    "no_local_decoder",
    "no_relation_attention",
    "no_state_power_gate",
)


def _data_root(merged: dict) -> Path:
    path = Path(merged["data_root"])
    return path if path.is_absolute() else ROOT / path


def _paper_architecture_variant(base: dict, name: str, seed: int) -> dict:
    if name not in ARCHITECTURE_VARIANTS:
        raise ValueError(f"Unknown architecture variant: {name}")
    cfg = copy.deepcopy(base)
    cfg["experiment_id"] = f"paper5w_table_vii_{name}_seed{seed}"
    arch = cfg["architecture"]

    if name == "single_scale_stem":
        arch["use_multiscale_stem"] = False
    elif name == "tcn_without_dilation":
        arch["max_dilation"] = 1
    elif name == "no_channel_attention":
        arch["task_attention"] = {"enabled": False}
    elif name == "no_local_decoder":
        arch["head_local_layers"] = 0
    elif name == "no_relation_attention":
        arch["cross_appliance"] = {"enabled": False}
    elif name == "no_state_power_gate":
        arch["gate_mode"] = "none"
        cfg["evaluation"]["state_calibration"]["apply_to_power"] = False

    return cfg


def _evaluate_missing(
    adapter,
    checkpoint: Path,
    run_dir: Path,
    splits: list[str],
) -> None:
    for split in splits:
        metrics_path = (
            run_dir / "validation_metrics.csv"
            if split == "validation"
            else run_dir / "test" / split / "metrics.csv"
        )
        if metrics_path.exists():
            print(f"Already evaluated: {metrics_path}", flush=True)
            continue
        evaluate_model(
            adapter,
            checkpoint,
            run_dir,
            split=split,
            show_cost_summary=False,
        )


def _run_one(
    experiment: dict,
    model_cfg: dict,
    run_dir: Path,
    seed: int,
    splits: list[str],
) -> None:
    merged = merge_configs(experiment, model_cfg)
    adapter = get_adapter(
        merged["model_name"], merged, data_root=str(_data_root(merged))
    )
    checkpoint = run_dir / "best.pt"
    if not checkpoint.exists():
        checkpoint = train_model(adapter, run_dir, seed=seed)
    else:
        print(f"Already trained: {checkpoint}", flush=True)
    _evaluate_missing(adapter, checkpoint, run_dir, splits)


def run_table_v(experiment_path: Path, seeds: list[int], models: list[str]) -> None:
    base_experiment = load_experiment(experiment_path)
    test_scenarios = list(resolve_test_scenarios(base_experiment))
    for model_name in models:
        base_model = load_model_config(MODEL_CONFIGS[model_name])
        for seed in seeds:
            experiment = copy.deepcopy(base_experiment)
            experiment["seed"] = seed
            model_cfg = copy.deepcopy(base_model)
            model_cfg["experiment_id"] = f"paper5w_table_v_{model_name}_seed{seed}"
            run_dir = ROOT / "runs" / "paper_5w" / "table_v" / model_name / f"seed_{seed}"
            print(f"\n=== Table V: {model_name}, seed {seed} ===", flush=True)
            _run_one(
                experiment,
                model_cfg,
                run_dir,
                seed,
                ["validation", *test_scenarios],
            )


def run_table_vii(
    experiment_path: Path,
    seeds: list[int],
    variants: list[str],
    evaluate_test: bool = False,
) -> None:
    base_experiment = load_experiment(experiment_path)
    base_model = load_model_config(MODEL_CONFIGS["mart_nilm"])
    splits = ["validation"]
    if evaluate_test:
        splits.extend(resolve_test_scenarios(base_experiment))
    for variant in variants:
        for seed in seeds:
            experiment = copy.deepcopy(base_experiment)
            experiment["seed"] = seed
            model_cfg = _paper_architecture_variant(base_model, variant, seed)
            run_dir = (
                ROOT
                / "runs"
                / "paper_5w"
                / "table_vii"
                / variant
                / f"seed_{seed}"
            )
            print(f"\n=== Table VII: {variant}, seed {seed} ===", flush=True)
            _run_one(experiment, model_cfg, run_dir, seed, splits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--table",
        choices=("v", "vii", "all"),
        default="all",
        help="Paper table experiment family.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_CONFIGS), default=list(MODEL_CONFIGS)
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=ARCHITECTURE_VARIANTS,
        default=list(ARCHITECTURE_VARIANTS),
    )
    parser.add_argument(
        "--vii-eval-test",
        action="store_true",
        help=(
            "Evaluate the frozen Table VII checkpoints on the held-out test "
            "houses. This must be used for reporting only, never model selection."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be distinct")
    if args.table in {"v", "all"}:
        run_table_v(args.experiment, args.seeds, args.models)
    if args.table in {"vii", "all"}:
        run_table_vii(
            args.experiment,
            args.seeds,
            args.variants,
            evaluate_test=args.vii_eval_test,
        )


if __name__ == "__main__":
    main()

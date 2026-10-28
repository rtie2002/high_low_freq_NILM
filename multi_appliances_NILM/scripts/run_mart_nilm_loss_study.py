#!/usr/bin/env python
"""Validation-only, one-factor MART-NILM loss ablation and coefficient study.

Every candidate inherits the frozen paper configuration except the named loss
setting.  Relative-energy loss is always disabled.  The screen stage evaluates
only validation data, so held-out houses cannot influence which settings are
confirmed later with three seeds.
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


DEFAULT_EXPERIMENT = ROOT / "config" / "experiment_mixed_ukdale_refit_5w_all_house_split.yaml"
DEFAULT_MODEL = ROOT / "config" / "models" / "mart_nilm.yaml"

# One controlled change per run.  The retained values are included only as the
# common reference; the screen compares each non-reference setting to it.
CANDIDATES: dict[str, dict[str, object]] = {
    "reference": {},
    "no_on_mse": {"power_on_weight": 0.0},
    "no_off_mse": {"power_off_weight": 0.0},
    "no_delta_mse": {"power_delta_weight": 0.0},
    "no_false_positive": {"state_fp_weight": 0.0},
    "on_mse_half": {"power_on_weight": 0.5},
    "on_mse_double": {"power_on_weight": 2.0},
    "off_mse_quarter": {"power_off_weight": 0.25},
    "off_mse_double": {"power_off_weight": 1.0},
    "delta_low": {"power_delta_weight": 0.05},
    "delta_high": {"power_delta_weight": 0.30},
    "delta_all_samples": {"power_delta_on_only": False},
    "fp_half": {"state_fp_weight": 0.5},
    "fp_double": {"state_fp_weight": 2.0},
    "lambda_state_low": {"lambda_state": 0.5},
    "lambda_state_high": {"lambda_state": 1.0},
    "pos_cap_low": {"pos_weight_cap": 6},
    "pos_cap_high": {"pos_weight_cap": 24},
    "balance_limit_low": {"task_balance_ratio_limit": 2.0},
    "balance_limit_high": {"task_balance_ratio_limit": 5.0},
}


def _data_root(merged: dict) -> Path:
    path = Path(merged["data_root"])
    return path if path.is_absolute() else ROOT / path


def _candidate_config(base: dict, name: str, seed: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["experiment_id"] = f"paper5w_loss_study_{name}_seed{seed}"
    # The completed three-seed ablation established that this term is removed.
    cfg["loss"]["power_energy_relative_weight"] = 0.0
    cfg["loss"].update(CANDIDATES[name])
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    parser.add_argument("--root", type=Path, default=ROOT / "runs" / "paper_5w" / "loss_study")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Skip report-only Table III test-house evaluation.",
    )
    args = parser.parse_args()
    unknown = sorted(set(args.candidates) - set(CANDIDATES))
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}. Choices: {sorted(CANDIDATES)}")

    experiment_base = load_experiment(args.experiment)
    model_base = load_model_config(args.model)
    evaluation_splits = ["validation"]
    if not args.validation_only:
        evaluation_splits.extend(resolve_test_scenarios(experiment_base))
    for name in args.candidates:
        for seed in args.seeds:
            experiment = copy.deepcopy(experiment_base)
            experiment["seed"] = seed
            model_cfg = _candidate_config(model_base, name, seed)
            merged = merge_configs(experiment, model_cfg)
            run_dir = args.root / name / f"seed_{seed}"
            print(f"\n=== Loss study: {name}; seed {seed} ===", flush=True)
            adapter = get_adapter(merged["model_name"], merged, data_root=str(_data_root(merged)))
            checkpoint = run_dir / "best.pt"
            if not checkpoint.exists():
                checkpoint = train_model(adapter, run_dir, seed=seed)
            for split in evaluation_splits:
                metrics_path = (
                    run_dir / "validation_metrics.csv"
                    if split == "validation"
                    else run_dir / "test" / split / "metrics.csv"
                )
                if metrics_path.exists():
                    continue
                evaluate_model(adapter, checkpoint, run_dir, split=split, show_cost_summary=False)


if __name__ == "__main__":
    main()

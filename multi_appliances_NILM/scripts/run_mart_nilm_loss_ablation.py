#!/usr/bin/env python
"""Run the fixed three-seed MART-NILM relative-energy-loss ablation.

This is a single controlled intervention: it sets the relative-energy loss
coefficient from 0.25 to 0.0.  The data split, architecture, labels,
augmentation, optimiser, validation-only checkpoint selection, and final
evaluation protocol are inherited unchanged from the frozen paper model.

The held-out test metrics are report-only.  They must not be used to choose a
new coefficient; a follow-up coefficient study is warranted only if this
ablation supports retaining the term on the validation houses.
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
DEFAULT_MODEL = ROOT / "config" / "models" / "mart_nilm.yaml"
DEFAULT_SEEDS = (2026, 2027, 2028)


def _data_root(merged: dict) -> Path:
    path = Path(merged["data_root"])
    return path if path.is_absolute() else ROOT / path


def _loss_ablation_config(base_model: dict, seed: int) -> dict:
    cfg = copy.deepcopy(base_model)
    cfg["experiment_id"] = f"paper5w_table_ix_no_relative_energy_seed{seed}"
    cfg["loss"]["power_energy_relative_weight"] = 0.0
    return cfg


def _evaluate_missing(adapter, checkpoint: Path, run_dir: Path, splits: list[str]) -> None:
    for split in splits:
        metrics_path = (
            run_dir / "validation_metrics.csv"
            if split == "validation"
            else run_dir / "test" / split / "metrics.csv"
        )
        if metrics_path.exists():
            print(f"Already evaluated: {metrics_path}", flush=True)
            continue
        evaluate_model(adapter, checkpoint, run_dir, split=split, show_cost_summary=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be distinct")

    base_experiment = load_experiment(args.experiment)
    base_model = load_model_config(args.model)
    splits = ["validation", *resolve_test_scenarios(base_experiment)]
    for seed in args.seeds:
        experiment = copy.deepcopy(base_experiment)
        experiment["seed"] = seed
        model_cfg = _loss_ablation_config(base_model, seed)
        merged = merge_configs(experiment, model_cfg)
        run_dir = ROOT / "runs" / "paper_5w" / "table_ix" / "no_relative_energy" / f"seed_{seed}"
        print(f"\n=== Relative-energy loss ablation, seed {seed} ===", flush=True)
        adapter = get_adapter(
            merged["model_name"], merged, data_root=str(_data_root(merged))
        )
        checkpoint = run_dir / "best.pt"
        if not checkpoint.exists():
            checkpoint = train_model(adapter, run_dir, seed=seed)
        else:
            print(f"Already trained: {checkpoint}", flush=True)
        _evaluate_missing(adapter, checkpoint, run_dir, splits)


if __name__ == "__main__":
    main()

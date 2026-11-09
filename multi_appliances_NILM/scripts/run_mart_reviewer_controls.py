"""Controlled no-LRE reviewer comparisons; validation selects checkpoints.

Reuse the final no-LRE reference. Change one mechanism per comparison and
evaluate frozen test houses only for reporting, never configuration selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARIANTS = ("random_recomposition", "global_balance", "unbounded_balance", "no_relation")


def control_config(base: dict, variant: str, seed: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["experiment_id"] = f"reviewer_no_lre_{variant}_seed{seed}"
    cfg["loss"]["power_energy_relative_weight"] = 0.0
    if variant == "random_recomposition":
        # Preserve 50% mixing, all loss settings and the real-data half.
        cfg["training"]["random_mix"]["focal_event_sampling"]["enabled"] = False
    elif variant == "global_balance":
        cfg["loss"]["task_balance"] = "equal"
    elif variant == "unbounded_balance":
        cfg["loss"]["task_balance"] = "per_appliance_equal"
    elif variant == "no_relation":
        cfg["architecture"]["cross_appliance"] = {"enabled": False}
    else:
        raise ValueError(variant)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from config import load_experiment, load_model_config, merge_configs
    experiment = load_experiment(ROOT / "config/experiment_mixed_ukdale_refit_5w_all_house_split.yaml")
    base = load_model_config(ROOT / "config/models/mart_nilm.yaml")
    for seed in args.seeds:
        for variant in args.variants:
            cfg = control_config(base, variant, seed)
            run = ROOT / "runs/paper_5w/reviewer_controls" / variant / f"seed_{seed}"
            if args.dry_run:
                print(json.dumps({"variant": variant, "seed": seed, "loss": cfg["loss"], "run": str(run)}))
                continue
            from data.dataloader import resolve_test_scenarios
            from main import get_adapter
            from runner import evaluate_model, train_model
            exp = copy.deepcopy(experiment)
            exp["seed"] = seed
            merged = merge_configs(exp, cfg)
            data_root = Path(merged["data_root"])
            if not data_root.is_absolute():
                data_root = ROOT / data_root
            print(f"=== {variant}, seed {seed} ===", flush=True)
            adapter = get_adapter(merged["model_name"], merged, data_root=str(data_root))
            checkpoint = run / "best.pt"
            # Skip completed runs only; a partial checkpoint must not be
            # mistaken for a completed training run after interruption.
            complete = run / "reviewer_training_complete.json"
            if not complete.exists():
                checkpoint = train_model(adapter, run, seed=seed)
                complete.write_text(json.dumps({"checkpoint": str(checkpoint), "seed": seed}), encoding="utf-8")
            for split in ["validation", *resolve_test_scenarios(exp)]:
                metrics = run / "validation_metrics.csv" if split == "validation" else run / "test" / split / "metrics.csv"
                if not metrics.exists():
                    evaluate_model(adapter, checkpoint, run, split=split, show_cost_summary=False)
            print(f"COMPLETED {variant}, seed {seed}", flush=True)


if __name__ == "__main__":
    main()

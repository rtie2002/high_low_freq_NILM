"""Compare metrics across models for one experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from adapters.common import PredictionBundle
from adapters.dataloader import resolve_state_thresholds_watts
from evaluation.metrics import evaluate_bundle
from evaluation.power_postprocess import resolve_power_postprocess
from evaluation.run_summary import enrich_compare_table


def _experiment_cfg_for_bundle(bundle: PredictionBundle, config_dir: Path) -> dict:
    for path in sorted(config_dir.glob("experiment_*.yaml")):
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg.get("experiment_id") == bundle.experiment_id:
            return cfg
    return {}


def compare_experiment(runs_dir: Path, experiment_id: str, config_dir: Path | None = None) -> pd.DataFrame:
    exp_dir = runs_dir / experiment_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"No runs found at {exp_dir}")

    config_dir = config_dir or Path("config")
    frames = []
    for model_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        pred_path = model_dir / "test_predictions.npz"
        if not pred_path.exists():
            continue
        bundle = PredictionBundle.load(pred_path)
        experiment_cfg = _experiment_cfg_for_bundle(bundle, config_dir)
        eval_cfg = experiment_cfg.get("evaluation", {})
        power_postprocess = (
            resolve_power_postprocess(experiment_cfg, bundle.appliances)
            if experiment_cfg
            else None
        )
        on_thresholds = (
            resolve_state_thresholds_watts(experiment_cfg, bundle.appliances)
            if experiment_cfg
            else None
        )
        frames.append(
            evaluate_bundle(
                bundle,
                sae_period=int(eval_cfg.get("sae_period", 1200)),
                on_threshold_watts=on_thresholds,
                state_label_source="threshold" if on_thresholds is not None else "auto",
                power_postprocess=power_postprocess,
            )
        )

    if not frames:
        raise FileNotFoundError(f"No test_predictions.npz under {exp_dir}")

    table = pd.concat(frames, ignore_index=True)
    table = enrich_compare_table(table, runs_dir, experiment_id)
    out_path = exp_dir / "compare_results.csv"
    table.to_csv(out_path, index=False)

    overall = table[table["appliance"] == "overall"].copy()
    if not overall.empty:
        show_cols = [
            c
            for c in [
                "model",
                "mae",
                "sae",
                "f1",
                "micro_f1",
                "parameters_m",
                "training_time",
                "checkpoint_mb",
                "best_epoch",
                "best_score",
            ]
            if c in overall.columns
        ]
        print("\nModel comparison (overall test metrics + cost):", flush=True)
        print(overall[show_cols].to_string(index=False), flush=True)
        print(f"\nSaved full table: {out_path}", flush=True)

    return table

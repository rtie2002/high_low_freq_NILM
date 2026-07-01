"""Compare metrics across models for one experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from adapters.types import PredictionBundle
from evaluation.metrics import evaluate_bundle


def compare_experiment(runs_dir: Path, experiment_id: str) -> pd.DataFrame:
    exp_dir = runs_dir / experiment_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"No runs found at {exp_dir}")

    frames = []
    for model_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        pred_path = model_dir / "test_predictions.npz"
        if not pred_path.exists():
            continue
        bundle = PredictionBundle.load(pred_path)
        frames.append(evaluate_bundle(bundle))

    if not frames:
        raise FileNotFoundError(f"No test_predictions.npz under {exp_dir}")

    table = pd.concat(frames, ignore_index=True)
    out_path = exp_dir / "compare_results.csv"
    table.to_csv(out_path, index=False)
    return table

"""Fit active-state power centers from source-split submeter watts (no manual labels).

Example (from multi_appliances_NILM/):

  python scripts/fit_active_state_centers.py \\
    --experiment config/experiment_ukdale.yaml \\
    --model config/models/multinilm_fractional.yaml \\
    --output config/active_state_centers_ukdale.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.config import load_yaml
from adapters.dataloader import MultiApplianceDataLoader, resolve_state_thresholds_watts
from evaluation.active_state_snap import (
    fit_centers_from_power_matrix,
    resolve_active_state_snap,
    save_centers,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None, help="Override fit_split (default: train)")
    args = parser.parse_args()

    experiment = load_yaml(args.experiment)
    model_cfg = load_yaml(args.model)
    # Ensure snap block exists for resolve (script may run before user enables eval)
    model_cfg.setdefault("evaluation", {}).setdefault(
        "active_state_snap",
        {
            "enabled": True,
            "method": "kmeans",
            "mode": "segment",
            "n_clusters": {
                "kettle": 1,
                "fridge": 2,
                "dishwasher": 3,
                "washingmachine": 3,
                "microwave": 1,
            },
            "fill_gaps_max": 20,
            "fit_split": "train",
        },
    )
    model_cfg["evaluation"]["active_state_snap"]["enabled"] = True

    cfg = resolve_active_state_snap(experiment, [], model_cfg)
    if cfg is None:
        raise SystemExit("active_state_snap config missing")

    loader = MultiApplianceDataLoader(experiment, model_cfg)
    appliances = loader.appliances
    # Re-resolve with real appliance list
    cfg = resolve_active_state_snap(experiment, appliances, model_cfg)
    assert cfg is not None
    split = args.split or cfg.fit_split
    _, power, _ = loader.get_raw_csv_arrays(split)
    thr = resolve_state_thresholds_watts(experiment, appliances)

    centers = fit_centers_from_power_matrix(power, appliances, cfg, thr)
    path = save_centers(
        args.output,
        centers,
        meta={
            "experiment_id": experiment.get("experiment_id"),
            "fit_split": split,
            "method": cfg.method,
            "n_clusters": cfg.n_clusters,
            "on_thresholds_watts": {a: float(t) for a, t in zip(appliances, thr)},
        },
    )
    print(f"Saved active-state centers -> {path}")
    for app, c in centers.items():
        print(f"  {app}: {np_round(c)}")


def np_round(c) -> list[float]:
    import numpy as np

    return [round(float(v), 1) for v in np.asarray(c)]


if __name__ == "__main__":
    main()

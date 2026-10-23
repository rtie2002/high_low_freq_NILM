#!/usr/bin/env python
"""Audit MART-NILM without validation state calibration or state power gating.

The script reuses frozen Table-V checkpoints and writes results to a separate
directory.  It never retrains a model or changes the original Table-V outputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_experiment, load_model_config, merge_configs
from data.dataloader import resolve_test_scenarios
from main import get_adapter
from runner import evaluate_model


DEFAULT_EXPERIMENT = (
    ROOT / "config" / "experiment_mixed_ukdale_refit_5w_all_house_split.yaml"
)
MART_CONFIG = ROOT / "config" / "models" / "mart_nilm.yaml"
TABLE_V_ROOT = ROOT / "runs" / "paper_5w" / "table_v" / "mart_nilm"
AUDIT_ROOT = ROOT / "runs" / "paper_5w" / "postprocess_audit" / "mart_nilm"


def raw_state_decoding_config(base: dict, seed: int) -> dict:
    """Return MART's frozen architecture with evaluator state rules disabled."""
    cfg = copy.deepcopy(base)
    cfg["experiment_id"] = f"paper5w_mart_raw_state_audit_seed{seed}"
    evaluation = cfg.setdefault("evaluation", {})
    # Keep only model-intrinsic outputs: no threshold sweep, temporal cleanup,
    # or evaluator-side hard state mask on the predicted power trace.
    evaluation["state_calibration"] = {"enabled": False}
    evaluation["gate_power_by_state"] = False
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be distinct")

    base_experiment = load_experiment(args.experiment)
    base_model = load_model_config(MART_CONFIG)
    test_scenarios = list(resolve_test_scenarios(base_experiment))

    for seed in args.seeds:
        checkpoint = TABLE_V_ROOT / f"seed_{seed}" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Frozen Table-V checkpoint is missing: {checkpoint}")

        experiment = copy.deepcopy(base_experiment)
        experiment["seed"] = seed
        model_cfg = raw_state_decoding_config(base_model, seed)
        merged = merge_configs(experiment, model_cfg)
        adapter = get_adapter(
            merged["model_name"], merged, data_root=str(ROOT / merged["data_root"])
        )
        run_dir = AUDIT_ROOT / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "audit_protocol.json").write_text(
            json.dumps(
                {
                    "source_checkpoint": str(checkpoint),
                    "state_calibration": "disabled",
                    "temporal_state_cleanup": "disabled",
                    "evaluator_state_power_gate": "disabled",
                    "power_postprocess": "unchanged from the common experiment configuration",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        for split in ["validation", *test_scenarios]:
            output_dir = run_dir if split == "validation" else run_dir / "test" / split
            metrics_path = output_dir / (
                "validation_metrics.csv" if split == "validation" else "metrics.csv"
            )
            if metrics_path.exists():
                print(f"Already evaluated: {metrics_path}", flush=True)
                continue
            print(f"Evaluating seed {seed}: {split}", flush=True)
            evaluate_model(adapter, checkpoint, run_dir, split=split, show_cost_summary=False)


if __name__ == "__main__":
    main()

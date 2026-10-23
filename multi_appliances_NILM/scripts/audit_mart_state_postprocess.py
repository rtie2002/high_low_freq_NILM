#!/usr/bin/env python
"""Audit frozen Table-V checkpoints under alternate state-decoding protocols.

The script writes results to a separate directory. It never retrains a model
or changes the original Table-V outputs.
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
MODEL_SPECS = {
    "mart_nilm": {
        "config": ROOT / "config" / "models" / "mart_nilm.yaml",
        "adapter": "multinilm_fractional",
    },
    "transfer_multi_appliance": {
        "config": ROOT / "config" / "models" / "transfer_multi_appliance.yaml",
        "adapter": "transfer_multi_appliance",
    },
}
TABLE_V_ROOT = ROOT / "runs" / "paper_5w" / "table_v"
AUDIT_ROOT = ROOT / "runs" / "paper_5w" / "postprocess_audit"


def state_decoding_config(base: dict, seed: int, protocol: str) -> dict:
    """Apply an evaluator-only state-decoding protocol to a frozen model."""
    cfg = copy.deepcopy(base)
    cfg["experiment_id"] = f"paper5w_{cfg['model_name']}_{protocol}_state_audit_seed{seed}"
    evaluation = cfg.setdefault("evaluation", {})
    if protocol == "raw":
        # Keep only model-intrinsic outputs: no threshold sweep, temporal
        # cleanup, or evaluator-side hard state mask on predicted power.
        evaluation["state_calibration"] = {"enabled": False}
    else:
        # Match the state-decoding overlay used in the completed MART, UNet,
        # and MAT Table-V runs. Thresholds are selected only on validation.
        evaluation["state_calibration"] = {
            "enabled": True,
            "split": "validation",
            "threshold_grid": [0.05, 0.98, 0.01],
            "apply_to_power": True,
            "postprocess": {
                "enabled": True,
                "min_on_seconds": {"microwave": 32},
                "merge_gap_seconds": {"microwave": 24},
            },
        }
    evaluation["gate_power_by_state"] = False
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), default="mart_nilm")
    parser.add_argument(
        "--protocol",
        choices=("raw", "calibrated"),
        default="raw",
        help="raw disables evaluator decoding; calibrated enables the validation-only overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be distinct")

    base_experiment = load_experiment(args.experiment)
    spec = MODEL_SPECS[args.model]
    base_model = load_model_config(spec["config"])
    test_scenarios = list(resolve_test_scenarios(base_experiment))

    for seed in args.seeds:
        checkpoint = TABLE_V_ROOT / args.model / f"seed_{seed}" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Frozen Table-V checkpoint is missing: {checkpoint}")

        experiment = copy.deepcopy(base_experiment)
        experiment["seed"] = seed
        model_cfg = state_decoding_config(base_model, seed, args.protocol)
        merged = merge_configs(experiment, model_cfg)
        data_root = Path(merged["data_root"])
        if not data_root.is_absolute():
            data_root = ROOT / data_root
        adapter = get_adapter(spec["adapter"], merged, data_root=str(data_root))
        # Preserve the original MART raw-audit location already used in the
        # experiment log; all other audit variants get an explicit protocol.
        if args.model == "mart_nilm" and args.protocol == "raw":
            run_dir = AUDIT_ROOT / args.model / f"seed_{seed}"
        else:
            run_dir = AUDIT_ROOT / args.model / args.protocol / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "audit_protocol.json").write_text(
            json.dumps(
                {
                    "source_checkpoint": str(checkpoint),
                    "state_decoding_protocol": args.protocol,
                    "state_calibration": "enabled on validation only"
                    if args.protocol == "calibrated"
                    else "disabled",
                    "temporal_state_cleanup": "microwave: 32 s minimum ON; 24 s maximum OFF gap"
                    if args.protocol == "calibrated"
                    else "disabled",
                    "evaluator_state_power_gate": "enabled"
                    if args.protocol == "calibrated"
                    else "disabled",
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

"""Re-evaluate a MATUDA checkpoint with the F1 cast bug fixed."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.config import load_experiment, load_model_config, merge_configs
from adapters.matuda import MATUDAAdapter
from runner import evaluate_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--experiment", type=Path, default=ROOT / "config/experiment_ukdale_matuda.yaml")
    ap.add_argument("--model-config", type=Path, default=ROOT / "config/models/matuda.yaml")
    args = ap.parse_args()
    exp = load_experiment(args.experiment)
    model_cfg = load_model_config(args.model_config)
    cfg = merge_configs(exp, model_cfg)
    adapter = MATUDAAdapter(cfg)
    evaluate_model(adapter, checkpoint=args.checkpoint, run_dir=args.checkpoint.parent)


if __name__ == "__main__":
    main()

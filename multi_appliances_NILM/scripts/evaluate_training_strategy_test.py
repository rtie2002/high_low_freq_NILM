"""Report frozen training-strategy checkpoints on both test houses; never train."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import merge_configs
from main import get_adapter
from runner import evaluate_model

VARIANTS = ("standard_joint_training", "focal_event_mixing", "appliance_wise_balancing", "complete")
HOUSES = ("ukdale_house2", "refit_house20")
METRICS = ("mae", "on_mae", "sae", "f1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, provenance = [], []
    for variant in VARIANTS:
        for seed in (2026, 2027, 2028):
            source = ROOT / "runs/paper_5w"
            source /= "table_v/mart_nilm" if variant == "complete" else f"table_viii/{variant}"
            source /= f"seed_{seed}"
            checkpoint = source / "best.pt"
            # Missing frozen checkpoints must never trigger training.
            assert checkpoint.is_file(), checkpoint
            saved = yaml.safe_load((source / "config_merged.yaml").read_text(encoding="utf-8"))
            cfg = merge_configs(saved["experiment"], saved["model"])
            destination = args.output / variant / f"seed_{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("config_merged.yaml", "state_calibration.json"):
                shutil.copy2(source / name, destination / name)
            adapter = get_adapter(cfg["model_name"], cfg, data_root=str(ROOT / cfg["data_root"]))
            timing = json.loads((source / "training_time.json").read_text(encoding="utf-8"))
            provenance.append({"variant": variant, "seed": seed, "checkpoint": str(checkpoint),
                               "best_epoch": timing["best_epoch"],
                               "energy_relative_weight": saved["model"].get("loss", {}).get("power_energy_relative_weight")})
            for house in HOUSES:
                metric_path = destination / "test" / house / "metrics.csv"
                if not metric_path.exists():
                    evaluate_model(adapter, checkpoint, destination, split=house, show_cost_summary=False)
                frame = pd.read_csv(metric_path).set_index("appliance")
                values = frame.loc[["kettle", "fridge", "dishwasher", "washingmachine", "microwave"], list(METRICS)].mean()
                rows.append({"variant": variant, "seed": seed, "house": house, **values.to_dict()})
    raw = pd.DataFrame(rows)
    raw.to_csv(args.output / "per_house_seed.csv", index=False)
    # Average houses within each seed before computing sample SD across seeds.
    per_seed = raw.groupby(["variant", "seed"], sort=False)[list(METRICS)].mean()
    per_seed.to_csv(args.output / "per_seed.csv")
    summary = per_seed.groupby("variant", sort=False).agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(args.output / "test_summary.csv")
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()

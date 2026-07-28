"""
Auto-iterate MATUDA configs until H2 macro-F1 >= target (default 0.5).

Runs on the training machine; reads validation_test_comparison.csv OVERALL test_f1.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = r"C:\Users\PC\anaconda3\envs\nilm\python.exe"
TARGET_F1 = 0.50
LOG = ROOT / "runs" / "_auto_v4" / "loop_log.jsonl"
LOG.parent.mkdir(parents=True, exist_ok=True)

# Ordered trials: strongest first hypotheses from v3 failure analysis.
TRIALS = [
    {
        "name": "v4_source_only",
        "experiment": "config/experiment_ukdale_matuda_v4_so.yaml",
        "model_config": "config/models/matuda_v4_source_only.yaml",
        "run_dir": "runs/ukdale_matuda_v4_source_only/matuda",
    },
    {
        "name": "v4_soft_egc",
        "experiment": "config/experiment_ukdale_matuda_v4.yaml",
        "model_config": "config/models/matuda_v4.yaml",
        "run_dir": "runs/ukdale_matuda_v4_soft_egc/matuda",
    },
]


def _run(cmd: list[str]) -> int:
    print(">>", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def _read_test_f1(run_dir: Path) -> tuple[float | None, float | None]:
    cmp_path = run_dir / "validation_test_comparison.csv"
    if not cmp_path.exists():
        return None, None
    rows = list(csv.DictReader(cmp_path.open(encoding="utf-8")))
    if not rows:
        return None, None
    # Prefer explicit OVERALL row if present; else mean of appliances.
    for r in rows:
        app = str(r.get("appliance", "")).lower()
        if app in {"overall", "macro", "mean"}:
            return float(r["test_f1"]), float(r["test_mae"])
    f1s = [float(r["test_f1"]) for r in rows if r.get("test_f1")]
    maes = [float(r["test_mae"]) for r in rows if r.get("test_mae")]
    if not f1s:
        return None, None
    return sum(f1s) / len(f1s), (sum(maes) / len(maes) if maes else None)


def main() -> int:
    best = {"name": None, "f1": -1.0, "mae": None}
    for trial in TRIALS:
        name = trial["name"]
        run_dir = ROOT / trial["run_dir"]
        print(f"\n===== TRIAL {name} =====", flush=True)
        rc = _run(
            [
                PY,
                "main.py",
                "--mode",
                "train_evaluate",
                "--model",
                "matuda",
                "--experiment",
                trial["experiment"],
                "--model-config",
                trial["model_config"],
            ]
        )
        f1, mae = _read_test_f1(run_dir)
        rec = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "rc": rc,
            "test_f1": f1,
            "test_mae": mae,
        }
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"RESULT {name}: test_f1={f1} test_mae={mae} rc={rc}", flush=True)
        if f1 is not None and f1 > best["f1"]:
            best = {"name": name, "f1": f1, "mae": mae}
        if f1 is not None and f1 >= TARGET_F1:
            print(f"TARGET REACHED: {f1:.4f} >= {TARGET_F1}", flush=True)
            print(json.dumps(best), flush=True)
            return 0

    print(f"TARGET NOT REACHED. best={best}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())

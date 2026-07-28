"""
Autonomous MATUDA experiment loop (design → train → score → next).

Sequence (publishable UK-DALE protocol, H1+H5 → H2, never H3/H4):
  1. Wait for any current matuda train to finish
  2. Train MATUDA v2 EGC
  3. Train Source-Only (same backbone)
  4. Train Global FC-UDA
  5. Compare test metrics; write SCOREBOARD.md
  6. If EGC H2 F1 < Source-Only + 0.05: flag NEED_REDESIGN

Run detached on training PC:
  powershell -File scripts/auto_experiment_loop.ps1
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = r"C:\Users\PC\anaconda3\envs\nilm\python.exe"
LOG_DIR = ROOT / "runs" / "_auto_loop"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SCOREBOARD = LOG_DIR / "SCOREBOARD.md"

JOBS = [
    {
        "name": "matuda_v2_egc",
        "experiment": "config/experiment_ukdale_matuda_v2.yaml",
        "model_config": "config/models/matuda_v2.yaml",
        "seed": 2026,
    },
    {
        "name": "matuda_v2_source_only",
        "experiment": "config/experiment_ukdale_matuda_v2.yaml",
        "model_config": "config/models/matuda_source_only.yaml",
        "seed": 2026,
        "run_dir": "runs/ukdale_matuda_v2_source_only/matuda",
    },
    {
        "name": "matuda_v2_global_uda",
        "experiment": "config/experiment_ukdale_matuda_v2.yaml",
        "model_config": "config/models/matuda_global_uda.yaml",
        "seed": 2026,
        "run_dir": "runs/ukdale_matuda_v2_global_uda/matuda",
    },
]


def log(msg: str) -> None:
    line = f"{datetime.now().isoformat()}  {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "loop.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def matuda_busy() -> bool:
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Where-Object { $_.CommandLine -match 'main.py.*matuda|train_matuda|run_journal' } | "
                "Measure-Object | Select-Object -ExpandProperty Count",
            ],
            text=True,
        ).strip()
        return int(out or "0") > 0
    except Exception:
        return False


def wait_gpu_free(poll_sec: int = 90) -> None:
    while matuda_busy():
        log("GPU busy — waiting for current MATUDA/P0 job to finish")
        time.sleep(poll_sec)
    log("GPU free")


def run_job(job: dict) -> Path:
    out_log = LOG_DIR / f"{job['name']}.log"
    cmd = [
        PY,
        "main.py",
        "--mode",
        "train_evaluate",
        "--model",
        "matuda",
        "--experiment",
        job["experiment"],
        "--model-config",
        job["model_config"],
        "--seed",
        str(job["seed"]),
    ]
    if job.get("run_dir"):
        cmd += ["--run-dir", job["run_dir"]]
    log(f"START {job['name']}: {' '.join(cmd)}")
    with open(out_log, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    log(f"EXIT {job['name']} code={proc.returncode} log={out_log}")
    if proc.returncode != 0:
        raise RuntimeError(f"{job['name']} failed, see {out_log}")
    return out_log


def _find_metrics(run_hint: str) -> dict | None:
    """Parse last evaluate summary from log or run_manifest / metrics files."""
    # Prefer test_metrics.json if present; else scrape log.
    candidates = list((ROOT / "runs").rglob("test_metrics.json"))
    candidates += list((ROOT / "runs").rglob("run_manifest.json"))
    best = None
    for p in sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True):
        if run_hint.replace("/", "\\") in str(p) or run_hint.split("/")[-1] in str(p):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                best = {"path": str(p), "data": data}
                break
            except Exception:
                continue
    return best


def scrape_log_metrics(log_path: Path) -> dict:
    """Best-effort parse of printed test metrics from main.py log."""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    out = {"f1": None, "mae": None, "sae": None}
    # Look for common printed patterns near end.
    for line in reversed(text.splitlines()[-200:]):
        low = line.lower()
        if out["f1"] is None and "f1" in low and "macro" in low:
            for tok in line.replace(",", " ").split():
                try:
                    v = float(tok)
                    if 0.0 <= v <= 1.0:
                        out["f1"] = v
                        break
                except ValueError:
                    pass
        if "mae" in low and out["mae"] is None:
            for tok in line.replace(",", " ").split():
                try:
                    v = float(tok)
                    if v > 1.0:  # watts-ish
                        out["mae"] = v
                        break
                except ValueError:
                    pass
    return out


def write_scoreboard(rows: list[dict]) -> None:
    lines = [
        "# MATUDA auto scoreboard",
        "",
        f"Updated: {datetime.now().isoformat()}",
        "",
        "| Method | H2 F1 | H2 MAE | H2 SAE | Notes |",
        "|--------|------:|-------:|-------:|-------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r.get('f1', '—')} | {r.get('mae', '—')} | {r.get('sae', '—')} | {r.get('notes', '')} |"
        )
    # Decision rule
    by = {r["name"]: r for r in rows}
    egc = by.get("matuda_v2_egc", {})
    so = by.get("matuda_v2_source_only", {})
    lines.append("")
    if egc.get("f1") is not None and so.get("f1") is not None:
        delta = float(egc["f1"]) - float(so["f1"])
        if delta >= 0.05:
            lines.append(f"**Decision:** KEEP EGC (ΔF1={delta:+.3f}). Next: multi-seed + MultiNILM DA baseline.")
        else:
            lines.append(
                f"**Decision:** NEED_REDESIGN (ΔF1={delta:+.3f} < 0.05). "
                "Next: stronger rare-app weighting, longer source weeks, λ sweep."
            )
    SCOREBOARD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Wrote {SCOREBOARD}")


def main() -> None:
    log("=== AUTO EXPERIMENT LOOP START ===")
    wait_gpu_free()
    rows = []
    for job in JOBS:
        wait_gpu_free()
        log_path = run_job(job)
        metrics = scrape_log_metrics(log_path)
        metrics["name"] = job["name"]
        metrics["notes"] = str(log_path)
        rows.append(metrics)
        write_scoreboard(rows)
    log("=== AUTO EXPERIMENT LOOP DONE ===")


if __name__ == "__main__":
    main()

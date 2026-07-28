"""
Journal P0 runner: chronological H2 split + 3 seeds x {Source-Only, Global FC-UDA, MATUDA}
then core ablations (single seed).

  C:\\Users\\PC\\anaconda3\\envs\\nilm\\python.exe scripts\\run_journal_p0.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SEEDS = [2024, 2025, 2026]
MAIN = [
    ROOT / "configs" / "matuda_s2_b0.yaml",
    ROOT / "configs" / "matuda_s2_b1.yaml",
    ROOT / "configs" / "matuda_s2_m0.yaml",
]
ABLATIONS = [
    ROOT / "configs" / "matuda_ablate_mmd_only.yaml",
    ROOT / "configs" / "matuda_ablate_coral_only.yaml",
    ROOT / "configs" / "matuda_ablate_egc_no_cond.yaml",
    ROOT / "configs" / "matuda_ablate_egc_no_entropy.yaml",
]


def run(cfg: Path, seed: int | None = None) -> None:
    cmd = [PY, str(ROOT / "scripts" / "train_matuda.py"), "--config", str(cfg)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    print("=" * 72, flush=True)
    print("RUN", " ".join(cmd), flush=True)
    log_dir = ROOT / "results" / "_journal_p0_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg.stem + (f"_seed{seed}" if seed is not None else "")
    log_path = log_dir / f"{tag}.log"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    print(f"EXIT {tag} code={proc.returncode} log={log_path}", flush=True)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def main() -> None:
    for cfg in MAIN:
        for seed in SEEDS:
            run(cfg, seed=seed)
    for cfg in ABLATIONS:
        run(cfg, seed=2026)
    print("JOURNAL_P0_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

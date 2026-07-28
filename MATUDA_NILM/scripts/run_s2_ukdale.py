"""Run Stage-2 UK-DALE experiments: B0 → B1 → M0 (EGC-DA)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CONFIGS = [
    ROOT / "configs" / "matuda_s2_b0.yaml",
    ROOT / "configs" / "matuda_s2_b1.yaml",
    ROOT / "configs" / "matuda_s2_m0.yaml",
]


def main() -> None:
    log_dir = ROOT / "results" / "_s2_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for cfg in CONFIGS:
        print("=" * 72, flush=True)
        print(f"START {cfg.name}", flush=True)
        log_path = log_dir / f"{cfg.stem}.log"
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                [PY, str(ROOT / "scripts" / "train_matuda.py"), "--config", str(cfg)],
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        print(f"EXIT {cfg.name} code={proc.returncode} log={log_path}", flush=True)
        if proc.returncode != 0:
            sys.exit(proc.returncode)
    print("S2 complete: B0 + B1 + M0", flush=True)


if __name__ == "__main__":
    main()

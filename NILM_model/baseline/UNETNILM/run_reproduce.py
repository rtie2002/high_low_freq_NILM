#!/usr/bin/env python
"""Reproduce UNet-NILM paper results (UK-DALE, Table 1).

Usage (from UNETNILM/):
  python run_reproduce.py --extract-data
  python run_reproduce.py --verify-results
  python run_reproduce.py --train --epochs 50
  python run_reproduce.py --train --epochs 1 --sample 5000   # quick smoke test
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA_DIR = ROOT / "data"  # extracted .npy files (not the src/data python package)


def _setup_src_path():
    """Prevent UNETNILM/data/ (npy folder) from shadowing src/data/ (python package)."""
    root = ROOT.resolve()
    src = SRC.resolve()
    cleaned = []
    for p in sys.path:
        if not p:
            continue
        try:
            if Path(p).resolve() == root:
                continue
        except OSError:
            pass
        cleaned.append(p)
    sys.path[:] = cleaned
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _require_src():
    load_data = SRC / "data" / "load_data.py"
    if load_data.exists():
        return
    raise SystemExit(
        f"Missing Python package at {load_data}\n"
        "The UNETNILM/src/ folder is not on this machine (incomplete clone/copy).\n"
        "Fix: git pull  OR copy the whole UNETNILM/src/ folder from your laptop.\n"
        f"Then run from: {ROOT}"
    )


def resolve_data_dir() -> Path:
    """Find ukdale npy root (handles accidental data/data/ nesting after re-extract)."""
    for candidate in (DATA_DIR, DATA_DIR / "data"):
        if (candidate / "ukdale" / "training" / "noise_inputs.npy").exists():
            return candidate
    return DATA_DIR


_setup_src_path()


def extract_data():
    data_zip = ROOT / "data.zip"
    if not data_zip.exists():
        raise FileNotFoundError(f"Missing {data_zip}")
    with zipfile.ZipFile(data_zip, "r") as zf:
        zf.extractall(ROOT)
    data_dir = resolve_data_dir()
    train_npy = data_dir / "ukdale" / "training" / "noise_inputs.npy"
    if not train_npy.exists():
        raise RuntimeError(
            f"Extraction failed: {train_npy} not found. "
            "Check data.zip or remove nested data/data/ and re-run --extract-data."
        )
    print(f"Extracted data to {data_dir}")


def verify_results():
    import numpy as np

    results_zip = ROOT / "results.zip"
    npy_path = ROOT / "results" / "ukdale_UNETNiLM_quantilesresults.npy"
    if not npy_path.exists() and results_zip.exists():
        with zipfile.ZipFile(results_zip, "r") as zf:
            zf.extractall(ROOT)
    if not npy_path.exists():
        raise FileNotFoundError("Run with bundled results.zip or train first.")

    r = np.load(npy_path, allow_pickle=True).item()
    print("=== Author saved results (UNet-NILM, UK-DALE) ===")
    print(r["results"])
    print("\nPaper Table 1 targets (UNet-NILM):")
    print("  F1-macro ~ 0.941")
    print("  MAE avg  ~ 11 W (per appliance, median quantile)")


def train(epochs: int, sample: int | None, batch_size: int, denoise: bool):
    _setup_src_path()
    _require_src()
    from data.load_data import ukdale_appliance_data
    from experiment import run_experiments

    data_dir = resolve_data_dir()
    train_npy = data_dir / "ukdale" / "training" / "noise_inputs.npy"
    if not train_npy.exists():
        extract_data()
        data_dir = resolve_data_dir()

    results, save_path = run_experiments(
        model_name="UNETNiLM",
        denoise=denoise,
        batch_size=batch_size,
        epochs=epochs,
        sequence_length=99,
        sample=sample,
        dropout=0.25,
        data="ukdale",
        benchmark="multi-appliance",
        appliances=list(ukdale_appliance_data.keys()),
        appliance_id=None,
        data_path=str(data_dir) + "/",
        checkpoint_path=str(ROOT / "checkpoints") + "/",
        results_path=str(ROOT / "results") + "/",
    )
    out = ROOT / "results" / "reproduction_UNETNiLM_multi_appliance.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.save(out, results)
    print(f"\nSaved run output to {out}")
    print(f"Checkpoint dir: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="UNet-NILM paper reproduction")
    parser.add_argument("--extract-data", action="store_true")
    parser.add_argument("--verify-results", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--sample", type=int, default=None, help="Use first N timesteps only")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--denoise", action="store_true", help="Use denoise_inputs instead of noise_inputs")
    args = parser.parse_args()

    if args.extract_data:
        extract_data()
    if args.verify_results:
        verify_results()
    if args.train:
        train(args.epochs, args.sample, args.batch_size, args.denoise)
    if not any([args.extract_data, args.verify_results, args.train]):
        parser.print_help()


if __name__ == "__main__":
    main()

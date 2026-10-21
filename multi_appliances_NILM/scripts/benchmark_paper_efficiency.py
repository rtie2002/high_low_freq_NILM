#!/usr/bin/env python
"""Benchmark frozen Table V checkpoints under one inference protocol.

Timing covers the complete adapter inference path: data loading, host-to-device
transfer, model forward passes, transfer of predictions to CPU, and overlap
reconstruction.  Model construction, checkpoint loading, calibration, metric
calculation, and plotting are deliberately outside the timed region.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import resolve_tensor_dtype
from main import get_adapter


MODEL_LABELS = {
    "unetnilm": "UNet-NILM",
    "mat_nilm": "MAT-Conv",
    "transfer_multi_appliance": "TransferNILM",
    "mart_nilm": "MART-NILM",
}
# Paper-facing result directories use ``mart_nilm`` while the registered
# runtime adapter retains the historical ``multinilm_fractional`` name.
MODEL_ADAPTER_NAMES = {
    "mart_nilm": "multinilm_fractional",
}
DEFAULT_MODELS = ("mart_nilm", "unetnilm")
DEFAULT_SEEDS = (2026, 2027, 2028)
DEFAULT_SPLITS = ("ukdale_house2", "refit_house20")


def _load_saved_config(run_dir: Path) -> dict:
    path = run_dir / "config_merged.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "model" not in config or "experiment" not in config:
        raise ValueError(f"Invalid merged config: {path}")
    return config


def _data_root(config: dict) -> Path:
    path = Path(config["data_root"])
    return path if path.is_absolute() else ROOT / path


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_split(adapter, model, device, split: str, warmup_batches: int) -> dict:
    # Warm-up is untimed and uses a separate loader, so every measured pass
    # starts from the first window of the house.
    warmup_loader = adapter.build_dataloader(split)
    adapter.predict_dataloader(
        model,
        warmup_loader,
        device,
        split=split,
        max_batches=warmup_batches,
    )
    _synchronise(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    loader = adapter.build_dataloader(split)
    started = time.perf_counter()
    bundle = adapter.predict_dataloader(model, loader, device, split=split)
    _synchronise(device)
    elapsed = time.perf_counter() - started
    n_points = int(len(bundle.y_pred_watts))
    return {
        "split": split,
        "seconds": elapsed,
        "covered_timesteps": n_points,
        "timesteps_per_second": n_points / elapsed,
        "windows": int(len(loader.dataset)),
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
            if device.type == "cuda"
            else None
        ),
    }


def benchmark_run(
    run_dir: Path,
    model_name: str,
    seed: int,
    splits: list[str],
    batch_size: int,
    warmup_batches: int,
) -> Path:
    checkpoint = run_dir / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    config = _load_saved_config(run_dir)
    config["model"]["training"]["batch_size"] = batch_size
    config["model"]["training"]["num_workers"] = 0
    config["model"]["training"]["persistent_workers"] = False
    config["model"]["training"].pop("prefetch_factor", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, tensor_dtype = resolve_tensor_dtype(config["model"])
    adapter_name = MODEL_ADAPTER_NAMES.get(model_name, model_name)
    adapter = get_adapter(adapter_name, config, data_root=str(_data_root(config)))
    model = adapter.build_model(device)
    if tensor_dtype == torch.float64:
        model = model.double()
    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    model.eval()

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results = [
        _benchmark_split(adapter, model, device, split, warmup_batches)
        for split in splits
    ]
    payload = {
        "model": MODEL_LABELS[model_name],
        "model_name": model_name,
        "seed": seed,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "batch_size": batch_size,
        "warmup_batches": warmup_batches,
        "timing_scope": "data loading + H2D + forward + D2H + output reconstruction",
        "parameters_trainable": parameter_count,
        "parameters_trainable_millions": parameter_count / 1_000_000.0,
        "checkpoint_size_mib": checkpoint.stat().st_size / (1024.0**2),
        "splits": results,
    }
    output = run_dir / "inference_benchmark.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved {output}", flush=True)
    for row in results:
        print(
            f"  {row['split']}: {row['seconds']:.3f} s, "
            f"{row['timesteps_per_second']:.1f} timesteps/s, "
            f"peak {row['peak_gpu_memory_mib']:.1f} MiB",
            flush=True,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs" / "paper_5w")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_LABELS), default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-batches", type=int, default=5)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.warmup_batches <= 0:
        raise ValueError("batch-size and warmup-batches must be positive")

    for model_name in args.models:
        for seed in args.seeds:
            run_dir = args.run_root / "table_v" / model_name / f"seed_{seed}"
            print(f"\n=== {MODEL_LABELS[model_name]}, seed {seed} ===", flush=True)
            benchmark_run(
                run_dir,
                model_name,
                seed,
                args.splits,
                args.batch_size,
                args.warmup_batches,
            )


if __name__ == "__main__":
    main()

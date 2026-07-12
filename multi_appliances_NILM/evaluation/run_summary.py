"""Training run metadata: model size, timing, and post-eval summary display."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn


def count_model_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters for fair model comparison."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "parameters_total": int(total),
        "parameters_trainable": int(trainable),
        "parameters_total_millions": round(total / 1_000_000, 3),
        "parameters_trainable_millions": round(trainable / 1_000_000, 3),
    }


def checkpoint_size_mb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    return round(path.stat().st_size / (1024 * 1024), 3)


def format_parameter_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def _format_params(count: int) -> str:
    return format_parameter_count(count)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_hardware_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        info["gpu_total_memory_gb"] = round(props.total_memory / (1024**3), 2)
    return info


def save_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_json(path) or {}
    existing.update(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def load_run_summary(run_dir: Path) -> dict[str, Any]:
    """Merge run_manifest.json and training_time.json when present."""
    summary: dict[str, Any] = {}
    manifest = _load_json(run_dir / "run_manifest.json")
    timing = _load_json(run_dir / "training_time.json")
    if manifest:
        summary.update(manifest)
    if timing:
        summary.update(timing)
    return summary


def print_run_cost_summary(run_dir: Path, *, title: str = "Run summary") -> None:
    """Print model size and training time after metrics evaluation."""
    summary = load_run_summary(run_dir)
    if not summary:
        print(f"\n{title}: no run_manifest.json / training_time.json in {run_dir}")
        return

    lines = [f"\n{title}:"]
    model_name = summary.get("model_name")
    if model_name:
        lines.append(f"  model:              {model_name}")

    exp_id = summary.get("experiment_id")
    if exp_id:
        lines.append(f"  experiment:         {exp_id}")

    total_params = summary.get("parameters_total")
    trainable_params = summary.get("parameters_trainable")
    if total_params is not None:
        trainable_note = ""
        if trainable_params is not None and trainable_params != total_params:
            trainable_note = f" ({_format_params(int(trainable_params))} trainable)"
        lines.append(f"  parameters:         {_format_params(int(total_params))}{trainable_note}")

    ckpt_mb = summary.get("checkpoint_size_mb")
    ckpt_name = summary.get("checkpoint_file", "best.pt")
    if ckpt_mb is not None:
        lines.append(f"  checkpoint:         {ckpt_name} ({ckpt_mb:.2f} MB)")

    total_fmt = summary.get("total_formatted")
    if total_fmt:
        epochs = summary.get("epochs_completed")
        best_epoch = summary.get("best_epoch")
        avg_epoch = summary.get("avg_epoch_formatted")
        train_line = f"  training time:      {total_fmt}"
        if epochs is not None:
            train_line += f" ({epochs} epochs"
            if best_epoch is not None:
                train_line += f", best epoch {best_epoch}"
            train_line += ")"
        if avg_epoch:
            train_line += f", avg {avg_epoch}/epoch"
        lines.append(train_line)

    best_score = summary.get("best_score")
    monitor = summary.get("checkpoint_monitor")
    if best_score is not None and monitor:
        lines.append(f"  best {monitor}:       {best_score:.4f}")

    batch_size = summary.get("batch_size")
    if batch_size is not None:
        lines.append(f"  batch size:         {batch_size}")

    device = summary.get("device")
    gpu_name = summary.get("gpu_name")
    if gpu_name:
        lines.append(f"  hardware:           {gpu_name}")
    elif device:
        lines.append(f"  hardware:           {device}")

    seed = summary.get("seed")
    if seed is not None:
        lines.append(f"  seed:               {seed}")

    print("\n".join(lines), flush=True)


def print_evaluation_report(
    metrics: pd.DataFrame,
    run_dir: Path,
    *,
    split: str,
    show_cost_summary: bool = True,
) -> None:
    """Print per-appliance metrics table plus optional model size / training cost."""
    per_app = metrics[metrics["appliance"] != "overall"]
    overall = metrics[metrics["appliance"] == "overall"]

    print(f"\n{split.capitalize()} metrics:", flush=True)
    if not per_app.empty:
        print(per_app[["appliance", "mae", "sae", "f1"]].to_string(index=False), flush=True)
    if not overall.empty:
        row = overall.iloc[0]
        print(
            f"overall  mae={row['mae']:.4f}  sae={row['sae']:.4f}  "
            f"f1={row['f1']:.4f}  micro_f1={row['micro_f1']:.4f}",
            flush=True,
        )
    if show_cost_summary:
        print_run_cost_summary(run_dir, title="Training cost & model size")


def print_val_test_comparison(run_dir: Path) -> None:
    """Compare validation vs test metrics to inspect generalization gap."""
    val_path = run_dir / "validation_metrics.csv"
    test_path = run_dir / "test_metrics.csv"
    if not val_path.exists() or not test_path.exists():
        missing = [p.name for p in (val_path, test_path) if not p.exists()]
        print(f"\nValidation vs test comparison skipped (missing: {', '.join(missing)})", flush=True)
        return

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    val_app = val_df[val_df["appliance"] != "overall"].set_index("appliance")
    test_app = test_df[test_df["appliance"] != "overall"].set_index("appliance")
    appliances = [app for app in val_app.index if app in test_app.index]

    rows = []
    for app in appliances:
        v = val_app.loc[app]
        t = test_app.loc[app]
        rows.append({
            "appliance": app,
            "val_mae": float(v["mae"]),
            "test_mae": float(t["mae"]),
            "mae_gap": float(t["mae"] - v["mae"]),
            "val_f1": float(v["f1"]),
            "test_f1": float(t["f1"]),
            "f1_gap": float(t["f1"] - v["f1"]),
            "val_sae": float(v["sae"]),
            "test_sae": float(t["sae"]),
        })

    compare_df = pd.DataFrame(rows)
    compare_path = run_dir / "validation_test_comparison.csv"
    compare_df.to_csv(compare_path, index=False)

    val_overall = val_df[val_df["appliance"] == "overall"].iloc[0]
    test_overall = test_df[test_df["appliance"] == "overall"].iloc[0]
    mae_gap = float(test_overall["mae"] - val_overall["mae"])
    f1_gap = float(test_overall["f1"] - val_overall["f1"])

    print("\nValidation vs test comparison:", flush=True)
    print(
        compare_df[
            ["appliance", "val_mae", "test_mae", "mae_gap", "val_f1", "test_f1", "f1_gap"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        flush=True,
    )
    print(
        f"overall  val_mae={val_overall['mae']:.4f}  test_mae={test_overall['mae']:.4f}  "
        f"mae_gap={mae_gap:+.4f}  "
        f"val_f1={val_overall['f1']:.4f}  test_f1={test_overall['f1']:.4f}  "
        f"f1_gap={f1_gap:+.4f}",
        flush=True,
    )
    if abs(mae_gap) < 5 and abs(f1_gap) < 0.05:
        print("  transfer note: validation and test are close — similar generalization.", flush=True)
    elif test_overall["mae"] > val_overall["mae"] or test_overall["f1"] < val_overall["f1"]:
        print("  transfer note: test is worse than validation — possible domain/house gap.", flush=True)
    else:
        print("  transfer note: test is better than validation — check split overlap or leakage.", flush=True)
    print(f"Saved comparison table: {compare_path}", flush=True)


def enrich_compare_table(table: pd.DataFrame, runs_dir: Path, experiment_id: str) -> pd.DataFrame:
    """Add parameter count and training time columns for cross-model comparison."""
    rows = []
    for model_name, group in table.groupby("model"):
        run_dir = runs_dir / experiment_id / str(model_name)
        summary = load_run_summary(run_dir)
        base = group.copy()
        base["parameters_m"] = summary.get("parameters_total_millions")
        base["trainable_parameters_m"] = summary.get("parameters_trainable_millions")
        base["training_time"] = summary.get("total_formatted")
        base["training_seconds"] = summary.get("total_seconds")
        base["checkpoint_mb"] = summary.get("checkpoint_size_mb")
        base["best_epoch"] = summary.get("best_epoch")
        base["best_score"] = summary.get("best_score")
        rows.append(base)
    return pd.concat(rows, ignore_index=True) if rows else table

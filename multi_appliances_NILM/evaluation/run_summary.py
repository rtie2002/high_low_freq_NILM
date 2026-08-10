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


def _box(title: str, width: int = 78) -> tuple[str, str, str]:
    """ASCII frame lines for a titled console section."""
    inner = width - 2
    title_bit = f" {title} "
    fill = max(0, inner - len(title_bit))
    left = fill // 2
    right = fill - left
    top = "+" + ("-" * left) + title_bit + ("-" * right) + "+"
    mid = "|" + (" " * inner) + "|"
    bot = "+" + ("-" * inner) + "+"
    return top, mid, bot


def _row(width: int, left: str, right: str = "") -> str:
    inner = width - 2
    text = f" {left}"
    if right:
        pad = max(1, inner - len(text) - len(right) - 1)
        text = text + (" " * pad) + right + " "
    if len(text) < inner:
        text = text + (" " * (inner - len(text)))
    return "|" + text[:inner] + "|"


def print_run_cost_summary(run_dir: Path, *, title: str = "Run summary") -> None:
    """Print model size and training time after metrics evaluation."""
    summary = load_run_summary(run_dir)
    if not summary:
        print(f"\n{title}: no run_manifest.json / training_time.json in {run_dir}")
        return

    width = 78
    top, _, bot = _box(title.upper(), width)
    lines = ["", top]

    model_name = summary.get("model_name")
    if model_name:
        lines.append(_row(width, "Model", str(model_name)))

    exp_id = summary.get("experiment_id")
    if exp_id:
        lines.append(_row(width, "Experiment", str(exp_id)))

    total_params = summary.get("parameters_total")
    trainable_params = summary.get("parameters_trainable")
    if total_params is not None:
        trainable_note = ""
        if trainable_params is not None and trainable_params != total_params:
            trainable_note = f" ({_format_params(int(trainable_params))} trainable)"
        lines.append(_row(width, "Parameters", f"{_format_params(int(total_params))}{trainable_note}"))

    ckpt_mb = summary.get("checkpoint_size_mb")
    ckpt_name = summary.get("checkpoint_file", "best.pt")
    if ckpt_mb is not None:
        lines.append(_row(width, "Checkpoint", f"{ckpt_name} ({ckpt_mb:.2f} MB)"))

    total_fmt = summary.get("total_formatted")
    if total_fmt:
        epochs = summary.get("epochs_completed")
        best_epoch = summary.get("best_epoch")
        avg_epoch = summary.get("avg_epoch_formatted")
        detail = str(total_fmt)
        if epochs is not None:
            detail += f"  |  {epochs} epochs"
            if best_epoch is not None:
                detail += f"  (best @{best_epoch})"
        if avg_epoch:
            detail += f"  |  avg {avg_epoch}/epoch"
        lines.append(_row(width, "Training", detail))

    best_score = summary.get("best_score")
    monitor = summary.get("checkpoint_monitor")
    if best_score is not None and monitor:
        lines.append(_row(width, f"Best {monitor}", f"{float(best_score):.4f}"))

    batch_size = summary.get("batch_size")
    if batch_size is not None:
        lines.append(_row(width, "Batch size", str(batch_size)))

    device = summary.get("device")
    gpu_name = summary.get("gpu_name")
    lines.append(_row(width, "Hardware", str(gpu_name or device or "n/a")))

    seed = summary.get("seed")
    if seed is not None:
        lines.append(_row(width, "Seed", str(seed)))

    lines.append(bot)
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

    width = 88
    top, _, bot = _box(f"{split.upper()} METRICS", width)
    print(f"\n{top}", flush=True)
    header = (
        f"{'appliance':<16}"
        f"{'MAE':>10}"
        f"{'SAE':>10}"
        f"{'maF1':>10}"
        f"{'miF1':>10}"
    )
    print(_row(width, header), flush=True)
    print(_row(width, "-" * 56), flush=True)

    if not per_app.empty:
        for _, r in per_app.iterrows():
            line = (
                f"{str(r['appliance']):<16}"
                f"{float(r['mae']):>10.4f}"
                f"{float(r['sae']):>10.4f}"
                f"{float(r['f1']):>10.4f}"
                f"{'—':>10}"
            )
            print(_row(width, line), flush=True)

    if not overall.empty:
        row = overall.iloc[0]
        print(_row(width, "-" * 56), flush=True)
        macro = float(row["macro_f1"]) if "macro_f1" in row.index and pd.notna(row["macro_f1"]) else float(row["f1"])
        micro = float(row["micro_f1"]) if pd.notna(row["micro_f1"]) else float("nan")
        overall_line = (
            f"{'OVERALL':<16}"
            f"{float(row['mae']):>10.4f}"
            f"{float(row['sae']):>10.4f}"
            f"{macro:>10.4f}"
            f"{micro:>10.4f}"
        )
        print(_row(width, overall_line), flush=True)
        print(
            _row(
                width,
                "maF1=macro mean of per-app F1; miF1=micro pooled TP/FP/FN",
            ),
            flush=True,
        )
    print(bot, flush=True)

    if show_cost_summary:
        print_run_cost_summary(run_dir, title="Training cost & model size")


def print_val_test_comparison(run_dir: Path) -> None:
    """Compare validation vs test metrics to inspect generalization gap."""
    from evaluation.plots import build_val_test_comparison_frame, save_val_test_comparison_figure

    val_path = run_dir / "validation_metrics.csv"
    test_path = run_dir / "test_metrics.csv"
    if not val_path.exists() or not test_path.exists():
        missing = [p.name for p in (val_path, test_path) if not p.exists()]
        print(f"\nValidation vs test comparison skipped (missing: {', '.join(missing)})", flush=True)
        return

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    compare_df = build_val_test_comparison_frame(val_df, test_df)
    # Appliance rows only in the CSV summary of gaps (overall included).
    compare_path = run_dir / "validation_test_comparison.csv"
    compare_df.to_csv(compare_path, index=False)

    # Same table as a PNG (final best-checkpoint evaluate).
    fig_path = run_dir / "validation_test_comparison.png"
    save_val_test_comparison_figure(
        val_df,
        test_df,
        fig_path,
        title="best ckpt val vs test",
    )

    width = 110
    top, _, bot = _box("VALIDATION vs TEST  (transfer / house gap)", width)
    print(f"\n{top}", flush=True)
    hdr = (
        f"{'appliance':<16}"
        f"{'val_MAE':>9}{'test_MAE':>10}{'MAE_gap':>9}"
        f"{'val_maF1':>10}{'test_maF1':>10}{'maF1_gap':>10}"
        f"{'val_miF1':>10}{'test_miF1':>10}{'miF1_gap':>10}"
    )
    print(_row(width, hdr), flush=True)
    print(_row(width, "-" * 98), flush=True)

    def _f1(x) -> str:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return f"{'—':>10}"
        return f"{float(x):>10.4f}"

    def _gap(x) -> str:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return f"{'—':>10}"
        return f"{float(x):>+10.4f}"

    for _, r in compare_df.iterrows():
        if str(r["appliance"]) == "overall":
            print(_row(width, "-" * 98), flush=True)
        line = (
            f"{str(r['appliance']):<16}"
            f"{float(r['val_MAE']):>9.2f}{float(r['test_MAE']):>10.2f}{float(r['MAE_gap']):>+9.2f}"
            f"{_f1(r['val_maF1'])}{_f1(r['test_maF1'])}{_gap(r['maF1_gap'])}"
            f"{_f1(r['val_miF1'])}{_f1(r['test_miF1'])}{_gap(r['miF1_gap'])}"
        )
        print(_row(width, line), flush=True)

    overall = compare_df[compare_df["appliance"] == "overall"]
    note = "maF1=macro; miF1=micro (pooled). "
    if not overall.empty:
        mae_gap = float(overall.iloc[0]["MAE_gap"])
        f1_gap = float(overall.iloc[0]["maF1_gap"])
        if abs(mae_gap) < 5 and abs(f1_gap) < 0.05:
            note += "Transfer: val ≈ test."
        elif mae_gap > 0 or f1_gap < 0:
            note += "Transfer: test weaker than val — domain/house gap remains."
        else:
            note += "Transfer: test better than val — check split/leakage."
    print(_row(width, note), flush=True)
    print(bot, flush=True)
    print(f"Saved comparison table: {compare_path}", flush=True)
    print(f"Saved comparison figure: {fig_path}", flush=True)


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

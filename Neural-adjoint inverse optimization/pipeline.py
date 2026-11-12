"""Neural-Adjoint NILM End-to-End Pipeline.

Runs the full two-stage workflow described in
  Benchmarking Deep Inverse Models over Time and the Neural-Adjoint Method
  (Broad et al. 2021)

Stages
------
1. Train the appliance-conditioned CNN forward model (Stage 1).
2. Freeze the decoder; run latent inverse optimisation on the validation
   split (Stage 2 — validation).
3. Run Stage-2 inference on two held-out test houses in parallel:
     • REFIT House 20
     • UK-DALE House 2

Everything is timed, logged to the terminal with rich progress banners, and
summarised in a JSON + markdown pipeline report.

Usage
-----
  python pipeline.py [--config configs/stage1.yaml] [--run-dir runs/pipeline_<timestamp>]
                     [--overwrite] [--smoke]

Flags
-----
  --config     Stage-1 config path (default: configs/stage1.yaml).
  --run-dir    Output root; subdirectories are created for each stage.
  --overwrite  Allow reusing an existing run directory.
  --smoke      One epoch, two batches, two inverse steps — fast sanity check.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ─── pretty terminal helpers ──────────────────────────────────────────────────

BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
RESET  = "\033[0m"
LINE   = "─" * 72


def banner(title: str, colour: str = CYAN) -> None:
    print(f"\n{colour}{BOLD}{LINE}{RESET}")
    print(f"{colour}{BOLD}  {title}{RESET}")
    print(f"{colour}{BOLD}{LINE}{RESET}")


def info(msg: str) -> None:
    print(f"  {msg}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, s = divmod(seconds, 60)
    return f"{int(minutes)} min {s:.1f} s"


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ─── model-size helpers ───────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def checkpoint_size_bytes(path: Path) -> int:
    return path.stat().st_size


# ─── stage 1 ─────────────────────────────────────────────────────────────────

def run_stage1(
    config_path: Path,
    run_dir: Path,
    *,
    smoke: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Train the forward model and return timing + metric info."""
    banner("STAGE 1 — Forward model training", CYAN)
    info(f"Config : {config_path}")
    info(f"Run dir: {run_dir}")
    info(f"Smoke  : {smoke}")

    # Import lazily so the script can at least print the banner even if
    # a dependency is missing.
    from stage1_forward.train import train as train_forward

    t0 = time.perf_counter()
    output_dir = train_forward(config_path, run_dir=run_dir, smoke=smoke, overwrite=overwrite)
    elapsed = time.perf_counter() - t0

    checkpoint = output_dir / "best.pt"
    manifest_path = output_dir / "run_manifest.json"
    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    from models.cnn import build_forward_model
    model_tmp = build_forward_model(ckpt["model_config"])
    n_params = count_parameters(model_tmp)
    del model_tmp

    ckpt_bytes = checkpoint_size_bytes(checkpoint)

    ok(f"Forward training finished in {fmt_seconds(elapsed)}")
    ok(f"Best epoch : {manifest['best_epoch']}  |  "
       f"Val MSE : {manifest['best_validation_mse']:.6g}")
    ok(f"Parameters : {n_params:,}")
    ok(f"Checkpoint : {fmt_bytes(ckpt_bytes)}  ({checkpoint})")

    return {
        "stage": "stage1_forward",
        "run_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        "elapsed_seconds": elapsed,
        "best_epoch": manifest["best_epoch"],
        "best_validation_mse": manifest["best_validation_mse"],
        "train_windows": manifest["train_windows"],
        "validation_windows": manifest["validation_windows"],
        "parameter_count": n_params,
        "checkpoint_bytes": ckpt_bytes,
    }


# ─── stage 2 (single split) ───────────────────────────────────────────────────

def _run_one_inverse(
    *,
    config_path: Path,
    checkpoint: Path,
    run_dir: Path,
    split: str,
    steps: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    train_mean_init: bool | None,
    prior_weight: float | None,
    max_windows: int | None,
    overwrite: bool,
    device: str | None,
    smoke: bool,
) -> dict[str, Any]:
    """Run Stage-2 inverse optimisation for one split and return timing + metrics."""
    from utils.config import load_yaml

    # Merge stage2 defaults with any overrides.
    config = load_yaml(config_path)
    if "stage1_config" in config:
        stage1_path = config_path.parent / config["stage1_config"]
        stage1_config = load_yaml(stage1_path)
        stage1_config["inference"] = config["inference"]
        config = stage1_config
        effective_config_path = stage1_path
    else:
        effective_config_path = config_path

    settings = config["inference"]
    resolved_steps        = steps         if steps         is not None else int(settings["steps"])
    resolved_lr           = learning_rate if learning_rate is not None else float(settings["learning_rate"])
    resolved_batch        = batch_size    if batch_size    is not None else int(settings["batch_size"])
    resolved_init         = train_mean_init if train_mean_init is not None else bool(settings.get("train_mean_init", False))
    resolved_prior        = prior_weight  if prior_weight  is not None else float(settings.get("prior_weight", 0.0))
    resolved_device_name  = device        if device        is not None else str(settings.get("device", "auto"))

    if smoke:
        resolved_steps = 2
        resolved_batch = 4
        max_windows    = max_windows or 2

    if resolved_device_name == "auto":
        resolved_device_name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(resolved_device_name)

    import math as _math
    seed = int(config["run"]["seed"])
    gen  = torch.Generator(device=dev).manual_seed(seed)

    ckpt = torch.load(checkpoint, map_location=dev, weights_only=True)
    from models.cnn import build_forward_model
    model = build_forward_model(ckpt["model_config"]).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    scale  = float(ckpt["power_scale_watts"])
    length = int(ckpt["window_length"])
    stride = int(config["data"]["window_stride"])

    # Freeze decoder weights — Stage-2 optimises only the latent codes z.
    model.eval().requires_grad_(False)

    from utils.protocol import load_protocol, save_inverse_evaluation
    from stage2_inverse.losses import estimate_constant_background, meter_mse, diagonal_latent_prior
    from stage2_inverse.inverse import solve_windows

    protocol = load_protocol(
        experiment_path=effective_config_path,
        window_length=length,
        stride=stride,
        batch_size=resolved_batch,
    )

    # Build train-mean initial code if requested.
    initial_code_center = None
    latent_prior        = None
    prior_starts        = []
    if resolved_init or resolved_prior:
        center_windows = int(settings.get("center_windows", 128))
        all_starts     = protocol.loader.build_dataset("train").indices
        count          = min(center_windows, len(all_starts))
        picks          = np.linspace(0, len(all_starts) - 1, count, dtype=int)
        prior_starts   = all_starts[picks].tolist()
        with torch.no_grad():
            train = protocol.loader.get_split("train")
            rows  = all_starts[picks, None] + np.arange(length)
            training_power = torch.as_tensor(
                train.targets[rows] / scale, device=dev
            ).transpose(1, 2)
            codes                = model.encode(training_power)
            initial_code_center  = codes.mean((0, 3), keepdim=True)
            spread               = codes.std((0, 3), correction=0, keepdim=True)
            prior_min_std        = float(settings.get("prior_min_std", 0.01))
            latent_prior         = (initial_code_center, spread.clamp_min(prior_min_std))

    dataset     = protocol.loader.build_dataset(split)
    mains_watts = protocol.loader.get_split(split).inputs
    starts      = dataset.indices[:max_windows]

    info(f"  Split: {split}  |  Windows: {len(starts)}  |  Steps: {resolved_steps}  |  "
         f"LR: {resolved_lr}  |  Device: {resolved_device_name}")

    offsets        = np.arange(length)
    window_outputs = []
    initial_losses = []
    best_losses    = []

    t0 = time.perf_counter()
    n_batches = math.ceil(len(starts) / resolved_batch)
    for batch_idx, offset in enumerate(range(0, len(starts), resolved_batch)):
        rows   = starts[offset : offset + resolved_batch, None] + offsets
        meter  = torch.as_tensor(mains_watts[rows] / scale, device=dev, dtype=torch.float32)
        init_c = (
            initial_code_center.expand(len(rows), -1, -1, _math.ceil(length / model.DOWNSAMPLE))
            if resolved_init else None
        )
        power, background, initial_mse, best_mse = solve_windows(
            model, meter,
            steps=resolved_steps,
            learning_rate=resolved_lr,
            generator=gen,
            initial_codes=init_c,
            latent_prior=latent_prior if resolved_prior else None,
            prior_weight=resolved_prior,
        )
        bg_trace    = background[:, None, None].expand(-1, 1, length)
        predictions = torch.cat((power, bg_trace), dim=1).transpose(1, 2)
        window_outputs.append(predictions.cpu().numpy() * scale)
        initial_losses.append(initial_mse.cpu().numpy())
        best_losses.append(best_mse.cpu().numpy())

        # Progress bar: print every 10 % (or every batch for small runs).
        report_every = max(1, n_batches // 10)
        if (batch_idx + 1) % report_every == 0 or batch_idx + 1 == n_batches:
            pct    = 100 * (batch_idx + 1) / n_batches
            so_far = time.perf_counter() - t0
            eta    = so_far / (batch_idx + 1) * (n_batches - batch_idx - 1)
            print(
                f"    [{split}] batch {batch_idx + 1}/{n_batches} "
                f"({pct:5.1f} %)  elapsed {fmt_seconds(so_far)}  "
                f"ETA {fmt_seconds(eta)}",
                flush=True,
            )
    elapsed = time.perf_counter() - t0

    merged, covered_rows = protocol.loader.reconstruct_timeline_from_windows(
        split, np.concatenate(window_outputs)
    )
    appliances_watts = merged[:, :-1]
    background_watts = merged[:, -1]
    meter_fit_watts  = appliances_watts.sum(axis=1) + background_watts
    observed_watts   = mains_watts[covered_rows]

    agg_mae = float(np.abs(meter_fit_watts - observed_watts).mean())
    init_rmse = float(np.sqrt(np.concatenate(initial_losses).mean()) * scale)
    best_rmse = float(np.sqrt(np.concatenate(best_losses).mean())   * scale)

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_base = {
        "checkpoint"               : str(checkpoint.resolve()),
        "config"                   : str(config_path),
        "split"                    : split,
        "device"                   : str(dev),
        "seed"                     : seed,
        "initialisation"           : "train_mean" if resolved_init else "random",
        "prior_weight"             : resolved_prior,
        "prior_train_window_starts": prior_starts,
        "steps"                    : resolved_steps,
        "learning_rate"            : resolved_lr,
        "batch_size"               : resolved_batch,
        "power_scale_watts"        : scale,
        "window_length"            : length,
        "window_stride"            : stride,
        "background"               : "nonnegative_constant_per_window",
        "evaluated_windows"        : len(starts),
        "available_windows"        : len(dataset),
        "covered_rows"             : len(covered_rows),
        "available_rows"           : len(mains_watts),
        "initial_window_meter_mse_watts2": float(np.concatenate(initial_losses).mean() * scale ** 2),
        "best_window_meter_mse_watts2"   : float(np.concatenate(best_losses).mean()    * scale ** 2),
        "selected_window_meter_mse_watts2": float(np.concatenate(best_losses).mean()   * scale ** 2),
        "aggregate_reconstruction_mae_watts": agg_mae,
        "inference_seconds"        : elapsed,
    }
    save_inverse_evaluation(
        protocol=protocol,
        split=split,
        run_dir=run_dir,
        model_name="appliance_conditioned_cnn_inverse",
        csv_timesteps=covered_rows,
        predicted_appliance_watts=appliances_watts,
        manifest=manifest_base,
        random_seed=seed,
        waveform_periods=int(config.get("evaluation", {}).get("on_periods", 5)),
        waveform_dpi=int(config.get("evaluation", {}).get("dpi", 300)),
    )
    np.savez_compressed(
        run_dir / "meter_diagnostics.npz",
        csv_timesteps=covered_rows,
        observed_watts=observed_watts,
        reconstructed_watts=meter_fit_watts,
        background_watts=background_watts,
        raw_appliance_watts=appliances_watts,
    )

    # Read the saved macro metrics for the summary table.
    metrics_csv = run_dir / "metrics.csv"
    macro_metrics: dict[str, float] = {}
    if metrics_csv.exists():
        import csv as _csv
        with metrics_csv.open(encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            for mrow in reader:
                if mrow.get("appliance", "").lower() == "macro":
                    for k, v in mrow.items():
                        try:
                            macro_metrics[k] = float(v)
                        except (ValueError, TypeError):
                            pass

    ok(f"[{split}] Inverse done in {fmt_seconds(elapsed)}  |  "
       f"AggMAE {agg_mae:.1f} W  |  "
       f"InitRMSE {init_rmse:.1f} W → BestRMSE {best_rmse:.1f} W")

    return {
        "stage"               : f"stage2_{split}",
        "split"               : split,
        "run_dir"             : str(run_dir),
        "elapsed_seconds"     : elapsed,
        "evaluated_windows"   : len(starts),
        "covered_rows"        : len(covered_rows),
        "init_meter_rmse_w"   : init_rmse,
        "best_meter_rmse_w"   : best_rmse,
        "aggregate_mae_w"     : agg_mae,
        "macro_metrics"       : macro_metrics,
    }


# ─── stage 2 dispatcher ───────────────────────────────────────────────────────

def run_stage2(
    config_path: Path,
    checkpoint: Path,
    base_run_dir: Path,
    *,
    smoke: bool,
    overwrite: bool,
) -> list[dict[str, Any]]:
    """Run Stage-2 inverse optimisation on validation and both test houses."""
    banner("STAGE 2 — Frozen-decoder inverse optimisation", CYAN)

    from utils.config import load_yaml
    stage2_config = config_path.parent / "stage2.yaml"
    if not stage2_config.exists():
        stage2_config = config_path  # Fallback: use stage1 config directly.

    # Splits: validation first, then both test houses (use their scenario IDs).
    # The two test houses are: refit_house20, ukdale_house2.
    # 'validation' always uses the standard validation split.
    # 'test' uses the combined test file. We also evaluate per-house scenarios.
    splits_to_run = [
        ("validation",    base_run_dir / "stage2_validation"),
        ("refit_house20", base_run_dir / "stage2_test_refit_house20"),
        ("ukdale_house2", base_run_dir / "stage2_test_ukdale_house2"),
    ]

    results = []
    for split, run_dir in splits_to_run:
        banner(f"  Stage 2 — split: {split}", YELLOW)
        try:
            result = _run_one_inverse(
                config_path=stage2_config,
                checkpoint=checkpoint,
                run_dir=run_dir,
                split=split,
                steps=None,
                batch_size=None,
                learning_rate=None,
                train_mean_init=True,   # Match the established best practice.
                prior_weight=None,
                max_windows=None,
                overwrite=overwrite,
                device=None,
                smoke=smoke,
            )
            results.append(result)
        except Exception as exc:
            warn(f"[{split}] failed: {exc}")
            results.append({
                "stage"           : f"stage2_{split}",
                "split"           : split,
                "run_dir"         : str(run_dir),
                "elapsed_seconds" : 0.0,
                "error"           : str(exc),
            })

    return results


# ─── final summary ────────────────────────────────────────────────────────────

def print_summary(
    stage1: dict[str, Any],
    stage2_results: list[dict[str, Any]],
    total_elapsed: float,
    report_path: Path,
) -> None:
    banner("PIPELINE SUMMARY", GREEN)

    # Stage 1
    info(f"{'Stage 1 — forward training':40s}  {fmt_seconds(stage1['elapsed_seconds'])}")
    info(f"  Best epoch : {stage1['best_epoch']}   "
         f"Val MSE : {stage1['best_validation_mse']:.6g}")
    info(f"  Parameters : {stage1['parameter_count']:,}   "
         f"Checkpoint : {fmt_bytes(stage1['checkpoint_bytes'])}")

    # Stage 2
    for r in stage2_results:
        split   = r.get("split", "?")
        elapsed = r.get("elapsed_seconds", 0.0)
        if "error" in r:
            warn(f"Stage 2 [{split:20s}]  ERROR: {r['error']}")
            continue
        agg_mae = r.get("aggregate_mae_w", float("nan"))
        m       = r.get("macro_metrics", {})
        f1      = m.get("macro_f1",  m.get("f1",  float("nan")))
        mae_w   = m.get("macro_mae", m.get("mae", float("nan")))
        info(
            f"Stage 2 [{split:20s}]  {fmt_seconds(elapsed):10s}  "
            f"AggMAE {agg_mae:7.1f} W  "
            f"MacroMAE {mae_w:7.1f} W  "
            f"MacroF1 {f1:.3f}"
        )

    print()
    ok(f"Total pipeline time : {fmt_seconds(total_elapsed)}")
    ok(f"Report written to   : {report_path}")


# ─── markdown report ──────────────────────────────────────────────────────────

def write_report(
    report_path: Path,
    stage1: dict[str, Any],
    stage2_results: list[dict[str, Any]],
    total_elapsed: float,
    args_dict: dict[str, Any],
) -> None:
    lines: list[str] = []
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines += [
        "# Neural-Adjoint NILM Pipeline Report",
        "",
        f"**Generated:** {ts}  ",
        f"**Python:** {sys.version.split()[0]}  ",
        f"**Platform:** {platform.platform()}  ",
        f"**PyTorch:** {torch.__version__}  ",
        f"**CUDA available:** {torch.cuda.is_available()}  ",
        "",
    ]

    if torch.cuda.is_available():
        lines += [
            f"**GPU:** {torch.cuda.get_device_name(0)}  ",
            "",
        ]

    # Pipeline args
    lines += [
        "## Pipeline Arguments",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    for k, v in args_dict.items():
        lines.append(f"| `{k}` | `{v}` |")
    lines.append("")

    # Stage 1
    lines += [
        "## Stage 1 — Forward Model Training",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Elapsed | {fmt_seconds(stage1['elapsed_seconds'])} |",
        f"| Best epoch | {stage1['best_epoch']} |",
        f"| Validation MSE (scaled) | {stage1['best_validation_mse']:.6g} |",
        f"| Trainable parameters | {stage1['parameter_count']:,} |",
        f"| Checkpoint size | {fmt_bytes(stage1['checkpoint_bytes'])} |",
        f"| Training windows | {stage1['train_windows']:,} |",
        f"| Validation windows | {stage1['validation_windows']:,} |",
        f"| Checkpoint path | `{stage1['checkpoint']}` |",
        "",
    ]

    # Stage 2
    lines += [
        "## Stage 2 — Inverse Optimisation Results",
        "",
        "| Split | Time | Windows | AggMAE (W) | MacroMAE (W) | Macro-F1 | InitRMSE→BestRMSE (W) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in stage2_results:
        split   = r.get("split", "?")
        elapsed = fmt_seconds(r.get("elapsed_seconds", 0.0))
        if "error" in r:
            lines.append(f"| {split} | — | — | ERROR | {r['error'][:60]} | — | — |")
            continue
        windows = r.get("evaluated_windows", "?")
        agg_mae = r.get("aggregate_mae_w", float("nan"))
        m       = r.get("macro_metrics", {})
        f1      = m.get("macro_f1",  m.get("f1",  float("nan")))
        mae_w   = m.get("macro_mae", m.get("mae", float("nan")))
        init_r  = r.get("init_meter_rmse_w", float("nan"))
        best_r  = r.get("best_meter_rmse_w", float("nan"))
        lines.append(
            f"| {split} | {elapsed} | {windows} | {agg_mae:.1f} | "
            f"{mae_w:.1f} | {f1:.3f} | {init_r:.1f} → {best_r:.1f} |"
        )
    lines.append("")

    # Per-appliance macro metrics per split
    lines += ["## Per-Split Macro Metrics", ""]
    for r in stage2_results:
        if "error" in r:
            continue
        split = r.get("split", "?")
        m     = r.get("macro_metrics", {})
        if not m:
            continue
        lines += [f"### {split}", "", "| Metric | Value |", "| --- | --- |"]
        for k, v in m.items():
            lines.append(f"| {k} | {v:.4g} |")
        lines.append("")

    # Total time
    lines += [
        "## Total Pipeline",
        "",
        f"**Total elapsed:** {fmt_seconds(total_elapsed)}",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ─── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neural-Adjoint NILM end-to-end pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
            # Quick smoke test (1 epoch, 2 batches, 2 inverse steps):
              python pipeline.py --smoke

            # Full pipeline with custom output directory:
              python pipeline.py --run-dir runs/pipeline_full

            # Resume / overwrite an existing directory:
              python pipeline.py --run-dir runs/pipeline_full --overwrite
        """),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "stage1.yaml",
        help="Stage-1 config YAML (default: configs/stage1.yaml).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Output root directory. Auto-named with a timestamp if omitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing run directories.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: 1 training epoch, 2 inverse steps, 2 windows.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        parser.error(f"Config not found: {config_path}")

    # Auto-name the run directory with a timestamp.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix    = "_smoke" if args.smoke else ""
    run_dir   = (args.run_dir or config_path.parent.parent / "runs" / f"pipeline_{timestamp}{suffix}").resolve()

    args_dict = {
        "config"   : str(config_path),
        "run_dir"  : str(run_dir),
        "overwrite": args.overwrite,
        "smoke"    : args.smoke,
        "pytorch"  : torch.__version__,
        "cuda"     : torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        args_dict["gpu"] = torch.cuda.get_device_name(0)

    banner("NEURAL-ADJOINT NILM — END-TO-END PIPELINE", GREEN)
    info(f"Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    info(f"Config   : {config_path}")
    info(f"Run dir  : {run_dir}")
    info(f"PyTorch  : {torch.__version__}")
    info(f"CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        info(f"GPU      : {torch.cuda.get_device_name(0)}")
    info(f"Smoke    : {args.smoke}")
    print()

    t_pipeline_start = time.perf_counter()

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    stage1_dir = run_dir / "stage1_forward"
    stage1_result = run_stage1(
        config_path, stage1_dir, smoke=args.smoke, overwrite=args.overwrite
    )
    checkpoint = Path(stage1_result["checkpoint"])

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    stage2_results = run_stage2(
        config_path, checkpoint, run_dir,
        smoke=args.smoke, overwrite=args.overwrite,
    )

    total_elapsed = time.perf_counter() - t_pipeline_start

    # ── Report ───────────────────────────────────────────────────────────────
    report_path = run_dir / "pipeline_report.md"
    write_report(report_path, stage1_result, stage2_results, total_elapsed, args_dict)

    # Save the full machine-readable summary.
    summary = {
        "pipeline_args"   : args_dict,
        "stage1"          : stage1_result,
        "stage2"          : stage2_results,
        "total_elapsed_s" : total_elapsed,
        "report"          : str(report_path),
    }
    summary_path = run_dir / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print_summary(stage1_result, stage2_results, total_elapsed, report_path)


if __name__ == "__main__":
    main()


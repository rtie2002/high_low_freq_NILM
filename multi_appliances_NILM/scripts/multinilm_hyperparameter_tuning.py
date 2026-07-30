#!/usr/bin/env python
"""Bayesian-style hyperparameter tuning for MultiNILM (Optuna TPE).

Tunes knobs from ``config/models/multinilm.yaml`` (windows, architecture, DA loss,
training). Every trial forces ``domain_adaptation.enabled: true``.

Run from repo root or multi_appliances_NILM:

    python scripts/multinilm_hyperparameter_tuning.py
    python scripts/multinilm_hyperparameter_tuning.py --n-trials 30 --fast
    # Optional: override fixed epoch count (default 150; epochs are never tuned)
    python scripts/multinilm_hyperparameter_tuning.py --epochs 150

    # Subset of knobs (others stay at values from --model-config)
    python scripts/multinilm_hyperparameter_tuning.py --tune learning_rate,lambda_domain,dropout

    # List available names for --tune
    python scripts/multinilm_hyperparameter_tuning.py --list-tune-params

Each trial trains MultiNILM and minimizes the same checkpoint score used during
normal training (``val_mae_minus_f1`` by default). Results are saved under
``results/multinilm_hyperparameter_tuning_<experiment_id>/``.

**Objective:** minimize ``normalized_val_MAE - macro_val_F1`` on validation
(H1+H5). Test (H2) is not used for tuning.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.config import load_experiment, load_model_config, merge_configs, resolve_training_targets
from adapters.multinilm import MultiNILMAdapter
from runner import train_model


# =============================================================================
# 1. Default configuration (override via CLI)
# =============================================================================
DEFAULT_EXPERIMENT = ROOT / "config" / "experiment_ukdale.yaml"
DEFAULT_MODEL_CONFIG = ROOT / "config" / "models" / "multinilm.yaml"
DEFAULT_SEED = 2026
DEFAULT_N_TRIALS = 30
# Fixed training length for every trial (not tuned).
DEFAULT_EPOCHS = 150

# Channel schedule = start_width * 2^{0..depth-1}.
# Two axes: scale (÷2 / base / ×2 vs yaml start=32) and depth (add/remove stages).
CHANNEL_START_CHOICES = (16, 32, 64)
CHANNEL_DEPTH_CHOICES_FAST = (2, 3, 4)
CHANNEL_DEPTH_CHOICES_FULL = (2, 3, 4, 5)


def _make_channel_schedule(start: int, depth: int) -> list[int]:
    return [int(start) * (2**i) for i in range(int(depth))]


def _channel_schedule_presets(*, fast_search: bool) -> dict[str, list[int]]:
    depths = CHANNEL_DEPTH_CHOICES_FAST if fast_search else CHANNEL_DEPTH_CHOICES_FULL
    return {
        f"start{start}_depth{depth}": _make_channel_schedule(start, depth)
        for start in CHANNEL_START_CHOICES
        for depth in depths
    }


# Static full grid (used for key lookup / study_config).
CHANNEL_SCHEDULE_PRESETS: dict[str, list[int]] = _channel_schedule_presets(fast_search=False)

GATE_MODE_CHOICES = ("soft", "hard", "soft_train_hard_eval")
DETAIL_KERNEL_PRESETS: dict[str, list[int]] = {
    "3_5_9": [3, 5, 9],
    "3_5": [3, 5],
    "5_9": [5, 9],
    "3_7_11": [3, 7, 11],
}
DOMAIN_SCALE_CHOICES = ("none", "equal")
# With training_targets: full_input, input and output MUST match.
# Powers of 2 so stride = window // {2,4,8,...} stays exact.
WINDOW_LENGTH_CHOICES = (64, 128, 256, 512, 1024, 2048)
INPUT_WINDOW_CHOICES = WINDOW_LENGTH_CHOICES
OUTPUT_WINDOW_CANDIDATES = WINDOW_LENGTH_CHOICES
MIN_WINDOWS_PER_SPLIT = 8
# Stride = window // divisor (÷2 pattern). Fixed Optuna space; absolute stride follows window.
STRIDE_DIVISOR_CHOICES_FAST = (2, 4, 8)  # 50% / 25% / 12.5% of window
STRIDE_DIVISOR_CHOICES_FULL = (2, 4, 8, 16)


def _stride_divisor_choices(*, fast_search: bool) -> list[int]:
    return list(STRIDE_DIVISOR_CHOICES_FAST if fast_search else STRIDE_DIVISOR_CHOICES_FULL)


def _stride_from_window(window_length: int, divisor: int) -> int:
    """stride = window // divisor (e.g. 600//2=300). Always >= 1."""
    return max(1, int(window_length) // max(1, int(divisor)))


def _nearest_stride_divisor(window_length: int, stride: int) -> int:
    """Map a yaml stride back onto the closest ÷2 divisor."""
    window = max(1, int(window_length))
    target = max(1, int(stride))
    candidates = list(STRIDE_DIVISOR_CHOICES_FULL) + [32]
    return min(candidates, key=lambda d: abs(_stride_from_window(window, d) - target))


def _stride_choices_for_window(input_length: int, *, fast_search: bool) -> list[int]:
    """Absolute strides implied by window // {2,4,8,...}."""
    return [
        _stride_from_window(input_length, d)
        for d in _stride_divisor_choices(fast_search=fast_search)
    ]


# Tunable knobs from config/models/multinilm.yaml (domain_adaptation.enabled always forced true).
# epochs is fixed at DEFAULT_EPOCHS (not tuned).
TUNABLE_PARAMETERS = (
    "input_window_length",
    "output_window_length",
    "input_stride",
    "eval_stride",
    "channel_schedule",
    "stem_kernel_size",
    "stage_kernel_size",
    "num_blocks",
    "kernel_size",
    "max_dilation",
    "dropout",
    "gate_mode",
    "gate_threshold",
    "head_local_layers",
    "head_kernel_size",
    "head_use_residual",
    "use_multiscale_stem",
    "detail_kernels",
    "detail_branch_channels",
    "domain_mu",
    "domain_scale",
    "lambda_domain",
    "batch_size",
    "learning_rate",
    "weight_decay",
)
DEFAULT_TUNE_PARAMETERS = TUNABLE_PARAMETERS
TUNE_ALIASES = {
    "lr": "learning_rate",
    "learning-rate": "learning_rate",
    "wd": "weight_decay",
    "weight-decay": "weight_decay",
    "gate": "gate_mode",
    "channels": "channel_schedule",
    "channel-schedule": "channel_schedule",
    "input-window": "input_window_length",
    "input_window": "input_window_length",
    "output-window": "output_window_length",
    "output_window": "output_window_length",
    "window": "window_length",
    "window_length": "window_length",
    "train-stride": "input_stride",
    "train_stride": "input_stride",
    "eval-stride": "eval_stride",
    "stride": "stride",
    "lambda_da": "lambda_domain",
    "lambda-domain": "lambda_domain",
    "domain-mu": "domain_mu",
    "domain-scale": "domain_scale",
    "batch": "batch_size",
}


# =============================================================================
# 2. Load Optuna (auto-install if missing)
# =============================================================================
try:
    import optuna
except ImportError:
    print("Optuna is not installed in this Python environment.")
    print("Installing Optuna with:")
    print(f"  {sys.executable} -m pip install optuna")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna/TPE hyperparameter tuning for MultiNILM.",
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Fixed epochs per trial (default: {DEFAULT_EPOCHS}; not tuned)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the compact search space",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Output folder (default: results/multinilm_hyperparameter_tuning_<experiment_id>)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override experiment data_root",
    )
    parser.add_argument(
        "--tune",
        type=str,
        default=None,
        help=(
            "Comma-separated hyperparameters to tune. "
            f"Available: {', '.join(TUNABLE_PARAMETERS)} (+ alias stride). "
            f"Default: {', '.join(DEFAULT_TUNE_PARAMETERS)}. "
            "Others stay at values from --model-config."
        ),
    )
    parser.add_argument(
        "--list-tune-params",
        action="store_true",
        help="Print tunable hyperparameter names and exit.",
    )
    return parser.parse_args()


def _canonical_tune_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    if key in {"stride", "window_length"}:
        return key
    canonical = TUNE_ALIASES.get(key, key)
    if canonical in {"stride", "window_length"}:
        return canonical
    if canonical not in TUNABLE_PARAMETERS:
        known = ", ".join(sorted(set(TUNABLE_PARAMETERS) | {"stride", "window_length"}))
        raise ValueError(f"Unknown hyperparameter {name!r}. Available: {known}")
    return canonical


def _parse_tune_set(raw: str | None) -> set[str]:
    if raw is None or not str(raw).strip():
        return set(DEFAULT_TUNE_PARAMETERS)
    names = [_canonical_tune_name(part) for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("--tune was provided but no hyperparameter names were parsed.")
    expanded: set[str] = set()
    for name in names:
        if name == "stride":
            expanded.update({"input_stride", "eval_stride"})
        elif name == "window_length":
            expanded.update({"input_window_length", "output_window_length"})
        else:
            expanded.add(name)
    return expanded


def _uses_full_input_training(model_cfg: dict) -> bool:
    return resolve_training_targets(model_cfg.get("windowing", {})) == "full_input"


def _channel_schedule_key(schedule: list[int]) -> str:
    for key, value in CHANNEL_SCHEDULE_PRESETS.items():
        if value == [int(v) for v in schedule]:
            return key
    if not schedule:
        return "empty"
    start = int(schedule[0])
    depth = len(schedule)
    return f"start{start}_depth{depth}"


def _infer_channel_start_depth(schedule: list[int]) -> tuple[int, int]:
    """Recover (start, depth) from a schedule; fall back to nearest start choice."""
    depth = max(1, len(schedule))
    start = int(schedule[0]) if schedule else 32
    if start not in CHANNEL_START_CHOICES:
        start = min(CHANNEL_START_CHOICES, key=lambda s: abs(s - start))
    return start, depth


def _detail_kernels_key(kernels: list[int]) -> str:
    for key, value in DETAIL_KERNEL_PRESETS.items():
        if value == [int(v) for v in kernels]:
            return key
    return "_".join(str(int(v)) for v in kernels)


def _baseline_param_values(model_cfg: dict) -> dict[str, Any]:
    train_cfg = model_cfg.get("training", {})
    arch_cfg = model_cfg.get("architecture", {})
    window_cfg = model_cfg.get("windowing", {})
    loss_cfg = model_cfg.get("loss", {})
    schedule = [int(v) for v in arch_cfg.get("channel_schedule", [32, 64, 128])]
    channel_start, channel_depth = _infer_channel_start_depth(schedule)
    detail_kernels = [int(v) for v in arch_cfg.get("detail_kernels", [3, 5, 9])]
    input_window = int(window_cfg.get("input_window_length", 600))
    output_window = int(window_cfg.get("output_window_length", 600))
    input_stride = int(window_cfg.get("input_stride", 300))
    eval_stride = int(window_cfg.get("eval_stride", 300))
    input_stride_divisor = _nearest_stride_divisor(input_window, input_stride)
    eval_stride_divisor = _nearest_stride_divisor(input_window, eval_stride)
    return {
        "input_window_length": input_window,
        "output_window_length": output_window,
        "input_stride_divisor": input_stride_divisor,
        "eval_stride_divisor": eval_stride_divisor,
        "input_stride": _stride_from_window(input_window, input_stride_divisor),
        "eval_stride": _stride_from_window(input_window, eval_stride_divisor),
        "channel_start": channel_start,
        "channel_depth": channel_depth,
        "channel_schedule": _make_channel_schedule(channel_start, channel_depth),
        "channel_schedule_key": f"start{channel_start}_depth{channel_depth}",
        "hidden_channels": int(arch_cfg.get("hidden_channels", schedule[-1])),
        "stem_kernel_size": int(arch_cfg.get("stem_kernel_size", 7)),
        "stage_kernel_size": int(arch_cfg.get("stage_kernel_size", 5)),
        "num_blocks": int(arch_cfg.get("num_blocks", 8)),
        "kernel_size": int(arch_cfg.get("kernel_size", 5)),
        "max_dilation": int(arch_cfg.get("max_dilation", 64)),
        "dropout": float(arch_cfg.get("dropout", 0.15)),
        "gate_mode": str(arch_cfg.get("gate_mode", "hard")).lower(),
        "gate_threshold": float(arch_cfg.get("gate_threshold", 0.5)),
        "head_local_layers": int(arch_cfg.get("head_local_layers", 2)),
        "head_kernel_size": int(arch_cfg.get("head_kernel_size", 3)),
        "head_use_residual": bool(arch_cfg.get("head_use_residual", True)),
        "use_multiscale_stem": bool(arch_cfg.get("use_multiscale_stem", True)),
        "detail_kernels": detail_kernels,
        "detail_kernels_key": _detail_kernels_key(detail_kernels),
        "detail_branch_channels": int(arch_cfg.get("detail_branch_channels", 16)),
        "domain_mu": float(loss_cfg.get("domain_mu", 0.4)),
        "domain_scale": str(loss_cfg.get("domain_scale", "equal")).lower(),
        "lambda_domain": float(loss_cfg.get("lambda_domain", 0.0)),
        "batch_size": int(train_cfg.get("batch_size", 32)),
        "epochs": DEFAULT_EPOCHS,
        "learning_rate": float(train_cfg.get("learning_rate", 1e-4)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
    }


def _resolve_data_root(experiment: dict, data_path: Path | None) -> Path | None:
    if data_path is not None:
        return data_path if data_path.is_absolute() else ROOT / data_path
    raw = experiment.get("data_root")
    if raw is None:
        return None
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def _deep_copy_merged(experiment: dict, model_cfg: dict) -> dict:
    merged = merge_configs(experiment, copy.deepcopy(model_cfg))
    merged["experiment"] = copy.deepcopy(experiment)
    merged["model"] = copy.deepcopy(model_cfg)
    return merged


def _domain_feature_layers_for_blocks(num_blocks: int) -> list[str]:
    """Late TCN hooks scaled to ``num_blocks`` (yaml uses temporal_4/6 for n=8)."""
    n = max(1, int(num_blocks))
    if n == 1:
        return ["temporal_0", "aligned"]
    # Same fractions as yaml defaults for n=8 → indices 4 and 6.
    a = min(n - 1, max(0, (n * 4) // 8))
    b = min(n - 1, max(0, (n * 6) // 8))
    if a == b:
        a = max(0, b - 1)
    return [f"temporal_{a}", f"temporal_{b}", "aligned"]


def _apply_tuning_overrides(model_cfg: dict, trial_params: dict, *, epochs: int) -> None:
    """Mutate model_cfg in place for one Optuna trial.

    Always forces ``domain_adaptation.enabled: true`` (required for DA loss).
    """
    arch_cfg = model_cfg.setdefault("architecture", {})
    schedule = [int(v) for v in trial_params["channel_schedule"]]
    # MultiNILM requires hidden_channels == channel_schedule[-1].
    hidden = int(schedule[-1])
    trial_params["hidden_channels"] = hidden
    num_blocks = int(trial_params["num_blocks"])

    arch_cfg["channel_schedule"] = schedule
    arch_cfg["hidden_channels"] = hidden
    arch_cfg["stem_kernel_size"] = int(trial_params["stem_kernel_size"])
    arch_cfg["stage_kernel_size"] = int(trial_params["stage_kernel_size"])
    arch_cfg["num_blocks"] = num_blocks
    arch_cfg["kernel_size"] = int(trial_params["kernel_size"])
    arch_cfg["max_dilation"] = int(trial_params["max_dilation"])
    arch_cfg["dropout"] = float(trial_params["dropout"])
    arch_cfg["gate_mode"] = str(trial_params["gate_mode"]).lower()
    arch_cfg["gate_threshold"] = float(trial_params["gate_threshold"])
    arch_cfg["head_local_layers"] = int(trial_params["head_local_layers"])
    arch_cfg["head_kernel_size"] = int(trial_params["head_kernel_size"])
    arch_cfg["head_use_residual"] = bool(trial_params["head_use_residual"])
    arch_cfg["use_multiscale_stem"] = bool(trial_params["use_multiscale_stem"])
    arch_cfg["detail_kernels"] = [int(v) for v in trial_params["detail_kernels"]]
    arch_cfg["detail_branch_channels"] = int(trial_params["detail_branch_channels"])
    # Keep DA hooks valid when num_blocks changes (temporal_6 invalid if n<7).
    arch_cfg["domain_feature_layers"] = _domain_feature_layers_for_blocks(num_blocks)

    loss_cfg = model_cfg.setdefault("loss", {})
    loss_cfg["domain_mu"] = float(trial_params["domain_mu"])
    loss_cfg["domain_scale"] = str(trial_params["domain_scale"]).lower()
    loss_cfg["lambda_domain"] = float(trial_params["lambda_domain"])

    # Always on during tuning (yaml may have enabled: false).
    model_cfg.setdefault("domain_adaptation", {})["enabled"] = True

    train_cfg = model_cfg.setdefault("training", {})
    train_cfg["batch_size"] = int(trial_params["batch_size"])
    train_cfg["learning_rate"] = float(trial_params["learning_rate"])
    train_cfg["weight_decay"] = float(trial_params["weight_decay"])
    train_cfg["epochs"] = int(epochs)
    train_cfg["scheduler"] = "none"

    window_cfg = model_cfg.setdefault("windowing", {})
    window_cfg["input_window_length"] = int(trial_params["input_window_length"])
    window_cfg["output_window_length"] = int(trial_params["output_window_length"])
    window_cfg["input_stride"] = int(trial_params["input_stride"])
    window_cfg["eval_stride"] = int(trial_params["eval_stride"])
    window_cfg.setdefault("eval_reconstruction", "auto")
    window_cfg.setdefault("training_targets", "auto")

    plots = train_cfg.setdefault("plots", {})
    plots["enabled"] = False
    plots["plot_mode"] = "end"
    feature_maps = plots.setdefault("feature_maps", {})
    feature_maps["enabled"] = False


def _output_window_choices(input_length: int, *, require_equal: bool) -> list[int]:
    """Valid output windows for the chosen input length."""
    input_len = int(input_length)
    if require_equal:
        return [input_len]
    return [value for value in OUTPUT_WINDOW_CANDIDATES if value <= input_len]


def _apply_strides_from_divisors(params: dict[str, Any]) -> None:
    """Recompute absolute strides from window // divisor (auto when window changes)."""
    window = int(params["input_window_length"])
    in_div = int(params.get("input_stride_divisor", 2))
    ev_div = int(params.get("eval_stride_divisor", in_div))
    params["input_stride"] = _stride_from_window(window, in_div)
    params["eval_stride"] = _stride_from_window(window, ev_div)


def _normalize_window_params(params: dict[str, Any], *, require_equal_windows: bool) -> None:
    input_len = int(params["input_window_length"])
    output_len = int(params["output_window_length"])
    if require_equal_windows:
        params["output_window_length"] = input_len
    elif output_len > input_len:
        params["output_window_length"] = input_len
    _apply_strides_from_divisors(params)


def _csv_row_count(csv_path: Path) -> int:
    with csv_path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _count_windows_for_split(
    n_timesteps: int,
    windowing: dict[str, Any],
    *,
    split: str,
) -> int:
    from adapters.dataloader import _count_windows

    stride_key = "input_stride" if split == "train" else "eval_stride"
    stride = int(windowing.get(stride_key, windowing.get("input_stride", 1)))
    return _count_windows(n_timesteps, windowing, max(1, stride))


def _validate_window_stride_config(
    model_cfg: dict,
    trial_params: dict[str, Any],
    experiment: dict,
    data_root: Path | None,
) -> None:
    """Reject window/stride combos that crash training or misalign labels."""
    windowing = copy.deepcopy(model_cfg.get("windowing", {}))
    windowing.update(
        {
            "input_window_length": int(trial_params["input_window_length"]),
            "output_window_length": int(trial_params["output_window_length"]),
            "input_stride": int(trial_params["input_stride"]),
            "eval_stride": int(trial_params["eval_stride"]),
        }
    )
    require_equal = resolve_training_targets(windowing) == "full_input"
    input_len = int(windowing["input_window_length"])
    output_len = int(windowing["output_window_length"])

    if input_len <= 0 or output_len <= 0:
        raise ValueError("Window lengths must be positive.")
    if output_len > input_len:
        raise ValueError(
            f"Unsafe window combo: output_window_length={output_len} > input_window_length={input_len}."
        )
    if require_equal and input_len != output_len:
        raise ValueError(
            "Unsafe window combo for training_targets=full_input: "
            f"input_window_length ({input_len}) must equal output_window_length ({output_len}). "
            "Otherwise train loss shapes mismatch (B,T,A) or val labels misalign with model center-crop."
        )

    if data_root is None:
        return

    csv_cfg = experiment.get("csv", {})
    split_files = {
        "train": csv_cfg.get("train_file"),
        "validation": csv_cfg.get("validation_file"),
    }
    for split, rel_path in split_files.items():
        if not rel_path:
            continue
        csv_path = Path(rel_path)
        if not csv_path.is_absolute():
            csv_path = data_root / csv_path
        if not csv_path.exists():
            continue
        n_rows = _csv_row_count(csv_path)
        n_windows = _count_windows_for_split(n_rows, windowing, split=split)
        if n_windows < MIN_WINDOWS_PER_SPLIT:
            raise ValueError(
                f"Unsafe window/stride for {split}: only {n_windows} windows "
                f"(need >= {MIN_WINDOWS_PER_SPLIT}) with "
                f"input={input_len}, stride={windowing['input_stride' if split == 'train' else 'eval_stride']}, "
                f"csv_rows={n_rows}."
            )


def _suggest_trial_params(
    trial: optuna.Trial,
    *,
    fast_search: bool,
    tune_set: set[str],
    baseline: dict[str, Any],
    model_cfg: dict,
) -> dict[str, Any]:
    params = dict(baseline)
    require_equal_windows = _uses_full_input_training(model_cfg)
    window_length_tuned = "input_window_length" in tune_set or "output_window_length" in tune_set

    if window_length_tuned and require_equal_windows:
        if "input_window_length" in tune_set or "output_window_length" in tune_set:
            params["input_window_length"] = trial.suggest_categorical(
                "window_length",
                list(WINDOW_LENGTH_CHOICES),
            )
            params["output_window_length"] = int(params["input_window_length"])

    if not (window_length_tuned and require_equal_windows) and "input_window_length" in tune_set:
        params["input_window_length"] = trial.suggest_categorical(
            "input_window_length",
            list(INPUT_WINDOW_CHOICES),
        )

    if not (window_length_tuned and require_equal_windows):
        if "output_window_length" in tune_set:
            params["output_window_length"] = trial.suggest_categorical(
                "output_window_length",
                list(OUTPUT_WINDOW_CANDIDATES),
            )
        elif require_equal_windows and "input_window_length" in tune_set:
            params["output_window_length"] = int(params["input_window_length"])

    # Stride search = divisor (÷2 pattern); absolute stride = window // divisor.
    divisor_choices = _stride_divisor_choices(fast_search=fast_search)
    tune_stride_together = "input_stride" in tune_set and "eval_stride" in tune_set
    if tune_stride_together:
        stride_divisor = trial.suggest_categorical("stride_divisor", divisor_choices)
        params["input_stride_divisor"] = int(stride_divisor)
        params["eval_stride_divisor"] = int(stride_divisor)
    else:
        if "input_stride" in tune_set:
            params["input_stride_divisor"] = trial.suggest_categorical(
                "input_stride_divisor", divisor_choices
            )
        if "eval_stride" in tune_set:
            params["eval_stride_divisor"] = trial.suggest_categorical(
                "eval_stride_divisor", divisor_choices
            )

    if "channel_schedule" in tune_set:
        # Two axes: scale start (÷2/base/×2) and depth (add/remove ×2 stages).
        channel_start = trial.suggest_categorical(
            "channel_start", list(CHANNEL_START_CHOICES)
        )
        depth_choices = (
            list(CHANNEL_DEPTH_CHOICES_FAST)
            if fast_search
            else list(CHANNEL_DEPTH_CHOICES_FULL)
        )
        channel_depth = trial.suggest_categorical("channel_depth", depth_choices)
        params["channel_start"] = int(channel_start)
        params["channel_depth"] = int(channel_depth)
        params["channel_schedule"] = _make_channel_schedule(channel_start, channel_depth)
        params["channel_schedule_key"] = f"start{channel_start}_depth{channel_depth}"
        params["hidden_channels"] = int(params["channel_schedule"][-1])

    if "stem_kernel_size" in tune_set:
        params["stem_kernel_size"] = trial.suggest_categorical(
            "stem_kernel_size", [3, 5, 7, 9]
        )

    if "stage_kernel_size" in tune_set:
        params["stage_kernel_size"] = trial.suggest_categorical(
            "stage_kernel_size", [3, 5, 7]
        )

    if "num_blocks" in tune_set:
        if fast_search:
            params["num_blocks"] = trial.suggest_categorical("num_blocks", [4, 6, 8])
        else:
            params["num_blocks"] = trial.suggest_int("num_blocks", 4, 12, step=2)

    if "kernel_size" in tune_set:
        params["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7])

    if "max_dilation" in tune_set:
        params["max_dilation"] = trial.suggest_categorical(
            "max_dilation", [16, 32, 64, 128]
        )

    if "dropout" in tune_set:
        if fast_search:
            params["dropout"] = trial.suggest_float("dropout", 0.08, 0.20)
        else:
            params["dropout"] = trial.suggest_float("dropout", 0.05, 0.25)

    if "gate_mode" in tune_set:
        params["gate_mode"] = trial.suggest_categorical("gate_mode", list(GATE_MODE_CHOICES))

    if "gate_threshold" in tune_set:
        if fast_search:
            params["gate_threshold"] = trial.suggest_categorical(
                "gate_threshold", [0.4, 0.5, 0.6]
            )
        else:
            params["gate_threshold"] = trial.suggest_float("gate_threshold", 0.3, 0.7)

    if "head_local_layers" in tune_set:
        params["head_local_layers"] = trial.suggest_int("head_local_layers", 0, 4)

    if "head_kernel_size" in tune_set:
        params["head_kernel_size"] = trial.suggest_categorical(
            "head_kernel_size", [3, 5]
        )

    if "head_use_residual" in tune_set:
        params["head_use_residual"] = trial.suggest_categorical(
            "head_use_residual", [True, False]
        )

    if "use_multiscale_stem" in tune_set:
        params["use_multiscale_stem"] = trial.suggest_categorical(
            "use_multiscale_stem", [True, False]
        )

    if "detail_kernels" in tune_set:
        detail_key = trial.suggest_categorical(
            "detail_kernels",
            list(DETAIL_KERNEL_PRESETS.keys()),
        )
        params["detail_kernels_key"] = detail_key
        params["detail_kernels"] = list(DETAIL_KERNEL_PRESETS[detail_key])

    if "detail_branch_channels" in tune_set:
        params["detail_branch_channels"] = trial.suggest_categorical(
            "detail_branch_channels", [8, 16, 24, 32]
        )

    if "domain_mu" in tune_set:
        if fast_search:
            params["domain_mu"] = trial.suggest_float("domain_mu", 0.2, 0.6)
        else:
            params["domain_mu"] = trial.suggest_float("domain_mu", 0.1, 0.8)

    if "domain_scale" in tune_set:
        params["domain_scale"] = trial.suggest_categorical(
            "domain_scale", list(DOMAIN_SCALE_CHOICES)
        )

    if "lambda_domain" in tune_set:
        # Keep > 0 so enabled DA actually contributes (yaml comment: needs lambda_domain > 0).
        if fast_search:
            params["lambda_domain"] = trial.suggest_float("lambda_domain", 0.2, 0.8)
        else:
            params["lambda_domain"] = trial.suggest_float("lambda_domain", 0.1, 0.9)

    if "batch_size" in tune_set:
        params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])

    if "learning_rate" in tune_set:
        if fast_search:
            params["learning_rate"] = trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True)
        else:
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)

    if "weight_decay" in tune_set:
        if trial.suggest_categorical("use_weight_decay", [False, True]):
            params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        else:
            params["weight_decay"] = 0.0

    # MultiNILM constraint: hidden_channels must equal channel_schedule[-1].
    params["hidden_channels"] = int(params["channel_schedule"][-1])
    _normalize_window_params(params, require_equal_windows=require_equal_windows)
    return params


def _read_trial_metrics(run_dir: Path) -> dict:
    timing_path = run_dir / "training_time.json"
    history_path = run_dir / "history.json"
    if not timing_path.exists():
        raise FileNotFoundError(f"Missing training summary: {timing_path}")

    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    best_score = timing.get("best_score")
    best_epoch = int(timing.get("best_epoch") or 0)
    if best_score is None or best_epoch <= 0:
        raise RuntimeError(f"No best checkpoint recorded in {timing_path}")

    metrics = {
        "best_score": float(best_score),
        "best_epoch": best_epoch,
        "epochs_completed": int(timing.get("epochs_completed") or 0),
        "checkpoint_monitor": str(timing.get("checkpoint_monitor", "")),
    }

    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        row = next((item for item in history if int(item.get("epoch", -1)) == best_epoch), None)
        if row:
            metrics["val_mae_watts"] = float(row.get("val_mae_watts", row.get("val_mae", 0.0)))
            metrics["val_mae_norm"] = float(row.get("val_mae_norm", 0.0))
            metrics["val_f1"] = float(row.get("val_f1", 0.0))
            metrics["val_loss"] = float(row.get("val_loss", 0.0))

    return metrics


def _trial_param_column(param: str, columns: pd.Index) -> str | None:
    candidates = [f"params_{param}"]
    if param in {"input_stride", "eval_stride"}:
        candidates = [
            "params_stride_divisor",
            "params_input_stride_divisor",
            "params_eval_stride_divisor",
            f"params_{param}",
        ]
    if param in {"input_window_length", "output_window_length"}:
        candidates.insert(0, "params_window_length")
    if param == "channel_schedule":
        candidates = ["params_channel_start", "params_channel_depth", "params_channel_schedule"]
    for column in candidates:
        if column in columns:
            return column
    return None


def _is_numeric_series(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna().sum() >= max(3, int(0.8 * len(series)))


def _compute_hyperparameter_trends(
    complete: pd.DataFrame,
    tune_set: set[str],
) -> list[dict[str, Any]]:
    """Estimate whether increasing each tuned parameter helps or hurts score."""
    scores = complete["value"].astype(float)
    trends: list[dict[str, Any]] = []

    for param in TUNABLE_PARAMETERS:
        if param not in tune_set:
            continue
        column = _trial_param_column(param, complete.columns)
        if column is None:
            continue

        values = complete[column]
        mask = values.notna() & scores.notna()
        if int(mask.sum()) < 3:
            continue

        param_values = values[mask]
        param_scores = scores[mask]

        entry: dict[str, Any] = {
            "parameter": param,
            "n_trials": int(mask.sum()),
            "score_direction": "lower_is_better",
        }

        if _is_numeric_series(param_values):
            x = pd.to_numeric(param_values, errors="coerce").astype(float)
            y = param_scores.astype(float)
            rho, p_value = stats.spearmanr(x, y)
            slope, intercept = np.polyfit(x.to_numpy(), y.to_numpy(), deg=1)

            if rho > 0.05:
                direction = "increase_worsens"
                direction_text = "Higher values tend to WORSE validation score"
            elif rho < -0.05:
                direction = "increase_improves"
                direction_text = "Higher values tend to BETTER validation score"
            else:
                direction = "weak_or_flat"
                direction_text = "No clear trend with higher values"

            entry.update(
                {
                    "kind": "numeric",
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "trend_direction": direction,
                    "trend_summary": direction_text,
                }
            )
        else:
            grouped = (
                pd.DataFrame({"level": param_values.astype(str), "score": param_scores.astype(float)})
                .groupby("level", as_index=False)
                .agg(mean=("score", "mean"), count=("score", "count"))
                .sort_values("mean", ascending=True)
            )
            best = grouped.iloc[0]
            worst = grouped.iloc[-1]
            entry.update(
                {
                    "kind": "categorical",
                    "levels": grouped.to_dict(orient="records"),
                    "best_level": str(best["level"]),
                    "best_mean_score": float(best["mean"]),
                    "worst_level": str(worst["level"]),
                    "worst_mean_score": float(worst["mean"]),
                    "trend_summary": (
                        f"Best level={best['level']} (mean score {best['mean']:.4f}), "
                        f"worst level={worst['level']} (mean score {worst['mean']:.4f})"
                    ),
                }
            )
        trends.append(entry)

    return trends


def _print_trend_report(trends: list[dict[str, Any]]) -> None:
    if not trends:
        print("No hyperparameter trend summary available (need >= 3 completed trials).")
        return

    print()
    print("Hyperparameter trend summary (validation checkpoint score; lower is better):")
    print("-" * 88)
    for item in trends:
        param = item["parameter"]
        if item["kind"] == "numeric":
            rho = item["spearman_rho"]
            print(
                f"  {param:22s} | rho={rho:+.3f} | p={item['p_value']:.3g} | "
                f"{item['trend_summary']}"
            )
        else:
            print(f"  {param:22s} | {item['trend_summary']}")
    print("-" * 88)


def _save_tuning_plots(
    study: optuna.Study,
    results_dir: Path,
    *,
    tune_set: set[str],
) -> list[dict[str, Any]]:
    trials_df = study.trials_dataframe()
    complete = trials_df[trials_df["state"] == "COMPLETE"].copy()
    if complete.empty:
        return []

    complete = complete.sort_values("number").reset_index(drop=True)
    trends = _compute_hyperparameter_trends(complete, tune_set)
    trends_path = results_dir / "hyperparameter_trends.json"
    trends_path.write_text(json.dumps(trends, indent=2), encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    trial_numbers = complete["number"] + 1
    scores = complete["value"]

    # 1. Optimization history
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trial_numbers, scores, marker="o", linewidth=1.8, markersize=4, label="Trial score")
    ax.plot(
        trial_numbers,
        scores.cummin(),
        marker="s",
        linewidth=2.4,
        markersize=4,
        label="Best score so far",
    )
    ax.set_title("MultiNILM Optuna/TPE Optimization History", fontsize=14, weight="bold")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation checkpoint score (lower is better)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="major", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(results_dir / "optimization_history_checkpoint_score.png", dpi=220)
    plt.close(fig)

    # 2. Validation MAE / F1 across trials
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_columns = {
        "user_attrs_val_mae_watts": "Val MAE (W)",
        "user_attrs_val_f1": "Val macro F1",
        "user_attrs_val_mae_norm": "Val MAE (norm)",
    }
    for column, label in metric_columns.items():
        if column in complete.columns:
            ax.plot(
                trial_numbers,
                complete[column],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=label,
            )
    ax.set_title("Validation Metrics Across Hyperparameter Trials", fontsize=14, weight="bold")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Metric value")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="major", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(results_dir / "validation_metrics_by_trial.png", dpi=220)
    plt.close(fig)

    # 3. Runtime vs score
    duration_seconds = complete["duration"].dt.total_seconds()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(duration_seconds, scores, s=42, alpha=0.8)
    best_idx = int(scores.idxmin())
    ax.scatter(
        duration_seconds.loc[best_idx],
        scores.loc[best_idx],
        s=90,
        marker="*",
        color="crimson",
        label="Best trial",
    )
    ax.set_title("Runtime vs Validation Checkpoint Score", fontsize=14, weight="bold")
    ax.set_xlabel("Trial Runtime (seconds)")
    ax.set_ylabel("Checkpoint score")
    ax.grid(True, which="major", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(results_dir / "runtime_vs_checkpoint_score.png", dpi=220)
    plt.close(fig)

    # 4. Hyperparameter importance (tuned params only)
    try:
        importances = optuna.importance.get_param_importances(study)
        importances = {k: v for k, v in importances.items() if k in tune_set or k == "use_weight_decay"}
    except Exception as exc:
        print(f"[warning] Could not compute hyperparameter importance: {exc}")
        importances = {}

    if importances:
        items = sorted(importances.items(), key=lambda item: item[1], reverse=True)
        names = [item[0] for item in items]
        values = [item[1] for item in items]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(names[::-1], values[::-1], color="#4c78a8")
        ax.set_title("Hyperparameter Importance (tuned parameters)", fontsize=14, weight="bold")
        ax.set_xlabel("Importance")
        ax.grid(True, axis="x", alpha=0.30)
        fig.tight_layout()
        fig.savefig(results_dir / "hyperparameter_importance.png", dpi=220)
        plt.close(fig)

    # 5. Slice plots with trend lines / level means
    param_columns: list[tuple[str, str]] = []
    seen_columns: set[str] = set()
    for name in TUNABLE_PARAMETERS:
        if name not in tune_set:
            continue
        column = _trial_param_column(name, complete.columns)
        if column is None or column in seen_columns:
            continue
        seen_columns.add(column)
        param_columns.append((name, column))
    if "weight_decay" in tune_set and "params_use_weight_decay" in complete.columns:
        param_columns.append(("use_weight_decay", "params_use_weight_decay"))

    if param_columns:
        n_cols = 3
        n_rows = int((len(param_columns) + n_cols - 1) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(4, 3.5 * n_rows)))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        trend_by_param = {item["parameter"]: item for item in trends if item["kind"] == "numeric"}

        for ax, (param_name, column) in zip(axes, param_columns):
            values = complete[column]
            if _is_numeric_series(values):
                x = pd.to_numeric(values, errors="coerce").astype(float)
                ax.scatter(x, scores, s=36, alpha=0.8)
                if param_name in trend_by_param and x.notna().sum() >= 2:
                    slope = trend_by_param[param_name]["slope"]
                    intercept = trend_by_param[param_name]["intercept"]
                    x_line = np.linspace(float(x.min()), float(x.max()), 50)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, color="crimson", linewidth=2.0, label="Trend")
                    rho = trend_by_param[param_name]["spearman_rho"]
                    ax.text(
                        0.03,
                        0.97,
                        f"rho={rho:+.2f}",
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=9,
                        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                    )
            else:
                categories = list(dict.fromkeys(values.astype(str)))
                x_values = values.astype(str).map({name: idx for idx, name in enumerate(categories)})
                ax.scatter(x_values, scores, s=36, alpha=0.8)
                means = (
                    pd.DataFrame({"x": x_values, "score": scores})
                    .groupby("x")["score"]
                    .mean()
                )
                ax.plot(
                    means.index,
                    means.values,
                    color="crimson",
                    linewidth=2.0,
                    marker="D",
                    markersize=5,
                    label="Level mean",
                )
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, rotation=25, ha="right")
            ax.set_title(param_name)
            ax.set_ylabel("Checkpoint score")
            ax.grid(True, alpha=0.30)

        for ax in axes[len(param_columns) :]:
            ax.axis("off")

        fig.suptitle("Hyperparameter Slice Plots (with trends)", fontsize=15, weight="bold")
        fig.tight_layout()
        fig.savefig(results_dir / "hyperparameter_slice_plots.png", dpi=220)
        plt.close(fig)

    # 6. Dedicated trend summary figure (one panel per tuned parameter)
    if trends:
        n_cols = 2
        n_rows = int((len(trends) + n_cols - 1) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, max(4, 3.2 * n_rows)))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax, item in zip(axes, trends):
            param = item["parameter"]
            column = _trial_param_column(param, complete.columns)
            if column is None:
                ax.axis("off")
                continue
            values = complete[column]
            if item["kind"] == "numeric":
                x = pd.to_numeric(values, errors="coerce").astype(float)
                ax.scatter(x, scores, s=40, alpha=0.85, color="#4c78a8")
                x_line = np.linspace(float(x.min()), float(x.max()), 50)
                y_line = item["slope"] * x_line + item["intercept"]
                ax.plot(x_line, y_line, color="crimson", linewidth=2.2)
                title_suffix = f"rho={item['spearman_rho']:+.2f}"
            else:
                levels = [row["level"] for row in item["levels"]]
                means = [row["mean"] for row in item["levels"]]
                ax.bar(levels, means, color="#4c78a8", alpha=0.85)
                ax.set_xlabel(param)
                title_suffix = f"best={item['best_level']}"
            ax.set_title(f"{param} ({title_suffix})", fontsize=11, weight="bold")
            ax.set_ylabel("Checkpoint score")
            ax.grid(True, alpha=0.30)
            if item["kind"] == "numeric":
                ax.set_xlabel(param)

        for ax in axes[len(trends) :]:
            ax.axis("off")

        fig.suptitle(
            "Hyperparameter Trends (lower checkpoint score is better)",
            fontsize=14,
            weight="bold",
        )
        fig.tight_layout()
        fig.savefig(results_dir / "hyperparameter_trend_plots.png", dpi=220)
        plt.close(fig)

    return trends


def _print_search_space_info(fast_search: bool, *, model_cfg: dict | None = None) -> None:
    divisor_choices = _stride_divisor_choices(fast_search=fast_search)
    require_equal = _uses_full_input_training(model_cfg or {"windowing": {"training_targets": "full_input"}})
    channel_presets = _channel_schedule_presets(fast_search=fast_search)
    print("\nDefault search spaces (from multinilm.yaml knobs):")
    print(f"  gate_mode: {list(GATE_MODE_CHOICES)}")
    print(
        "  channel_schedule: start∈"
        f"{list(CHANNEL_START_CHOICES)} (÷2/base/×2) × depth∈"
        f"{list(CHANNEL_DEPTH_CHOICES_FAST if fast_search else CHANNEL_DEPTH_CHOICES_FULL)} "
        f"-> {list(channel_presets.values())}"
    )
    print(f"  detail_kernels: {list(DETAIL_KERNEL_PRESETS.keys())}")
    print(f"  domain_scale: {list(DOMAIN_SCALE_CHOICES)}")
    print("  domain_adaptation.enabled: always true")
    print(f"  epochs: fixed at {DEFAULT_EPOCHS} (not tuned)")
    if require_equal:
        print(f"  window_length (input=output): {list(WINDOW_LENGTH_CHOICES)}")
        print("  Safe rule: training_targets=full_input -> input_window_length must equal output_window_length.")
    else:
        print(f"  input_window_length: {list(INPUT_WINDOW_CHOICES)}")
        print(f"  output_window_length: any of {list(OUTPUT_WINDOW_CANDIDATES)} with output <= input")
    print(f"  stride_divisor: {divisor_choices}  ->  stride = window // divisor")
    print("  Example (window -> strides):")
    for input_len in WINDOW_LENGTH_CHOICES:
        print(f"    {input_len} -> {_stride_choices_for_window(input_len, fast_search=fast_search)}")
    print("  Aliases: 'window_length' (both windows), 'stride' (both stride divisors).")
    print("  eval_reconstruction/training_targets: auto (derived from window + stride).\n")


def main() -> None:
    args = parse_args()
    if args.list_tune_params:
        print("Tunable hyperparameters (--tune accepts comma-separated names):")
        for name in TUNABLE_PARAMETERS:
            print(f"  - {name}")
        print("\nAlways forced:")
        print("  - domain_adaptation.enabled = true")
        print(f"  - epochs = {DEFAULT_EPOCHS} (constant; not tuned)")
        print("\nSpecial aliases:")
        print("  - window_length -> input_window_length + output_window_length (kept equal for full_input)")
        print("  - stride        -> input/eval stride_divisor together (stride = window // divisor)")
        print("\nAliases:", ", ".join(f"{k}->{v}" for k, v in sorted(TUNE_ALIASES.items())))
        _print_search_space_info(fast_search=True, model_cfg=load_model_config(DEFAULT_MODEL_CONFIG))
        return

    tune_set = _parse_tune_set(args.tune)
    experiment = load_experiment(args.experiment)
    base_model_cfg = load_model_config(args.model_config)
    baseline_params = _baseline_param_values(base_model_cfg)
    helper_keys = {
        "channel_schedule_key",
        "detail_kernels_key",
        "channel_start",
        "channel_depth",
        "input_stride_divisor",
        "eval_stride_divisor",
    }
    fixed_params = {
        key: baseline_params[key]
        for key in baseline_params
        if key not in tune_set and key not in helper_keys
    }
    experiment_id = str(experiment["experiment_id"])
    fast_search = bool(args.fast)

    # Epochs are never tuned; constant DEFAULT_EPOCHS unless --epochs overrides.
    fixed_epochs = int(args.epochs) if args.epochs is not None else DEFAULT_EPOCHS
    baseline_params["epochs"] = fixed_epochs
    fixed_params["epochs"] = fixed_epochs

    results_dir = args.results_dir or (
        ROOT / "results" / f"multinilm_hyperparameter_tuning_{experiment_id}"
    )
    trials_root = results_dir / "trials"
    results_dir.mkdir(parents=True, exist_ok=True)
    trials_root.mkdir(parents=True, exist_ok=True)

    data_root = _resolve_data_root(experiment, args.data_path)
    checkpoint_monitor = str(
        base_model_cfg.get("training", {}).get("checkpoint_monitor", "val_mae_minus_f1")
    )
    appliances = merge_configs(experiment, base_model_cfg)["appliances"]

    best_params_path = results_dir / "best_hyperparameters.json"
    trials_log_path = results_dir / "hyperparameter_trials.csv"
    study_config_path = results_dir / "study_config.json"

    study_config = {
        "experiment": str(args.experiment),
        "model_config": str(args.model_config),
        "experiment_id": experiment_id,
        "n_trials": int(args.n_trials),
        "epochs_per_trial": fixed_epochs,
        "fast_search_space": fast_search,
        "seed": int(args.seed),
        "checkpoint_monitor": checkpoint_monitor,
        "domain_adaptation_enabled": True,
        "appliances": appliances,
        "objective_metric": {
            "name": checkpoint_monitor,
            "direction": "minimize",
            "formula": "normalized_val_MAE - macro_val_F1 (when checkpoint_monitor=val_mae_minus_f1)",
            "split_used_for_selection": "validation (H1+H5 last week)",
            "split_not_used_for_selection": "test (H2 cross-house)",
            "logged_per_trial": [
                "best_score",
                "val_mae_watts",
                "val_mae_norm",
                "val_f1",
                "val_loss",
                "best_epoch",
            ],
            "full_eval_after_tuning": [
                "MAE (W per appliance)",
                "SAE",
                "EA",
                "macro/micro F1",
                "accuracy",
            ],
        },
        "tuned_parameters": sorted(tune_set),
        "fixed_parameters": fixed_params,
        "fixed_settings": {
            "domain_adaptation.enabled": True,
            "scheduler": "none",
            "training_plots": "disabled during tuning",
            "search_space": {
                "gate_mode": list(GATE_MODE_CHOICES),
                "channel_start": list(CHANNEL_START_CHOICES),
                "channel_depth": list(
                    CHANNEL_DEPTH_CHOICES_FAST if fast_search else CHANNEL_DEPTH_CHOICES_FULL
                ),
                "channel_schedule": _channel_schedule_presets(fast_search=fast_search),
                "detail_kernels": DETAIL_KERNEL_PRESETS,
                "domain_scale": list(DOMAIN_SCALE_CHOICES),
                "input_window_length": list(INPUT_WINDOW_CHOICES),
                "window_length_choices_when_full_input": list(WINDOW_LENGTH_CHOICES),
                "window_safety_rule": (
                    "training_targets=full_input requires input_window_length == output_window_length"
                ),
                "output_window_candidates": list(OUTPUT_WINDOW_CANDIDATES),
                "output_window_rule": "output <= input",
                "stride_divisor_choices": _stride_divisor_choices(fast_search=fast_search),
                "stride_rule": "stride = input_window_length // stride_divisor",
            },
        },
    }
    study_config_path.write_text(json.dumps(study_config, indent=2), encoding="utf-8")

    print("MultiNILM surrogate hyperparameter tuning")
    print(f"Experiment: {args.experiment}")
    print(f"Model config: {args.model_config}")
    print(f"Trials: {args.n_trials} | Epochs/trial: {fixed_epochs} (fixed) | Fast space: {fast_search}")
    print("domain_adaptation.enabled: always true")
    print(f"Tuning: {', '.join(sorted(tune_set))}")
    print(f"Fixed from yaml: {', '.join(f'{k}={v}' for k, v in sorted(fixed_params.items()))}")
    print(f"Objective: minimize {checkpoint_monitor}")
    print(f"Results: {results_dir}")
    if tune_set & {"input_window_length", "output_window_length", "input_stride", "eval_stride"}:
        print("[note] Window/stride: stride = window // divisor (auto-recomputed when window changes).")
    _print_search_space_info(fast_search, model_cfg=base_model_cfg)
    if _uses_full_input_training(base_model_cfg):
        print("[note] full_input training: tuner keeps input_window_length == output_window_length.")
    print(flush=True)

    start_time = perf_counter()
    require_equal_windows = _uses_full_input_training(base_model_cfg)

    def objective(trial: optuna.Trial) -> float:
        trial_params = _suggest_trial_params(
            trial,
            fast_search=fast_search,
            tune_set=tune_set,
            baseline=baseline_params,
            model_cfg=base_model_cfg,
        )
        trial_epochs = fixed_epochs
        trial_params["epochs"] = trial_epochs
        model_cfg = copy.deepcopy(base_model_cfg)
        _apply_tuning_overrides(model_cfg, trial_params, epochs=trial_epochs)
        _validate_window_stride_config(
            model_cfg,
            trial_params,
            experiment,
            data_root,
        )
        merged = _deep_copy_merged(experiment, model_cfg)

        trial_dir = trials_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        trial_start = perf_counter()
        print()
        print("=" * 88, flush=True)
        print(f"Trial {trial.number + 1}/{args.n_trials} started", flush=True)
        print(f"Parameters: {trial_params}", flush=True)
        print(f"Run dir: {trial_dir}", flush=True)

        adapter = MultiNILMAdapter(merged, data_root=str(data_root) if data_root else None)
        try:
            train_model(adapter, trial_dir, epochs=trial_epochs, seed=args.seed)
            metrics = _read_trial_metrics(trial_dir)
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            print(f"Trial {trial.number + 1} failed: {exc}", flush=True)
            if isinstance(exc, ValueError) and "Unsafe window" in str(exc):
                raise optuna.TrialPruned(str(exc)) from exc
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            raise

        trial_elapsed = perf_counter() - trial_start
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("trial_run_dir", str(trial_dir))
        for key, value in trial_params.items():
            trial.set_user_attr(f"param_{key}", value)

        print(
            f"Trial {trial.number + 1}/{args.n_trials} finished in {trial_elapsed / 60:.1f} min",
            flush=True,
        )
        print(
            f"best_score={metrics['best_score']:.4f} | "
            f"best_epoch={metrics['best_epoch']} | "
            f"val_mae={metrics.get('val_mae_watts', float('nan')):.2f} W | "
            f"val_f1={metrics.get('val_f1', float('nan')):.4f}",
            flush=True,
        )

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return metrics["best_score"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(args.seed)),
        study_name=f"multinilm_{experiment_id}",
    )
    study.optimize(objective, n_trials=int(args.n_trials), show_progress_bar=True)

    best_params = study.best_params.copy()
    use_weight_decay = best_params.pop("use_weight_decay", None)
    if use_weight_decay is False:
        best_params["weight_decay"] = 0.0

    if "channel_start" in best_params or "channel_depth" in best_params:
        channel_start = int(best_params.get("channel_start", baseline_params["channel_start"]))
        channel_depth = int(best_params.get("channel_depth", baseline_params["channel_depth"]))
        best_params["channel_start"] = channel_start
        best_params["channel_depth"] = channel_depth
        best_params["channel_schedule"] = _make_channel_schedule(channel_start, channel_depth)
        best_params["channel_schedule_key"] = f"start{channel_start}_depth{channel_depth}"
        best_params["hidden_channels"] = int(best_params["channel_schedule"][-1])

    if "detail_kernels" in best_params and isinstance(best_params["detail_kernels"], str):
        detail_key = best_params["detail_kernels"]
        best_params["detail_kernels_key"] = detail_key
        best_params["detail_kernels"] = list(DETAIL_KERNEL_PRESETS[detail_key])

    if "window_length" in best_params:
        window_value = int(best_params.pop("window_length"))
        best_params["input_window_length"] = window_value
        best_params["output_window_length"] = window_value

    if "stride_divisor" in best_params:
        stride_divisor = int(best_params.pop("stride_divisor"))
        best_params["input_stride_divisor"] = stride_divisor
        best_params["eval_stride_divisor"] = stride_divisor

    for key in TUNABLE_PARAMETERS:
        if key not in tune_set:
            best_params[key] = baseline_params[key]
    if "channel_schedule" not in tune_set:
        best_params["channel_schedule"] = baseline_params["channel_schedule"]
        best_params["channel_schedule_key"] = baseline_params["channel_schedule_key"]
        best_params["channel_start"] = baseline_params["channel_start"]
        best_params["channel_depth"] = baseline_params["channel_depth"]
    if "detail_kernels" not in tune_set:
        best_params["detail_kernels"] = baseline_params["detail_kernels"]
        best_params["detail_kernels_key"] = baseline_params["detail_kernels_key"]
    if "input_stride" not in tune_set:
        best_params["input_stride_divisor"] = baseline_params["input_stride_divisor"]
    if "eval_stride" not in tune_set:
        best_params["eval_stride_divisor"] = baseline_params["eval_stride_divisor"]

    best_params["epochs"] = fixed_epochs
    best_params["hidden_channels"] = int(best_params["channel_schedule"][-1])
    _normalize_window_params(best_params, require_equal_windows=require_equal_windows)

    best_trial = study.best_trial

    arch_patch: dict[str, Any] = {}
    for key in (
        "channel_schedule",
        "stem_kernel_size",
        "stage_kernel_size",
        "num_blocks",
        "kernel_size",
        "max_dilation",
        "dropout",
        "gate_mode",
        "gate_threshold",
        "head_local_layers",
        "head_kernel_size",
        "head_use_residual",
        "use_multiscale_stem",
        "detail_kernels",
        "detail_branch_channels",
    ):
        if key in tune_set:
            arch_patch[key] = best_params[key]
    if "channel_schedule" in tune_set or "num_blocks" in tune_set:
        arch_patch["hidden_channels"] = best_params["hidden_channels"]
    if "num_blocks" in tune_set:
        arch_patch["domain_feature_layers"] = _domain_feature_layers_for_blocks(
            int(best_params["num_blocks"])
        )

    loss_patch: dict[str, Any] = {}
    for key in ("domain_mu", "domain_scale", "lambda_domain"):
        if key in tune_set:
            loss_patch[key] = best_params[key]

    window_patch: dict[str, Any] = {}
    for key in ("input_window_length", "output_window_length", "input_stride", "eval_stride"):
        if key in tune_set:
            window_patch[key] = best_params[key]

    train_patch: dict[str, Any] = {"scheduler": "none", "epochs": fixed_epochs}
    for key in ("batch_size", "learning_rate", "weight_decay"):
        if key in tune_set:
            train_patch[key] = best_params.get(key, baseline_params.get(key))

    result = {
        "best_checkpoint_score": study.best_value,
        "checkpoint_monitor": checkpoint_monitor,
        "best_params": best_params,
        "tuned_parameters": sorted(tune_set),
        "fixed_parameters": fixed_params,
        "best_trial_metrics": {
            key: best_trial.user_attrs.get(key)
            for key in (
                "best_epoch",
                "epochs_completed",
                "val_mae_watts",
                "val_mae_norm",
                "val_f1",
                "val_loss",
                "trial_run_dir",
            )
        },
        "recommended_yaml_patch": {
            **({"architecture": arch_patch} if arch_patch else {}),
            **({"loss": loss_patch} if loss_patch else {}),
            **({"windowing": window_patch} if window_patch else {}),
            "domain_adaptation": {"enabled": True},
            "training": train_patch,
        },
        **study_config,
    }

    best_params_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_log_path, index=False)
    trends = _save_tuning_plots(study, results_dir, tune_set=tune_set)
    _print_trend_report(trends)

    elapsed = perf_counter() - start_time
    print()
    print("MultiNILM hyperparameter tuning completed.")
    print(f"Trials: {args.n_trials}")
    print(f"Best validation checkpoint score: {study.best_value:.4f}")
    print(f"Best parameters: {best_params}")
    print(f"Saved best parameters: {best_params_path}")
    print(f"Saved trial log: {trials_log_path}")
    print(f"Saved trend summary: {results_dir / 'hyperparameter_trends.json'}")
    print(f"Saved plots under: {results_dir}")
    print(f"  - hyperparameter_trend_plots.png (increase/decrease vs score)")
    print(f"  - hyperparameter_slice_plots.png (scatter + trend line)")
    print(f"Elapsed time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()

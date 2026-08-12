"""Load and merge experiment + model YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Dataset side: CSV paths, columns, normalization, evaluation."""
    return load_yaml(path)


def load_model_config(path: str | Path) -> dict[str, Any]:
    """Model side: windowing, architecture, loss, training hyperparameters."""
    return load_yaml(path)


def model_name_from_config(model_cfg: dict[str, Any]) -> str:
    """Read model id from config/models/*.yaml (top-level model_name)."""
    if name := model_cfg.get("model_name"):
        return str(name)
    if nested := model_cfg.get("model", {}).get("name"):
        return str(nested)
    raise ValueError("Model yaml must define model_name (e.g. model_name: multinilm)")


def appliance_list(experiment: dict[str, Any], model_cfg: dict[str, Any] | None = None) -> list[str]:
    """Appliance channel order for targets, model outputs, and metrics."""
    if model_cfg:
        if apps := model_cfg.get("data", {}).get("appliances"):
            return list(apps)
    return list(experiment["csv"]["appliances"])


def appliance_off_norm_normalized(experiment: dict[str, Any], appliances: list[str]) -> list[float]:
    """Normalized power target when an appliance is OFF (0 W in raw space).

    With z-score targets (w - mean) / std, 0 W maps to -mean/std per appliance.
    State gating must blend toward this value when OFF, not toward 0, otherwise
    denorm(0) = mean and waveforms show a constant watt spike (e.g. fridge 50 W).
    """
    norm = experiment.get("normalization", {})
    app_cfg = norm.get("appliances", {})
    values: list[float] = []
    for app in appliances:
        stats = app_cfg.get(app)
        if not stats or "mean" not in stats or "std" not in stats:
            values.append(0.0)
            continue
        std = float(stats["std"])
        mean = float(stats["mean"])
        values.append(-mean / std if std else 0.0)
    return values


def resolve_tensor_dtype(model_cfg: dict[str, Any]) -> tuple[np.dtype, torch.dtype]:
    """Resolve training tensor dtype from model yaml (baseline transfer uses float64)."""
    raw = (
        model_cfg.get("training", {}).get("tensor_dtype")
        or model_cfg.get("data", {}).get("tensor_dtype")
        or "float32"
    )
    name = str(raw).lower()
    if name in {"float64", "double", "f64"}:
        return np.float64, torch.float64
    return np.float32, torch.float32


def resolve_eval_reconstruction(windowing: dict[str, Any], *, split: str = "validation") -> str:
    """Resolve timeline reconstruction mode for inference plots and metrics.

    When eval stride is smaller than the output window, windows overlap on the
    CSV timeline. ``flat`` concatenation produces non-monotonic csv_timesteps
    and zig-zag waveform plots; use ``overlap_mean`` in that case.

    ``eval_reconstruction: auto`` (recommended) picks:
        overlap_mean  if eval_stride < output_window_length
        flat          otherwise
    """
    mode = str(windowing.get("eval_reconstruction", "auto")).lower()
    out_len = int(windowing.get("output_window_length", 1))
    stride = int(windowing.get("eval_stride", windowing.get("input_stride", 1)))
    if mode in {"", "auto", "default"}:
        return "overlap_mean" if stride < out_len else "flat"
    if mode == "flat" and stride < out_len:
        return "overlap_mean"
    return mode


def resolve_training_targets(windowing: dict[str, Any]) -> str:
    """Resolve which timesteps are supervised during training.

    ``training_targets: auto`` (recommended) picks:
        full_input     when input_window_length == output_window_length
        output_window  otherwise (end-aligned slice on val/test always)
    """
    raw = str(windowing.get("training_targets", "auto")).lower()
    if raw in {"", "auto", "default"}:
        input_len = int(windowing.get("input_window_length", 1))
        output_len = int(windowing.get("output_window_length", input_len))
        return "full_input" if input_len == output_len else "output_window"
    if raw not in {"full_input", "output_window"}:
        raise ValueError(
            "windowing.training_targets must be one of: auto, full_input, output_window"
        )
    return raw


def resolve_lr_scheduler_settings(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve LR scheduler switch and parameters from model training yaml.

    Primary switch (on/off + type):

        training.scheduler: none | reduce_on_plateau | step_lr
        # or
        training.scheduler:
          type: reduce_on_plateau

    Secondary preset (used when scheduler is enabled; safe to keep while scheduler=none):

        training.lr_scheduler:
          monitor: val_mae_minus_f1
          patience: 5
          factor: 0.5
          min_lr: 0.000001
          step_size: 100
          gamma: 0.1

    Legacy flat keys (still supported): scheduler_monitor, scheduler_patience, ...
    """
    secondary: dict[str, Any] = {}
    if isinstance(raw_secondary := train_cfg.get("lr_scheduler"), dict):
        secondary = dict(raw_secondary)

    primary = train_cfg.get("scheduler", "none")
    if isinstance(primary, dict):
        sched_type = str(primary.get("type", primary.get("name", "none"))).lower()
        params = {**secondary, **primary}
    else:
        sched_type = str(primary or "none").lower()
        params = dict(secondary)
        legacy_map = {
            "scheduler_monitor": "monitor",
            "scheduler_patience": "patience",
            "scheduler_factor": "factor",
            "scheduler_min_lr": "min_lr",
            "scheduler_step_size": "step_size",
            "scheduler_gamma": "gamma",
            "decay_step": "step_size",
            "gamma": "gamma",
        }
        for legacy_key, param_key in legacy_map.items():
            if legacy_key in train_cfg and param_key not in params:
                params[param_key] = train_cfg[legacy_key]

    disabled = sched_type in {"", "none", "off", "disabled", "null", "false"}
    monitor_default = str(train_cfg.get("checkpoint_monitor", "val_mae")).lower()
    preset_type = str(params.get("type", sched_type)).lower()

    return {
        "type": sched_type,
        "preset_type": preset_type,
        "enabled": not disabled,
        "monitor": str(params.get("monitor", monitor_default)).lower(),
        "patience": int(params.get("patience", 5)),
        "factor": float(params.get("factor", 0.5)),
        "min_lr": float(params.get("min_lr", 1e-6)),
        "step_size": int(params.get("step_size", 100)),
        "gamma": float(params.get("gamma", 0.1)),
    }


def resolve_experiment_id(experiment: dict[str, Any], model_cfg: dict[str, Any]) -> str:
    """Run-folder name under ``runs/<experiment_id>/<model_name>/``.

    Split of responsibilities:
      - ``config/experiment_*.yaml`` → data only (``dataset_id``, CSVs, norms)
      - ``config/models/*.yaml`` → training setup; set ``experiment_id`` here

    Example::

        # experiment_ukdale.yaml
        dataset_id: ukdale
        # multinilm_fractional.yaml
        experiment_id: ukdale(20wk source, no DA)
        # -> runs/ukdale(20wk source, no DA)/multinilm_fractional/

    Fallbacks (legacy / quick runs):
      1. model ``experiment_id`` (preferred)
      2. ``dataset_id`` (+ optional model ``run_tag``)
      3. experiment ``experiment_id`` (+ optional ``run_tag``)
    """
    if model_cfg.get("experiment_id"):
        return str(model_cfg["experiment_id"]).strip()

    base = experiment.get("dataset_id") or experiment.get("experiment_id")
    if not base:
        raise ValueError(
            "Set experiment_id in the model yaml (run folder name), "
            "or dataset_id in the experiment yaml."
        )
    base = str(base).strip()
    tag = model_cfg.get("run_tag")
    if tag is None or str(tag).strip() == "":
        return base
    return f"{base}, {str(tag).strip()}"


def merge_configs(experiment: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Single runtime config passed to adapters and runner."""
    experiment_id = resolve_experiment_id(experiment, model_cfg)
    # Inject so adapters that read experiment["experiment_id"] stay consistent.
    experiment_runtime = {**experiment, "experiment_id": experiment_id}
    return {
        "experiment": experiment_runtime,
        "model": model_cfg,
        "model_name": model_name_from_config(model_cfg),
        "experiment_id": experiment_id,
        "appliances": appliance_list(experiment_runtime, model_cfg),
        "data_root": experiment_runtime.get("data_root"),
        "evaluation": experiment_runtime["evaluation"],
        "seed": experiment_runtime.get("seed"),
    }

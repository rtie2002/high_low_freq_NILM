"""Read the single project configuration and its dataset settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_data_root(experiment_path: Path, experiment: dict[str, Any]) -> Path:
    """Keep legacy root-relative and config-relative dataset paths readable."""
    configured = Path(experiment["data_root"])
    if configured.is_absolute():
        return configured
    candidates = (experiment_path.parent / configured, experiment_path.parent.parent / configured)
    return next((path for path in candidates if path.exists()), candidates[-1])


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Dataset side: CSV paths, columns, normalization, evaluation."""
    experiment_path = Path(path).resolve()
    config = load_yaml(experiment_path)
    config = config.get("dataset", config)
    if normalization_file := config.get("normalization_file"):
        stats_path = Path(normalization_file)
        if not stats_path.is_absolute():
            stats_path = experiment_path.parent / stats_path
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Normalization statistics not found: {stats_path}. "
                "Regenerate the mixed dataset before training."
            )
        with stats_path.open(encoding="utf-8") as file:
            config["normalization"] = json.load(file)
    return config


def appliance_list(experiment: dict[str, Any], model_cfg: dict[str, Any] | None = None) -> list[str]:
    """Appliance channel order for targets, model outputs, and metrics."""
    if model_cfg:
        if apps := model_cfg.get("data", {}).get("appliances"):
            return list(apps)
    return list(experiment["csv"]["appliances"])


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

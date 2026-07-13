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


def merge_configs(experiment: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Single runtime config passed to adapters and runner."""
    return {
        "experiment": experiment,
        "model": model_cfg,
        "model_name": model_name_from_config(model_cfg),
        "experiment_id": experiment["experiment_id"],
        "appliances": appliance_list(experiment, model_cfg),
        "data_root": experiment.get("data_root"),
        "evaluation": experiment["evaluation"],
        "seed": experiment.get("seed"),
    }

"""Load experiment + model YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def load_model_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def appliance_list(experiment: dict[str, Any], model_cfg: dict[str, Any] | None = None) -> list[str]:
    """Appliance order from model data.appliances override, else csv.appliances keys."""
    if model_cfg:
        if apps := model_cfg.get("data", {}).get("appliances"):
            return list(apps)
    return list(experiment["csv"]["appliances"])


def merge_configs(experiment: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "model": model_cfg,
        "experiment_id": experiment["experiment_id"],
        "appliances": appliance_list(experiment, model_cfg),
        "data_root": experiment.get("data_root"),
        "evaluation": experiment["evaluation"],
        "seed": experiment.get("seed"),
    }

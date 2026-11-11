"""One readable evaluation entry point for every inverse NILM method.

The inverse solver supplies only `(csv_timesteps, predicted_appliance_watts)`.
This module attaches held-out labels solely for evaluation, then saves the same
prediction bundle, metric tables, background diagnostics, and waveforms used by
this project's frozen benchmark protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import appliance_list, load_experiment

from .data_loader import NILMDataLoader
from .metrics import background_fpr_table, evaluate_bundle, resolve_power_postprocess
from .output_format import save_result_json, save_result_table
from .prediction import PredictionBundle
from .waveform_capture import save_appliance_on_waveforms


@dataclass(frozen=True)
class NILMProtocol:
    """Fixed split and evaluation settings used by an inverse experiment."""

    experiment: dict[str, Any]
    appliances: list[str]
    loader: NILMDataLoader
    thresholds_watts: np.ndarray


def load_protocol(
    *,
    experiment_path: Path,
    window_length: int,
    stride: int,
    batch_size: int,
) -> NILMProtocol:
    """Load the local benchmark split with inverse-method window settings."""
    experiment_path = Path(experiment_path).resolve()
    experiment = load_experiment(experiment_path)
    appliances = appliance_list(experiment)
    inverse_window_cfg = {
        "windowing": {
            "input_window_length": window_length,
            "output_window_length": window_length,
            "input_stride": stride,
            "eval_stride": stride,
            "output_alignment": "end",
            "training_targets": "full_input",
        },
        "training": {"batch_size": batch_size, "num_workers": 0},
        "data": {"state_label_source": "csv"},
    }
    data_root = Path(experiment["data_root"])
    if not data_root.is_absolute():
        data_root = experiment_path.parent / data_root
    loader = NILMDataLoader(experiment, inverse_window_cfg, data_root)
    thresholds = np.asarray(
        [experiment["evaluation"]["on_thresholds_watts"][app] for app in appliances],
        dtype=np.float32,
    )
    return NILMProtocol(experiment, appliances, loader, thresholds)


def save_inverse_evaluation(
    *,
    protocol: NILMProtocol,
    split: str,
    run_dir: Path,
    model_name: str,
    csv_timesteps: np.ndarray,
    predicted_appliance_watts: np.ndarray,
    manifest: dict[str, Any],
    waveform_periods: int = 5,
    waveform_dpi: int = 200,
    random_seed: int = 2026,
) -> PredictionBundle:
    """Evaluate one inverse timeline with this project's fixed protocol."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    raw = protocol.loader.get_split(split)
    csv_timesteps = np.asarray(csv_timesteps, dtype=np.int64)
    prediction = np.asarray(predicted_appliance_watts, dtype=np.float32)
    expected_shape = (len(csv_timesteps), len(protocol.appliances))
    if prediction.shape != expected_shape:
        raise ValueError(f"Predictions must have shape {expected_shape}, got {prediction.shape}")

    bundle = PredictionBundle(
        experiment_id=str(protocol.experiment.get("dataset_id", "legacy_dataset")),
        model_name=model_name,
        split=split,
        appliances=protocol.appliances,
        sample_index=np.arange(len(csv_timesteps), dtype=np.int64),
        y_true_watts=raw.targets[csv_timesteps],
        y_pred_watts=prediction,
        y_true_on=raw.states[csv_timesteps].astype(np.int32),
        y_pred_on=(prediction > protocol.thresholds_watts).astype(np.int32),
        # Deterministic inverse models do not have a calibrated state score.
        y_pred_state_prob=None,
        csv_timesteps=csv_timesteps,
        segment_ids=raw.segment_ids[csv_timesteps],
    )
    bundle.save(run_dir / "predictions.npz")

    postprocess = resolve_power_postprocess(protocol.experiment, protocol.appliances)
    evaluation_cfg = protocol.experiment["evaluation"]
    sample_seconds = float(protocol.experiment["csv"]["sample_seconds"])
    metrics = evaluate_bundle(
        bundle,
        sae_period=int(evaluation_cfg.get("sae_period", 1200)),
        on_threshold_watts=protocol.thresholds_watts,
        state_label_source="csv",
        power_postprocess=postprocess,
        sample_seconds=sample_seconds,
        event_tolerance_seconds=float(evaluation_cfg.get("event_tolerance_seconds", 0.0)),
    )
    save_result_table(metrics, run_dir / "metrics.csv")

    aggregate_watts = raw.inputs[csv_timesteps]
    background_metrics = background_fpr_table(
        bundle,
        aggregate_watts,
        sample_seconds=sample_seconds,
        state_label_source="csv",
        on_threshold_watts=protocol.thresholds_watts,
        power_postprocess=postprocess,
    )
    save_result_table(background_metrics, run_dir / "background_fpr.csv")
    save_result_json(run_dir / "run_manifest.json", manifest)
    save_appliance_on_waveforms(
        run_dir / "waveforms",
        appliances=protocol.appliances,
        y_true_watts=bundle.y_true_watts,
        y_pred_watts=bundle.y_pred_watts,
        y_true_on=bundle.y_true_on,
        y_pred_on=bundle.y_pred_on,
        state_label_source="csv",
        aggregate=aggregate_watts,
        csv_timesteps=csv_timesteps,
        segment_ids=bundle.segment_ids,
        n_periods=int(waveform_periods),
        dpi=int(waveform_dpi),
        title_prefix=f"{model_name} — ",
        rng=np.random.default_rng(random_seed),
    )
    return bundle

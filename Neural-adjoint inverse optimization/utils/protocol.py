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

from .config import appliance_list, load_experiment, resolve_data_root

from .data_loader import NILMDataLoader
from .metrics import apply_power_postprocess_pair, background_fpr_table, evaluate_bundle, resolve_power_postprocess
from .output_format import save_result_json, save_result_table
from .prediction import PredictionBundle
from .state_postprocess import apply_state_postprocess_arrays
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
    csv_sequence_only: bool = False,
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
    data_root = resolve_data_root(experiment_path, experiment)
    loader_experiment = experiment
    if csv_sequence_only:
        # Match Stage 1 windows, which use the preprocessed sequence IDs.
        csv_cfg = dict(experiment["csv"])
        csv_cfg.pop("sample_seconds", None)
        loader_experiment = {**experiment, "csv": csv_cfg}
    loader = NILMDataLoader(loader_experiment, inverse_window_cfg, data_root)
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
    waveform_dpi: int = 300,
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

    sample_seconds = float(protocol.experiment["csv"]["sample_seconds"])
    segments = raw.segment_ids[csv_timesteps]
    microwave = np.asarray([app == "microwave" for app in protocol.appliances])
    # Match the paper's duration rules and power gate. The inverse baseline
    # thresholds watts because it has no learned state head.
    pred_on = apply_state_postprocess_arrays(
        prediction,
        thresholds=protocol.thresholds_watts.tolist(),
        min_on_samples=np.where(microwave, round(32 / sample_seconds), 1).tolist(),
        merge_gap_samples=np.where(microwave, round(24 / sample_seconds), 0).tolist(),
        segment_ids=segments,
    )
    bundle = PredictionBundle(
        experiment_id=str(protocol.experiment.get("dataset_id", "legacy_dataset")),
        model_name=model_name,
        split=split,
        appliances=protocol.appliances,
        sample_index=np.arange(len(csv_timesteps), dtype=np.int64),
        y_true_watts=raw.targets[csv_timesteps],
        y_pred_watts=prediction * pred_on,
        y_true_on=raw.states[csv_timesteps].astype(np.int32),
        y_pred_on=pred_on,
        # Deterministic inverse models do not have a calibrated state score.
        y_pred_state_prob=None,
        csv_timesteps=csv_timesteps,
        segment_ids=segments,
    )
    bundle.save(run_dir / "predictions.npz")

    postprocess = resolve_power_postprocess(protocol.experiment, protocol.appliances)
    evaluation_cfg = protocol.experiment["evaluation"]
    metrics = evaluate_bundle(
        bundle,
        sae_period=int(evaluation_cfg.get("sae_period", 1200)),
        on_threshold_watts=protocol.thresholds_watts,
        state_label_source="csv",
        power_postprocess=postprocess,
        sample_seconds=sample_seconds,
        event_tolerance_seconds=float(evaluation_cfg.get("event_tolerance_seconds", 0.0)),
        state_scores=prediction,  # Rank by raw watts before thresholding or gating.
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
    save_result_json(run_dir / "run_manifest.json", {
        **manifest,
        "average_precision_score": "ungated_predicted_watts",
        "on_thresholds_watts": dict(zip(protocol.appliances, protocol.thresholds_watts.tolist())),
        "microwave_min_on_seconds": 32,
        "microwave_merge_gap_seconds": 24,
        "power_gated_by_decoded_state": True,
        "waveform_context_scale": 10,
    })
    _, plotted_watts = apply_power_postprocess_pair(bundle.y_true_watts, bundle.y_pred_watts, postprocess)
    save_appliance_on_waveforms(
        run_dir / "waveforms",
        appliances=protocol.appliances,
        y_true_watts=bundle.y_true_watts,
        y_pred_watts=plotted_watts,
        y_true_on=bundle.y_true_on,
        y_pred_on=bundle.y_pred_on,
        state_label_source="csv",
        aggregate=aggregate_watts,
        csv_timesteps=csv_timesteps,
        segment_ids=bundle.segment_ids,
        n_periods=int(waveform_periods),
        dpi=int(waveform_dpi),
        context_scale=10.0,
        title_prefix=f"{model_name} — ",
        rng=np.random.default_rng(random_seed),
    )
    return bundle

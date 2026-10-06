"""Compare two saved NILM runs on identical events and background conditions.

The script uses saved best-checkpoint ``predictions.npz`` files, so it does not
train or rerun inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.event_diagnostics import (
    event_waveform_table,
    paired_event_differences,
    run_event_summary,
    save_paired_waveform_plots,
    snr_summary_table,
)


@dataclass
class SavedPredictionBundle:
    """Torch-free view of the arrays required by the comparison diagnostics."""

    experiment_id: str
    model_name: str
    split: str
    appliances: list[str]
    y_true_watts: np.ndarray
    y_pred_watts: np.ndarray
    y_true_on: np.ndarray | None
    y_pred_on: np.ndarray | None
    csv_timesteps: np.ndarray | None
    segment_ids: np.ndarray | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare event waveforms from two saved best-checkpoint runs."
    )
    parser.add_argument("--run-a", type=Path, required=True, help="First model run directory")
    parser.add_argument("--run-b", type=Path, required=True, help="Second model run directory")
    parser.add_argument("--label-a", default="run_a", help="Short label used in tables/plots")
    parser.add_argument("--label-b", default="run_b", help="Short label used in tables/plots")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["refit_house20", "ukdale_house2"],
        help="Saved test scenario names",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _load_merged_config(run_dir: Path) -> tuple[dict, dict]:
    path = run_dir / "config_merged.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing merged config: {path}")
    with path.open(encoding="utf-8") as file:
        merged = yaml.safe_load(file) or {}
    return merged["experiment"], merged["model"]


def _resolve_data_root(experiment: dict) -> Path:
    data_root = Path(experiment["data_root"])
    return data_root if data_root.is_absolute() else PROJECT_ROOT / data_root


def _prediction_path(run_dir: Path, split: str) -> Path:
    path = run_dir / "test" / split / "predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved predictions: {path}")
    return path


def _optional_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    values = data[key]
    return None if values.size == 0 else values


def _load_bundle(path: Path) -> SavedPredictionBundle:
    with np.load(path, allow_pickle=True) as data:
        return SavedPredictionBundle(
            experiment_id=str(data["experiment_id"]),
            model_name=str(data["model_name"]),
            split=str(data["split"]),
            appliances=[str(item) for item in data["appliances"].tolist()],
            y_true_watts=data["y_true_watts"],
            y_pred_watts=data["y_pred_watts"],
            y_true_on=_optional_array(data, "y_true_on"),
            y_pred_on=_optional_array(data, "y_pred_on"),
            csv_timesteps=_optional_array(data, "csv_timesteps"),
            segment_ids=_optional_array(data, "segment_ids"),
        )


def _assert_aligned(a: SavedPredictionBundle, b: SavedPredictionBundle) -> None:
    if a.appliances != b.appliances:
        raise ValueError(f"Appliance order differs: {a.appliances} vs {b.appliances}")
    if a.csv_timesteps is None or b.csv_timesteps is None:
        raise ValueError("Both bundles need csv_timesteps for exact event alignment")
    if not np.array_equal(a.csv_timesteps, b.csv_timesteps):
        raise ValueError("Prediction bundles do not cover identical CSV timesteps")
    if a.y_true_watts.shape != b.y_true_watts.shape or not np.allclose(
        a.y_true_watts, b.y_true_watts, equal_nan=True
    ):
        raise ValueError("Prediction bundles do not contain identical ground truth")
    if a.y_pred_on is None or b.y_pred_on is None:
        raise ValueError("Both bundles need calibrated y_pred_on states")


def _scenario_csv_path(experiment: dict, split: str) -> Path:
    csv_cfg = experiment["csv"]
    scenarios = csv_cfg.get("test_scenarios", {})
    relative = scenarios.get(split)
    if not relative:
        raise ValueError(f"No csv.test_scenarios entry for {split}")
    path = Path(relative)
    return path if path.is_absolute() else _resolve_data_root(experiment) / path


def _load_raw_timeline(
    experiment: dict,
    split: str,
    appliances: list[str],
    csv_timesteps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw watts/labels using the same required-column dropna rule as training."""
    csv_cfg = experiment["csv"]
    mains_column = str(csv_cfg.get("mains_column", "aggregate"))
    power_columns = [csv_cfg["appliances"][app]["power"] for app in appliances]
    state_columns = [csv_cfg["appliances"][app]["state"] for app in appliances]
    required = list(dict.fromkeys([mains_column, *power_columns, *state_columns]))
    frame = pd.read_csv(_scenario_csv_path(experiment, split), usecols=required).dropna(
        subset=required
    )
    rows = np.asarray(csv_timesteps, dtype=np.int64)
    if rows.size and (rows.min() < 0 or rows.max() >= len(frame)):
        raise IndexError(f"Prediction CSV rows are outside the {split} CSV timeline")
    aggregate = frame[mains_column].to_numpy(dtype=np.float32)[rows]
    true_watts = frame[power_columns].to_numpy(dtype=np.float32)[rows]
    true_on = frame[state_columns].to_numpy(dtype=np.int32)[rows]
    return aggregate, true_watts, true_on


def _best_epoch(run_dir: Path) -> int | None:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file).get("best_epoch")


def main() -> None:
    args = _parse_args()
    run_a = args.run_a.resolve()
    run_b = args.run_b.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_b / "comparisons" / f"{args.label_a}_vs_{args.label_b}"
    )
    output.mkdir(parents=True, exist_ok=True)

    experiment, _ = _load_merged_config(run_a)
    all_events: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    all_differences: list[pd.DataFrame] = []

    for split in args.splits:
        bundle_a = _load_bundle(_prediction_path(run_a, split))
        bundle_b = _load_bundle(_prediction_path(run_b, split))
        _assert_aligned(bundle_a, bundle_b)
        csv_rows = bundle_a.csv_timesteps
        aggregate, true_watts, true_on = _load_raw_timeline(
            experiment,
            split,
            bundle_a.appliances,
            csv_rows,
        )
        segment_ids = bundle_a.segment_ids
        if segment_ids is None:
            segment_ids = np.zeros(len(csv_rows), dtype=np.int64)
        sample_seconds = float(experiment["csv"]["sample_seconds"])
        bundles = {args.label_a: bundle_a, args.label_b: bundle_b}

        split_events = pd.concat(
            [
                event_waveform_table(
                    bundle,
                    run_label=label,
                    aggregate_watts=aggregate,
                    true_appliance_watts=true_watts,
                    true_on=true_on,
                    segment_ids=segment_ids,
                    sample_seconds=sample_seconds,
                )
                for label, bundle in bundles.items()
            ],
            ignore_index=True,
        )
        all_events.append(split_events)
        all_summaries.append(
            run_event_summary(
                split_events,
                bundles,
                true_on=true_on,
                segment_ids=segment_ids,
                sample_seconds=sample_seconds,
            )
        )
        all_differences.append(
            paired_event_differences(split_events, args.label_a, args.label_b)
        )
        save_paired_waveform_plots(
            split_events,
            bundles,
            aggregate_watts=aggregate,
            true_appliance_watts=true_watts,
            true_on=true_on,
            segment_ids=segment_ids,
            sample_seconds=sample_seconds,
            output_dir=output / "paired_waveforms" / split,
            dpi=args.dpi,
        )

    events = pd.concat(all_events, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)
    differences = pd.concat(all_differences, ignore_index=True)
    snr_summary = snr_summary_table(events)
    events.to_csv(output / "event_metrics.csv", index=False)
    summary.to_csv(output / "run_summary.csv", index=False)
    snr_summary.to_csv(output / "snr_summary.csv", index=False)
    differences.to_csv(output / "paired_event_differences.csv", index=False)

    metadata = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "best_epoch_a": _best_epoch(run_a),
        "best_epoch_b": _best_epoch(run_b),
        "splits": args.splits,
        "sample_seconds": float(experiment["csv"]["sample_seconds"]),
        "delta_definition": "run_b minus run_a",
    }
    with (output / "comparison_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nEvent-level comparison (best saved predictions):")
    print(summary.to_string(index=False))
    print(f"\nSaved diagnostics: {output}")


if __name__ == "__main__":
    main()

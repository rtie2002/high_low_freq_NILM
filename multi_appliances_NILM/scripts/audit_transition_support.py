"""Audit whether predicted appliance transitions are supported by mains edges.

This is a read-only diagnostic for saved ``predictions.npz`` bundles.  It
separates predicted ON transitions into matched onsets, false onsets while the
target is OFF, and fragmented onsets inside a true ON period.  For every onset
it measures the strongest positive aggregate edge in a small temporal
neighbourhood.

Example
-------
python scripts/audit_transition_support.py ^
  --run-dir runs/multinilm_clipped_per_appliance_balance_house_split/multinilm_fractional ^
  --dataset-root datasets/mixed_ukdale_refit_5w_house_split
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCENARIO_CSV = {
    "validation": Path("validating/multi_appliance_validating.csv"),
    "refit_house20": Path("testing/refit_house20/multi_appliance_testing.csv"),
    "ukdale_house2": Path("testing/ukdale_house2/multi_appliance_testing.csv"),
}


@dataclass(frozen=True)
class Bundle:
    appliances: list[str]
    true_on: np.ndarray
    pred_on: np.ndarray
    csv_timesteps: np.ndarray
    segment_ids: np.ndarray


def load_bundle(path: Path) -> Bundle:
    with np.load(path, allow_pickle=True) as data:
        segment_ids = np.asarray(data["segment_ids"], dtype=np.int64)
        if segment_ids.size == 0:
            segment_ids = np.zeros(len(data["y_true_on"]), dtype=np.int64)
        return Bundle(
            appliances=[str(value) for value in data["appliances"].tolist()],
            true_on=np.asarray(data["y_true_on"], dtype=bool),
            pred_on=np.asarray(data["y_pred_on"], dtype=bool),
            csv_timesteps=np.asarray(data["csv_timesteps"], dtype=np.int64),
            segment_ids=segment_ids,
        )


def load_aggregate(path: Path) -> np.ndarray:
    values: list[float] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "aggregate" not in reader.fieldnames:
            raise ValueError(f"Missing aggregate column in {path}")
        for row in reader:
            values.append(float(row["aggregate"]))
    return np.asarray(values, dtype=np.float64)


def onset_indices(state: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    previous = np.zeros_like(state, dtype=bool)
    same_segment = segment_ids[1:] == segment_ids[:-1]
    previous[1:] = state[:-1] & same_segment
    return np.flatnonzero(state & ~previous)


def nearest_onset_distance(indices: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.empty(0, dtype=np.int64)
    if len(reference) == 0:
        return np.full(len(indices), np.iinfo(np.int32).max, dtype=np.int64)
    positions = np.searchsorted(reference, indices)
    left = np.maximum(positions - 1, 0)
    right = np.minimum(positions, len(reference) - 1)
    return np.minimum(np.abs(indices - reference[left]), np.abs(indices - reference[right]))


def positive_edge_support(
    aggregate: np.ndarray,
    csv_rows: np.ndarray,
    segment_ids: np.ndarray,
    indices: np.ndarray,
    radius: int,
) -> np.ndarray:
    edge = np.zeros_like(aggregate)
    edge[1:] = np.maximum(aggregate[1:] - aggregate[:-1], 0.0)
    support = np.zeros(len(indices), dtype=np.float64)
    for output_index, onset in enumerate(indices):
        lo = max(0, int(onset) - radius)
        hi = min(len(csv_rows), int(onset) + radius + 1)
        valid = segment_ids[lo:hi] == segment_ids[onset]
        rows = csv_rows[lo:hi][valid]
        support[output_index] = float(np.max(edge[rows])) if len(rows) else 0.0
    return support


def summarize(values: np.ndarray) -> dict[str, float | int]:
    if len(values) == 0:
        return {"count": 0, "median_w": np.nan, "q10_w": np.nan, "q90_w": np.nan,
                "ge25_rate": np.nan, "ge50_rate": np.nan}
    return {
        "count": int(len(values)),
        "median_w": float(np.median(values)),
        "q10_w": float(np.quantile(values, 0.10)),
        "q90_w": float(np.quantile(values, 0.90)),
        "ge25_rate": float(np.mean(values >= 25.0)),
        "ge50_rate": float(np.mean(values >= 50.0)),
    }


def prediction_path(run_dir: Path, scenario: str) -> Path:
    if scenario == "validation":
        return run_dir / "validation_predictions.npz"
    return run_dir / "test" / scenario / "predictions.npz"


def audit_scenario(
    run_dir: Path,
    dataset_root: Path,
    scenario: str,
    radius: int,
) -> list[dict[str, object]]:
    bundle = load_bundle(prediction_path(run_dir, scenario))
    aggregate_full = load_aggregate(dataset_root / SCENARIO_CSV[scenario])
    if bundle.csv_timesteps.max(initial=-1) >= len(aggregate_full):
        raise ValueError(f"CSV row index exceeds aggregate length for {scenario}")

    rows: list[dict[str, object]] = []
    for app_index, appliance in enumerate(bundle.appliances):
        true_state = bundle.true_on[:, app_index]
        pred_state = bundle.pred_on[:, app_index]
        true_onsets = onset_indices(true_state, bundle.segment_ids)
        pred_onsets = onset_indices(pred_state, bundle.segment_ids)
        distance = nearest_onset_distance(pred_onsets, true_onsets)
        support = positive_edge_support(
            aggregate_full,
            bundle.csv_timesteps,
            bundle.segment_ids,
            pred_onsets,
            radius,
        )

        categories = {
            "matched": distance <= radius,
            "false_while_off": (distance > radius) & ~true_state[pred_onsets],
            "fragment_inside_on": (distance > radius) & true_state[pred_onsets],
        }
        true_support = positive_edge_support(
            aggregate_full,
            bundle.csv_timesteps,
            bundle.segment_ids,
            true_onsets,
            radius,
        )
        for category, mask in categories.items():
            rows.append({
                "scenario": scenario,
                "appliance": appliance,
                "category": category,
                **summarize(support[mask]),
            })
        rows.append({
            "scenario": scenario,
            "appliance": appliance,
            "category": "all_true_onsets",
            **summarize(true_support),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--radius-samples", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    for scenario in SCENARIO_CSV:
        results.extend(
            audit_scenario(
                args.run_dir,
                args.dataset_root,
                scenario,
                args.radius_samples,
            )
        )

    output = args.output or args.run_dir / "transition_support_audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {output}")
    for row in results:
        if row["appliance"] in {"fridge", "microwave"}:
            print(
                f"{row['scenario']:14s} {row['appliance']:9s} "
                f"{row['category']:18s} n={row['count']:4d} "
                f"median={row['median_w']:7.1f}W q10={row['q10_w']:7.1f}W "
                f"q90={row['q90_w']:7.1f}W >=50W={row['ge50_rate']:.3f}"
            )


if __name__ == "__main__":
    main()

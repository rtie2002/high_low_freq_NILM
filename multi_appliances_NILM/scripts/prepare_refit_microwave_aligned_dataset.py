#!/usr/bin/env python
"""Create a separate REFIT microwave sensor-aligned evaluation protocol.

REFIT exports aggregate and appliance readings under one row timestamp, but
the underlying meters can expose a small event-dependent lag.  This script
does not overwrite the established benchmark.  It copies an existing mixed
dataset and shifts each *complete REFIT microwave ON event* to the strongest
physically plausible aggregate rising edge within a fixed +/-3-sample search
window.  Aggregate power, every other appliance, UK-DALE, and split membership
remain unchanged.

The rule is deterministic and identical for train, validation, and test.  The
result is a secondary sensor-aligned protocol; original REFIT metrics must
still be reported for comparison with prior work.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "datasets" / "mixed_ukdale_refit_5w_house_split"
DEFAULT_OUTPUT = ROOT / "datasets" / "mixed_ukdale_refit_5w_house_split_mw_aligned"
CSV_RELATIVE_PATHS = (
    Path("training/multi_appliance_training.csv"),
    Path("validating/multi_appliance_validating.csv"),
    Path("testing/multi_appliance_testing.csv"),
    Path("testing/refit_house20/multi_appliance_testing.csv"),
    Path("testing/ukdale_house2/multi_appliance_testing.csv"),
)


def _runs(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.r_[0, np.asarray(state, dtype=np.int8), 0]
    changes = np.diff(padded)
    return np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)


def align_sequence(
    aggregate: np.ndarray,
    microwave_power: np.ndarray,
    microwave_on: np.ndarray,
    *,
    max_lag_samples: int = 3,
    minimum_microwave_jump_watts: float = 100.0,
    minimum_aggregate_edge_watts: float = 100.0,
    minimum_edge_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Align complete microwave events in one uninterrupted sequence."""
    aggregate = np.asarray(aggregate, dtype=np.float64)
    power = np.asarray(microwave_power, dtype=np.float64)
    state = np.asarray(microwave_on, dtype=np.int8)
    aligned_power = power.copy()
    aligned_state = state.copy()
    aggregate_delta = np.diff(aggregate, prepend=aggregate[0])
    starts, ends = _runs(state)
    lag_counts: Counter[int] = Counter()
    rejected = 0

    for start, end in zip(starts, ends):
        if start == 0 or end >= len(state):
            rejected += 1
            continue
        jump = float(power[start] - power[start - 1])
        if jump < minimum_microwave_jump_watts:
            rejected += 1
            continue

        low = max(1, int(start) - max_lag_samples)
        high = min(len(state) - 1, int(start) + max_lag_samples)
        candidate_rows = np.arange(low, high + 1, dtype=np.int64)
        best_row = int(candidate_rows[np.argmax(aggregate_delta[candidate_rows])])
        lag = best_row - int(start)
        best_edge = float(aggregate_delta[best_row])
        required_edge = max(
            minimum_aggregate_edge_watts,
            minimum_edge_fraction * jump,
        )
        shifted_start = int(start) + lag
        shifted_end = int(end) + lag
        if best_edge < required_edge or shifted_start < 0 or shifted_end > len(state):
            rejected += 1
            continue

        lag_counts[lag] += 1
        if lag == 0:
            continue

        event_power = power[start:end].copy()
        aligned_power[start:end] = 0.0
        aligned_state[start:end] = 0
        aligned_power[shifted_start:shifted_end] = np.maximum(
            aligned_power[shifted_start:shifted_end], event_power
        )
        aligned_state[shifted_start:shifted_end] = 1

    audit: dict[str, object] = {
        "events": int(len(starts)),
        "accepted_events": int(sum(lag_counts.values())),
        "rejected_events": int(rejected),
        "changed_events": int(sum(count for lag, count in lag_counts.items() if lag != 0)),
        "lag_counts": {str(lag): int(lag_counts.get(lag, 0)) for lag in range(-max_lag_samples, max_lag_samples + 1)},
    }
    return aligned_power, aligned_state, audit


def align_frame(frame: pd.DataFrame, **settings: object) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    required = {
        "dataset", "house", "sequence_id", "aggregate",
        "microwave_power", "microwave_on",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    result = frame.copy()
    refit = result["dataset"].astype(str).str.lower().eq("refit")
    audits: list[dict[str, object]] = []
    selected = result.loc[refit]
    for (house, sequence_id), group in selected.groupby(
        ["house", "sequence_id"], sort=False
    ):
        positions = group.index.to_numpy()
        power, state, audit = align_sequence(
            group["aggregate"].to_numpy(),
            group["microwave_power"].to_numpy(),
            group["microwave_on"].to_numpy(),
            **settings,
        )
        result.loc[positions, "microwave_power"] = power
        result.loc[positions, "microwave_on"] = state
        audits.append({"house": int(house), "sequence_id": int(sequence_id), **audit})
    return result, audits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-lag-samples", type=int, default=3)
    parser.add_argument("--minimum-microwave-jump-watts", type=float, default=100.0)
    parser.add_argument("--minimum-aggregate-edge-watts", type=float, default=100.0)
    parser.add_argument("--minimum-edge-fraction", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if args.max_lag_samples < 0:
        raise ValueError("max-lag-samples must be non-negative")

    shutil.copytree(source, output)
    settings = {
        "max_lag_samples": int(args.max_lag_samples),
        "minimum_microwave_jump_watts": float(args.minimum_microwave_jump_watts),
        "minimum_aggregate_edge_watts": float(args.minimum_aggregate_edge_watts),
        "minimum_edge_fraction": float(args.minimum_edge_fraction),
    }
    audit: dict[str, object] = {
        "source_dataset": str(source),
        "output_dataset": str(output),
        "protocol": "secondary REFIT microwave sensor-aligned protocol",
        "aggregate_changed": False,
        "ukdale_changed": False,
        "settings": settings,
        "files": {},
    }
    for relative_path in CSV_RELATIVE_PATHS:
        path = output / relative_path
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        aligned, rows = align_frame(frame, **settings)
        aligned.to_csv(path, index=False)
        audit["files"][str(relative_path)] = rows
        print(f"Aligned {relative_path}: {sum(row['changed_events'] for row in rows)} changed events")

    (output / "microwave_alignment_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# REFIT microwave sensor-aligned protocol\n\n"
        "This is a secondary diagnostic protocol generated from the established "
        "house-held-out split. Only complete REFIT microwave target events may be "
        "shifted by the deterministic rule recorded in "
        "`microwave_alignment_audit.json`. Aggregate power, UK-DALE, all other "
        "appliances, houses, and date ranges are unchanged. Report the original "
        "REFIT benchmark results alongside this protocol.\n",
        encoding="utf-8",
    )
    print(f"Saved aligned dataset: {output}")


if __name__ == "__main__":
    main()

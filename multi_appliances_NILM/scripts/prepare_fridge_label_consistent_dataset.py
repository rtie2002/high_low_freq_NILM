#!/usr/bin/env python
"""Create a house-split dataset with consistent REFIT-11 fridge supervision.

The existing five-week training split contains a legacy REFIT house 11 fridge
label: 98k samples are marked ON although power is below the experiment's
50 W ON definition (median 35 W). Validation and test houses do not show this
shift. This script copies the established split, recomputes only that training
house's fridge state with the current 50 W / duration rule, and moves power
from newly-OFF samples into the unmodelled residual by setting the supervised
fridge target to zero. Aggregate measurements and every evaluation row remain
unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .prepare_mixed_ukdale_refit_3week_split import APPS_5, training_normalization
except ImportError:  # Direct execution: python scripts/prepare_...py
    from prepare_mixed_ukdale_refit_3week_split import APPS_5, training_normalization


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "datasets" / "mixed_ukdale_refit_5w_house_split"
DEFAULT_OUTPUT = ROOT / "datasets" / "mixed_ukdale_refit_5w_house_split_fridge_consistent"


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    changes = np.diff(padded)
    return np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)


def _label_one_segment(
    power: np.ndarray,
    *,
    threshold_watts: float,
    min_off_samples: int,
    min_on_samples: int,
) -> np.ndarray:
    state = (np.asarray(power) >= float(threshold_watts)).astype(np.int8)
    starts, ends = _runs(state)
    for index in range(len(starts) - 1):
        gap = int(starts[index + 1] - ends[index])
        if 0 < gap <= int(min_off_samples):
            state[ends[index]:starts[index + 1]] = 1

    starts, ends = _runs(state)
    for start, end in zip(starts, ends):
        if 0 < end - start < int(min_on_samples):
            state[start:end] = 0
    return state


def relabel_refit11_fridge(
    frame: pd.DataFrame,
    *,
    threshold_watts: float = 50.0,
    min_off_samples: int = 2,
    min_on_samples: int = 8,
) -> dict[str, int | float]:
    required = {
        "dataset", "house", "sequence_id", "fridge_power", "fridge_on"
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")

    selected = frame["dataset"].astype(str).str.lower().eq("refit") & frame["house"].eq(11)
    selected_index = np.flatnonzero(selected.to_numpy())
    if selected_index.size == 0:
        raise ValueError("No REFIT house 11 rows found in training CSV")

    old_state = frame.loc[selected, "fridge_on"].to_numpy(np.int8)
    power = frame.loc[selected, "fridge_power"].to_numpy(np.float64)
    sequence = frame.loc[selected, "sequence_id"].to_numpy(np.int64)
    new_state = np.zeros_like(old_state)
    boundaries = np.flatnonzero(np.r_[True, sequence[1:] != sequence[:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        new_state[start:end] = _label_one_segment(
            power[start:end],
            threshold_watts=threshold_watts,
            min_off_samples=min_off_samples,
            min_on_samples=min_on_samples,
        )

    newly_off = (old_state == 1) & (new_state == 0)
    newly_on = (old_state == 0) & (new_state == 1)
    frame.loc[selected, "fridge_on"] = new_state
    cleaned_power = power.copy()
    cleaned_power[new_state == 0] = 0.0
    frame.loc[selected, "fridge_power"] = cleaned_power
    return {
        "selected_rows": int(selected_index.size),
        "old_on_samples": int(old_state.sum()),
        "new_on_samples": int(new_state.sum()),
        "newly_off_samples": int(newly_off.sum()),
        "newly_on_samples": int(newly_on.sum()),
        "removed_target_watt_samples": float(power[newly_off].sum()),
        "threshold_watts": float(threshold_watts),
        "min_off_samples": int(min_off_samples),
        "min_on_samples": int(min_on_samples),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not source.is_dir():
        raise FileNotFoundError(source)

    shutil.copytree(source, output)
    training_path = output / "training" / "multi_appliance_training.csv"
    training = pd.read_csv(training_path)
    audit = relabel_refit11_fridge(training)
    training.to_csv(training_path, index=False)

    stats = training_normalization(training, list(APPS_5))
    (output / "normalization_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    audit.update({
        "source_dataset": str(source),
        "output_dataset": str(output),
        "scope": "training REFIT house 11 fridge only",
        "aggregate_changed": False,
        "validation_or_test_changed": False,
    })
    (output / "fridge_label_consistency_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

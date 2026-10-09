"""Audit aggregate/microwave edge alignment in each dataset split.

The REFIT CSV timestamp is shared by all exported columns, but the physical
aggregate sensor and individual appliance monitors are not synchronized.  This
script measures the *observed* edge offset after preprocessing; it does not
claim to recover the unavailable per-sensor acquisition timestamps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LAGS = np.arange(-3, 4, dtype=np.int64)
USECOLS = ["dataset", "house", "sequence_id", "aggregate", "microwave_power"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--threshold-watts",
        type=float,
        default=200.0,
        help="Microwave power threshold used to identify physical starts.",
    )
    parser.add_argument(
        "--minimum-jump-watts",
        type=float,
        default=100.0,
        help="Minimum microwave rise at a retained start.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def split_files(root: Path) -> list[tuple[str, Path]]:
    return [
        ("train", root / "training" / "multi_appliance_training.csv"),
        ("validation", root / "validating" / "multi_appliance_validating.csv"),
        (
            "test",
            root / "testing" / "refit_house20" / "multi_appliance_testing.csv",
        ),
        (
            "test",
            root / "testing" / "ukdale_house2" / "multi_appliance_testing.csv",
        ),
    ]


def valid_starts(frame: pd.DataFrame, threshold: float, minimum_jump: float) -> np.ndarray:
    power = frame["microwave_power"].to_numpy(dtype=np.float64)
    sequence = frame["sequence_id"].to_numpy()
    rise = np.diff(power, prepend=power[0])
    starts = np.flatnonzero(
        (power >= threshold)
        & (np.roll(power, 1) < threshold)
        & (rise >= minimum_jump)
        & (sequence == np.roll(sequence, 1))
    )

    # Every tested lag needs a valid preceding point for its aggregate delta.
    starts = starts[(starts >= 4) & (starts < len(frame) - 3)]
    if not len(starts):
        return starts
    complete_context = np.asarray(
        [np.all(sequence[start - 4 : start + 4] == sequence[start]) for start in starts]
    )
    return starts[complete_context]


def summarize_group(
    split: str,
    dataset: str,
    house: int,
    frame: pd.DataFrame,
    threshold: float,
    minimum_jump: float,
) -> dict[str, object]:
    aggregate = frame["aggregate"].to_numpy(dtype=np.float64)
    microwave = frame["microwave_power"].to_numpy(dtype=np.float64)
    aggregate_delta = np.diff(aggregate, prepend=aggregate[0])
    starts = valid_starts(frame, threshold, minimum_jump)

    row: dict[str, object] = {
        "split": split,
        "dataset": dataset,
        "house": int(house),
        "events": int(len(starts)),
    }
    if not len(starts):
        return row

    edge_matrix = np.stack([aggregate_delta[starts + lag] for lag in LAGS], axis=1)
    best_lags = LAGS[np.argmax(edge_matrix, axis=1)]
    for lag in LAGS:
        row[f"best_lag_{lag:+d}"] = int(np.sum(best_lags == lag))
        row[f"median_edge_{lag:+d}_w"] = float(
            np.median(edge_matrix[:, int(lag - LAGS[0])])
        )

    row["dominant_lag"] = int(
        LAGS[np.argmax([row[f"median_edge_{lag:+d}_w"] for lag in LAGS])]
    )
    for offset in range(3):
        # A submeter cannot physically exceed the simultaneous whole-house
        # aggregate. Violations reveal timestamp/measurement inconsistency.
        impossible = aggregate[starts + offset] + 5.0 < microwave[starts + offset]
        row[f"aggregate_below_microwave_t+{offset}_pct"] = float(impossible.mean() * 100.0)
    return row


def audit_file(
    split: str,
    path: Path,
    threshold: float,
    minimum_jump: float,
) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=USECOLS)
    rows = []
    for (dataset, house), group in frame.groupby(["dataset", "house"], sort=True):
        rows.append(
            summarize_group(
                split,
                str(dataset),
                int(house),
                group.reset_index(drop=True),
                threshold,
                minimum_jump,
            )
        )
    return rows


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for split, path in split_files(args.dataset_root):
        rows.extend(
            audit_file(split, path, args.threshold_watts, args.minimum_jump_watts)
        )

    result = pd.DataFrame(rows).sort_values(["split", "dataset", "house"])
    display_columns = [
        "split",
        "dataset",
        "house",
        "events",
        "dominant_lag",
        "best_lag_+0",
        "best_lag_+1",
        "best_lag_+2",
        "aggregate_below_microwave_t+0_pct",
        "aggregate_below_microwave_t+1_pct",
        "aggregate_below_microwave_t+2_pct",
    ]
    print(result.reindex(columns=display_columns).to_string(index=False, float_format="%.1f"))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, float_format="%.3f")
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

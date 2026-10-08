#!/usr/bin/env python
"""Copy a prepared dataset while excluding one house from training only.

This is a data-quality ablation, not a new split protocol: validation and test
files are copied unchanged, and normalization is refitted on the remaining
training rows.  It is useful when one training submeter uses an incompatible
label or calibration convention that cannot be repaired without guessing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

try:
    from .prepare_mixed_ukdale_refit_3week_split import APPS_5, training_normalization
except ImportError:
    from prepare_mixed_ukdale_refit_3week_split import APPS_5, training_normalization


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "datasets" / "mixed_ukdale_refit_5w_house_split"
DEFAULT_OUTPUT = ROOT / "datasets" / "mixed_ukdale_refit_5w_house_split_no_refit11"


def exclude_training_house(
    frame: pd.DataFrame,
    *,
    dataset: str,
    house: int,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    required = {"dataset", "house"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")

    selected = (
        frame["dataset"].astype(str).str.lower().eq(str(dataset).lower())
        & frame["house"].eq(int(house))
    )
    removed = int(selected.sum())
    if removed == 0:
        raise ValueError(f"No {dataset} house {house} rows found in training CSV")
    kept = frame.loc[~selected].reset_index(drop=True)
    return kept, {
        "excluded_dataset": str(dataset).lower(),
        "excluded_house": int(house),
        "removed_training_rows": removed,
        "remaining_training_rows": int(len(kept)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default="refit")
    parser.add_argument("--house", type=int, default=11)
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
    training, audit = exclude_training_house(
        training,
        dataset=args.dataset,
        house=args.house,
    )
    training.to_csv(training_path, index=False)

    stats = training_normalization(training, list(APPS_5))
    (output / "normalization_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    audit.update({
        "source_dataset": str(source),
        "output_dataset": str(output),
        "validation_or_test_changed": False,
        "normalization_refitted_on_remaining_training_rows": True,
    })
    (output / "training_house_exclusion_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

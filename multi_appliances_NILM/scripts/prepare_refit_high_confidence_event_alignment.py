"""Create a controlled REFIT timing-correction ablation.

The source house split, timestamps, aggregate input, UK-DALE rows, and split
membership are preserved exactly.  Only high-confidence REFIT appliance events
are shifted by ``realign_high_confidence_events``.  This makes the experiment a
timing-label ablation instead of silently changing the selected calendar data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "multi_appliances_NILM"
sys.path.insert(0, str(ROOT / "dataset_preprocess"))
sys.path.insert(0, str(PROJECT / "scripts"))

from refit_processing_multi_appliance import (  # noqa: E402
    realign_high_confidence_events,
)
from prepare_mixed_ukdale_refit_3week_split import training_normalization  # noqa: E402


APPLIANCES = ["kettle", "fridge", "dishwasher", "washingmachine", "microwave"]
DEFAULT_SOURCE = PROJECT / "datasets" / "mixed_ukdale_refit_5w_house_split"
DEFAULT_OUTPUT = PROJECT / "datasets" / "mixed_ukdale_refit_5w_house_split_event_aligned"
DEFAULT_CONFIG = ROOT / "config" / "preprocess" / "refit.yaml"
CSV_FILES = [
    Path("training/multi_appliance_training.csv"),
    Path("validating/multi_appliance_validating.csv"),
    Path("testing/multi_appliance_testing.csv"),
    Path("testing/refit_house20/multi_appliance_testing.csv"),
    Path("testing/ukdale_house2/multi_appliance_testing.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-lag-samples", type=int, default=2)
    return parser.parse_args()


def align_csv(
    source: Path,
    output: Path,
    config: dict,
    max_lag_samples: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame = pd.read_csv(source)
    original_columns = list(frame.columns)
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    summaries: list[dict[str, object]] = []
    events: list[dict[str, object]] = []

    if "dataset" not in frame.columns or not frame["dataset"].eq("refit").any():
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.drop(columns="_row_order").to_csv(output, index=False)
        return summaries, events

    pieces: list[pd.DataFrame] = [frame.loc[~frame["dataset"].eq("refit")].copy()]
    for house_value, house_frame in frame.loc[frame["dataset"].eq("refit")].groupby(
        "house", sort=False
    ):
        house = int(house_value)
        original_labels = {
            app: house_frame[f"{app}_on"].to_numpy(copy=True) for app in APPLIANCES
        }
        indexed = house_frame.copy()
        indexed.index = pd.to_datetime(indexed["readable_time"], utc=True)
        corrected, house_summaries, house_events = realign_high_confidence_events(
            indexed,
            APPLIANCES,
            config,
            house,
            max_lag_samples=max_lag_samples,
            apply=True,
        )
        for app, old_labels in original_labels.items():
            changed = int(np.sum(corrected[f"{app}_on"].to_numpy() != old_labels))
            for summary in house_summaries:
                if summary["appliance"] == app:
                    summary["state_rows_changed"] = changed
                    summary["file"] = str(source.relative_to(source.parents[2]))
                    break
        pieces.append(corrected)
        summaries.extend(house_summaries)
        events.extend(house_events)

    result = (
        pd.concat(pieces, ignore_index=True)
        .sort_values("_row_order", kind="stable")
        .drop(columns="_row_order")
    )
    if list(result.columns) != original_columns:
        result = result[original_columns]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return summaries, events


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if args.max_lag_samples < 0:
        raise ValueError("--max-lag-samples must be non-negative")

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    shutil.copytree(source, output)

    all_summaries: list[dict[str, object]] = []
    all_events: list[dict[str, object]] = []
    for relative in CSV_FILES:
        summaries, events = align_csv(
            source / relative,
            output / relative,
            config,
            args.max_lag_samples,
        )
        all_summaries.extend(summaries)
        all_events.extend(events)
        print(f"aligned {relative}: accepted={len(events):,}", flush=True)

    training = pd.read_csv(output / CSV_FILES[0])
    normalization = training_normalization(training, APPLIANCES)
    (output / "normalization_stats.json").write_text(
        json.dumps(normalization, indent=2), encoding="utf-8"
    )
    pd.DataFrame(all_summaries).to_csv(output / "alignment_summary.csv", index=False)
    pd.DataFrame(all_events).to_csv(output / "alignment_events.csv", index=False)
    (output / "alignment_meta.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "max_lag_samples": int(args.max_lag_samples),
                "aggregate_modified": False,
                "rule": "complete event; matching ON/OFF aggregate edges; ambiguous events rejected",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved controlled alignment dataset: {output}")


if __name__ == "__main__":
    main()

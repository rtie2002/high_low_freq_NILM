#!/usr/bin/env python
"""Build a duration-controlled, house-held-out UK-DALE/REFIT dataset.

Protocol
--------
Training (a contiguous block centred on the established active 5-week block):
  UK-DALE 1; REFIT 3, 5, 9, 11

Validation (complete previously selected 8-week blocks):
  UK-DALE 5; REFIT 2

Final test (complete previously selected 8-week blocks):
  UK-DALE 2; REFIT 20

The established 5-week selection is the training anchor: a longer duration
contains those exact rows and extends it as evenly as possible on both sides
using the current corrected original 8-second house CSVs. This keeps the
active events and calendar period used by the baseline instead of selecting
an unrelated season. Validation and test rows are copied exactly from the
5-week reference dataset. Normalization statistics are fitted on the extended
five-house training split only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_mixed_ukdale_refit_3week_split import (
    APPS_5,
    REFIT_DIR,
    SAMPLE_SECONDS,
    TIME_COL,
    UKDALE_DIR,
    count_events,
    frame_sequence_ids,
    resolve_refit,
    resolve_ukdale,
    summarize_split,
    training_normalization,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "datasets" / "mixed_ukdale_refit_5w_house_split"
DEFAULT_EVAL_SELECTION = (
    ROOT / "datasets" / "mixed_ukdale_refit_8w" / "selection_summary.csv"
)
DEFAULT_TRAIN_ANCHOR = DEFAULT_OUT_DIR / "selection_summary.csv"
DEFAULT_REFERENCE_DATASET = DEFAULT_OUT_DIR

TRAIN_HOUSES = (("ukdale", 1), ("refit", 3), ("refit", 5), ("refit", 9), ("refit", 11))
VALIDATION_HOUSES = (("ukdale", 5), ("refit", 2))
TEST_HOUSES = (("ukdale", 2), ("refit", 20))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a duration-controlled split with whole-house validation."
    )
    parser.add_argument("--ukdale-dir", type=Path, default=UKDALE_DIR)
    parser.add_argument("--refit-dir", type=Path, default=REFIT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--eval-selection-summary",
        type=Path,
        default=DEFAULT_EVAL_SELECTION,
        help="8-week selection summary whose time bounds are reused for validation/test.",
    )
    parser.add_argument(
        "--train-anchor-summary",
        type=Path,
        default=DEFAULT_TRAIN_ANCHOR,
        help="Selection summary containing the established 5-week training blocks.",
    )
    parser.add_argument(
        "--reference-dataset-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DATASET,
        help=(
            "Existing 5-week dataset whose training core and held-out splits "
            "must remain unchanged."
        ),
    )
    parser.add_argument("--train-weeks", type=float, default=5.0)
    return parser.parse_args()


def _source_path(dataset: str, house: int, ukdale_dir: Path, refit_dir: Path) -> Path:
    if dataset == "ukdale":
        return resolve_ukdale(house, ukdale_dir)
    if dataset == "refit":
        return resolve_refit(house, refit_dir)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _load_slice_chunked(
    csv_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Read only one selected time interval from a potentially large house CSV."""
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        timestamps = (
            pd.to_datetime(chunk["unix_time"], unit="s", errors="coerce", utc=True)
            if "unix_time" in chunk.columns
            else pd.to_datetime(chunk[TIME_COL], errors="coerce", utc=True)
        )
        keep = timestamps.notna() & (timestamps >= start) & (timestamps <= end)
        if not bool(keep.any()):
            continue
        part = chunk.loc[keep].copy()
        part[TIME_COL] = timestamps.loc[keep]
        selected.append(part)
    if not selected:
        raise RuntimeError(f"No rows found in {csv_path} for {start} -> {end}")
    return (
        pd.concat(selected, ignore_index=True)
        .sort_values(TIME_COL, kind="stable")
        .reset_index(drop=True)
    )


def _tag_dataset(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    frame = frame.copy()
    if "dataset" in frame.columns:
        frame["dataset"] = dataset
    else:
        frame.insert(1, "dataset", dataset)
    return frame


def _inventory_row(
    frame: pd.DataFrame,
    *,
    dataset: str,
    house: int,
    role: str,
    source_csv: Path,
    requested_weeks: float,
    selection_note: str,
) -> dict[str, object]:
    apps = list(APPS_5)
    segments = frame_sequence_ids(frame)
    powers = frame[[f"{app}_power" for app in apps]].sum(axis=1).to_numpy(float)
    residual = np.clip(frame["aggregate"].to_numpy(float) - powers, 0.0, None)
    row: dict[str, object] = {
        "dataset": dataset,
        "house": house,
        "role": role,
        "source_csv": str(source_csv.resolve()),
        "start": str(frame[TIME_COL].iloc[0]),
        "end": str(frame[TIME_COL].iloc[-1]),
        "requested_weeks": requested_weeks,
        "n_rows": len(frame),
        "effective_days": len(frame) * SAMPLE_SECONDS / 86_400.0,
        "mean_residual_w": float(np.mean(residual)),
        "median_residual_w": float(np.median(residual)),
        "residual_0_100_pct": 100.0 * float(np.mean((residual >= 0) & (residual < 100))),
        "residual_100_200_pct": 100.0 * float(np.mean((residual >= 100) & (residual < 200))),
        "residual_200_400_pct": 100.0 * float(np.mean((residual >= 200) & (residual < 400))),
        "residual_400_800_pct": 100.0 * float(np.mean((residual >= 400) & (residual < 800))),
        "residual_800_inf_pct": 100.0 * float(np.mean(residual >= 800)),
        "selection_note": selection_note,
    }
    for app in apps:
        state = pd.to_numeric(frame[f"{app}_on"], errors="coerce").fillna(0).gt(0).to_numpy(np.int8)
        row[f"events_{app}"] = count_events(state, segments)
        row[f"on_hours_{app}"] = float(state.sum() * SAMPLE_SECONDS / 3600.0)
    return row


def _established_bounds(
    summary_path: Path,
) -> dict[tuple[str, int], tuple[pd.Timestamp, pd.Timestamp]]:
    summary = pd.read_csv(summary_path)
    required = {"dataset", "house", "start", "end"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"{summary_path} is missing columns: {sorted(missing)}")
    bounds: dict[tuple[str, int], tuple[pd.Timestamp, pd.Timestamp]] = {}
    for row in summary.itertuples(index=False):
        key = (str(row.dataset).lower(), int(row.house))
        bounds[key] = (pd.Timestamp(row.start), pd.Timestamp(row.end))
    needed = set(TRAIN_HOUSES) | set(VALIDATION_HOUSES) | set(TEST_HOUSES)
    absent = needed.difference(bounds)
    if absent:
        raise ValueError(f"8-week selection summary lacks houses: {sorted(absent)}")
    return bounds


def _training_anchor_bounds(
    summary_path: Path,
) -> dict[tuple[str, int], tuple[pd.Timestamp, pd.Timestamp]]:
    """Read the fixed active training blocks used by the 5-week baseline."""
    summary = pd.read_csv(summary_path)
    training = summary.loc[summary["role"].astype(str).str.lower().eq("train")]
    bounds = {
        (str(row.dataset).lower(), int(row.house)): (
            pd.Timestamp(row.start),
            pd.Timestamp(row.end),
        )
        for row in training.itertuples(index=False)
    }
    absent = set(TRAIN_HOUSES).difference(bounds)
    if absent:
        raise ValueError(f"Training anchor summary lacks houses: {sorted(absent)}")
    return bounds


def _expand_anchor(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    target_weeks: float,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Expand an inclusive 8-second block while preserving the full anchor."""
    target_samples = int(round(target_weeks * 7 * 86_400 / SAMPLE_SECONDS))
    anchor_samples = int(round((end - start).total_seconds() / SAMPLE_SECONDS)) + 1
    if target_samples < anchor_samples:
        raise ValueError(
            "--train-weeks cannot be shorter than the established anchor; "
            "use the original block-search procedure for shorter ablations"
        )
    extra_samples = target_samples - anchor_samples
    before_samples = (extra_samples + 1) // 2
    after_samples = extra_samples - before_samples
    step = pd.Timedelta(seconds=SAMPLE_SECONDS)
    return start - before_samples * step, end + after_samples * step


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"  wrote {path} ({len(frame):,} rows)", flush=True)


def _house_frame(frame: pd.DataFrame, dataset: str, house: int) -> pd.DataFrame:
    """Select one house without relying on house numbers being globally unique."""
    keep = (
        frame["dataset"].astype(str).str.lower().eq(dataset)
        & pd.to_numeric(frame["house"], errors="coerce").eq(house)
    )
    selected = frame.loc[keep].copy()
    if selected.empty:
        raise RuntimeError(f"Reference split lacks {dataset} house {house}")
    selected[TIME_COL] = pd.to_datetime(selected[TIME_COL], errors="coerce", utc=True)
    return selected


def main() -> None:
    args = parse_args()
    train_set = set(TRAIN_HOUSES)
    val_set = set(VALIDATION_HOUSES)
    test_set = set(TEST_HOUSES)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Train, validation and test houses must be disjoint")

    print("=== Duration-controlled training / whole-house validation split ===", flush=True)
    print(f"  train={TRAIN_HOUSES}", flush=True)
    print(f"  validation={VALIDATION_HOUSES}", flush=True)
    print(f"  test={TEST_HOUSES}", flush=True)
    print(
        f"  training blocks={args.train_weeks:g} weeks; "
        "validation/test reuse established 8-week bounds",
        flush=True,
    )

    established_bounds = _established_bounds(args.eval_selection_summary)
    training_anchors = _training_anchor_bounds(args.train_anchor_summary)
    reference_train = pd.read_csv(
        args.reference_dataset_dir / "training" / "multi_appliance_training.csv"
    )
    reference_validation = pd.read_csv(
        args.reference_dataset_dir / "validating" / "multi_appliance_validating.csv"
    )
    reference_test = pd.read_csv(
        args.reference_dataset_dir / "testing" / "multi_appliance_testing.csv"
    )
    split_parts: dict[str, list[pd.DataFrame]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    inventory: list[dict[str, object]] = []

    jobs = [
        *(dataset_house + ("train",) for dataset_house in TRAIN_HOUSES),
        *(dataset_house + ("validation",) for dataset_house in VALIDATION_HOUSES),
        *(dataset_house + ("test",) for dataset_house in TEST_HOUSES),
    ]
    for dataset, house, role in jobs:
        csv_path = _source_path(dataset, house, args.ukdale_dir, args.refit_dir)
        print(f"\n[{role} | {dataset} house {house}] {csv_path}", flush=True)
        if role == "train":
            anchor_start, anchor_end = training_anchors[(dataset, house)]
            start, end = _expand_anchor(
                anchor_start,
                anchor_end,
                target_weeks=float(args.train_weeks),
            )
            print(
                "  fixed active anchor: "
                f"{anchor_start} -> {anchor_end}",
                flush=True,
            )
            expanded = _tag_dataset(_load_slice_chunked(csv_path, start, end), dataset)
            anchor = _house_frame(reference_train, dataset, house)
            outside_anchor = expanded.loc[
                expanded[TIME_COL].lt(anchor_start) | expanded[TIME_COL].gt(anchor_end)
            ]
            frame = (
                pd.concat([outside_anchor, anchor], ignore_index=True)
                .sort_values(TIME_COL, kind="stable")
                .reset_index(drop=True)
            )
            selection_note = (
                f"{args.train_weeks:g}w_extension_preserving_exact_5w_core"
            )
            requested_weeks = float(args.train_weeks)
        else:
            start, end = established_bounds[(dataset, house)]
            reference = reference_validation if role == "validation" else reference_test
            frame = _house_frame(reference, dataset, house)
            selection_note = "exact_held_out_rows_from_5w_baseline"
            requested_weeks = 8.0

        print(f"  selected rows: {frame[TIME_COL].iloc[0]} -> {frame[TIME_COL].iloc[-1]}", flush=True)
        print(" ", summarize_split(role.upper(), frame, list(APPS_5)), flush=True)
        split_parts[role].append(frame)
        inventory.append(
            _inventory_row(
                frame,
                dataset=dataset,
                house=house,
                role=role,
                source_csv=csv_path,
                requested_weeks=requested_weeks,
                selection_note=selection_note,
            )
        )

    train_df = pd.concat(split_parts["train"], ignore_index=True)
    validation_df = pd.concat(split_parts["validation"], ignore_index=True)
    test_df = pd.concat(split_parts["test"], ignore_index=True)

    out_dir = args.out_dir
    _write_csv(train_df, out_dir / "training" / "multi_appliance_training.csv")
    _write_csv(validation_df, out_dir / "validating" / "multi_appliance_validating.csv")
    _write_csv(test_df, out_dir / "testing" / "multi_appliance_testing.csv")
    for (dataset, house), scenario in test_df.groupby(["dataset", "house"], sort=False):
        _write_csv(
            scenario.reset_index(drop=True),
            out_dir
            / "testing"
            / f"{str(dataset).lower()}_house{int(house)}"
            / "multi_appliance_testing.csv",
        )

    normalization_path = out_dir / "normalization_stats.json"
    normalization_path.write_text(
        json.dumps(training_normalization(train_df, list(APPS_5)), indent=2),
        encoding="utf-8",
    )
    inventory_path = out_dir / "selection_summary.csv"
    pd.DataFrame(inventory).to_csv(inventory_path, index=False)
    meta = {
        "protocol": "duration-controlled training blocks with whole-house validation",
        "sample_seconds": SAMPLE_SECONDS,
        "train_weeks": float(args.train_weeks),
        "validation_test_weeks": 8.0,
        "train_houses": [list(item) for item in TRAIN_HOUSES],
        "validation_houses": [list(item) for item in VALIDATION_HOUSES],
        "test_houses": [list(item) for item in TEST_HOUSES],
        "established_8week_bounds_source": str(args.eval_selection_summary.resolve()),
        "training_anchor_source": str(args.train_anchor_summary.resolve()),
        "normalization_source": "training split only",
    }
    (out_dir / "selection_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("\n=== Final split ===", flush=True)
    print(summarize_split("TRAIN", train_df, list(APPS_5)), flush=True)
    print(summarize_split("VALIDATION", validation_df, list(APPS_5)), flush=True)
    print(summarize_split("TEST", test_df, list(APPS_5)), flush=True)
    print(f"  wrote {normalization_path}", flush=True)
    print(f"  wrote {inventory_path}", flush=True)


if __name__ == "__main__":
    main()

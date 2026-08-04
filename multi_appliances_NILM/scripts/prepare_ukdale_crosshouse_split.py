#!/usr/bin/env python
"""Prepare a cross-house UK-DALE split from full-house CSV files.

Default (expanded source data):
    - House 1 / 5: 10-week block -> first 8 weeks train, last 2 weeks validation
    - House 2:     4-week block -> all test  (unchanged for fair comparison)

Previous default was 4 weeks/house (3 train + 1 val). Override with --source-weeks.

Source CSVs (under datasets/ukdale/):
    ukdale_house1_lf_6s.csv
    ukdale_house2_lf_6s.csv
    ukdale_house5_lf_6s.csv
(Legacy alias still accepted: multi_appliance_FULL_house{N}.csv)

Outputs:
    datasets/ukdale/training/multi_appliance_training.csv
    datasets/ukdale/validating/multi_appliance_validating.csv
    datasets/ukdale/testing/multi_appliance_testing.csv

Example:
    python scripts/prepare_ukdale_crosshouse_split.py
    python scripts/prepare_ukdale_crosshouse_split.py --source-weeks 20 --val-weeks 4
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
UKDALE_DIR = ROOT / "datasets" / "ukdale"

TRAIN_OUT = UKDALE_DIR / "training" / "multi_appliance_training.csv"
VAL_OUT = UKDALE_DIR / "validating" / "multi_appliance_validating.csv"
TEST_OUT = UKDALE_DIR / "testing" / "multi_appliance_testing.csv"

TIME_COL = "readable_time"

# End dates kept from the original high-ON-activity weekly analysis;
# block start = end − (source_weeks − 1 day) so the active weeks stay near the end.
HOUSE1_BLOCK_END = "2014-10-20 23:59:59"
HOUSE5_BLOCK_END = "2014-08-04 23:59:59"

# Test house fixed (do not enlarge when expanding source — keeps old runs comparable).
HOUSE2_TEST_START = "2013-07-09 00:00:00"
HOUSE2_TEST_END = "2013-08-05 23:59:59"


def resolve_house_csv(ukdale: Path, house: int) -> Path:
    """Prefer ukdale_house{N}_lf_6s.csv; accept legacy FULL / lf names."""
    candidates = [
        ukdale / f"ukdale_house{house}_lf_6s.csv",
        ukdale / f"multi_appliance_FULL_house{house}.csv",
        ukdale / f"multi_appliance_house{house}_lf.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing house {house} CSV under {ukdale}. Tried:\n  - "
        + "\n  - ".join(str(p) for p in candidates)
    )


def _parse_end(end: str) -> datetime:
    return datetime.strptime(end, "%Y-%m-%d %H:%M:%S")


def _block_start_from_end(end: str, weeks: int) -> str:
    """Inclusive start so [start, end] spans ``weeks`` calendar weeks (~7*weeks days)."""
    end_dt = _parse_end(end)
    start_dt = (end_dt - timedelta(days=7 * weeks - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return start_dt.strftime("%Y-%m-%d %H:%M:%S")


def _load_slice(csv_path: Path, start: str, end: str) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}.")
    df = pd.read_csv(csv_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    out = df[(df[TIME_COL] >= start) & (df[TIME_COL] <= end)].copy()
    return out.sort_values(TIME_COL).reset_index(drop=True)


def _split_train_val_by_weeks(
    df: pd.DataFrame,
    *,
    source_weeks: int,
    val_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last ``val_weeks`` of the block -> val; earlier part -> train."""
    if df.empty:
        raise ValueError("Selected block is empty.")
    if val_weeks <= 0 or val_weeks >= source_weeks:
        raise ValueError(
            f"Need 0 < val_weeks < source_weeks, got val={val_weeks}, source={source_weeks}"
        )
    # Time-based cut: keep last (val_weeks / source_weeks) of the time span.
    t0 = df[TIME_COL].iloc[0]
    t1 = df[TIME_COL].iloc[-1]
    span = t1 - t0
    if span <= pd.Timedelta(0):
        raise ValueError("Block has zero time span.")
    val_frac = float(val_weeks) / float(source_weeks)
    cut_time = t1 - span * val_frac
    train = df[df[TIME_COL] < cut_time].copy()
    val = df[df[TIME_COL] >= cut_time].copy()
    if train.empty or val.empty:
        # Fallback: row fraction (same ratios) if timestamps are sparse/gaps.
        split_idx = max(1, min(len(df) - 1, int(len(df) * (1.0 - val_frac))))
        train = df.iloc[:split_idx].copy()
        val = df.iloc[split_idx:].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build UK-DALE cross-house train/val/test CSVs.")
    p.add_argument(
        "--source-weeks",
        type=int,
        default=10,
        help="Weeks per source house (H1 and H5) before train/val split (default: 10).",
    )
    p.add_argument(
        "--val-weeks",
        type=int,
        default=2,
        help="Weeks reserved for validation at the end of each source block (default: 2).",
    )
    p.add_argument(
        "--ukdale-dir",
        type=Path,
        default=UKDALE_DIR,
        help="Directory with ukdale_house*_lf_6s.csv and train/val/test folders.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ukdale = args.ukdale_dir
    house1_csv = resolve_house_csv(ukdale, 1)
    house2_csv = resolve_house_csv(ukdale, 2)
    house5_csv = resolve_house_csv(ukdale, 5)
    train_out = ukdale / "training" / "multi_appliance_training.csv"
    val_out = ukdale / "validating" / "multi_appliance_validating.csv"
    test_out = ukdale / "testing" / "multi_appliance_testing.csv"

    source_weeks = int(args.source_weeks)
    val_weeks = int(args.val_weeks)
    train_weeks = source_weeks - val_weeks

    h1_start = _block_start_from_end(HOUSE1_BLOCK_END, source_weeks)
    h5_start = _block_start_from_end(HOUSE5_BLOCK_END, source_weeks)

    print(
        f"Building split: H1+H5 = {source_weeks} wk/house "
        f"({train_weeks} train + {val_weeks} val); H2 test = 4 wk (fixed).",
        flush=True,
    )
    print(f"  house1 CSV: {house1_csv.name}", flush=True)
    print(f"  house2 CSV: {house2_csv.name}", flush=True)
    print(f"  house5 CSV: {house5_csv.name}", flush=True)

    train_h1_full = _load_slice(house1_csv, h1_start, HOUSE1_BLOCK_END)
    train_h5_full = _load_slice(house5_csv, h5_start, HOUSE5_BLOCK_END)
    test_h2 = _load_slice(house2_csv, HOUSE2_TEST_START, HOUSE2_TEST_END)

    for name, block, start, end in (
        ("house1", train_h1_full, h1_start, HOUSE1_BLOCK_END),
        ("house5", train_h5_full, h5_start, HOUSE5_BLOCK_END),
        ("house2", test_h2, HOUSE2_TEST_START, HOUSE2_TEST_END),
    ):
        if block.empty:
            raise ValueError(
                f"{name} slice empty for {start} -> {end}. Check CSV date coverage."
            )
        print(
            f"  {name}: rows={len(block):,}  "
            f"time={block[TIME_COL].iloc[0]} -> {block[TIME_COL].iloc[-1]}",
            flush=True,
        )

    train_h1, val_h1 = _split_train_val_by_weeks(
        train_h1_full, source_weeks=source_weeks, val_weeks=val_weeks
    )
    train_h5, val_h5 = _split_train_val_by_weeks(
        train_h5_full, source_weeks=source_weeks, val_weeks=val_weeks
    )

    train_df = (
        pd.concat([train_h1, train_h5], ignore_index=True)
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat([val_h1, val_h5], ignore_index=True)
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )
    test_df = test_h2

    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    print("Wrote:", flush=True)
    print(f"  train: {train_out}  rows={len(train_df):,}", flush=True)
    print(f"  val:   {val_out}  rows={len(val_df):,}", flush=True)
    print(f"  test:  {test_out}  rows={len(test_df):,}", flush=True)
    print(
        f"  per source house: ~{train_weeks} wk train + ~{val_weeks} wk val "
        f"(H2 test unchanged at 4 wk).",
        flush=True,
    )


if __name__ == "__main__":
    main()

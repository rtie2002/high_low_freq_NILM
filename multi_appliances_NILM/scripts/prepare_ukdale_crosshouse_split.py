#!/usr/bin/env python
"""Prepare a cross-house UK-DALE split from full-house CSV files.

Design:
    - House 1: selected 4-week block -> first 3 weeks train, last 1 week validation
    - House 5: selected 4-week block -> first 3 weeks train, last 1 week validation
    - House 2: selected 4-week block -> all 4 weeks test

Outputs:
    datasets/ukdale/training/multi_appliance_training.csv
    datasets/ukdale/validating/multi_appliance_validating.csv
    datasets/ukdale/testing/multi_appliance_testing.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
UKDALE_DIR = ROOT / "datasets" / "ukdale"

HOUSE1_CSV = UKDALE_DIR / "multi_appliance_FULL_house1.csv"
HOUSE2_CSV = UKDALE_DIR / "multi_appliance_FULL_house2.csv"
HOUSE5_CSV = UKDALE_DIR / "multi_appliance_FULL_house5.csv"

TRAIN_OUT = UKDALE_DIR / "training" / "multi_appliance_training.csv"
VAL_OUT = UKDALE_DIR / "validating" / "multi_appliance_validating.csv"
TEST_OUT = UKDALE_DIR / "testing" / "multi_appliance_testing.csv"

TIME_COL = "readable_time"

# Selected from weekly ON-activity analysis.
HOUSE1_BLOCK_START = "2014-09-23 00:00:00"
HOUSE1_BLOCK_END = "2014-10-20 23:59:59"

HOUSE5_BLOCK_START = "2014-07-08 00:00:00"
HOUSE5_BLOCK_END = "2014-08-04 23:59:59"

HOUSE2_TEST_START = "2013-07-09 00:00:00"
HOUSE2_TEST_END = "2013-08-05 23:59:59"


def _load_slice(csv_path: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    out = df[(df[TIME_COL] >= start) & (df[TIME_COL] <= end)].copy()
    return out.sort_values(TIME_COL).reset_index(drop=True)


def _split_3week_train_1week_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Selected block is empty.")
    split_idx = int(len(df) * 0.75)
    train = df.iloc[:split_idx].copy()
    val = df.iloc[split_idx:].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True)


def main() -> None:
    train_h1_full = _load_slice(HOUSE1_CSV, HOUSE1_BLOCK_START, HOUSE1_BLOCK_END)
    train_h5_full = _load_slice(HOUSE5_CSV, HOUSE5_BLOCK_START, HOUSE5_BLOCK_END)
    test_h2 = _load_slice(HOUSE2_CSV, HOUSE2_TEST_START, HOUSE2_TEST_END)

    train_h1, val_h1 = _split_3week_train_1week_val(train_h1_full)
    train_h5, val_h5 = _split_3week_train_1week_val(train_h5_full)

    train_df = pd.concat([train_h1, train_h5], ignore_index=True).sort_values(TIME_COL).reset_index(drop=True)
    val_df = pd.concat([val_h1, val_h5], ignore_index=True).sort_values(TIME_COL).reset_index(drop=True)
    test_df = test_h2.sort_values(TIME_COL).reset_index(drop=True)

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    VAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEST_OUT.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print("Saved cross-house UK-DALE split:")
    print(f"  train: {TRAIN_OUT}  rows={len(train_df):,}")
    print(f"  val:   {VAL_OUT}  rows={len(val_df):,}")
    print(f"  test:  {TEST_OUT}  rows={len(test_df):,}")
    print()
    print("Selected ranges:")
    print(f"  house1 block: {HOUSE1_BLOCK_START} -> {HOUSE1_BLOCK_END}")
    print(f"  house5 block: {HOUSE5_BLOCK_START} -> {HOUSE5_BLOCK_END}")
    print(f"  house2 test:  {HOUSE2_TEST_START} -> {HOUSE2_TEST_END}")


if __name__ == "__main__":
    main()

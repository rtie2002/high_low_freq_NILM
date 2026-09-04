#!/usr/bin/env python
"""Prepare UK-DALE cross-house data at 1-minute resolution.

Protocol:
  - train/validation source: house 1
  - test source: house 2
  - resolution: 1 minute

Inputs default to existing full-house CSVs:
  multi_appliances_NILM/datasets/ukdale/ukdale_house1_lf_6s.csv
  multi_appliances_NILM/datasets/ukdale/ukdale_house2_lf_6s.csv

Outputs:
  multi_appliances_NILM/datasets/ukdale_h1_h2_1min/training/multi_appliance_training.csv
  multi_appliances_NILM/datasets/ukdale_h1_h2_1min/validating/multi_appliance_validating.csv
  multi_appliances_NILM/datasets/ukdale_h1_h2_1min/testing/multi_appliance_testing.csv

Example:
  python multi_appliances_NILM/scripts/prepare_ukdale_h1_train_h2_test_1min.py

Use all house-1 rows for training and write an empty validation file:
  python multi_appliances_NILM/scripts/prepare_ukdale_h1_train_h2_test_1min.py --val-fraction 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
NILM_DIR = PROJECT_DIR / "multi_appliances_NILM"
DEFAULT_UKDALE_DIR = NILM_DIR / "datasets" / "ukdale"
DEFAULT_OUT_DIR = NILM_DIR / "datasets" / "ukdale_h1_h2_1min"

TIME_COL = "readable_time"
HOUSE_COL = "house"
STATE_SUFFIX = "_on"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 1-minute UK-DALE cross-house split: train H1, test H2."
    )
    parser.add_argument(
        "--ukdale-dir",
        type=Path,
        default=DEFAULT_UKDALE_DIR,
        help="Directory containing ukdale_house1_lf_6s.csv and ukdale_house2_lf_6s.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--house1-csv",
        type=Path,
        default=None,
        help="Optional explicit house 1 CSV path.",
    )
    parser.add_argument(
        "--house2-csv",
        type=Path,
        default=None,
        help="Optional explicit house 2 CSV path.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Last fraction of house 1 reserved for validation. Use 0 for no validation.",
    )
    parser.add_argument(
        "--start-house1",
        type=str,
        default=None,
        help="Optional inclusive start time for house 1, e.g. '2014-07-01 00:00:00'.",
    )
    parser.add_argument(
        "--end-house1",
        type=str,
        default=None,
        help="Optional inclusive end time for house 1.",
    )
    parser.add_argument(
        "--start-house2",
        type=str,
        default=None,
        help="Optional inclusive start time for house 2.",
    )
    parser.add_argument(
        "--end-house2",
        type=str,
        default=None,
        help="Optional inclusive end time for house 2.",
    )
    return parser.parse_args()


def resolve_house_csv(ukdale_dir: Path, house: int, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit
    else:
        path = ukdale_dir / f"ukdale_house{house}_lf_6s.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    return path


def classify_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    value_cols: list[str] = []
    state_cols: list[str] = []
    for col in columns:
        if col == TIME_COL or col == HOUSE_COL:
            continue
        if col.endswith(STATE_SUFFIX):
            state_cols.append(col)
        else:
            value_cols.append(col)
    if "aggregate" not in value_cols:
        raise ValueError("Input CSV must contain an 'aggregate' column.")
    return value_cols, state_cols


def load_and_resample_1min(
    csv_path: Path,
    *,
    house: int,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    print(f"Loading house {house}: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    if TIME_COL not in df.columns:
        raise ValueError(f"{csv_path} has no '{TIME_COL}' column.")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    bad_time = int(df[TIME_COL].isna().sum())
    if bad_time:
        print(f"  warning: dropping {bad_time:,} rows with invalid {TIME_COL}", flush=True)
        df = df.dropna(subset=[TIME_COL]).copy()

    if start:
        df = df[df[TIME_COL] >= pd.Timestamp(start)]
    if end:
        df = df[df[TIME_COL] <= pd.Timestamp(end)]
    if df.empty:
        raise ValueError(f"House {house} has no rows after time filtering.")

    value_cols, state_cols = classify_columns(list(df.columns))
    keep_cols = [TIME_COL, *value_cols, *state_cols]
    df = df[keep_cols].copy()
    df = df.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL], keep="first")
    df = df.set_index(TIME_COL)

    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in state_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    values = df[value_cols].resample("1min").mean()
    states = df[state_cols].resample("1min").max() if state_cols else pd.DataFrame(index=values.index)
    out = values.join(states, how="left")

    out = out.dropna(subset=["aggregate"]).copy()
    power_cols = [col for col in value_cols if col != "aggregate"]
    out[power_cols] = out[power_cols].fillna(0.0)
    for col in state_cols:
        out[col] = out[col].fillna(0).astype(int)

    out = out.reset_index()
    out.insert(1, HOUSE_COL, house)
    ordered = [TIME_COL, HOUSE_COL, *value_cols, *state_cols]
    out = out[ordered]

    print(
        f"  rows={len(out):,}  time={out[TIME_COL].iloc[0]} -> {out[TIME_COL].iloc[-1]}",
        flush=True,
    )
    return out


def split_house1(df: pd.DataFrame, val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("--val-fraction must be >= 0 and < 1.")
    if val_fraction == 0.0:
        return df.reset_index(drop=True), df.iloc[0:0].copy()

    split_idx = int(round(len(df) * (1.0 - val_fraction)))
    split_idx = max(1, min(len(df) - 1, split_idx))
    train = df.iloc[:split_idx].copy()
    val = df.iloc[split_idx:].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True)


def write_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path} rows={len(df):,}", flush=True)


def main() -> None:
    args = parse_args()
    house1_csv = resolve_house_csv(args.ukdale_dir, 1, args.house1_csv)
    house2_csv = resolve_house_csv(args.ukdale_dir, 2, args.house2_csv)

    house1 = load_and_resample_1min(
        house1_csv,
        house=1,
        start=args.start_house1,
        end=args.end_house1,
    )
    house2 = load_and_resample_1min(
        house2_csv,
        house=2,
        start=args.start_house2,
        end=args.end_house2,
    )

    train, val = split_house1(house1, args.val_fraction)
    test = house2.reset_index(drop=True)

    train_path = args.out_dir / "training" / "multi_appliance_training.csv"
    val_path = args.out_dir / "validating" / "multi_appliance_validating.csv"
    test_path = args.out_dir / "testing" / "multi_appliance_testing.csv"

    write_split(train, train_path)
    write_split(val, val_path)
    write_split(test, test_path)

    meta = {
        "dataset": "ukdale",
        "resolution": "1min",
        "protocol": "train house 1, test house 2",
        "house1_csv": str(house1_csv),
        "house2_csv": str(house2_csv),
        "train_file": str(train_path),
        "validation_file": str(val_path),
        "test_file": str(test_path),
        "val_fraction": args.val_fraction,
        "rows": {
            "train": int(len(train)),
            "validation": int(len(val)),
            "test": int(len(test)),
        },
    }
    meta_path = args.out_dir / "split_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()

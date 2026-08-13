#!/usr/bin/env python
"""Prepare a cross-house REDD split from whole-house 6 s CSVs.

Default protocol (UK-DALE-style analogy):
    - Source: House 1 + House 3  (labeled)
      each house: first (1 - val_frac) -> train, last val_frac -> validation
    - Test:   House 2            (all available days)

REDD low-freq coverage is only ~5–6 weeks/house, so we use a **time fraction**
instead of calendar weeks.

Source CSVs (preferred under datasets/redd/, else created_data/REDD/):
    redd_house1_lf_6s.csv
    redd_house2_lf_6s.csv
    redd_house3_lf_6s.csv

Outputs:
    datasets/redd/training/multi_appliance_training.csv
    datasets/redd/validating/multi_appliance_validating.csv
    datasets/redd/testing/multi_appliance_testing.csv

Example:
    python scripts/prepare_redd_crosshouse_split.py
    python scripts/prepare_redd_crosshouse_split.py --val-frac 0.25
    python scripts/prepare_redd_crosshouse_split.py --train-houses 1,3 --test-house 2
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
REDD_DIR = ROOT / "datasets" / "redd"
CREATED_REDD = REPO / "dataset_preprocess" / "created_data" / "REDD"

TIME_COL = "readable_time"
ON_COLS = [
    "microwave_on",
    "fridge_on",
    "dishwasher_on",
    "washingmachine_on",
]


def resolve_house_csv(redd: Path, house: int, *, created: Path) -> Path:
    """Prefer datasets/redd/redd_house{N}_lf_6s.csv; fall back to created_data."""
    candidates = [
        redd / f"redd_house{house}_lf_6s.csv",
        redd / f"multi_appliance_house{house}_lf.csv",
        created / f"redd_house{house}_lf_6s.csv",
        created / f"multi_appliance_house{house}_lf.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing house {house} CSV. Tried:\n  - "
        + "\n  - ".join(str(p) for p in candidates)
    )


def ensure_house_copies(
    redd: Path, created: Path, houses: list[int], *, force: bool = False
) -> None:
    """Copy created_data CSVs into datasets/redd/ when missing or stale."""
    redd.mkdir(parents=True, exist_ok=True)
    for house in houses:
        dest = redd / f"redd_house{house}_lf_6s.csv"
        src = created / f"redd_house{house}_lf_6s.csv"
        if not src.is_file():
            continue
        if dest.is_file() and not force:
            if dest.stat().st_mtime >= src.stat().st_mtime:
                continue
        print(f"  copying {src.name} -> {dest}", flush=True)
        shutil.copy2(src, dest)


def _load_house(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    parsed = pd.to_datetime(df[TIME_COL], errors="coerce")
    n_bad = int(parsed.isna().sum())
    if n_bad:
        print(
            f"  warning: {csv_path.name}: dropping {n_bad:,} rows with bad {TIME_COL}",
            flush=True,
        )
        df = df.loc[parsed.notna()].copy()
        parsed = parsed.loc[parsed.notna()]
    df[TIME_COL] = parsed
    return df.sort_values(TIME_COL).reset_index(drop=True)


def _split_train_val_by_frac(
    df: pd.DataFrame,
    *,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last ``val_frac`` of the time span -> val; earlier -> train."""
    if df.empty:
        raise ValueError("Selected house CSV is empty.")
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"Need 0 < val_frac < 1, got {val_frac}")
    t0 = df[TIME_COL].iloc[0]
    t1 = df[TIME_COL].iloc[-1]
    span = t1 - t0
    if span <= pd.Timedelta(0):
        raise ValueError("House block has zero time span.")
    cut_time = t1 - span * float(val_frac)
    train = df[df[TIME_COL] < cut_time].copy()
    val = df[df[TIME_COL] >= cut_time].copy()
    if train.empty or val.empty:
        split_idx = max(1, min(len(df) - 1, int(len(df) * (1.0 - val_frac))))
        train = df.iloc[:split_idx].copy()
        val = df.iloc[split_idx:].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True)


def _on_summary(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"  {label}: empty", flush=True)
        return
    parts = [f"rows={len(df):,}"]
    for col in ON_COLS:
        if col not in df.columns:
            continue
        on = int(pd.to_numeric(df[col], errors="coerce").fillna(0).gt(0).sum())
        parts.append(f"{col.replace('_on', '')}_ON={on:,}")
    t0 = df[TIME_COL].iloc[0]
    t1 = df[TIME_COL].iloc[-1]
    print(f"  {label}: {t0} -> {t1}  |  " + "  ".join(parts), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build REDD cross-house train/val/test CSVs.")
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.25,
        help="Fraction of each source house reserved for validation (default: 0.25).",
    )
    p.add_argument(
        "--train-houses",
        type=str,
        default="1,3",
        help="Comma list of labeled source houses (default: 1,3).",
    )
    p.add_argument(
        "--test-house",
        type=int,
        default=2,
        help="Unlabeled/eval target house (default: 2).",
    )
    p.add_argument(
        "--redd-dir",
        type=Path,
        default=REDD_DIR,
        help="datasets/redd with house CSVs and train/val/test folders.",
    )
    p.add_argument(
        "--created-dir",
        type=Path,
        default=CREATED_REDD,
        help="Fallback folder with redd_house*_lf_6s.csv from preprocess.",
    )
    p.add_argument(
        "--force-copy",
        action="store_true",
        help="Overwrite datasets/redd/redd_house*_lf_6s.csv from created_data even if newer.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    redd = args.redd_dir
    created = args.created_dir
    train_houses = [int(x.strip()) for x in args.train_houses.split(",") if x.strip()]
    test_house = int(args.test_house)
    if test_house in train_houses:
        raise ValueError(f"test house {test_house} must not be in train houses {train_houses}")
    val_frac = float(args.val_frac)

    all_houses = sorted(set(train_houses) | {test_house})
    ensure_house_copies(redd, created, all_houses, force=bool(args.force_copy))

    train_out = redd / "training" / "multi_appliance_training.csv"
    val_out = redd / "validating" / "multi_appliance_validating.csv"
    test_out = redd / "testing" / "multi_appliance_testing.csv"

    print(
        f"Building REDD split: source houses={train_houses} "
        f"(val_frac={val_frac:g}); test house={test_house}.",
        flush=True,
    )

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for house in train_houses:
        path = resolve_house_csv(redd, house, created=created)
        print(f"  source house {house}: {path}", flush=True)
        full = _load_house(path)
        _on_summary(full, f"house{house} full")
        train_h, val_h = _split_train_val_by_frac(full, val_frac=val_frac)
        _on_summary(train_h, f"house{house} train")
        _on_summary(val_h, f"house{house} val")
        train_parts.append(train_h)
        val_parts.append(val_h)

    test_path = resolve_house_csv(redd, test_house, created=created)
    print(f"  test house {test_house}: {test_path}", flush=True)
    test_df = _load_house(test_path)
    _on_summary(test_df, f"house{test_house} test")

    train_df = (
        pd.concat(train_parts, ignore_index=True)
        .sort_values([TIME_COL, "house"] if "house" in train_parts[0].columns else [TIME_COL])
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_parts, ignore_index=True)
        .sort_values([TIME_COL, "house"] if "house" in val_parts[0].columns else [TIME_COL])
        .reset_index(drop=True)
    )

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
    if test_house == 2:
        print(
            "  note: REDD house-2 washingmachine often has near-zero ON labels "
            "in this export — expect weak WM transfer metrics.",
            flush=True,
        )


if __name__ == "__main__":
    main()

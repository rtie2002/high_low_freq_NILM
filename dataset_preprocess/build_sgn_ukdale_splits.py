"""Build SGN paper-style UK-DALE train / validating / testing CSV splits.

Paper protocol (Shin et al. AAAI 2019):
  - Train houses: 1, 3, 4, 5 (we use 1 + 5; houses 3/4 unavailable for all appliances)
  - Test house: 2
  - Duration: last 1 week per house in the paper; default here is 28 days (4 weeks)

Split modes (``--val_source``):
  train_houses (default):
    - training:   all but the last ``val_days`` calendar days (per train house)
    - validating: last ``val_days`` calendar days (per train house, concatenated)
  test_house (cross-house transfer — recommended for H2 generalization):
    - training:   full ``last_days`` window from train houses (1 + 5)
    - validating: last ``val_days`` from the test house window (proxy target house)
    - testing:    earlier ``last_days - val_days`` from the test house (disjoint from val)

Outputs in NILM_model/data/ (``train_houses`` mode):
  - multi_appliance_training.csv
  - multi_appliance_validating.csv
  - multi_appliance_testing.csv

Outputs (``test_house`` mode):
  - multi_appliance_training_cross_house.csv
  - multi_appliance_validating_cross_house.csv
  - multi_appliance_testing_cross_house.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SAMPLE_SECONDS = 6
TIME_COL = "readable_time"
HOUSE_COL = "house"


def resolve_house_csv(data_dir: Path, house: int) -> Path:
    """Prefer ukdale_house{N}_lf_6s.csv, fall back to multi_appliance_house{N}_lf.csv."""
    candidates = [
        data_dir / f"ukdale_house{house}_lf_6s.csv",
        data_dir / f"multi_appliance_house{house}_lf.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def project_paths(data_dir: Path) -> dict[int, Path]:
    return {house: resolve_house_csv(data_dir, house) for house in (1, 2, 5)}


def rename_on_to_state(df: pd.DataFrame) -> pd.DataFrame:
    rename = {col: col.replace("_on", "_state") for col in df.columns if col.endswith("_on")}
    return df.rename(columns=rename)


def export_framework_splits(
    training: pd.DataFrame,
    validating: pd.DataFrame,
    testing: pd.DataFrame,
    export_dir: Path,
    *,
    rename_state_columns: bool = True,
) -> None:
    """Write multi_appliances_NILM/datasets/ukdale/{training,validating,testing}/data.csv."""
    if rename_state_columns:
        training = rename_on_to_state(training)
        validating = rename_on_to_state(validating)
        testing = rename_on_to_state(testing)

    for split, frame in [("training", training), ("validating", validating), ("testing", testing)]:
        out_dir = export_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "data.csv"
        frame.to_csv(out_path, index=False)
        print(f"  exported {split}: {out_path} ({len(frame):,} rows)")


def load_last_days(csv_path: Path, days: float, *, tail_buffer_rows: int | None = None) -> pd.DataFrame:
    """Load the last `days` calendar days from a house CSV (handles large files)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing house CSV: {csv_path}")

    approx_rows = int(days * 86400 / SAMPLE_SECONDS) + 10_000
    keep_rows = tail_buffer_rows or max(approx_rows, 150_000)
    print(f"  reading tail ~{keep_rows:,} rows from {csv_path.name} ...")

    buffer: pd.DataFrame | None = None
    for chunk in pd.read_csv(csv_path, chunksize=500_000):
        buffer = chunk if buffer is None else pd.concat([buffer, chunk], ignore_index=True)
        if len(buffer) > keep_rows:
            buffer = buffer.iloc[-keep_rows:].reset_index(drop=True)

    if buffer is None or buffer.empty:
        raise ValueError(f"No rows read from {csv_path}")

    buffer[TIME_COL] = pd.to_datetime(buffer[TIME_COL])
    end_time = buffer[TIME_COL].max()
    start_time = end_time - pd.Timedelta(days=days)
    trimmed = buffer[buffer[TIME_COL] >= start_time].sort_values(TIME_COL).reset_index(drop=True)
    if trimmed.empty:
        raise ValueError(f"No rows in last {days:g} days for {csv_path}")
    return trimmed


def load_full(csv_path: Path) -> pd.DataFrame:
    print(f"  reading full file {csv_path.name} ...")
    df = pd.read_csv(csv_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df.sort_values(TIME_COL).reset_index(drop=True)


def temporal_train_val_split_by_ratio(
    frames: list[pd.DataFrame],
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each house timeline by row ratio (used for --full_range only)."""
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for frame in frames:
        ordered = frame.sort_values(TIME_COL).reset_index(drop=True)
        split_idx = int(len(ordered) * train_ratio)
        if split_idx <= 432 or split_idx >= len(ordered) - 432:
            raise ValueError(
                f"Temporal split index {split_idx} invalid for house pool length {len(ordered)}."
            )
        train_parts.append(ordered.iloc[:split_idx].copy())
        val_parts.append(ordered.iloc[split_idx:].copy())

    training = pd.concat(train_parts, ignore_index=True).sort_values([HOUSE_COL, TIME_COL]).reset_index(drop=True)
    validating = pd.concat(val_parts, ignore_index=True).sort_values([HOUSE_COL, TIME_COL]).reset_index(drop=True)
    return training, validating


def temporal_train_val_split(
    frames: list[pd.DataFrame],
    val_days: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last ``val_days`` from each house; remainder is training."""
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    min_rows = 432 + 1
    for frame in frames:
        ordered = frame.sort_values(TIME_COL).reset_index(drop=True)
        if len(ordered) < 2 * min_rows:
            raise ValueError(f"House pool too short for split: {len(ordered)} rows")
        end_time = ordered[TIME_COL].max()
        val_start = end_time - pd.Timedelta(days=val_days)
        val_mask = ordered[TIME_COL] >= val_start
        val_part = ordered[val_mask].copy()
        train_part = ordered[~val_mask].copy()
        if len(train_part) < min_rows or len(val_part) < min_rows:
            raise ValueError(
                f"Temporal split left train={len(train_part)} val={len(val_part)} rows; "
                f"need at least {min_rows} each. Try smaller --val_days or larger --last_days."
            )
        train_parts.append(train_part)
        val_parts.append(val_part)

    training = pd.concat(train_parts, ignore_index=True).sort_values([HOUSE_COL, TIME_COL]).reset_index(drop=True)
    validating = pd.concat(val_parts, ignore_index=True).sort_values([HOUSE_COL, TIME_COL]).reset_index(drop=True)
    return training, validating


def concat_train_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Use the full loaded window from each train house (no internal val holdout)."""
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values([HOUSE_COL, TIME_COL])
        .reset_index(drop=True)
    )


def split_test_house_val_test(
    test_pool: pd.DataFrame,
    val_days: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out last val_days of the test-house window for validation; remainder is test."""
    ordered = test_pool.sort_values(TIME_COL).reset_index(drop=True)
    min_rows = 432 + 1
    if len(ordered) < 2 * min_rows:
        raise ValueError(f"Test house pool too short for split: {len(ordered)} rows")
    end_time = ordered[TIME_COL].max()
    val_start = end_time - pd.Timedelta(days=val_days)
    val_mask = ordered[TIME_COL] >= val_start
    validating = ordered[val_mask].copy()
    testing = ordered[~val_mask].copy()
    if len(testing) < min_rows or len(validating) < min_rows:
        raise ValueError(
            f"Test-house split left test={len(testing)} val={len(validating)} rows; "
            f"need at least {min_rows} each. Try smaller --val_days or larger --last_days."
        )
    return testing, validating


def summarize(name: str, df: pd.DataFrame) -> None:
    houses = sorted(df[HOUSE_COL].unique())
    start = df[TIME_COL].min()
    end = df[TIME_COL].max()
    print(f"\n{name}")
    print(f"  rows   : {len(df):,}")
    print(f"  houses : {houses}")
    print(f"  time   : {start} -> {end}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SGN UK-DALE train/val/test CSV splits.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "NILM_model" / "data",
        help="Directory containing multi_appliance_house{1,2,5}_lf.csv",
    )
    parser.add_argument(
        "--last_days",
        type=float,
        default=28.0,
        help="Use last N days per house (paper=7; default 28=4 weeks for stable training).",
    )
    parser.add_argument(
        "--full_range",
        action="store_true",
        help="Use full house CSVs instead of last N days (Scenario A; slow to train).",
    )
    parser.add_argument(
        "--val_days",
        type=float,
        default=4.0,
        help="Last N calendar days per train house reserved for validating (default 4).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=None,
        help="Deprecated: use --val_days instead. If set, overrides --val_days as fraction of last_days.",
    )
    parser.add_argument(
        "--train_houses",
        type=str,
        default="1,5",
        help="Comma-separated train house ids.",
    )
    parser.add_argument(
        "--test_house",
        type=int,
        default=2,
        help="Holdout test house id.",
    )
    parser.add_argument(
        "--val_source",
        choices=["train_houses", "test_house"],
        default="test_house",
        help=(
            "test_house (default): cross-house transfer val from test house; "
            "train_houses: val = last val_days from each train house."
        ),
    )
    parser.add_argument(
        "--test_full",
        action="store_true",
        help="Use the full test-house CSV (recommended for deployment-style H2 holdout).",
    )
    parser.add_argument(
        "--export_dir",
        type=Path,
        default=None,
        help=(
            "Also write training/data.csv, validating/data.csv, testing/data.csv "
            "under this directory (multi_appliances_NILM layout)."
        ),
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    train_houses = [int(item.strip()) for item in args.train_houses.split(",") if item.strip()]
    house_paths = project_paths(data_dir)

    val_days = float(args.val_days)
    if args.train_ratio is not None and not args.full_range:
        val_days = max(1.0, float(args.last_days) * (1.0 - float(args.train_ratio)))

    print("=" * 72)
    print("SGN UK-DALE SPLIT BUILDER")
    print("=" * 72)
    print(f"data_dir    : {data_dir}")
    print(f"train houses: {train_houses}")
    print(f"test house  : {args.test_house}")
    print(f"mode        : {'full_range' if args.full_range else f'last {args.last_days:g} days'}")
    if args.full_range:
        print("train/val   : 85% / 15% temporal per train house (full-range mode)")
    elif args.val_source == "test_house":
        print(
            f"train/val   : train H{','.join(str(h) for h in train_houses)} "
            f"({args.last_days:g}d each) + val H{args.test_house} (last {val_days:g}d); "
            f"test H{args.test_house} (first {args.last_days - val_days:g}d)"
        )
    else:
        print(f"train/val   : first {args.last_days - val_days:g} days train + last {val_days:g} days val (per house)")

    loader = load_full if args.full_range else lambda path: load_last_days(path, args.last_days)

    train_frames = []
    for house_id in train_houses:
        path = house_paths.get(house_id)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Expected CSV for house {house_id}: {path}")
        print(f"\n[train pool] house {house_id}")
        train_frames.append(loader(path))

    test_path = house_paths.get(args.test_house)
    if test_path is None or not test_path.exists():
        raise FileNotFoundError(f"Expected CSV for test house {args.test_house}: {test_path}")
    print(f"\n[test pool] house {args.test_house}")
    if args.test_full or (args.val_source == "train_houses" and not args.full_range):
        test_pool_df = load_full(test_path)
    else:
        test_pool_df = loader(test_path)

    if args.full_range:
        training_df, validating_df = temporal_train_val_split_by_ratio(train_frames, 0.85)
        test_df = test_pool_df
    elif args.val_source == "test_house":
        if val_days >= args.last_days:
            raise ValueError(
                f"--val_days ({val_days}) must be smaller than --last_days ({args.last_days}) "
                "when --val_source test_house"
            )
        training_df = concat_train_frames(train_frames)
        test_df, validating_df = split_test_house_val_test(test_pool_df, val_days)
    else:
        if val_days >= args.last_days:
            raise ValueError(f"--val_days ({val_days}) must be smaller than --last_days ({args.last_days})")
        training_df, validating_df = temporal_train_val_split(train_frames, val_days)
        test_df = test_pool_df

    if args.val_source == "test_house":
        out_train = data_dir / "multi_appliance_training_cross_house.csv"
        out_val = data_dir / "multi_appliance_validating_cross_house.csv"
        out_test = data_dir / "multi_appliance_testing_cross_house.csv"
    else:
        out_train = data_dir / "multi_appliance_training.csv"
        out_val = data_dir / "multi_appliance_validating.csv"
        out_test = data_dir / "multi_appliance_testing.csv"

    training_df.to_csv(out_train, index=False)
    validating_df.to_csv(out_val, index=False)
    test_df.to_csv(out_test, index=False)

    print("\n" + "=" * 72)
    print("SAVED SPLITS")
    print("=" * 72)
    summarize(out_train.name, training_df)
    summarize(out_val.name, validating_df)
    summarize(out_test.name, test_df)

    if args.export_dir is not None:
        print("\nFramework export")
        export_framework_splits(
            training_df,
            validating_df,
            test_df,
            args.export_dir.resolve(),
        )

    if args.val_source == "test_house":
        print("\nDone. Use csv_config: baseline/SGN/configs/training_data_ukdale_cross_house.json")
        print("      model_config: baseline/SGN/configs/sgn_ukdale_cross_house.json")
    else:
        print("\nDone. Point csv_config at the saved CSV paths under NILM_model/data/")


if __name__ == "__main__":
    main()

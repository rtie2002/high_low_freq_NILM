"""
Filter high-frequency feature CSVs to appliance ON periods with row buffer.

Keeps rows where on_off == 1, plus N rows before and after each ON period.
By default N=2, so with 6-second feature rows this keeps a 12-second margin
before and after each detected event.

Usage
-----
    python dataset_preprocess/high_frequency_data_extract/filter_on_period_with_buffer.py
    python dataset_preprocess/high_frequency_data_extract/filter_on_period_with_buffer.py --buffer_steps 2
    python dataset_preprocess/high_frequency_data_extract/filter_on_period_with_buffer.py --input_dir path/to/csvs --output_dir path/to/out
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def build_on_buffer_mask(on_off: pd.Series, buffer_steps: int) -> pd.Series:
    """
    Return True for ON rows and nearby rows within +/- buffer_steps.

    This uses row positions rather than timestamps, which matches the extracted
    HF feature grid and avoids problems if timestamps are strings.
    """
    on_mask = on_off.fillna(0).astype(int).eq(1)
    keep_mask = on_mask.copy()

    for step in range(1, buffer_steps + 1):
        keep_mask |= on_mask.shift(step, fill_value=False)
        keep_mask |= on_mask.shift(-step, fill_value=False)

    return keep_mask


def event_count(on_off: pd.Series) -> int:
    on_mask = on_off.fillna(0).astype(int).eq(1)
    starts = on_mask & ~on_mask.shift(1, fill_value=False)
    return int(starts.sum())


def filter_file(csv_path: str, output_dir: str, buffer_steps: int) -> dict:
    df = pd.read_csv(csv_path)
    if "on_off" not in df.columns:
        raise ValueError(f"Missing on_off column: {csv_path}")

    if "readable_time" in df.columns:
        df = df.sort_values("readable_time").reset_index(drop=True)

    keep_mask = build_on_buffer_mask(df["on_off"], buffer_steps)
    df_filtered = df.loc[keep_mask].copy()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(csv_path))
    df_filtered.to_csv(output_path, index=False)

    n_rows = len(df)
    n_kept = len(df_filtered)
    n_on = int(df["on_off"].fillna(0).astype(int).eq(1).sum())

    return {
        "file": os.path.basename(csv_path),
        "rows_original": n_rows,
        "rows_on": n_on,
        "events": event_count(df["on_off"]),
        "rows_kept": n_kept,
        "kept_ratio": n_kept / n_rows if n_rows else 0.0,
        "output_path": output_path,
    }


def filter_folder(input_dir: str, output_dir: str, buffer_steps: int) -> pd.DataFrame:
    csv_paths = [
        os.path.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if name.lower().endswith(".csv")
    ]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    records = []
    for csv_path in csv_paths:
        rec = filter_file(csv_path, output_dir, buffer_steps)
        records.append(rec)
        print(
            f"[filter] {rec['file']}: "
            f"{rec['rows_original']} -> {rec['rows_kept']} rows "
            f"(ON={rec['rows_on']}, events={rec['events']}, kept={rec['kept_ratio']:.2%})"
        )

    summary = pd.DataFrame(records)
    summary_path = os.path.join(output_dir, "on_period_buffer_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")
    return summary


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Keep ON-period rows plus a small before/after row buffer."
    )
    parser.add_argument(
        "--input_dir",
        default=os.path.join(
            PROJECT_ROOT, "dataset_preprocess", "high_frequency_data_extract", "output"
        ),
        help="Folder containing appliance CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            PROJECT_ROOT,
            "dataset_preprocess",
            "high_frequency_data_extract",
            "output_on_period_buffer2",
        ),
        help="Folder where filtered CSV files will be written.",
    )
    parser.add_argument(
        "--buffer_steps",
        type=int,
        default=2,
        help="Number of rows to keep before and after ON periods.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    filter_folder(args.input_dir, args.output_dir, args.buffer_steps)

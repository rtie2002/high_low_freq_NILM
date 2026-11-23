"""Build a 60/20/20 chronological split for every house and recording.

The input is the existing cross-house dataset. This script combines its three
CSV files, then divides every dataset/house/sequence_id recording by time:
first 60% for training, next 20% for validation, final 20% for testing.

The result evaluates future data from houses already seen during training.
It is therefore a within-house temporal protocol, not a cross-house protocol.
"""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_DIR / "dataset" / "mixed_ukdale_refit_5w_house_split"
OUTPUT_ROOT = PROJECT_DIR / "dataset" / "mixed_ukdale_refit_5w_temporal_602020"
SPLIT_NAMES = ("training", "validating", "testing")
GROUP_COLUMNS = ["dataset", "house", "sequence_id"]


def main():
    # The original three files contain different houses. Combining them lets
    # every house contribute to all three new time-ordered splits.
    source_tables = [
        pd.read_csv(SOURCE_ROOT / split / f"multi_appliance_{split}.csv")
        for split in SPLIT_NAMES
    ]
    data = pd.concat(source_tables, ignore_index=True)
    data = data.sort_values(GROUP_COLUMNS + ["unix_time"])

    parts = {name: [] for name in SPLIT_NAMES}
    for _, recording in data.groupby(GROUP_COLUMNS, sort=False):
        count = len(recording)
        train_end = int(count * 0.60)
        validation_end = int(count * 0.80)

        # Each part remains a separate CSV, so a later sliding window cannot
        # cross the 60% or 80% temporal boundary.
        parts["training"].append(recording.iloc[:train_end])
        parts["validating"].append(recording.iloc[train_end:validation_end])
        parts["testing"].append(recording.iloc[validation_end:])

    for split in SPLIT_NAMES:
        output_path = OUTPUT_ROOT / split / f"multi_appliance_{split}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = pd.concat(parts[split], ignore_index=True)
        output.to_csv(output_path, index=False)
        print(f"{split:10s} {len(output):,} rows -> {output_path}")

    print("Protocol: every house appears in train, validation, and test.")
    print("Order: first 60% train, next 20% validation, final 20% test.")


if __name__ == "__main__":
    main()

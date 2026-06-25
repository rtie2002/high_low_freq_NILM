"""Quick ON-event counts for SGN train/val/test CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

APPS = ["microwave", "dishwasher", "kettle", "washingmachine", "fridge"]
DATA_DIR = Path(__file__).resolve().parents[1] / "NILM_model" / "data"


def count_events(series: pd.Series) -> int:
    on = (series >= 0.5).astype(int)
    return int(((on.diff().fillna(on)) == 1).sum())


def main() -> None:
    for split in ("training", "validating", "testing"):
        path = DATA_DIR / f"multi_appliance_{split}.csv"
        df = pd.read_csv(path)
        print(f"\n=== {split} ({len(df):,} rows) ===")
        for app in APPS:
            col = f"{app}_on"
            if col not in df.columns:
                continue
            print(
                f"  {app:16s} events={count_events(df[col]):4d}  "
                f"ON_rows={int((df[col] >= 0.5).sum()):6d}"
            )
        if "house" in df.columns:
            for house_id in sorted(df["house"].unique()):
                sub = df[df["house"] == house_id]
                parts = []
                for app in APPS:
                    col = f"{app}_on"
                    if col in sub.columns:
                        parts.append(f"{app[:2]}={count_events(sub[col])}")
                print(f"    house {house_id}: {', '.join(parts)}")


if __name__ == "__main__":
    main()

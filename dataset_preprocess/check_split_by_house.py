"""Per-house breakdown of SGN train / validating / testing CSVs."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "NILM_model" / "data"
APPS = ["microwave", "dishwasher", "kettle", "washingmachine", "fridge"]


def main() -> None:
    for stem in ("training", "validating", "testing"):
        path = DATA / f"multi_appliance_{stem}.csv"
        cols = ["house", "readable_time"] + [f"{a}_on" for a in APPS]
        df = pd.read_csv(path, usecols=cols)
        print("=" * 64)
        print(f"{stem.upper()}  ({path.name})")
        print(f"Total rows: {len(df):,}")
        print(f"Houses in file: {sorted(df['house'].unique().tolist())}")
        print()
        for house in sorted(df["house"].unique()):
            sub = df[df["house"] == house]
            print(f"  House {house}: {len(sub):,} rows")
            print(f"    Time: {sub['readable_time'].min()}  ->  {sub['readable_time'].max()}")
            for app in APPS:
                on_rows = int((sub[f"{app}_on"] >= 0.5).sum())
                pct = 100.0 * on_rows / len(sub) if len(sub) else 0.0
                print(f"    {app:15} ON rows = {on_rows:6,}  ({pct:.2f}%)")
            print()


if __name__ == "__main__":
    main()

"""Count ON events (rising edges in *_state) per house/appliance/split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

APPS = ["kettle", "fridge", "microwave", "dishwasher", "washingmachine"]


def count_on_events(states: np.ndarray) -> int:
    if len(states) == 0:
        return 0
    s = states.astype(np.int8)
    prev = np.empty_like(s)
    prev[0] = 0
    prev[1:] = s[:-1]
    return int(np.sum((s == 1) & (prev == 0)))


def analyze_split(path: Path) -> pd.DataFrame:
    usecols = ["readable_time", "house"] + [f"{app}_state" for app in APPS]
    df = pd.read_csv(path, usecols=usecols)
    df["readable_time"] = pd.to_datetime(df["readable_time"])
    rows = []
    for house in sorted(df["house"].unique()):
        h = df[df["house"] == house].sort_values("readable_time")
        for app in APPS:
            col = f"{app}_state"
            rows.append(
                {
                    "house": int(house),
                    "appliance": app,
                    "on_rows": int(h[col].sum()),
                    "on_events": count_on_events(h[col].to_numpy()),
                    "timesteps": len(h),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "multi_appliances_NILM" / "datasets" / "ukdale"
    splits = {
        "training": base / "training" / "data.csv",
        "validating": base / "validating" / "data.csv",
        "testing": base / "testing" / "data.csv",
    }

    parts = []
    for split, path in splits.items():
        part = analyze_split(path)
        part["split"] = split
        parts.append(part)
    res = pd.concat(parts, ignore_index=True)

    for split in ["training", "validating", "testing"]:
        print("=" * 72)
        print(split.upper())
        sub = res[res["split"] == split]
        for house in sorted(sub["house"].unique()):
            print(f"  House {house}")
            hsub = sub[sub["house"] == house]
            for _, r in hsub.iterrows():
                print(
                    f"    {r['appliance']:<16} events={int(r['on_events']):>5}  "
                    f"on_rows={int(r['on_rows']):>8,}  steps={int(r['timesteps']):>9,}"
                )
            print(f"    {'TOTAL':<16} events={int(hsub['on_events'].sum()):>5}")
        by_app = sub.groupby("appliance")["on_events"].sum()
        print("  Split total by appliance:")
        for app in APPS:
            print(f"    {app:<16} {int(by_app[app]):>5} events")
        print(f"  GRAND TOTAL events: {int(sub['on_events'].sum())}")

    print()
    print("PIVOT (on_events: split x appliance)")
    pivot = res.groupby(["split", "appliance"])["on_events"].sum().unstack(fill_value=0)
    print(pivot.to_string())


if __name__ == "__main__":
    main()

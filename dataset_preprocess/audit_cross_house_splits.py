"""Audit UK-DALE cross-house SGN split CSVs for label/house sanity."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

APPLIANCES = ["kettle", "fridge", "microwave", "dishwasher", "washingmachine"]


def resolve_path(data_dir: Path, stem: str) -> Path | None:
    for name in (f"{stem}.csv", f"{stem}_cross_house.csv"):
        path = data_dir / name
        if path.exists():
            return path
    return None


def audit_file(label: str, path: Path) -> dict:
    cols = ["house", "readable_time", "aggregate"]
    for app in APPLIANCES:
        cols.extend([f"{app}_power", f"{app}_on"])
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    df["readable_time"] = pd.to_datetime(df["readable_time"])

    summary: dict = {
        "label": label,
        "file": path.name,
        "rows": len(df),
        "houses": sorted(int(h) for h in df["house"].unique()),
        "time_start": str(df["readable_time"].min()),
        "time_end": str(df["readable_time"].max()),
        "aggregate_std": float(df["aggregate"].std()),
    }
    for app in APPLIANCES:
        on_col = f"{app}_on"
        power_col = f"{app}_power"
        on = df[on_col].astype(int)
        on_rows = int(on.sum())
        events = int((on.diff().fillna(0) == 1).sum())
        on_power = df.loc[on == 1, power_col]
        summary[f"{app}_on_rows"] = on_rows
        summary[f"{app}_events"] = events
        summary[f"{app}_on_pct"] = 100.0 * on_rows / max(len(df), 1)
        summary[f"{app}_on_mean_w"] = float(on_power.mean()) if len(on_power) else 0.0
    return summary


def print_summary(summary: dict) -> None:
    print("=" * 72)
    print(f"{summary['label'].upper()}: {summary['file']}")
    print(f"  rows          : {summary['rows']:,}")
    print(f"  houses        : {summary['houses']}")
    print(f"  time          : {summary['time_start']} -> {summary['time_end']}")
    print(f"  aggregate std : {summary['aggregate_std']:.1f} W")
    for app in APPLIANCES:
        print(
            f"  {app:16s}: events={summary[f'{app}_events']:4d}  "
            f"ON rows={summary[f'{app}_on_rows']:7,d} ({summary[f'{app}_on_pct']:.3f}%)  "
            f"mean ON power={summary[f'{app}_on_mean_w']:.0f} W"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cross-house SGN split CSVs.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "NILM_model" / "data",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    stems = {
        "train": "multi_appliance_training",
        "val": "multi_appliance_validating",
        "test": "multi_appliance_testing",
    }
    summaries = []
    for label, stem in stems.items():
        path = resolve_path(data_dir, stem)
        if path is None:
            print(f"MISSING: {stem}.csv or {stem}_cross_house.csv in {data_dir}")
            continue
        s = audit_file(label, path)
        summaries.append(s)
        print_summary(s)

    if len(summaries) == 3:
        train_h = set(summaries[0]["houses"])
        val_h = set(summaries[1]["houses"])
        test_h = set(summaries[2]["houses"])
        print("\n" + "=" * 72)
        print("SPLIT LOGIC CHECK")
        if train_h <= {1, 5} and val_h == {2} and test_h == {2}:
            print("  OK: cross-house split (train H1+H5, val/test H2 only)")
        elif val_h <= {1, 5}:
            print("  WARNING: val is from train houses (H1/H5), NOT H2 cross-house val")
        else:
            print(f"  UNEXPECTED house sets: train={train_h} val={val_h} test={test_h}")

        dw_train = summaries[0]["dishwasher_events"]
        dw_val = summaries[1]["dishwasher_events"]
        dw_test = summaries[2]["dishwasher_events"]
        print(f"  dishwasher events: train={dw_train} val={dw_val} test={dw_test}")
        if dw_val < 3:
            print("  WARNING: very few dishwasher val events — F1/early-stop will be unstable")
        if dw_test < 5:
            print("  WARNING: very few dishwasher test events")


if __name__ == "__main__":
    main()

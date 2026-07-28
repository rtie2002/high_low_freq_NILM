"""Verify UK-DALE multi-appliance CSVs used by MATUDA (houses, ON rates, NaNs)."""
from pathlib import Path

import pandas as pd

BASE = Path(r"D:\Raymond\high_low_freq_NILM\multi_appliances_NILM\datasets\ukdale")
APPS = ["kettle", "fridge", "dishwasher", "washingmachine", "microwave"]
FILES = {
    "training": BASE / "training" / "multi_appliance_training.csv",
    "validating": BASE / "validating" / "multi_appliance_validating.csv",
    "testing": BASE / "testing" / "multi_appliance_testing.csv",
}


def main() -> None:
    for name, path in FILES.items():
        assert path.exists(), path
        df = pd.read_csv(path)
        print("=" * 60)
        print(name, path, "rows=", len(df))
        print("houses:", df["house"].value_counts().sort_index().to_dict())
        print("agg mean/std:", float(df["aggregate"].mean()), float(df["aggregate"].std()))
        print("NaNs:", int(df.isna().sum().sum()))
        for a in APPS:
            on = df[f"{a}_on"]
            print(
                f"  {a}: on_rate={float(on.mean()):.4f} "
                f"power_mean={float(df[f'{a}_power'].mean()):.1f}"
            )
        # Protocol checks
        if name in ("training", "validating"):
            assert set(df["house"].unique()) <= {1, 5}, name
        if name == "testing":
            assert set(df["house"].unique()) == {2}, name
    print("DATA_OK")


if __name__ == "__main__":
    main()

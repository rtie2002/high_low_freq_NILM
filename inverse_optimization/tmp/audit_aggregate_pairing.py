"""Read-only check of current exported reference values and aggregate mismatch."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Reference_Conditioned_Inverse_NILM/dataset/appliance_waveform reference"
stats = []
for app in ["washingmachine", "kettle", "microwave", "dishwasher", "fridge"]:
    parts = []
    cols = ["dataset", "house", "channel", "appliance_power_w", "aggregate_power_w"]
    for c in pd.read_csv(DATA / app / "events.csv", usecols=cols, chunksize=200000):
        c["rows"] = 1
        c["high_power_rows"] = (c.appliance_power_w >= 1000).astype(int)
        c["high_power_violations"] = ((c.appliance_power_w >= 1000) &
                                      (c.appliance_power_w > c.aggregate_power_w + 100)).astype(int)
        c["violations_100w"] = (c.appliance_power_w > c.aggregate_power_w + 100).astype(int)
        parts.append(c.groupby(["dataset", "house", "channel"])[
            ["rows", "high_power_rows", "high_power_violations", "violations_100w"]].sum())
    result = pd.concat(parts).groupby(level=[0, 1, 2]).sum().reset_index()
    result.insert(0, "appliance", app)
    result["high_power_violation_pct"] = 100 * result.high_power_violations / result.high_power_rows.replace(0, np.nan)
    stats.append(result)
    print(result.to_string(index=False), flush=True)
pd.concat(stats).to_csv(Path(__file__).with_name("aggregate_pairing_audit.csv"), index=False)

# Full House 1 washing-machine export comparison against local input rows.
parts = []
for c in pd.read_csv(DATA / "washingmachine/events.csv", chunksize=100000):
    parts.append(c[(c.dataset == "refit") & (c.house == 1)])
saved = pd.concat(parts)
raw = pd.read_csv(ROOT / "dataset_preprocess/REFIT/House_1.csv",
                  usecols=["Unix", "Aggregate", "Appliance5"])
for saved_col, raw_col in [("unix_time", "Unix"), ("appliance_power_w", "Appliance5"),
                           ("aggregate_power_w", "Aggregate")]:
    assert np.array_equal(saved[saved_col], raw[raw_col].iloc[saved.source_idx])
print("HOUSE1 RAW MATCH:", len(saved), "exported sample rows, all three columns exact", flush=True)
e=saved[saved.event_id == "refit_washingmachine_h01_a5_000008"]
print("PICTURED EVENT",len(e), "rows",flush=True)
high=e[e.appliance_power_w>=2000]
print("High>=2000W range",high.unix_time.min(),high.unix_time.max(),
      "aggregate range",high.aggregate_power_w.min(),high.aggregate_power_w.max(),flush=True)

# Small lag diagnostic, using a regular grid and raw context around this event.
# This is a diagnostic only; no corrected data is written.
lo,hi=int(e.unix_time.min())-300,int(e.unix_time.max())+300
r=raw[raw.Unix.between(lo,hi)].copy()
r["time"]=pd.to_datetime(r.Unix,unit="s")
r=r.set_index("time")[["Aggregate","Appliance5"]].resample("8s").mean().interpolate(limit=2)
target=r.Appliance5>=2000
for lag_sec in [-120,-60,-32,-16,-8,0,8,16,32,60,120]:
    agg=r.Aggregate.shift(lag_sec//8)
    valid=target & agg.notna()
    fraction=(r.loc[valid,"Appliance5"] > agg[valid]+100).mean()
    print("LAG",lag_sec,"seconds: high-power violation fraction",round(fraction,3),flush=True)

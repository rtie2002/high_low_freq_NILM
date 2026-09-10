import os
from pathlib import Path
import numpy as np
import pandas as pd

APPS = ["washingmachine", "dishwasher", "fridge", "kettle", "microwave"]
# Power floor for "clearly active" check (above typical standby)
HI = {
    "washingmachine": 200,
    "dishwasher": 200,
    "fridge": 50,
    "kettle": 500,
    "microwave": 500,
}

roots = [
    Path("multi_appliances_NILM/datasets/ukdale"),
    Path("multi_appliances_NILM/datasets/refit"),
]
files = sorted([*roots[0].glob("*_lf_8s.csv"), *roots[1].glob("*_lf_8s.csv")])
print(f"files={len(files)}")

rows = []
for f in files:
    name = f.name
    # ukdale_house5_lf_8s.csv / refit_house2_lf_8s.csv
    parts = name.replace("_lf_8s.csv","").split("_house")
    ds, house = parts[0], int(parts[1])
    usecols = ["sequence_id"]
    for a in APPS:
        usecols += [f"{a}_power", f"{a}_on"]
    # some cols may be missing?
    header = pd.read_csv(f, nrows=0).columns.tolist()
    usecols = [c for c in usecols if c in header]
    present = [a for a in APPS if f"{a}_power" in usecols and f"{a}_on" in usecols]
    stats = {a: {"n":0,"on":0,"hi":0,"hi_on":0,"hi_off":0,"max":0.0,"on_at_hi_off_max":0.0} for a in present}
    for chunk in pd.read_csv(f, usecols=usecols, chunksize=400000):
        for a in present:
            pw = chunk[f"{a}_power"].to_numpy(float)
            on = chunk[f"{a}_on"].to_numpy()
            hi = pw >= HI[a]
            st = stats[a]
            st["n"] += len(chunk)
            st["on"] += int(on.sum())
            st["hi"] += int(hi.sum())
            st["hi_on"] += int((hi & (on!=0)).sum())
            st["hi_off"] += int((hi & (on==0)).sum())
            m = float(pw.max()) if len(pw) else 0
            st["max"] = max(st["max"], m)
            if (hi & (on==0)).any():
                st["on_at_hi_off_max"] = max(st["on_at_hi_off_max"], float(pw[hi & (on==0)].max()))
    for a in present:
        st = stats[a]
        cov = st["hi_on"]/st["hi"] if st["hi"] else 1.0
        rows.append({
            "dataset": ds, "house": house, "app": a,
            "hi_thr": HI[a], "hi": st["hi"], "hi_off": st["hi_off"],
            "hi_cov": round(cov,4), "on_rows": st["on"],
            "maxW": round(st["max"],1), "miss_maxW": round(st["on_at_hi_off_max"],1),
        })

out = pd.DataFrame(rows).sort_values(["app","hi_cov","dataset","house"])
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)
print("\n=== ALL house×appliance high-power ON coverage ===")
print(out.to_string(index=False))
print("\n=== PROBLEMATIC (hi_cov < 0.95 and hi_off > 100) ===")
bad = out[(out.hi_cov < 0.95) & (out.hi_off > 100)].sort_values(["app","hi_cov"])
print(bad.to_string(index=False) if len(bad) else "(none)")
out.to_csv("tmp_label_audit.csv", index=False)
print("\nwrote tmp_label_audit.csv")

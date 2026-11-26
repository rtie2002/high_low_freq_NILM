"""Read-only audit of the staged reference bank before promotion."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "Reference_Conditioned_Inverse_NILM" / "dataset" / "appliance_waveform reference"
STAGING = ROOT / "_rebuild_timestamp_rules"


def audit():
    old = pd.read_csv(ROOT / "manifest.csv")
    new = pd.read_csv(STAGING / "manifest.csv")
    results = {}
    for app, threshold, minimum in [("kettle", 200, 12), ("microwave", 200, 6),
                                    ("fridge", 50, 60), ("dishwasher", 50, 960),
                                    ("washingmachine", 20, 960)]:
        records = new[new.file.str.startswith(app + "\\")]
        baseline = old[old.file.str.startswith(app + "\\")]
        assert (records.duration_sec >= minimum).all()
        total_rows = 0
        ids = set()
        for c in pd.read_csv(STAGING / app / "events.csv", chunksize=100000):
            event_on = c.source_idx.between(c.on_start_idx, c.on_end_idx)
            above = c.appliance_power_w >= threshold
            assert (c.is_on == event_on).all()
            assert (c.event_on == event_on).all()
            assert (c.above_threshold == above).all()
            assert not ((~event_on) & above).any()
            assert np.isfinite(c.appliance_power_w).all()
            assert np.isfinite(c.aggregate_power_w).all()
            widths = c.sample_end_unix - c.unix_time
            assert widths.between(0, 16).all()
            total_rows += len(c)
            ids.update(c.event_id.unique())
        assert ids == set(records.event_id)
        assert total_rows == records.samples.sum()
        assert np.array_equal(records.sample_start_row.to_numpy(),
                              np.r_[0, records.samples.cumsum().to_numpy()[:-1]])
        assert (records.sample_end_row - records.sample_start_row + 1 == records.samples).all()
        results[app] = {"old_events": len(baseline), "new_events": len(records),
                        "sample_rows": total_rows,
                        "incomplete_boundary_events": int(records.duration_censored.sum()),
                        "off_context_threshold_violations": 0}
        print(app, results[app], flush=True)

    # Compare the pictured event to the original source, including its aggregate lag.
    c = pd.read_csv(STAGING / "kettle/events.csv", nrows=200)
    e = c[c.event_id == "refit_kettle_h04_a9_000004"]
    assert len(e)
    raw = pd.read_csv(ROOT.parents[2] / "dataset_preprocess/REFIT/House_4.csv",
                      usecols=["Unix", "Aggregate", "Appliance9"])
    np.testing.assert_array_equal(e.unix_time, raw.Unix.iloc[e.source_idx])
    np.testing.assert_array_equal(e.appliance_power_w, raw.Appliance9.iloc[e.source_idx])
    np.testing.assert_array_equal(e.aggregate_power_w, raw.Aggregate.iloc[e.source_idx])
    print("Pictured event: raw timestamps and both powers unchanged; duration", e.duration_sec.iloc[0], flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    audit()

"""Audit dishwasher ON/OFF labels for one week per house (1, 2, 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ukdale_processing import apply_algorithm1_labeling, resolve_appliance_setting
from ukdale_processing_multi_appliance import fill_short_appliance_gaps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset_preprocess" / "UK_DALE"
TZ = "Europe/London"
PERIOD = "6s"
THR = 50
CHANNELS = {1: 6, 2: 13, 5: 22}

WINDOWS = {
    1: ("2017-04-19", "2017-04-26"),
    2: ("2013-10-03", "2013-10-10"),
    5: ("2014-11-06", "2014-11-13"),
}


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    diff = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(diff == 1)[0], np.where(diff == -1)[0]))


def load_dishwasher(
    cfg: dict, house: int, start: str, end: str
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray] | tuple[None, None, None]:
    app_cfg = cfg["appliances"]["dishwasher"]
    algo = cfg.get("algorithm1", {})
    ch = CHANNELS[house]
    start_ts = pd.Timestamp(start, tz=TZ).tz_convert("UTC").timestamp()
    end_ts = pd.Timestamp(end, tz=TZ).tz_convert("UTC").timestamp()
    path = DATA / f"house_{house}" / f"channel_{ch}.dat"
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1], chunksize=500_000):
        sub = chunk[(chunk[0] >= start_ts) & (chunk[0] <= end_ts)]
        if len(sub):
            chunks.append(sub)
    if not chunks:
        return None, None, None

    raw = pd.concat(chunks)
    raw["time"] = pd.to_datetime(raw[0], unit="s", utc=True).dt.tz_convert(TZ)
    series = raw.set_index("time")[1].resample(PERIOD).mean()
    gap = int(
        resolve_appliance_setting(app_cfg, "resample_gap_fill", house, algo.get("resample_gap_fill", 3))
    )
    filled = fill_short_appliance_gaps(series, gap).fillna(0.0)
    power = filled.to_numpy(dtype=float)
    min_off = resolve_appliance_setting(app_cfg, "min_off_duration", house, 1)
    min_on = resolve_appliance_setting(app_cfg, "min_on_duration", house, 1)
    label = apply_algorithm1_labeling(
        power, THR, min_off_duration=min_off, min_on_duration=min_on, l_window=0
    )
    return filled.index, power, label


def audit_event(
    event_idx: int,
    times: pd.DatetimeIndex,
    power: np.ndarray,
    label: np.ndarray,
    s: int,
    e: int,
) -> dict:
    seg_p = power[s:e]
    dur_min = (e - s) * 6 / 60
    below = seg_p < THR
    below_frac = float(below.mean())
    low_runs = runs(below)
    max_low_min = max(((b - a) * 6 / 60 for a, b in low_runs), default=0.0)
    long_low = sum(1 for a, b in low_runs if (b - a) >= 5)

    high = seg_p >= 500
    high_runs = runs(high)
    peak_med = float(np.median(seg_p[high])) if high.any() else float(np.median(seg_p))
    mid_med = float(np.median(seg_p[(seg_p >= THR) & (seg_p < 500)])) if ((seg_p >= THR) & (seg_p < 500)).any() else 0.0

    # naive: count separate >=THR segments without gap closing
    naive_segments = len(runs(seg_p >= THR))

    false_merge_risk = dur_min > 180 and below_frac > 0.20
    two_peak_cycle = len(high_runs) >= 2 and long_low >= 1

    return {
        "event": event_idx,
        "start": times[s].strftime("%Y-%m-%d %H:%M"),
        "end": times[e - 1].strftime("%Y-%m-%d %H:%M"),
        "dur_min": round(dur_min, 1),
        "peak_med_W": round(peak_med, 0),
        "mid_med_W": round(mid_med, 0),
        "pct_below_thr": round(100 * below_frac, 1),
        "longest_low_min": round(max_low_min, 1),
        "high_power_segments": len(high_runs),
        "naive_thr_segments": naive_segments,
        "two_peak_cycle": two_peak_cycle,
        "false_merge_risk": false_merge_risk,
    }


def main() -> None:
    with open(ROOT / "config" / "preprocess" / "ukdale.yaml") as f:
        cfg = yaml.safe_load(f)

    print("=" * 90)
    print("DISHWASHER 1-WEEK LABEL AUDIT — houses 1, 2, 5")
    print("=" * 90)

    for house, (start, end) in WINDOWS.items():
        loaded = load_dishwasher(cfg, house, start, end)
        if loaded[0] is None:
            print(f"\nHouse {house}: no data")
            continue
        times, power, label = loaded
        y = label.astype(bool)
        on_runs = runs(y)

        print(f"\n### House {house} | {start} -> {end} ###")
        print(f"  samples: {len(power):,} ({len(power) * 6 / 86400:.2f} days)")
        print(f"  ON events: {len(on_runs)}")
        print(f"  ON fraction: {100 * y.mean():.2f}%")

        rows = [audit_event(i + 1, times, power, label, s, e) for i, (s, e) in enumerate(on_runs)]

        print(f"\n  {'Ev':>2} {'Start':>16} {'Dur':>6} {'Peak':>6} {'Mid':>5} {'<50W%':>6} {'HiSeg':>5} {'Naive':>5}  Notes")
        print("  " + "-" * 85)
        for r in rows:
            notes = []
            if r["two_peak_cycle"]:
                notes.append("2-peak cycle (middle ~100W)")
            if r["false_merge_risk"]:
                notes.append("CHECK: long ON with much <50W")
            if r["naive_thr_segments"] > r["high_power_segments"] + 1:
                notes.append("gap-closed middle")
            note = ", ".join(notes) if notes else "OK"
            print(
                f"  {r['event']:2d} {r['start'][5:16]:>16} {r['dur_min']:5.0f}m "
                f"{r['peak_med_W']:5.0f}W {r['mid_med_W']:5.0f}W {r['pct_below_thr']:5.1f}% "
                f"{r['high_power_segments']:5d} {r['naive_thr_segments']:5d}  {note}"
            )

        risks = [r for r in rows if r["false_merge_risk"]]
        two_peak = sum(1 for r in rows if r["two_peak_cycle"])
        print(f"\n  Summary: {two_peak}/{len(rows)} events are 2-peak cycles (middle kept ON on purpose)")
        if risks:
            print(f"  WARNING: {len(risks)} event(s) may merge separate runs — inspect events {[r['event'] for r in risks]}")
        else:
            print("  No evidence of separate OFF periods wrongly merged into one long ON event.")


if __name__ == "__main__":
    main()

"""Audit ON/OFF labels: missed ON, false ON, spikes, standby bands."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "NILM_model" / "data"
THRESH = {
    "kettle": {1: 200, 2: 200, 5: 200},
    "microwave": {1: 200, 2: 200, 5: 200},
    "fridge": {1: 50, 2: 50, 5: 50},
    "dishwasher": {1: 50, 2: 50, 5: 50},
    "washingmachine": {1: 20, 2: 20, 5: 25},
}
APPS = list(THRESH.keys())
SPLITS = ["training", "validating", "testing"]


def spike_like_off(p: np.ndarray, y: np.ndarray, thr: float, window: int = 5) -> int:
    off_idx = np.where(~y)[0]
    count = 0
    for i in off_idx:
        if p[i] < thr:
            continue
        lo = max(0, i - window // 2)
        hi = min(len(p), i + window // 2 + 1)
        neigh = np.concatenate([p[lo:i], p[i + 1 : hi]])
        if len(neigh) and np.median(neigh) < thr * 0.5 and p[i] >= thr:
            count += 1
    return count


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    diff = np.diff(np.r_[0, mask.astype(int), 0])
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def audit_split(name: str, df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    print(f"\n### {name.upper()} ({len(df):,} rows) ###")
    for app in APPS:
        pcol, ocol = f"{app}_power", f"{app}_on"
        for house_id in sorted(df.house.unique()):
            s = df[df.house == house_id].sort_values("readable_time").reset_index(drop=True)
            thr = THRESH[app][int(house_id)]
            p = s[pcol].to_numpy(dtype=float)
            y = s[ocol].to_numpy() >= 0.5

            miss = (~y) & (p >= thr)
            miss_n = int(miss.sum())
            miss_pct = 100 * miss_n / max((~y).sum(), 1)

            false_on = y & (p < thr)
            fo_n = int(false_on.sum())
            fo_pct = 100 * fo_n / max(y.sum(), 1)

            off_50_199 = int((~y & (p >= 50) & (p < thr)).sum()) if thr > 50 else 0
            spike_n = spike_like_off(p, y, thr) if miss_n > 0 else 0
            on_zero = int((y & (p < 5)).sum())
            max_off = float(p[~y].max()) if (~y).any() else 0.0

            miss_runs = contiguous_runs(miss)
            short_miss = [(e - s) for s, e in miss_runs if (e - s) <= 2]
            long_miss = [(s, e) for s, e in miss_runs if (e - s) > 2]

            flags: list[str] = []
            if miss_pct > 0.05:
                flags.append("MISSED_ON")
            if fo_pct > 15:
                flags.append("FALSE_ON_HIGH")
            if on_zero > 0:
                flags.append("ON_AT_ZERO")
            if spike_n > 10:
                flags.append("SPIKES")
            if off_50_199 > 10_000 and app in ("microwave", "kettle"):
                flags.append("STANDBY_BAND")

            row = {
                "split": name,
                "house": int(house_id),
                "app": app,
                "thr": thr,
                "miss": miss_n,
                "miss_pct": miss_pct,
                "false_on": fo_n,
                "fo_pct": fo_pct,
                "spike": spike_n,
                "on_zero": on_zero,
                "off_50_199": off_50_199,
                "max_off": max_off,
                "short_miss_runs": len(short_miss),
                "long_miss_runs": len(long_miss),
            }
            rows.append(row)

            if flags or miss_n > 0 or fo_n > 50 or on_zero > 0:
                tag = " ".join(flags) if flags else "minor"
                print(f"  H{house_id} {app} (thr={thr}W) [{tag}]")
                print(
                    f"    missed ON (OFF & p>={thr}W): {miss_n:,} ({miss_pct:.3f}% of OFF) "
                    f"| max OFF power={max_off:.0f}W"
                )
                if short_miss:
                    print(
                        f"      short missed bursts (<=12s): {len(short_miss)} runs, "
                        f"{sum(short_miss)} rows"
                    )
                for s, e in long_miss[:2]:
                    seg = p[s:e]
                    print(
                        f"      long missed run: {e - s} samples, "
                        f"med={np.median(seg):.0f}W max={seg.max():.0f}W"
                    )
                print(
                    f"    false ON (ON & p<{thr}W): {fo_n:,} ({fo_pct:.1f}% of ON) "
                    f"| ON at <5W: {on_zero:,}"
                )
                if spike_n:
                    print(f"    spike-like OFF (isolated p>={thr}W): ~{spike_n}")
                if off_50_199:
                    print(f"    OFF but 50-{thr - 1}W standby: {off_50_199:,} rows")
    return rows


def main() -> None:
    frames = {s: pd.read_csv(DATA / f"multi_appliance_{s}.csv") for s in SPLITS}
    all_df = pd.concat(frames.values(), ignore_index=True)

    print("=" * 90)
    print("LABEL AUDIT: missed ON | false ON | spikes | standby")
    print("=" * 90)

    all_rows: list[dict] = []
    for split in SPLITS:
        all_rows.extend(audit_split(split, frames[split]))
    all_rows.extend(audit_split("ALL", all_df))

    print("\n" + "=" * 90)
    print("TOTALS (all splits pooled)")
    print("=" * 90)
    print(
        f"{'H':>3} {'Appliance':>16} {'Thr':>4} {'MissedON':>9} {'%OFF':>7} "
        f"{'FalseON':>9} {'%ON':>7} {'ON@0W':>7} {'Spikes':>7} {'StbyOFF':>9}"
    )
    for r in sorted([x for x in all_rows if x["split"] == "ALL"], key=lambda x: (x["house"], x["app"])):
        print(
            f"{r['house']:3d} {r['app']:16} {r['thr']:4d} {r['miss']:9,} {r['miss_pct']:6.3f}% "
            f"{r['false_on']:9,} {r['fo_pct']:6.1f}% {r['on_zero']:7,} {r['spike']:7,} "
            f"{r['off_50_199']:9,}"
        )

    # worst examples: missed ON with highest power
    print("\n" + "=" * 90)
    print("TOP MISSED-ON EXAMPLES (OFF label but highest power)")
    print("=" * 90)
    for app in APPS:
        pcol, ocol = f"{app}_power", f"{app}_on"
        worst = all_df[(all_df[ocol] == 0) & (all_df[pcol] > 0)].nlargest(5, pcol)
        if len(worst):
            print(f"\n{app}:")
            cols = ["readable_time", "house", pcol, ocol, "aggregate"]
            print(worst[cols].to_string(index=False))


if __name__ == "__main__":
    main()

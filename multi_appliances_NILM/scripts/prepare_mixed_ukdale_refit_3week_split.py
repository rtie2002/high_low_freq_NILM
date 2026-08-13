#!/usr/bin/env python
"""Build a mixed UK-DALE + REFIT train/val/test split (3 weeks / house).

Protocol
--------
Source houses (labeled, 3-week best-activity blocks, then 80/20 train/val):
  UK-DALE: 1, 5
  REFIT:   2, 3, 5, 9, 11

Test houses (3-week best-activity blocks, held out):
  UK-DALE: 2
  REFIT:   20   (typical full 5-app house with solid WM / DW activity)

Block selection (per house)
---------------------------
Slide contiguous ``--block-weeks`` windows (default 3) with ``--step-days``
(default 1). Keep only windows that:
  1) have enough samples (coverage),
  2) pass hard ON floors for every appliance on the FULL block,
  3) pass scaled ON floors on the LAST 20% of the block (so val is active).

Among valid windows, pick maximin score:
  score = min_a (n_events(a) / min_events(a))
so the weakest appliance is as strong as possible.

Train / val
-----------
Within each selected source block: first 80% -> train, last 20% -> val
(time-contiguous). Val activity is guaranteed by the selection constraint.

Outputs (under datasets/mixed_ukdale_refit_3w/ by default)
----------------------------------------------------------
  training/multi_appliance_training.csv
  validating/multi_appliance_validating.csv
  testing/multi_appliance_testing.csv
  selection_summary.csv   (chosen windows + ON stats)

Example
-------
  python scripts/prepare_mixed_ukdale_refit_3week_split.py
  python scripts/prepare_mixed_ukdale_refit_3week_split.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UKDALE_DIR = ROOT / "datasets" / "ukdale"
REFIT_DIR = ROOT / "datasets" / "refit"
OUT_DIR = ROOT / "datasets" / "mixed_ukdale_refit_3w"

TIME_COL = "readable_time"
SAMPLE_SECONDS = 6.0

APPS_5 = ["kettle", "fridge", "dishwasher", "washingmachine", "microwave"]

# Source / test house lists
UKDALE_SOURCE = [1, 5]
UKDALE_TEST = [2]
REFIT_SOURCE = [2, 3, 5, 9, 11]
REFIT_TEST = [20]  # typical REFIT eval house

# Hard floors for a FULL 3-week block (events = contiguous ON runs).
# Val (last 20% ≈ 0.6 week) uses ceil(0.2 * these), with a minimum of 1
# for rare apps so val cannot be empty.
FULL_MIN_EVENTS = {
    "kettle": 6,
    "microwave": 6,
    "dishwasher": 2,
    "washingmachine": 2,
    "fridge": 30,  # many short cycles
}
FULL_MIN_ON_MINUTES = {
    "kettle": 10.0,
    "microwave": 10.0,
    "dishwasher": 60.0,
    "washingmachine": 60.0,
    "fridge": 0.0,  # use event count / frac instead
}
FRIDGE_MIN_ON_FRAC = 0.15


@dataclass
class BlockStats:
    start: pd.Timestamp
    end: pd.Timestamp
    n_rows: int
    coverage: float
    n_events: dict[str, int]
    on_minutes: dict[str, float]
    on_frac: dict[str, float]
    score: float
    valid: bool
    reject_reason: str = ""


def resolve_ukdale(house: int, ukdale: Path) -> Path:
    candidates = [
        ukdale / f"ukdale_house{house}_lf_6s.csv",
        ukdale / f"multi_appliance_house{house}_lf.csv",
        ukdale / f"multi_appliance_FULL_house{house}.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"UK-DALE house {house} CSV not found under {ukdale}")


def resolve_refit(house: int, refit: Path) -> Path:
    candidates = [
        refit / f"refit_house{house}_lf_6s.csv",
        refit / f"multi_appliance_house{house}_lf.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"REFIT house {house} CSV not found under {refit}")


def on_cols(apps: list[str]) -> list[str]:
    return [f"{a}_on" for a in apps]


def count_events(on: np.ndarray) -> int:
    """Number of contiguous ON runs (0->1 edges + leading 1)."""
    if on.size == 0:
        return 0
    x = on.astype(np.int8)
    return int(x[0]) + int(np.sum((x[1:] == 1) & (x[:-1] == 0)))


def load_on_timeline(csv_path: Path, apps: list[str]) -> tuple[pd.DatetimeIndex, np.ndarray]:
    cols = [TIME_COL] + on_cols(apps)
    print(f"  loading ON timeline: {csv_path.name}", flush=True)
    df = pd.read_csv(csv_path, usecols=cols)
    times = pd.to_datetime(df[TIME_COL], errors="coerce")
    ok = times.notna()
    if not bool(ok.all()):
        n_bad = int((~ok).sum())
        print(f"    drop {n_bad:,} bad timestamps", flush=True)
        df = df.loc[ok].copy()
        times = times.loc[ok]
    times = pd.DatetimeIndex(times)
    mat = np.column_stack(
        [df[f"{a}_on"].fillna(0).astype(np.int8).to_numpy() for a in apps]
    )
    order = np.argsort(times.values)
    return times[order], mat[order]


def load_full_slice(csv_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    t = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.loc[t.notna()].copy()
    df[TIME_COL] = t.loc[t.notna()]
    out = df[(df[TIME_COL] >= start) & (df[TIME_COL] <= end)].copy()
    return out.sort_values(TIME_COL).reset_index(drop=True)


def val_floors(full_events: dict[str, int], full_minutes: dict[str, float]) -> tuple[dict, dict]:
    """Scale full-block floors to ~20% duration; keep rare apps >= 1 event."""
    ev = {}
    mins = {}
    for app in APPS_5:
        ev[app] = max(1, int(np.ceil(0.2 * full_events[app])))
        # Fridge: keep a softer but non-zero cycle floor
        if app == "fridge":
            ev[app] = max(5, ev[app])
        mins[app] = 0.2 * full_minutes[app]
    return ev, mins


def _daily_tables(
    times: pd.DatetimeIndex, on_mat: np.ndarray, apps: list[str]
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate to calendar days: row counts, ON sample counts, ON event counts.

    Events that cross midnight are attributed to the day where the ON run starts
    (good enough for week selection).
    """
    day = times.floor("D")
    # row counts / on counts via groupby
    df = pd.DataFrame({"day": day, "ones": 1})
    for i, app in enumerate(apps):
        df[f"on_{app}"] = on_mat[:, i]
    g = df.groupby("day", sort=True)
    days = pd.DatetimeIndex(g.size().index)
    n_rows = g["ones"].sum().to_numpy(dtype=np.int64)
    on_counts = np.column_stack(
        [g[f"on_{app}"].sum().to_numpy(dtype=np.int64) for app in apps]
    )

    # event starts per day
    event_counts = np.zeros_like(on_counts)
    for i, app in enumerate(apps):
        x = on_mat[:, i].astype(np.int8)
        starts = np.zeros(len(x), dtype=np.int8)
        starts[0] = x[0]
        starts[1:] = ((x[1:] == 1) & (x[:-1] == 0)).astype(np.int8)
        edf = pd.DataFrame({"day": day, "s": starts})
        event_counts[:, i] = (
            edf.groupby("day", sort=True)["s"].sum().reindex(days, fill_value=0).to_numpy()
        )
    return days, n_rows, on_counts, event_counts


def _window_from_daily(
    days: pd.DatetimeIndex,
    n_rows: np.ndarray,
    on_counts: np.ndarray,
    event_counts: np.ndarray,
    apps: list[str],
    i0: int,
    i1: int,
    *,
    expected_rows: float,
    min_events: dict[str, int],
    min_on_minutes: dict[str, float],
    fridge_min_frac: float,
) -> BlockStats:
    """Stats for day indices [i0, i1) inclusive-exclusive on ``days``."""
    start = days[i0]
    end = days[i1 - 1] + pd.Timedelta(days=1) - pd.Timedelta(seconds=SAMPLE_SECONDS)
    n = int(n_rows[i0:i1].sum())
    coverage = float(n) / max(expected_rows, 1.0)
    n_events = {apps[j]: int(event_counts[i0:i1, j].sum()) for j in range(len(apps))}
    on_minutes = {
        apps[j]: float(on_counts[i0:i1, j].sum()) * SAMPLE_SECONDS / 60.0
        for j in range(len(apps))
    }
    on_frac = {
        apps[j]: (float(on_counts[i0:i1, j].sum()) / float(n)) if n else 0.0
        for j in range(len(apps))
    }

    reasons: list[str] = []
    if coverage < 0.80:
        reasons.append(f"coverage={coverage:.2f}<0.80")
    ratios: list[float] = []
    for app in apps:
        need_e = int(min_events.get(app, 1))
        need_m = float(min_on_minutes.get(app, 0.0))
        if app == "fridge":
            # Pass if enough cycles OR enough ON fraction (fridge is dense).
            if n_events[app] < need_e and on_frac[app] < fridge_min_frac:
                reasons.append(
                    f"fridge_events={n_events[app]}<{need_e} and "
                    f"frac={on_frac[app]:.3f}<{fridge_min_frac}"
                )
        else:
            if n_events[app] < need_e:
                reasons.append(f"{app}_events={n_events[app]}<{need_e}")
            if on_minutes[app] < need_m:
                reasons.append(f"{app}_min={on_minutes[app]:.1f}<{need_m}")
        ratios.append(n_events[app] / max(need_e, 1))
    score = float(min(ratios)) if ratios else -1.0
    return BlockStats(
        start=start,
        end=end,
        n_rows=n,
        coverage=coverage,
        n_events=n_events,
        on_minutes=on_minutes,
        on_frac=on_frac,
        score=score,
        valid=len(reasons) == 0,
        reject_reason="; ".join(reasons),
    )


def select_best_block(
    times: pd.DatetimeIndex,
    on_mat: np.ndarray,
    apps: list[str],
    *,
    block_weeks: float,
    step_days: float,
    require_active_val_tail: bool,
) -> BlockStats:
    if len(times) == 0:
        raise ValueError("Empty timeline")

    block_days = int(round(7.0 * block_weeks))
    step = max(1, int(round(step_days)))
    expected_full = (block_days * 24 * 3600) / SAMPLE_SECONDS
    val_days = max(1, int(round(0.2 * block_days)))
    train_days = block_days - val_days

    full_ev = FULL_MIN_EVENTS
    full_min = FULL_MIN_ON_MINUTES
    val_ev, val_min = val_floors(FULL_MIN_EVENTS, FULL_MIN_ON_MINUTES)

    days, n_rows_d, on_counts, event_counts = _daily_tables(times, on_mat, apps)
    # Fill missing calendar days with zeros so windows are true calendar spans
    if len(days) == 0:
        raise ValueError("No days in timeline")
    full_range = pd.date_range(days[0], days[-1], freq="D")
    if len(full_range) != len(days):
        idx = pd.Index(full_range)
        n_rows_s = pd.Series(n_rows_d, index=days).reindex(idx, fill_value=0).to_numpy(np.int64)
        on_s = (
            pd.DataFrame(on_counts, index=days, columns=apps)
            .reindex(idx, fill_value=0)
            .to_numpy(np.int64)
        )
        ev_s = (
            pd.DataFrame(event_counts, index=days, columns=apps)
            .reindex(idx, fill_value=0)
            .to_numpy(np.int64)
        )
        days = pd.DatetimeIndex(idx)
        n_rows_d, on_counts, event_counts = n_rows_s, on_s, ev_s

    best: BlockStats | None = None
    n_cand = 0
    n_valid = 0
    i = 0
    while i + block_days <= len(days):
        n_cand += 1
        full = _window_from_daily(
            days,
            n_rows_d,
            on_counts,
            event_counts,
            apps,
            i,
            i + block_days,
            expected_rows=expected_full,
            min_events=full_ev,
            min_on_minutes=full_min,
            fridge_min_frac=FRIDGE_MIN_ON_FRAC,
        )
        ok = full.valid
        if ok and require_active_val_tail:
            tail = _window_from_daily(
                days,
                n_rows_d,
                on_counts,
                event_counts,
                apps,
                i + train_days,
                i + block_days,
                expected_rows=expected_full * (val_days / block_days),
                min_events=val_ev,
                min_on_minutes=val_min,
                fridge_min_frac=0.10,
            )
            ok = tail.valid
        if ok:
            n_valid += 1
            if best is None or full.score > best.score:
                best = full
        i += step

    if best is None:
        print("    WARN: no window passed hard floors; using soft maximin fallback", flush=True)
        soft_best: BlockStats | None = None
        soft_key: tuple[float, float, float] | None = None
        i = 0
        while i + block_days <= len(days):
            full = _window_from_daily(
                days,
                n_rows_d,
                on_counts,
                event_counts,
                apps,
                i,
                i + block_days,
                expected_rows=expected_full,
                min_events={a: 1 for a in apps},
                min_on_minutes={a: 0.0 for a in apps},
                fridge_min_frac=0.0,
            )
            # Sparse houses (e.g. REFIT H9 with long gaps) may never hit 80%.
            # Still require a usable amount of samples (~>=40% of a full 3 weeks).
            if full.n_rows < int(0.40 * expected_full):
                i += step
                continue
            bonus = 0.0
            if require_active_val_tail:
                tail = _window_from_daily(
                    days,
                    n_rows_d,
                    on_counts,
                    event_counts,
                    apps,
                    i + train_days,
                    i + block_days,
                    expected_rows=expected_full * (val_days / block_days),
                    min_events={a: 1 for a in apps},
                    min_on_minutes={a: 0.0 for a in apps},
                    fridge_min_frac=0.0,
                )
                if min(tail.n_events.values()) >= 1:
                    bonus = 100.0
            # Prefer: denser window, then active val, then maximin event score.
            key = (full.coverage, bonus, full.score)
            if soft_key is None or key > soft_key:
                soft_key = key
                soft_best = BlockStats(
                    start=full.start,
                    end=full.end,
                    n_rows=full.n_rows,
                    coverage=full.coverage,
                    n_events=full.n_events,
                    on_minutes=full.on_minutes,
                    on_frac=full.on_frac,
                    score=full.score + bonus,
                    valid=True,
                    reject_reason=f"soft_fallback(cov={full.coverage:.2f})",
                )
            i += step
        if soft_best is None:
            raise RuntimeError(
                "Could not find any 3-week window with >=40% sample coverage. "
                "Check that the house CSV is complete (not truncated) and has ON labels."
            )
        print(
            f"    soft pick coverage={soft_best.coverage:.2f} "
            f"rows={soft_best.n_rows:,} note={soft_best.reject_reason}",
            flush=True,
        )
        best = soft_best

    print(
        f"    candidates={n_cand}, valid_hard={n_valid}, "
        f"picked {best.start} -> {best.end} score={best.score:.2f}",
        flush=True,
    )
    return best


def split_80_20(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split empty frame")
    t0 = df[TIME_COL].iloc[0]
    t1 = df[TIME_COL].iloc[-1]
    cut = t0 + 0.8 * (t1 - t0)
    train = df[df[TIME_COL] < cut].copy()
    val = df[df[TIME_COL] >= cut].copy()
    if train.empty or val.empty:
        idx = max(1, min(len(df) - 1, int(len(df) * 0.8)))
        train = df.iloc[:idx].copy()
        val = df.iloc[idx:].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True)


def summarize_split(name: str, df: pd.DataFrame, apps: list[str]) -> str:
    parts = [f"{name}: rows={len(df):,}"]
    if df.empty:
        return parts[0] + " EMPTY"
    parts.append(f"time={df[TIME_COL].iloc[0]} -> {df[TIME_COL].iloc[-1]}")
    houses = sorted(df["house"].unique().tolist()) if "house" in df.columns else []
    parts.append(f"houses={houses}")
    for a in apps:
        col = f"{a}_on"
        if col not in df.columns:
            continue
        on = df[col].fillna(0).to_numpy()
        parts.append(
            f"{a}: ON%={100.0 * on.mean():.2f} events={count_events(on)} "
            f"min={on.sum() * SAMPLE_SECONDS / 60.0:.1f}"
        )
    return " | ".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mixed UK-DALE+REFIT 3-week active split.")
    p.add_argument("--ukdale-dir", type=Path, default=UKDALE_DIR)
    p.add_argument("--refit-dir", type=Path, default=REFIT_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--block-weeks", type=float, default=3.0)
    p.add_argument("--step-days", type=float, default=1.0)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only select windows and print summary; do not write large CSVs.",
    )
    p.add_argument(
        "--refit-test-house",
        type=int,
        default=20,
        help="REFIT house held out for testing (default: 20).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apps = list(APPS_5)
    refit_test = [int(args.refit_test_house)]
    refit_source = [h for h in REFIT_SOURCE if h not in refit_test]
    # If user picks a source house as test, rebuild source from defaults minus test
    if not refit_source:
        refit_source = [h for h in [2, 3, 5, 9, 11, 20] if h not in refit_test]

    print("=== Mixed UK-DALE + REFIT 3-week split ===", flush=True)
    print(f"  UK-DALE source={UKDALE_SOURCE}  test={UKDALE_TEST}", flush=True)
    print(f"  REFIT   source={refit_source}  test={refit_test}", flush=True)
    print(
        f"  block={args.block_weeks} wk, step={args.step_days} d, "
        f"train/val=80/20 with active val-tail constraint",
        flush=True,
    )

    selections: list[dict] = []
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    jobs = (
        [("ukdale", h, "source") for h in UKDALE_SOURCE]
        + [("refit", h, "source") for h in refit_source]
        + [("ukdale", h, "test") for h in UKDALE_TEST]
        + [("refit", h, "test") for h in refit_test]
    )

    for dataset, house, role in jobs:
        path = (
            resolve_ukdale(house, args.ukdale_dir)
            if dataset == "ukdale"
            else resolve_refit(house, args.refit_dir)
        )
        print(f"\n[{dataset} house {house} | {role}] {path}", flush=True)
        times, on_mat = load_on_timeline(path, apps)
        # Test houses: still want active apps, but do not require val-tail
        # (whole block goes to test).
        require_val = role == "source"
        block = select_best_block(
            times,
            on_mat,
            apps,
            block_weeks=float(args.block_weeks),
            step_days=float(args.step_days),
            require_active_val_tail=require_val,
        )
        rec = {
            "dataset": dataset,
            "house": house,
            "role": role,
            "csv": str(path),
            "start": str(block.start),
            "end": str(block.end),
            "n_rows": block.n_rows,
            "coverage": round(block.coverage, 4),
            "score": round(block.score, 4),
            "reject_or_note": block.reject_reason,
            **{f"events_{a}": block.n_events[a] for a in apps},
            **{f"on_min_{a}": round(block.on_minutes[a], 2) for a in apps},
            **{f"on_frac_{a}": round(block.on_frac[a], 4) for a in apps},
        }
        selections.append(rec)
        print(
            "    events="
            + ", ".join(f"{a}:{block.n_events[a]}" for a in apps),
            flush=True,
        )

        if args.dry_run:
            continue

        print("    loading full rows for selected window...", flush=True)
        slice_df = load_full_slice(path, block.start, block.end)
        # Tag dataset for downstream DA / debugging
        slice_df.insert(1, "dataset", dataset)

        if role == "source":
            tr, va = split_80_20(slice_df)
            print("   ", summarize_split("train", tr, apps), flush=True)
            print("   ", summarize_split("val", va, apps), flush=True)
            # Soft assert: val has >=1 event per app
            for a in apps:
                col = f"{a}_on"
                if count_events(va[col].fillna(0).to_numpy()) < 1:
                    print(
                        f"    WARN: val has 0 events for {a} "
                        f"(house {house}); consider re-running with softer floors",
                        flush=True,
                    )
            train_parts.append(tr)
            val_parts.append(va)
        else:
            print("   ", summarize_split("test", slice_df, apps), flush=True)
            test_parts.append(slice_df)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "selection_summary.csv"
    pd.DataFrame(selections).to_csv(summary_path, index=False)
    meta_path = out_dir / "selection_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "block_weeks": args.block_weeks,
                "step_days": args.step_days,
                "train_val_ratio": "80/20",
                "ukdale_source": UKDALE_SOURCE,
                "ukdale_test": UKDALE_TEST,
                "refit_source": refit_source,
                "refit_test": refit_test,
                "full_min_events": FULL_MIN_EVENTS,
                "full_min_on_minutes": FULL_MIN_ON_MINUTES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote selection summary: {summary_path}", flush=True)

    if args.dry_run:
        print("Dry-run only; skipped writing train/val/test CSVs.", flush=True)
        return

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    train_out = out_dir / "training" / "multi_appliance_training.csv"
    val_out = out_dir / "validating" / "multi_appliance_validating.csv"
    test_out = out_dir / "testing" / "multi_appliance_testing.csv"
    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    print("\n=== Wrote split CSVs ===", flush=True)
    print(summarize_split("TRAIN", train_df, apps), flush=True)
    print(summarize_split("VAL", val_df, apps), flush=True)
    print(summarize_split("TEST", test_df, apps), flush=True)
    print(f"  {train_out}", flush=True)
    print(f"  {val_out}", flush=True)
    print(f"  {test_out}", flush=True)


if __name__ == "__main__":
    main()

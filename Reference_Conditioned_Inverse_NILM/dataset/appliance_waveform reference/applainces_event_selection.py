"""Extract appliance ON events with short OFF context from REFIT and UK-DALE.

Run this file directly. Samples from every detected event are appended to:

    dataset/appliance_waveform reference/<appliance>/events.csv

The dataset, house, channel, and event ID columns identify the source of every
sample. Appliance and aggregate powers use eight-second bins in watts. A
manifest with one summary row per event is saved at the dataset root.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================
OUTPUT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_ROOT.parents[1]
RAW_DATA_ROOT = PROJECT_ROOT.parent / "dataset_preprocess"


# ============================================================
# 2. Maximum OFF context saved around each ON event
# ============================================================
OFF_CONTEXT_SAMPLES = {
    "kettle": (4, 4),
    "microwave": (4, 4),
    "fridge": (8, 8),
    "dishwasher": (16, 16),
    "washingmachine": (16, 16),
}


# ============================================================
# 3. Physical event rules in seconds (shared by both datasets)
# Thresholds are unchanged. Durations follow the existing preprocessing YAMLs.
# ============================================================
EVENT_RULES = {
    "kettle": {"threshold_w": 200, "min_on_sec": 12, "max_off_gap_sec": 0},
    "microwave": {"threshold_w": 200, "min_on_sec": 6, "max_off_gap_sec": 30},
    "fridge": {"threshold_w": 50, "min_on_sec": 60, "max_off_gap_sec": 12},
    "dishwasher": {"threshold_w": 50, "min_on_sec": 960, "max_off_gap_sec": 1800},
    "washingmachine": {"threshold_w": 20, "min_on_sec": 960, "max_off_gap_sec": 960},
}

SAMPLE_SECONDS = 8
MAX_RAW_FILL_GAP_SEC = 16  # Only short, bounded polling gaps may be held.

# A long near-standby interval between two plausible programs makes a
# washer/dishwasher reference ambiguous. Screen it rather than inventing a cut.
AMBIGUOUS_IDLE_W = 5
AMBIGUOUS_IDLE_SEC = 600

# Conservative screening, NOT the strict official Issues rule. These are
# adjustable quality-control heuristics, not universal appliance physics.
# Reject only sustained large target-vs-aggregate deficits inside the ON event.
# Keep short sensor lags. Never delete individual samples or shift signals.
FILTER_SEVERE_REFIT_MISMATCH = True
SEVERE_DEFICIT_W = 300
SEVERE_DEFICIT_FRACTION = 0.30
MIN_SEVERE_DURATION_SEC = 60
# Fridge reference cycles need observed OFF on BOTH sides, not invented zeros.
FRIDGE_MIN_OFF_CONTEXT_SAMPLES = 8
# Dishwasher references also need a complete OFF -> cycle -> OFF observation.
DISHWASHER_MIN_OFF_CONTEXT_SAMPLES = 16
# Short heating events can be entirely mismatched yet last less than 60 seconds.
# Reject when >=90% of above-threshold duration has a large deficit (same
# 300 W / 30% rule). This screens pair usability; it does not diagnose the sensor.
SHORT_APPLIANCE_MAX_BAD_FRACTION = 0.90
REFIT_POWER_COLUMNS = [f"Appliance{i}" for i in range(1, 10)]

# Bound CSV buffering so long events do not accumulate across a whole house.
WRITE_BATCH_ROWS = 100_000
PROGRESS_EVERY_SECONDS = 5


# ============================================================
# 4. Clean reference channels selected from incomplete houses
#
# REFIT numbers refer to ApplianceN columns.
# UK-DALE numbers refer to channel_N.dat files.
# REFIT H12 A4 and A3 conflict with the cleaned release's appliance list.
# Leave both out of the reference bank until the local H12 version is verified.
# ============================================================
REFERENCE_CHANNELS = {
    "refit": {
        "kettle": {4: [9], 6: [7], 7: [9], 8: [9], 13: [9], 19: [5]},
        "fridge": {
            1: [1], 4: [1, 3], 7: [1], 8: [1], 10: [4], 12: [1],
            16: [1, 2], 17: [2], 18: [1, 3], 21: [1],
        },
        "dishwasher": {1: [6], 6: [3], 7: [6], 10: [6], 13: [4], 16: [6], 18: [6], 21: [4]},
        "washingmachine": {
            1: [5], 4: [4, 5], 6: [2], 7: [5], 8: [4], 10: [5],
            13: [3], 16: [5], 17: [4], 18: [5], 19: [2], 21: [3],
        },
        "microwave": {4: [8], 6: [6], 8: [8], 10: [8], 13: [8], 17: [7], 18: [9], 19: [4]},
    },
    "ukdale": {
        "kettle": {3: [2]},
        "fridge": {},
        "dishwasher": {},
        "washingmachine": {},
        "microwave": {},
    },
}


def refit_issues(data: pd.DataFrame) -> np.ndarray:
    """Reconstruct the published sum-of-IAMs consistency flag.

    All nine channels are required, not only the five target appliances.
    Missing/nonfinite readings are additionally rejected because their sum
    cannot establish consistency; they must not silently become zeros.
    """
    power = data[REFIT_POWER_COLUMNS].to_numpy(dtype=np.float64)
    aggregate = data["Aggregate"].to_numpy(dtype=np.float64)
    return ((power.sum(axis=1) > aggregate)
            | ~np.isfinite(power).all(axis=1) | ~np.isfinite(aggregate))


def resample_power(data, columns):
    """Common UTC grid: mean in [t,t+8), bounded one-bin forward hold.

    Hold only completely empty bins bracketed by raw timestamps <=16 s apart.
    No interpolation, zero padding, or extrapolation across long outages.
    """
    times = pd.to_datetime(data["unix_time"], unit="s", utc=True)
    source = data[columns].replace([np.inf, -np.inf], np.nan).set_index(times).sort_index()
    bins = source.resample(f"{SAMPLE_SECONDS}s", origin="epoch", closed="left", label="left")
    grid = bins.mean()
    raw_seconds = source.index.as_unit("ns").asi8 // 1_000_000_000
    grid_seconds = grid.index.as_unit("ns").asi8 // 1_000_000_000
    previous = np.searchsorted(raw_seconds, grid_seconds, side="right") - 1
    following = np.searchsorted(raw_seconds, grid_seconds, side="left")
    bounded = (previous >= 0) & (following < len(raw_seconds))
    positions = np.flatnonzero(bounded)
    bounded[positions] &= (
        raw_seconds[following[positions]] - raw_seconds[previous[positions]] <= MAX_RAW_FILL_GAP_SEC
    )
    sample_observed = bins.size().gt(0).to_numpy()
    fill = ~sample_observed & bounded
    grid.loc[fill] = grid.ffill(limit=1).loc[fill]
    grid["sample_observed"] = sample_observed
    grid.index.name = "timestamp"
    return grid


def read_refit_house(house):
    data = pd.read_csv(RAW_DATA_ROOT / "REFIT" / f"House_{house}.csv",
                       usecols=["Unix", "Aggregate"] + REFIT_POWER_COLUMNS)
    data.rename(columns={"Unix": "unix_time"}, inplace=True)
    grid = resample_power(data, ["Aggregate"] + REFIT_POWER_COLUMNS).reset_index()
    grid["unix_time"] = grid["timestamp"].astype("int64") // 1_000_000_000
    grid["timestamp"] = grid["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    grid["refit_issues"] = refit_issues(grid)
    return grid.rename(columns={"Aggregate": "aggregate_power_w"})


def read_refit(house: int, channel: int) -> pd.DataFrame:
    """Read one REFIT appliance channel and its aligned aggregate."""
    appliance_column = f"Appliance{channel}"
    data = read_refit_house(house)
    return data[["timestamp", "unix_time", "aggregate_power_w", appliance_column, "refit_issues",
                 "sample_observed"]].rename(
        columns={appliance_column: "appliance_power_w"}
    ).dropna(subset=["aggregate_power_w", "appliance_power_w"]).reset_index(drop=True)


def read_ukdale_channel(path: Path, power_name: str) -> pd.DataFrame:
    """Read and place one UK-DALE channel on the common 8-second grid."""
    data = pd.read_csv(path, sep=r"\s+", names=["unix_time", power_name])
    return resample_power(data, [power_name])


def read_ukdale(house: int, channel: int) -> pd.DataFrame:
    """Read one UK-DALE appliance channel and align it with aggregate power."""
    house_dir = RAW_DATA_ROOT / "UK_DALE" / f"house_{house}"
    aggregate = read_ukdale_channel(
        house_dir / "channel_1.dat", "aggregate_power_w"
    )[["aggregate_power_w"]]
    appliance = read_ukdale_channel(
        house_dir / f"channel_{channel}.dat", "appliance_power_w"
    )

    data = aggregate.join(appliance, how="inner").dropna().reset_index()
    data["unix_time"] = data["timestamp"].astype("int64") // 1_000_000_000
    data["timestamp"] = data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return data.reset_index(drop=True)


def fill_short_off_gaps(is_on: np.ndarray, unix_time: np.ndarray, max_off_gap: int) -> np.ndarray:
    """Join parts of one operation separated by a short below-threshold gap."""
    state = is_on.copy()
    if max_off_gap == 0 or not state.any():
        return state

    changes = np.diff(np.r_[0, state.astype(np.int8), 0])
    on_starts = np.flatnonzero(changes == 1)
    on_ends = np.flatnonzero(changes == -1) - 1

    gap_starts = on_ends[:-1] + 1
    gap_ends = on_starts[1:]
    # Measure from the first OFF timestamp to the next ON timestamp.
    short_gaps = unix_time[gap_ends] - unix_time[gap_starts] <= max_off_gap
    for gap_start, gap_end in zip(gap_starts[short_gaps], gap_ends[short_gaps]):
        state[gap_start:gap_end] = True
    return state


def sample_end_times(unix_time: np.ndarray) -> np.ndarray:
    """Each retained grid bin represents [timestamp, timestamp + 8 seconds)."""
    return unix_time + SAMPLE_SECONDS


def remove_short_on_events(is_on: np.ndarray, unix_time: np.ndarray, min_on_sec: int) -> np.ndarray:
    """Remove segments shorter than the configured physical duration."""
    state = is_on.copy()
    changes = np.diff(np.r_[0, state.astype(np.int8), 0])
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    right_edges = sample_end_times(unix_time)
    short_events = right_edges[ends - 1] - unix_time[starts] < min_on_sec
    for start, end_exclusive in zip(starts[short_events], ends[short_events]):
        state[start:end_exclusive] = False
    return state


def find_on_events(
    power_w: np.ndarray,
    unix_time: np.ndarray,
    appliance: str,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return events without allowing one event to cross a recording gap."""
    rule = EVENT_RULES[appliance]
    final_is_on = np.zeros(len(power_w), dtype=bool)
    events = []

    # Timestamp discontinuities start a new recording sequence.
    dt = np.diff(unix_time)
    sequence_breaks = np.flatnonzero(dt != SAMPLE_SECONDS) + 1
    boundaries = np.r_[0, sequence_breaks, len(power_w)]

    for sequence_start, sequence_end in zip(boundaries[:-1], boundaries[1:]):
        state = power_w[sequence_start:sequence_end] >= rule["threshold_w"]
        times = unix_time[sequence_start:sequence_end]
        state = fill_short_off_gaps(state, times, rule["max_off_gap_sec"])
        state = remove_short_on_events(state, times, rule["min_on_sec"])
        final_is_on[sequence_start:sequence_end] = state

        changes = np.diff(np.r_[0, state.astype(np.int8), 0])
        starts = np.flatnonzero(changes == 1) + sequence_start
        ends = np.flatnonzero(changes == -1) - 1 + sequence_start
        events.extend(zip(starts.tolist(), ends.tolist()))

    return final_is_on, events


def available_off_context(
    is_on: np.ndarray,
    on_start: int,
    on_end: int,
    max_pre: int,
    max_post: int,
    unix_time: np.ndarray,
) -> tuple[int, int]:
    """Count consecutive OFF samples without crossing an event or data gap."""
    pre = 0
    while pre < max_pre and on_start - pre - 1 >= 0:
        if is_on[on_start - pre - 1]:
            break
        if unix_time[on_start - pre] - unix_time[on_start - pre - 1] != SAMPLE_SECONDS:
            break
        pre += 1

    post = 0
    while post < max_post and on_end + post + 1 < len(is_on):
        if is_on[on_end + post + 1]:
            break
        if unix_time[on_end + post + 1] - unix_time[on_end + post] != SAMPLE_SECONDS:
            break
        post += 1
    return pre, post


def has_ambiguous_idle_gap(power_w, above_threshold, min_cycle_sec):
    """Check for an idle gap with a plausible program-length span on each side."""
    idle = power_w <= AMBIGUOUS_IDLE_W
    edges = np.diff(np.r_[0, idle.astype(np.int8), 0])
    gap_starts = np.flatnonzero(edges == 1)
    gap_ends = np.flatnonzero(edges == -1)
    active = np.flatnonzero(above_threshold)
    for start, end in zip(gap_starts, gap_ends):
        if (end - start) * SAMPLE_SECONDS < AMBIGUOUS_IDLE_SEC:
            continue
        left = np.searchsorted(active, start)
        right = np.searchsorted(active, end)
        if left == 0 or right == len(active):
            continue
        left_sec = (active[left - 1] - active[0] + 1) * SAMPLE_SECONDS
        right_sec = (active[-1] - active[right] + 1) * SAMPLE_SECONDS
        if min(left_sec, right_sec) >= min_cycle_sec:
            return True
    return False


def longest_severe_mismatch(power_w, aggregate_w, unix_time, right_edges):
    """Longest continuous large deficit, measured in seconds, within an event.

    Other household loads need not resemble the appliance waveform. Only the
    target substantially exceeding the aggregate is screened here. This is a
    quality flag, not proof of the underlying sensor fault or its cause.
    """
    deficit = power_w - aggregate_w
    bad = ((deficit > SEVERE_DEFICIT_W)
           & (deficit > SEVERE_DEFICIT_FRACTION * power_w))
    changes = np.diff(np.r_[0, bad.astype(np.int8), 0])
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    if not len(starts):
        return 0
    return int((right_edges[ends] - unix_time[starts]).max())


def severe_active_fraction(power_w, aggregate_w, unix_time, right_edges, threshold):
    """Time fraction of measured active power severely unsupported by aggregate.

    Actual irregular sample widths are used. Joined OFF phases are excluded.
    No alignment search, time shifting, or waveform changes are performed.
    """
    widths = right_edges - unix_time
    active = power_w >= threshold
    deficit = power_w - aggregate_w
    bad = active & (deficit > SEVERE_DEFICIT_W) & (deficit > SEVERE_DEFICIT_FRACTION * power_w)
    active_seconds = widths[active].sum()
    return float(widths[bad].sum() / active_seconds) if active_seconds else 0.0


def save_source_events(
    data: pd.DataFrame,
    dataset: str,
    appliance: str,
    house: int,
    channel: int,
    first_sample_row: int,
) -> tuple[list[dict], int]:
    """Append events from one source channel to its appliance CSV."""
    power_w = data["appliance_power_w"].to_numpy(dtype=np.float64)
    unix_time = data["unix_time"].to_numpy(dtype=np.int64)
    sample_observed = data["sample_observed"].to_numpy(dtype=bool)
    print("  Detecting ON events...", flush=True)
    is_on, events = find_on_events(power_w, unix_time, appliance)
    threshold = EVENT_RULES[appliance]["threshold_w"]
    above_threshold = power_w >= threshold
    # Invalid readings also stop OFF context; they are not evidence of OFF.
    context_blocked = above_threshold | ~np.isfinite(power_w)
    right_edges = sample_end_times(unix_time)
    max_pre, max_post = OFF_CONTEXT_SAMPLES[appliance]
    minimum_context = {"fridge": FRIDGE_MIN_OFF_CONTEXT_SAMPLES,
                       "dishwasher": DISHWASHER_MIN_OFF_CONTEXT_SAMPLES}.get(appliance, 1)
    off_context = {}
    complete = []
    for start, end in events:
        pre, post = available_off_context(
            context_blocked, start, end, max_pre, max_post, unix_time
        )
        # A forward-held bin can preserve continuity, but cannot establish OFF.
        if (pre >= minimum_context and post >= minimum_context
                and sample_observed[start - pre:start].any()
                and sample_observed[end + 1:end + 1 + post].any()):
            complete.append((start, end))
            off_context[start] = (pre, post)
    print(f"  Completeness: {len(events) - len(complete):,} references excluded", flush=True)
    events = complete

    if appliance in ("dishwasher", "washingmachine"):
        min_cycle_sec = EVENT_RULES[appliance]["min_on_sec"]
        retained = []
        for start, end in events:
            window = slice(start, end + 1)
            if not has_ambiguous_idle_gap(power_w[window], above_threshold[window], min_cycle_sec):
                retained.append((start, end))
        print(f"  Ambiguous cycle boundaries: {len(events) - len(retained):,} references excluded",
              flush=True)
        events = retained
    mismatch_seconds = {}
    mismatch_fraction = {}
    if dataset == "refit":
        aggregate_w = data["aggregate_power_w"].to_numpy(dtype=np.float64)
        for start, end in events:
            window = slice(start, end + 1)
            mismatch_seconds[start] = longest_severe_mismatch(
                power_w[window], aggregate_w[window], unix_time[window], right_edges[window]
            )
            if appliance in ("kettle", "microwave"):
                mismatch_fraction[start] = severe_active_fraction(
                    power_w[window], aggregate_w[window], unix_time[window], right_edges[window], threshold
                )
        if FILTER_SEVERE_REFIT_MISMATCH:
            retained = [(start, end) for start, end in events
                        if mismatch_seconds[start] < MIN_SEVERE_DURATION_SEC
                        and mismatch_fraction.get(start, 0) < SHORT_APPLIANCE_MAX_BAD_FRACTION]
            print(f"  Severe mismatch screening: {len(events):,} candidates | "
                  f"{len(events) - len(retained):,} rejected | {len(retained):,} retained",
                  flush=True)
            events = retained
    print(f"  Found {len(events):,} events; writing CSV...", flush=True)
    appliance_dir = OUTPUT_ROOT / appliance
    appliance_dir.mkdir(parents=True, exist_ok=True)
    events_file = appliance_dir / "events.csv"
    channel_label = f"A{channel}" if dataset == "refit" else f"C{channel}"
    manifest_rows = []
    next_sample_row = first_sample_row
    sample_columns = ["timestamp", "unix_time", "appliance_power_w", "aggregate_power_w"]
    source_columns = {name: data[name].to_numpy() for name in sample_columns}
    buffered_events = []
    buffered_rows = 0
    last_progress = perf_counter()

    write_header = not events_file.exists()
    with events_file.open("a", encoding="utf-8", newline="") as stream:
        for event_number, (on_start, on_end) in enumerate(events, start=1):
            off_pre, off_post = off_context[on_start]
            ref_start = on_start - off_pre
            ref_end = on_end + off_post
            event_id = (
                f"{dataset}_{appliance}_h{house:02d}_"
                f"{channel_label.lower()}_{event_number:06d}"
            )

            on_power = power_w[on_start:on_end + 1]
            duration_sec = int(right_edges[on_end] - unix_time[on_start])
            start_censored = on_start == 0 or (
                unix_time[on_start] - unix_time[on_start - 1] != SAMPLE_SECONDS
            )
            end_censored = on_end == len(unix_time) - 1 or (
                unix_time[on_end + 1] - unix_time[on_end] != SAMPLE_SECONDS
            )

            metadata = {
                "event_id": event_id,
                "dataset": dataset,
                "house": house,
                "channel": channel_label,
                "on_start_idx": on_start,
                "on_end_idx": on_end,
                "ref_start_idx": ref_start,
                "ref_end_idx": ref_end,
                "off_pre_samples": off_pre,
                "off_post_samples": off_post,
                "duration_sec": duration_sec,
                "peak_power_w": float(on_power.max()),
                "mean_on_power_w": float(on_power.mean()),
                "duration_censored": bool(start_censored or end_censored),
                "on_end_unix": int(right_edges[on_end]),
                "sample_period_sec": SAMPLE_SECONDS,
            }

            # Build columns together instead of repeatedly inserting columns.
            event_data = pd.DataFrame({
                "event_id": event_id,
                "dataset": dataset,
                "house": house,
                "channel": channel_label,
                "event_on": is_on[ref_start:ref_end + 1].astype(np.int8),
                "above_threshold": above_threshold[ref_start:ref_end + 1].astype(np.int8),
                "source_idx": np.arange(ref_start, ref_end + 1),
                **{name: values[ref_start:ref_end + 1]
                   for name, values in source_columns.items()},
            })
            buffered_events.append(event_data)
            buffered_rows += len(event_data)
            if buffered_rows >= WRITE_BATCH_ROWS or event_number == len(events):
                pd.concat(buffered_events, ignore_index=True).to_csv(
                    stream, index=False, header=write_header
                )
                write_header = False
                buffered_events.clear()
                buffered_rows = 0

            if perf_counter() - last_progress >= PROGRESS_EVERY_SECONDS or event_number == len(events):
                print(f"  Events {event_number:,}/{len(events):,} "
                      f"({event_number / len(events):.0%})", flush=True)
                last_progress = perf_counter()
            sample_end_row = next_sample_row + len(event_data) - 1
            manifest_rows.append(
                {
                    **metadata,
                    "samples": len(event_data),
                    "sample_start_row": next_sample_row,
                    "sample_end_row": sample_end_row,
                    "file": str(events_file.relative_to(OUTPUT_ROOT)),
                }
            )
            next_sample_row = sample_end_row + 1

    return manifest_rows, next_sample_row


def main() -> None:
    """Extract all configured REFIT and UK-DALE reference events."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    next_sample_row = {appliance: 0 for appliance in OFF_CONTEXT_SAMPLES}
    run_start = perf_counter()
    tasks = {}
    for dataset, appliance_map in REFERENCE_CHANNELS.items():
        for appliance, house_map in appliance_map.items():
            for house, channels in house_map.items():
                for channel in channels:
                    tasks.setdefault((dataset, house), []).append((appliance, channel))
    total_sources = sum(len(sources) for sources in tasks.values())
    completed_sources = 0
    print(f"Extracting {total_sources} channels from {len(tasks)} houses", flush=True)
    print(f"Output: {OUTPUT_ROOT}", flush=True)

    # Start a new deterministic dataset when the script is run again.
    for appliance in OFF_CONTEXT_SAMPLES:
        appliance_dir = OUTPUT_ROOT / appliance
        appliance_dir.mkdir(parents=True, exist_ok=True)
        events_file = appliance_dir / "events.csv"
        events_file.unlink(missing_ok=True)

    # Process one house at a time: parse each large REFIT CSV only once.
    for dataset, house in sorted(tasks):
        sources = tasks[(dataset, house)]
        print(f"\nLoading {dataset} house {house:02d}...", flush=True)
        load_start = perf_counter()
        if dataset == "refit":
            house_data = read_refit_house(house)
            print(f"  Loaded {len(house_data):,} rows in "
                  f"{perf_counter() - load_start:.1f}s", flush=True)

        for appliance, channel in sources:
            source_start = perf_counter()
            print(f"[{completed_sources + 1}/{total_sources}] {dataset} | "
                  f"house {house:02d} | {appliance} | channel {channel}", flush=True)
            if dataset == "refit":
                data = house_data[[
                    "timestamp", "unix_time", "aggregate_power_w", f"Appliance{channel}",
                    "refit_issues", "sample_observed",
                ]].rename(columns={f"Appliance{channel}": "appliance_power_w"}).dropna(
                    subset=["aggregate_power_w", "appliance_power_w"]
                ).reset_index(drop=True)
            else:
                data = read_ukdale(house, channel)
                print(f"  Loaded {len(data):,} aligned rows in "
                      f"{perf_counter() - source_start:.1f}s", flush=True)

            rows, next_sample_row[appliance] = save_source_events(
                data, dataset, appliance, house, channel, next_sample_row[appliance]
            )
            manifest_rows.extend(rows)
            completed_sources += 1
            print(f"  Done: {len(rows):,} events | "
                  f"channel {perf_counter() - source_start:.1f}s | "
                  f"total elapsed {(perf_counter() - run_start) / 60:.1f} min", flush=True)
        del data
        if dataset == "refit":
            del house_data

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUTPUT_ROOT / "manifest.csv", index=False)
    print(f"\nSaved {len(manifest):,} events")
    print(f"Manifest: {OUTPUT_ROOT / 'manifest.csv'}")


if __name__ == "__main__":
    main()

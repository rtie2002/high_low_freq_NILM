"""Extract appliance ON events with short OFF context from REFIT and UK-DALE.

Run this file directly. Samples from every detected event are appended to:

    dataset/appliance_waveform reference/<appliance>/events.csv

The dataset, house, channel, and event ID columns identify the source of every
sample. The original appliance and aggregate powers are kept in watts. A
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
# 3. ON-event rules at the common 8-second resolution
# ============================================================
EVENT_RULES = {
    "kettle": {"threshold_w": 200, "min_on_samples": 2, "max_off_gap": 0},
    "microwave": {"threshold_w": 200, "min_on_samples": 1, "max_off_gap": 4},
    "fridge": {"threshold_w": 50, "min_on_samples": 8, "max_off_gap": 2},
    "dishwasher": {"threshold_w": 50, "min_on_samples": 120, "max_off_gap": 225},
    "washingmachine": {"threshold_w": 20, "min_on_samples": 120, "max_off_gap": 120},
}

SAMPLE_SECONDS = 8

# Bound CSV buffering so long events do not accumulate across a whole house.
WRITE_BATCH_ROWS = 100_000
PROGRESS_EVERY_SECONDS = 5


# ============================================================
# 4. Clean reference channels selected from incomplete houses
#
# REFIT numbers refer to ApplianceN columns.
# UK-DALE numbers refer to channel_N.dat files.
# ============================================================
REFERENCE_CHANNELS = {
    "refit": {
        "kettle": {4: [9], 6: [7], 7: [9], 8: [9], 12: [4], 13: [9], 19: [5]},
        "fridge": {
            1: [1], 4: [1, 3], 7: [1], 8: [1], 10: [4], 12: [1],
            16: [1, 2], 17: [2], 18: [1, 3], 21: [1],
        },
        "dishwasher": {1: [6], 6: [3], 7: [6], 10: [6], 13: [4], 16: [6], 18: [6], 21: [4]},
        "washingmachine": {
            1: [5], 4: [4, 5], 6: [2], 7: [5], 8: [4], 10: [5],
            13: [3], 16: [5], 17: [4], 18: [5], 19: [2], 21: [3],
        },
        "microwave": {4: [8], 6: [6], 8: [8], 10: [8], 12: [3], 13: [8], 17: [7], 18: [9], 19: [4]},
    },
    "ukdale": {
        "kettle": {3: [2]},
        "fridge": {},
        "dishwasher": {},
        "washingmachine": {},
        "microwave": {},
    },
}


def read_refit(house: int, channel: int) -> pd.DataFrame:
    """Read one REFIT appliance channel and its aligned aggregate."""
    source = RAW_DATA_ROOT / "REFIT" / f"House_{house}.csv"
    appliance_column = f"Appliance{channel}"
    data = pd.read_csv(
        source,
        usecols=["Time", "Unix", "Aggregate", appliance_column],
    )
    return data.rename(
        columns={
            "Time": "timestamp",
            "Unix": "unix_time",
            "Aggregate": "aggregate_power_w",
            appliance_column: "appliance_power_w",
        }
    ).reset_index(drop=True)


def read_ukdale_channel(path: Path, power_name: str) -> pd.DataFrame:
    """Read and place one UK-DALE channel on the common 8-second grid."""
    data = pd.read_csv(path, sep=r"\s+", names=["unix_time", power_name])
    data["timestamp"] = pd.to_datetime(data["unix_time"], unit="s", utc=True)
    data = data.set_index("timestamp")[[power_name]]
    return data.resample(f"{SAMPLE_SECONDS}s").mean().interpolate(limit=2)


def read_ukdale(house: int, channel: int) -> pd.DataFrame:
    """Read one UK-DALE appliance channel and align it with aggregate power."""
    house_dir = RAW_DATA_ROOT / "UK_DALE" / f"house_{house}"
    aggregate = read_ukdale_channel(
        house_dir / "channel_1.dat", "aggregate_power_w"
    )
    appliance = read_ukdale_channel(
        house_dir / f"channel_{channel}.dat", "appliance_power_w"
    )

    data = aggregate.join(appliance, how="inner").dropna().reset_index()
    data["unix_time"] = data["timestamp"].astype("int64") // 1_000_000_000
    data["timestamp"] = data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return data.reset_index(drop=True)


def fill_short_off_gaps(is_on: np.ndarray, max_off_gap: int) -> np.ndarray:
    """Join parts of one operation separated by a short below-threshold gap."""
    state = is_on.copy()
    if max_off_gap == 0 or not state.any():
        return state

    changes = np.diff(np.r_[0, state.astype(np.int8), 0])
    on_starts = np.flatnonzero(changes == 1)
    on_ends = np.flatnonzero(changes == -1) - 1

    gap_starts = on_ends[:-1] + 1
    gap_ends = on_starts[1:]
    short_gaps = gap_ends - gap_starts <= max_off_gap
    for gap_start, gap_end in zip(gap_starts[short_gaps], gap_ends[short_gaps]):
        state[gap_start:gap_end] = True
    return state


def remove_short_on_events(is_on: np.ndarray, min_on_samples: int) -> np.ndarray:
    """Remove threshold crossings that are too short to be a real operation."""
    state = is_on.copy()
    changes = np.diff(np.r_[0, state.astype(np.int8), 0])
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    short_events = ends - starts < min_on_samples
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

    # A gap larger than two expected samples starts a new recording sequence.
    sequence_breaks = np.flatnonzero(np.diff(unix_time) > 2 * SAMPLE_SECONDS) + 1
    boundaries = np.r_[0, sequence_breaks, len(power_w)]

    for sequence_start, sequence_end in zip(boundaries[:-1], boundaries[1:]):
        state = power_w[sequence_start:sequence_end] >= rule["threshold_w"]
        state = fill_short_off_gaps(state, rule["max_off_gap"])
        state = remove_short_on_events(state, rule["min_on_samples"])
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
        if unix_time[on_start - pre] - unix_time[on_start - pre - 1] > 2 * SAMPLE_SECONDS:
            break
        pre += 1

    post = 0
    while post < max_post and on_end + post + 1 < len(is_on):
        if is_on[on_end + post + 1]:
            break
        if unix_time[on_end + post + 1] - unix_time[on_end + post] > 2 * SAMPLE_SECONDS:
            break
        post += 1
    return pre, post


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
    print("  Detecting ON events...", flush=True)
    is_on, events = find_on_events(power_w, unix_time, appliance)
    print(f"  Found {len(events):,} events; writing CSV...", flush=True)
    max_pre, max_post = OFF_CONTEXT_SAMPLES[appliance]

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
            off_pre, off_post = available_off_context(
                is_on, on_start, on_end, max_pre, max_post, unix_time
            )
            ref_start = on_start - off_pre
            ref_end = on_end + off_post
            event_id = (
                f"{dataset}_{appliance}_h{house:02d}_"
                f"{channel_label.lower()}_{event_number:06d}"
            )

            on_power = power_w[on_start:on_end + 1]
            duration_sec = int(unix_time[on_end] - unix_time[on_start] + SAMPLE_SECONDS)

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
            }

            # Build columns together instead of repeatedly inserting columns.
            event_data = pd.DataFrame({
                **metadata,
                "is_on": is_on[ref_start:ref_end + 1].astype(np.int8),
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
            columns = ["Time", "Unix", "Aggregate"] + sorted({
                f"Appliance{channel}" for _, channel in sources
            })
            house_data = pd.read_csv(
                RAW_DATA_ROOT / "REFIT" / f"House_{house}.csv", usecols=columns
            ).rename(columns={
                "Time": "timestamp", "Unix": "unix_time",
                "Aggregate": "aggregate_power_w",
            })
            print(f"  Loaded {len(house_data):,} rows in "
                  f"{perf_counter() - load_start:.1f}s", flush=True)

        for appliance, channel in sources:
            source_start = perf_counter()
            print(f"[{completed_sources + 1}/{total_sources}] {dataset} | "
                  f"house {house:02d} | {appliance} | channel {channel}", flush=True)
            if dataset == "refit":
                data = house_data[[
                    "timestamp", "unix_time", "aggregate_power_w", f"Appliance{channel}"
                ]].rename(columns={f"Appliance{channel}": "appliance_power_w"})
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

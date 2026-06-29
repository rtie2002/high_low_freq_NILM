"""Build MATNILM Table I (S2) UK-DALE train / val / test CSV splits.

Preprocessing follows MATNILM Section IV-A / SGN [18] (NILMTK-style):
  - 6 s fixed grid, resample mean, aggregate + appliances aligned
  - Drop segments with 20 consecutive missing or 1200 consecutive unchanged values
  - Backward-fill remaining missing values
  - Keep only segments longer than 1 hour
  - Store raw Watts in CSV
  - ON labels: simple15 (MATNILM/SGN 15 W) or algorithm1 (ukdale.yaml Kelly rules)

Default split (MATNILM S2 cross-house):
  Train : H1, 2017-04-23 (1 day)
  Val   : H1, 2017-04-25 (1 day)
  Test  : H2, 2013-05-21 -> 2013-06-03 (2 weeks; all 5 appliances online in UKDALE2017)
          Override with --test_start/--test_end. Apr 16-30 lacks H2 ch 12-15 data.

SGN training applies X/612, Y/612. ON labels:
  simple15   -> threshold_watts (Y > 15 W) at SGN load time
  algorithm1 -> read *_on columns from CSV (on_label_source: csv)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ukdale_processing import apply_algorithm1_labeling, resolve_appliance_setting  # noqa: E402
from ukdale_processing_multi_appliance import (  # noqa: E402
    load_appliance,
    load_mains,
    resolve_paths,
    selected_appliances,
)

LABEL_MODES = ("simple15", "algorithm1", "both")

SAMPLE_SECONDS = 6
MIN_SEGMENT_SAMPLES = 3600 // SAMPLE_SECONDS  # 1 hour @ 6 s
MAX_MISSING_RUN = 20
MAX_UNCHANGED_RUN = 1200
ON_THRESHOLD_WATTS = 15.0

DEFAULT_APPLIANCES = ["kettle", "fridge", "microwave", "dishwasher", "washingmachine"]

SPLITS = {
    "train": {"house": 1, "start": "2017-04-23 00:00:00", "end": "2017-04-24 00:00:00"},
    "val": {"house": 1, "start": "2017-04-25 00:00:00", "end": "2017-04-26 00:00:00"},
    # H2 dishwasher/fridge/microwave/washingmachine (ch 12-15) start 2013-05-20 in UKDALE2017.
    # Apr 16-30 only has mains + kettle; use first 2 weeks with all five appliances for test.
    "test": {"house": 2, "start": "2013-05-21 00:00:00", "end": "2013-06-04 00:00:00"},
}


def consecutive_true_runs(mask: np.ndarray, min_length: int) -> np.ndarray:
    """Mark positions inside a True-run of length >= min_length."""
    out = np.zeros(len(mask), dtype=bool)
    if min_length <= 0 or len(mask) == 0:
        return out
    start = 0
    while start < len(mask):
        if not mask[start]:
            start += 1
            continue
        end = start + 1
        while end < len(mask) and mask[end]:
            end += 1
        if end - start >= min_length:
            out[start:end] = True
        start = end
    return out


def stuck_value_mask(values: np.ndarray, min_length: int) -> np.ndarray:
    """Mark rows inside a window of min_length consecutive identical finite values."""
    out = np.zeros(len(values), dtype=bool)
    if len(values) < min_length:
        return out
    for start in range(len(values) - min_length + 1):
        window = values[start : start + min_length]
        if np.all(np.isfinite(window)) and np.ptp(window) == 0.0:
            out[start : start + min_length] = True
    return out


def bad_row_mask(df: pd.DataFrame, signal_cols: list[str]) -> np.ndarray:
    """Rows to exclude before segment splitting (any signal triggers exclusion)."""
    bad = np.zeros(len(df), dtype=bool)
    for col in signal_cols:
        values = df[col].to_numpy(dtype=np.float64)
        bad |= consecutive_true_runs(np.isnan(values), MAX_MISSING_RUN)
        bad |= stuck_value_mask(values, MAX_UNCHANGED_RUN)
    return bad


def split_valid_segments(df: pd.DataFrame, bad: np.ndarray) -> list[pd.DataFrame]:
    """Split timeline on bad rows; keep segments with more than 1 hour of samples."""
    segments: list[pd.DataFrame] = []
    start: int | None = None
    for index, is_bad in enumerate(bad):
        if is_bad:
            if start is not None and index - start >= MIN_SEGMENT_SAMPLES:
                segments.append(df.iloc[start:index].copy())
            start = None
        elif start is None:
            start = index
    if start is not None and len(df) - start >= MIN_SEGMENT_SAMPLES:
        segments.append(df.iloc[start:].copy())
    return segments


def backward_fill_signals(df: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    """Backward-fill missing values, then forward-fill any leading gaps at segment start."""
    filled = df.copy()
    for col in signal_cols:
        series = filled[col]
        series = series.bfill()
        series = series.ffill()
        filled[col] = series
    return filled


def date_bounds(start: str, end: str, tz: str) -> tuple[float, float]:
    start_ts = pd.to_datetime(start).tz_localize(tz).timestamp()
    end_ts = pd.to_datetime(end).tz_localize(tz).timestamp()
    return start_ts, end_ts


def load_appliance_aligned(
    data_dir: str,
    house: int,
    appliance: str,
    channel_id: int,
    start_ts: float,
    end_ts: float,
    tz: str,
    sample_period: str,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Load appliance power aligned to index; NaN when channel has no rows in range."""
    try:
        app_resampled = load_appliance(
            data_dir,
            house,
            appliance,
            channel_id,
            start_ts,
            end_ts,
            tz,
            sample_period,
        )
        return app_resampled[appliance].reindex(index)
    except (ValueError, FileNotFoundError) as exc:
        print(f"      {appliance:<15} channel={channel_id:<3} no data in range ({exc})")
        return pd.Series(np.nan, index=index, name=appliance)


def assign_on_labels(
    power: np.ndarray,
    appliance: str,
    house: int,
    config: dict,
    label_mode: str,
) -> np.ndarray:
    if label_mode == "simple15":
        return (power > ON_THRESHOLD_WATTS).astype(np.int8)
    app_cfg = config["appliances"][appliance]
    algorithm_cfg = config.get("algorithm1", {})
    labels = apply_algorithm1_labeling(
        power.copy(),
        x_threshold=resolve_appliance_setting(app_cfg, "on_power_threshold", house, 50),
        l_window=algorithm_cfg.get("window_length", 0),
        x_noise=algorithm_cfg.get("x_noise", 0),
        remove_spikes=algorithm_cfg.get("remove_spikes", True),
        spike_window=algorithm_cfg.get("spike_window", 5),
        spike_threshold=algorithm_cfg.get("spike_threshold", 3.0),
        background_threshold=algorithm_cfg.get("background_threshold", 50),
        min_off_duration=resolve_appliance_setting(app_cfg, "min_off_duration", house, 1),
        min_on_duration=resolve_appliance_setting(app_cfg, "min_on_duration", house, 1),
    )
    return labels.astype(np.int8)


def count_episodes(on: np.ndarray) -> int:
    series = pd.Series(on > 0)
    return int((series.diff().fillna(series.iloc[0]) == 1).sum())


def build_house_frame(
    data_dir: str,
    house: int,
    appliances: list[str],
    config: dict,
    start_ts: float,
    end_ts: float,
    tz: str,
    sample_period: str,
) -> pd.DataFrame:
    """Load mains + appliances, align, apply MATNILM gap filtering, return export frame."""
    mains = load_mains(data_dir, house, start_ts, end_ts, tz, sample_period)
    combined = mains.copy()

    for app in appliances:
        channel_id = config["appliances"][app].get("channel_map", {}).get(house)
        if channel_id is None:
            raise ValueError(f"No channel_map for {app} in house {house}")
        combined[app] = load_appliance_aligned(
            data_dir,
            house,
            app,
            channel_id,
            start_ts,
            end_ts,
            tz,
            sample_period,
            combined.index,
        )
        valid = int(combined[app].notna().sum())
        print(f"      {app:<15} channel={channel_id:<3} aligned rows={valid:,}")

    combined = combined.sort_index()
    signal_cols = ["aggregate", *appliances]
    combined[signal_cols] = combined[signal_cols].astype(np.float64)

    bad = bad_row_mask(combined, signal_cols)
    segments = split_valid_segments(combined, bad)
    if not segments:
        raise ValueError(
            f"No valid segments (>1 h) for house {house} after MATNILM gap filtering."
        )

    processed_parts: list[pd.DataFrame] = []
    for segment in segments:
        filled = backward_fill_signals(segment, signal_cols)
        filled = filled.dropna(subset=signal_cols)
        if len(filled) < MIN_SEGMENT_SAMPLES:
            continue

        aggregate = filled["aggregate"].to_numpy(dtype=np.float32)
        export = pd.DataFrame({"aggregate": aggregate}, index=filled.index)
        for app in appliances:
            power = np.minimum(
                filled[app].to_numpy(dtype=np.float32),
                aggregate,
            )
            power = np.clip(power, 0.0, None)
            export[f"{app}_power"] = power
        processed_parts.append(export)

    if not processed_parts:
        raise ValueError(f"All segments dropped for house {house}.")

    merged = pd.concat(processed_parts).sort_index()
    merged = merged.reset_index(names="time")
    merged["readable_time"] = merged["time"].dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M:%S")
    merged.insert(1, "house", house)
    merged.drop(columns=["time"], inplace=True)

    power_cols = [f"{app}_power" for app in appliances]
    return merged[["readable_time", "house", "aggregate", *power_cols]]


def summarize(name: str, df: pd.DataFrame, *, label_mode: str) -> None:
    print(f"\n{name}  [{label_mode}]")
    print(f"  rows   : {len(df):,}")
    print(f"  house  : {sorted(df['house'].unique())}")
    print(f"  time   : {df['readable_time'].min()} -> {df['readable_time'].max()}")
    for col in [c for c in df.columns if c.endswith("_on")]:
        on = df[col].to_numpy()
        rate = float(on.mean()) * 100.0
        episodes = count_episodes(on)
        print(f"  {col:<22} ON rate {rate:5.2f}%  episodes {episodes:>4}")


def output_paths(output_dir: Path, label_mode: str) -> dict[str, Path]:
    suffix = "matnilm" if label_mode == "simple15" else "matnilm_house_threshold"
    return {
        "train": output_dir / f"multi_appliance_training_{suffix}.csv",
        "val": output_dir / f"multi_appliance_validating_{suffix}.csv",
        "test": output_dir / f"multi_appliance_testing_{suffix}.csv",
    }


def label_mode_description(label_mode: str) -> str:
    if label_mode == "simple15":
        return f"power > {ON_THRESHOLD_WATTS:g} W (MATNILM/SGN paper rule)"
    return "ukdale.yaml per-house thresholds + Algorithm 1 min-on/min-off"


def main() -> None:
    default_config = SCRIPT_DIR.parent / "config" / "preprocess" / "ukdale.yaml"
    default_data = SCRIPT_DIR / "UK_DALE" / "UKDALE2017"
    default_out = SCRIPT_DIR.parent / "NILM_model" / "data"

    parser = argparse.ArgumentParser(description="Build MATNILM-style UK-DALE SGN CSV splits.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=default_data,
        help="UK-DALE root with house_1/, house_2/, ... (default: UK_DALE/UKDALE2017).",
    )
    parser.add_argument("--output_dir", type=Path, default=default_out)
    parser.add_argument(
        "--appliances",
        type=str,
        default=",".join(DEFAULT_APPLIANCES),
        help="Comma-separated appliance list.",
    )
    parser.add_argument(
        "--test_start",
        type=str,
        default=SPLITS["test"]["start"],
        help="Test window start (local Europe/London). Default: first 2 weeks with all H2 appliances.",
    )
    parser.add_argument(
        "--test_end",
        type=str,
        default=SPLITS["test"]["end"],
        help="Test window end (exclusive). Apr 2013 lacks H2 ch 12-15; see script docstring.",
    )
    parser.add_argument(
        "--label_mode",
        choices=LABEL_MODES,
        default="both",
        help="simple15=MATNILM 15W; algorithm1=ukdale.yaml house thresholds; both=write both CSV sets.",
    )
    args = parser.parse_args()

    splits = dict(SPLITS)
    splits["test"] = {"house": 2, "start": args.test_start, "end": args.test_end}

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths, _ = resolve_paths(config, str(args.config))
    data_dir = str(args.data_dir.resolve())
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Missing UK-DALE data directory: {data_dir}")

    global_params = config["global_params"]
    tz = global_params.get("timezone", "Europe/London")
    sample_seconds = int(global_params.get("sample_seconds", SAMPLE_SECONDS))
    sample_period = f"{sample_seconds}s"
    appliances = selected_appliances(config, args.appliances)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_modes = ["simple15", "algorithm1"] if args.label_mode == "both" else [args.label_mode]

    print("=" * 72)
    print("MATNILM UK-DALE SPLIT BUILDER (Section IV-A / SGN [18])")
    print("=" * 72)
    print(f"data_dir    : {data_dir}")
    print(f"output_dir  : {args.output_dir}")
    print(f"appliances  : {appliances}")
    print(f"label_mode  : {args.label_mode}")
    print("train scale : applied at SGN load time (X/612, Y/612)")

    # Load and gap-filter once; relabel per mode (same power columns).
    base_frames: dict[str, pd.DataFrame] = {}
    for split_name, spec in splits.items():
        start_ts, end_ts = date_bounds(spec["start"], spec["end"], tz)
        print(f"\n[{split_name}] house {spec['house']}  {spec['start']} -> {spec['end']}")
        base_frames[split_name] = build_house_frame(
            data_dir,
            spec["house"],
            appliances,
            config,
            start_ts,
            end_ts,
            tz,
            sample_period,
        )

    for label_mode in label_modes:
        print("\n" + "=" * 72)
        print(f"LABEL MODE: {label_mode.upper()} — {label_mode_description(label_mode)}")
        print("=" * 72)
        outputs: dict[str, pd.DataFrame] = {}
        for split_name, base in base_frames.items():
            house = int(base["house"].iloc[0])
            frame = base.copy()
            for app in appliances:
                power = frame[f"{app}_power"].to_numpy(dtype=np.float32)
                frame[f"{app}_on"] = assign_on_labels(power, app, house, config, label_mode)
            power_cols = [f"{app}_power" for app in appliances]
            on_cols = [f"{app}_on" for app in appliances]
            outputs[split_name] = frame[
                ["readable_time", "house", "aggregate", *power_cols, *on_cols]
            ]

        paths = output_paths(args.output_dir, label_mode)
        outputs["train"].to_csv(paths["train"], index=False)
        outputs["val"].to_csv(paths["val"], index=False)
        outputs["test"].to_csv(paths["test"], index=False)

        print("\nSAVED:")
        for split_name, path in paths.items():
            summarize(path.name, outputs[split_name], label_mode=label_mode)

    print("\nDone. SGN configs:")
    print("  MATNILM 15W (paper) : training_data_ukdale_matnilm_s2.json + sgn_ukdale_matnilm_s2.json")
    print("  House thresholds    : training_data_ukdale_matnilm_s2_house_threshold.json + sgn_ukdale_matnilm_s2_house_threshold.json")


if __name__ == "__main__":
    main()

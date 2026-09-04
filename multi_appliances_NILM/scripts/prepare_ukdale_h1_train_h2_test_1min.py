#!/usr/bin/env python
"""Prepare raw UK-DALE cross-house data at 1-minute resolution.

Protocol:
  - train/validation source: house 1
  - test source: house 2
  - resolution: 1 minute

This script reads raw UK-DALE .dat files directly, like
dataset_preprocess/ukdale_processing_multi_appliance.py. It does not read or
resample any previously-created 6-second CSV.

Default raw input:
  dataset_preprocess/UK_DALE/UKDALE2017/house_1/mains.dat
  dataset_preprocess/UK_DALE/UKDALE2017/house_2/mains.dat

Outputs:
  multi_appliances_NILM/datasets/ukdale_h1_h2_1min/training/multi_appliance_training.csv
  multi_appliances_NILM/datasets/ukdale_h1_h2_1min/validating/multi_appliance_validating.csv
  multi_appliances_NILM/datasets/ukdale_h1_h2_1min/testing/multi_appliance_testing.csv

Example:
  python multi_appliances_NILM/scripts/prepare_ukdale_h1_train_h2_test_1min.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
NILM_DIR = PROJECT_DIR / "multi_appliances_NILM"
PREPROCESS_DIR = PROJECT_DIR / "dataset_preprocess"

if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from ukdale_processing import apply_algorithm1_labeling, resolve_appliance_setting


DEFAULT_CONFIG = PROJECT_DIR / "config" / "preprocess" / "ukdale.yaml"
DEFAULT_RAW_DIR = PROJECT_DIR / "dataset_preprocess" / "UK_DALE" / "UKDALE2017"
DEFAULT_OUT_DIR = NILM_DIR / "datasets" / "ukdale_h1_h2_1min"

TIME_COL = "readable_time"
HOUSE_COL = "house"
RAW_SAMPLE_SECONDS = 6
TARGET_SAMPLE_SECONDS = 60
SAMPLE_PERIOD = "1min"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build raw UK-DALE 1-minute cross-house split: train H1, test H2."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="UK-DALE preprocessing config with appliance channel maps and thresholds.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Raw UK-DALE directory containing house_1/, house_2/, ...",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--appliances",
        type=str,
        default=None,
        help="Comma list. Default uses global_params.appliances_to_process from config.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Last fraction of house 1 reserved for validation. Use 0 for no validation.",
    )
    parser.add_argument("--start-house1", type=str, default=None)
    parser.add_argument("--end-house1", type=str, default=None)
    parser.add_argument("--start-house2", type=str, default=None)
    parser.add_argument("--end-house2", type=str, default=None)
    parser.add_argument(
        "--no-trim-to-common-start",
        action="store_true",
        help="Keep leading rows before every appliance channel has non-zero readings.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_existing_path(path: Path) -> Path:
    """Resolve relative paths against cwd first, then project root."""
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (PROJECT_DIR / path).resolve()


def resolve_raw_dir(raw_dir: Path) -> Path:
    """Accept both UK_DALE/UKDALE2017 and UK_DALE raw layouts."""
    base = resolve_existing_path(raw_dir)
    candidates = [
        base,
        base / "UKDALE2017",
        PROJECT_DIR / "dataset_preprocess" / "UK_DALE" / "UKDALE2017",
        PROJECT_DIR / "dataset_preprocess" / "UK_DALE",
    ]
    tried: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in tried:
            continue
        tried.append(candidate)
        if (candidate / "house_1" / "mains.dat").is_file() and (
            candidate / "house_2" / "mains.dat"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find raw UK-DALE mains files. Tried:\n  - "
        + "\n  - ".join(str(path) for path in tried)
        + "\nExpected a folder containing house_1\\mains.dat and house_2\\mains.dat."
    )


def selected_appliances(config: dict[str, Any], appliances_arg: str | None) -> list[str]:
    if appliances_arg:
        appliances = [item.strip() for item in appliances_arg.split(",") if item.strip()]
    else:
        appliances = list(config["global_params"].get("appliances_to_process", []))
    if not appliances:
        appliances = ["kettle", "fridge", "microwave", "dishwasher", "washingmachine"]
    missing = [app for app in appliances if app not in config["appliances"]]
    if missing:
        raise ValueError(f"Unknown appliance(s) in config: {missing}")
    return appliances


def raw_house_dir(raw_dir: Path, house: int) -> Path:
    house_dir = raw_dir / f"house_{house}"
    if house_dir.is_dir():
        return house_dir
    nested = raw_dir / "UKDALE2017" / f"house_{house}"
    if nested.is_dir():
        return nested
    raise FileNotFoundError(f"Missing raw house directory for house {house}: {house_dir}")


def dat_path(raw_dir: Path, house: int, channel: int | None = None) -> Path:
    house_dir = raw_house_dir(raw_dir, house)
    name = "mains.dat" if channel is None else f"channel_{channel}.dat"
    path = house_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing raw file: {path}")
    return path


def first_timestamp(path: Path) -> float:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return float(line.split()[0])
    raise ValueError(f"Could not read first timestamp from {path}")


def last_timestamp(path: Path) -> float:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        pos = handle.tell()
        buffer = bytearray()
        while pos > 0:
            pos -= 1
            handle.seek(pos)
            char = handle.read(1)
            if char == b"\n":
                line = bytes(reversed(buffer)).decode("utf-8", errors="ignore").strip()
                if line:
                    return float(line.split()[0])
                buffer.clear()
            else:
                buffer.append(char[0])
        line = bytes(reversed(buffer)).decode("utf-8", errors="ignore").strip()
        if line:
            return float(line.split()[0])
    raise ValueError(f"Could not read last timestamp from {path}")


def common_time_range(
    raw_dir: Path,
    house: int,
    appliances: list[str],
    config: dict[str, Any],
) -> tuple[float, float]:
    paths = [dat_path(raw_dir, house)]
    for app in appliances:
        channel = config["appliances"][app].get("channel_map", {}).get(house)
        if channel is None:
            channel = config["appliances"][app].get("channel_map", {}).get(str(house))
        if channel is None:
            raise ValueError(f"No channel_map entry for {app} in house {house}")
        paths.append(dat_path(raw_dir, house, int(channel)))
    start = max(first_timestamp(path) for path in paths)
    end = min(last_timestamp(path) for path in paths)
    if start >= end:
        raise ValueError(f"No common raw time overlap for house {house}")
    return start, end


def parse_time(value: str | None, timezone: str) -> float | None:
    if not value:
        return None
    return float(pd.to_datetime(value).tz_localize(timezone).timestamp())


def final_time_bounds(
    raw_dir: Path,
    house: int,
    appliances: list[str],
    config: dict[str, Any],
    start_arg: str | None,
    end_arg: str | None,
) -> tuple[float, float]:
    timezone = config["global_params"].get("timezone", "UTC")
    overlap_start, overlap_end = common_time_range(raw_dir, house, appliances, config)
    start = parse_time(start_arg, timezone) if start_arg else overlap_start
    end = parse_time(end_arg, timezone) if end_arg else overlap_end
    start = max(start, overlap_start)
    end = min(end, overlap_end)
    if start >= end:
        raise ValueError(f"Empty selected time range for house {house}")
    return start, end


def load_mains(raw_dir: Path, house: int, start_ts: float, end_ts: float, timezone: str) -> pd.DataFrame:
    path = dat_path(raw_dir, house)
    print(f"  mains: {path}", flush=True)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, sep=r"\s+", header=None, engine="c", chunksize=1_000_000):
        chunk = chunk[(chunk[0] >= start_ts) & (chunk[0] <= end_ts)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No mains rows for house {house}")
    df = pd.concat(chunks, ignore_index=True)
    if house == 2:
        df["aggregate"] = df[1]
    elif df.shape[1] >= 3:
        df["aggregate"] = df[1] + df[2]
    else:
        df["aggregate"] = df[1]
    df = df[[0, "aggregate"]].rename(columns={0: "time"})
    df = df.drop_duplicates(subset=["time"], keep="first")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(timezone)
    df = df.set_index("time").sort_index()
    return df.resample(SAMPLE_PERIOD).mean()


def load_appliance(
    raw_dir: Path,
    house: int,
    appliance: str,
    channel: int,
    start_ts: float,
    end_ts: float,
    timezone: str,
) -> pd.DataFrame:
    path = dat_path(raw_dir, house, channel)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        dtype={0: np.float64, 1: np.float32},
        engine="c",
        chunksize=1_000_000,
    ):
        chunk = chunk[(chunk[0] >= start_ts) & (chunk[0] <= end_ts)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No rows for {appliance} house {house}")
    df = pd.concat(chunks, ignore_index=True)
    df.columns = ["time", appliance]
    df = df.drop_duplicates(subset=["time"], keep="first")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(timezone)
    df = df.set_index("time").sort_index()
    return df.resample(SAMPLE_PERIOD).mean()


def fill_short_gaps(series: pd.Series, limit: int) -> pd.Series:
    if series.empty:
        return series
    return series.interpolate(method="linear", limit=limit, limit_area="inside").ffill(limit=1).bfill(limit=1)


def convert_sample_count(value: Any, *, minimum: int = 0) -> int:
    if value is None:
        return minimum
    seconds = float(value) * RAW_SAMPLE_SECONDS
    samples = int(math.ceil(seconds / TARGET_SAMPLE_SECONDS))
    return max(minimum, samples)


def make_1min_labels(
    power: np.ndarray,
    app_cfg: dict[str, Any],
    algorithm_cfg: dict[str, Any],
    house: int,
) -> np.ndarray:
    threshold = resolve_appliance_setting(app_cfg, "on_power_threshold", house, 50)
    min_off = convert_sample_count(
        resolve_appliance_setting(
            app_cfg,
            "min_off_duration",
            house,
            algorithm_cfg.get("min_off_duration_default", 1),
        ),
        minimum=0,
    )
    min_on = convert_sample_count(
        resolve_appliance_setting(
            app_cfg,
            "min_on_duration",
            house,
            algorithm_cfg.get("min_on_duration_default", 1),
        ),
        minimum=1,
    )
    l_window = convert_sample_count(algorithm_cfg.get("window_length", 0), minimum=0)
    return apply_algorithm1_labeling(
        power,
        x_threshold=threshold,
        l_window=l_window,
        x_noise=algorithm_cfg.get("x_noise", 0),
        remove_spikes=algorithm_cfg.get("remove_spikes", True),
        spike_window=max(1, convert_sample_count(algorithm_cfg.get("spike_window", 5), minimum=1)),
        spike_threshold=algorithm_cfg.get("spike_threshold", 3.0),
        background_threshold=algorithm_cfg.get("background_threshold", 50),
        min_off_duration=min_off,
        min_on_duration=min_on,
    )


def trim_to_common_appliance_start(
    combined: pd.DataFrame,
    appliances: list[str],
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    starts: list[pd.Timestamp] = []
    for app in appliances:
        col = f"{app}_power"
        active = combined[col] > 0
        if not active.any():
            raise ValueError(f"{col} has no non-zero readings; cannot find common start.")
        starts.append(combined.index[active][0])
    common_start = max(starts)
    return combined.loc[common_start:].copy(), common_start


def build_house_raw_1min(
    raw_dir: Path,
    house: int,
    appliances: list[str],
    config: dict[str, Any],
    start_arg: str | None,
    end_arg: str | None,
    *,
    trim_common_start: bool,
) -> pd.DataFrame:
    timezone = config["global_params"].get("timezone", "UTC")
    algorithm_cfg = config.get("algorithm1", {})
    start_ts, end_ts = final_time_bounds(raw_dir, house, appliances, config, start_arg, end_arg)
    start_label = pd.to_datetime(start_ts, unit="s", utc=True).tz_convert(timezone)
    end_label = pd.to_datetime(end_ts, unit="s", utc=True).tz_convert(timezone)

    print("=" * 72, flush=True)
    print(f"Building house {house} from raw .dat at 1-minute resolution", flush=True)
    print(f"  time: {start_label:%Y-%m-%d %H:%M:%S} -> {end_label:%Y-%m-%d %H:%M:%S}", flush=True)

    combined = load_mains(raw_dir, house, start_ts, end_ts, timezone)
    print(f"  mains rows after 1min resample: {len(combined):,}", flush=True)

    for app in appliances:
        app_cfg = config["appliances"][app]
        channel = app_cfg.get("channel_map", {}).get(house)
        if channel is None:
            channel = app_cfg.get("channel_map", {}).get(str(house))
        if channel is None:
            raise ValueError(f"No channel_map entry for {app} in house {house}")

        app_resampled = load_appliance(raw_dir, house, app, int(channel), start_ts, end_ts, timezone)
        aligned = combined[["aggregate"]].join(app_resampled, how="left")
        gap_6s = resolve_appliance_setting(
            app_cfg,
            "resample_gap_fill",
            house,
            algorithm_cfg.get("resample_gap_fill", 3),
        )
        gap_1min = convert_sample_count(gap_6s, minimum=0)
        aligned[app] = fill_short_gaps(aligned[app], limit=gap_1min)
        aligned = aligned.dropna(subset=["aggregate"]).copy()
        aligned[app] = aligned[app].fillna(0.0)

        power = np.minimum(
            aligned[app].to_numpy(dtype=np.float32),
            aligned["aggregate"].to_numpy(dtype=np.float32),
        )
        labels = make_1min_labels(power.copy(), app_cfg, algorithm_cfg, house)
        app_frame = pd.DataFrame(
            {
                f"{app}_power": power,
                f"{app}_on": labels.astype(int),
            },
            index=aligned.index,
        )
        combined = combined.join(app_frame, how="left")
        print(
            f"  {app:<15} channel={int(channel):<3} rows={len(app_frame):,} "
            f"ON rows={int(labels.sum()):,}",
            flush=True,
        )

    combined = combined.dropna(subset=["aggregate"]).copy()
    for app in appliances:
        combined[f"{app}_power"] = pd.to_numeric(combined[f"{app}_power"], errors="coerce").fillna(0.0)
        combined[f"{app}_on"] = pd.to_numeric(combined[f"{app}_on"], errors="coerce").fillna(0).astype(int)

    if trim_common_start:
        before = len(combined)
        combined, common_start = trim_to_common_appliance_start(combined, appliances)
        dropped = before - len(combined)
        print(
            f"  trimmed {dropped:,} leading rows before common appliance start "
            f"({common_start:%Y-%m-%d %H:%M:%S})",
            flush=True,
        )

    combined = combined.reset_index()
    combined[TIME_COL] = combined["time"].dt.tz_convert(timezone).dt.strftime("%Y-%m-%d %H:%M:%S")
    combined = combined.drop(columns=["time"])
    combined.insert(1, HOUSE_COL, house)
    ordered = [
        TIME_COL,
        HOUSE_COL,
        "aggregate",
        *[f"{app}_power" for app in appliances],
        *[f"{app}_on" for app in appliances],
    ]
    combined = combined[ordered]
    print(
        f"  final rows={len(combined):,}  "
        f"{combined[TIME_COL].iloc[0]} -> {combined[TIME_COL].iloc[-1]}",
        flush=True,
    )
    return combined


def split_house1(df: pd.DataFrame, val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("--val-fraction must be >= 0 and < 1.")
    if val_fraction == 0.0:
        return df.reset_index(drop=True), df.iloc[0:0].copy()
    split_idx = int(round(len(df) * (1.0 - val_fraction)))
    split_idx = max(1, min(len(df) - 1, split_idx))
    train = df.iloc[:split_idx].copy()
    val = df.iloc[split_idx:].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True)


def write_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path} rows={len(df):,}", flush=True)


def main() -> None:
    args = parse_args()
    config_path = resolve_existing_path(args.config)
    raw_dir = resolve_raw_dir(args.raw_dir)
    out_dir = resolve_existing_path(args.out_dir)
    config = load_config(config_path)
    appliances = selected_appliances(config, args.appliances)
    trim_common_start = not args.no_trim_to_common_start

    print("Raw UK-DALE 1-minute cross-house preparation", flush=True)
    print(f"config    : {config_path}", flush=True)
    print(f"raw_dir   : {raw_dir}", flush=True)
    print(f"out_dir   : {out_dir}", flush=True)
    print(f"appliances: {appliances}", flush=True)

    house1 = build_house_raw_1min(
        raw_dir,
        1,
        appliances,
        config,
        args.start_house1,
        args.end_house1,
        trim_common_start=trim_common_start,
    )
    house2 = build_house_raw_1min(
        raw_dir,
        2,
        appliances,
        config,
        args.start_house2,
        args.end_house2,
        trim_common_start=trim_common_start,
    )

    train, val = split_house1(house1, args.val_fraction)
    test = house2.reset_index(drop=True)

    train_path = out_dir / "training" / "multi_appliance_training.csv"
    val_path = out_dir / "validating" / "multi_appliance_validating.csv"
    test_path = out_dir / "testing" / "multi_appliance_testing.csv"

    write_split(train, train_path)
    write_split(val, val_path)
    write_split(test, test_path)

    meta = {
        "dataset": "ukdale",
        "resolution": "1min",
        "source": "raw_dat",
        "protocol": "train house 1, test house 2",
        "raw_dir": str(raw_dir),
        "config": str(config_path),
        "appliances": appliances,
        "train_file": str(train_path),
        "validation_file": str(val_path),
        "test_file": str(test_path),
        "val_fraction": args.val_fraction,
        "rows": {
            "train": int(len(train)),
            "validation": int(len(val)),
            "test": int(len(test)),
        },
    }
    meta_path = out_dir / "split_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()

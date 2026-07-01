"""
Low-frequency-only multi-appliance UK-DALE preprocessing.

This script creates one aligned CSV containing:

    readable_time
    aggregate
    <appliance>_power for each selected appliance
    <appliance>_on for each selected appliance

It is the low-frequency equivalent of:

    high_frequency_data_extract_multi_appliance.py

but it does not read high-frequency FLAC files and does not compute HF
features. It only uses UK-DALE mains.dat + appliance channel_*.dat files.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import yaml

from ukdale_processing import apply_algorithm1_labeling, resolve_appliance_setting


def get_arguments() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, "config", "preprocess", "ukdale.yaml")

    parser = argparse.ArgumentParser(
        description="Create one low-frequency multi-appliance UK-DALE CSV."
    )
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--house", type=int, default=None, help="Single house id.")
    parser.add_argument(
        "--houses",
        type=str,
        default=None,
        help="Comma list of houses to concatenate, e.g. 1,5. Overrides --house.",
    )
    parser.add_argument("--start", type=str, default=None, help="Override start time.")
    parser.add_argument("--end", type=str, default=None, help="Override end time.")
    parser.add_argument(
        "--full_range",
        action="store_true",
        help="Ignore start/end from ukdale.yaml and use each house's full available range.",
    )
    parser.add_argument(
        "--last_days",
        type=float,
        default=None,
        help="Use the last N days available for each house, e.g. 7 for paper-style last week.",
    )
    parser.add_argument(
        "--appliances",
        type=str,
        default=None,
        help="Comma list, e.g. kettle,fridge,microwave,dishwasher,washingmachine.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default is save_path/multi_appliance_house<ids>_lf.csv.",
    )
    parser.add_argument(
        "--output_mode",
        choices=["real", "zscore", "both"],
        default="real",
        help="real saves watts. zscore saves normalized power. both saves both column sets.",
    )
    parser.add_argument(
        "--allow_missing_appliances",
        action="store_true",
        help="Fill missing appliance channels with zeros instead of failing.",
    )
    parser.add_argument(
        "--split_houses",
        type=str,
        default=None,
        help=(
            "Write one CSV per house instead of merging, e.g. 1,2,5. "
            "Use with --full_range or --last_days for each house separately."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for --split_houses. Default: paths.save_path from config.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override UK-DALE root (house_1/, house_2/, ...). Default: paths.data_dir from config.",
    )
    parser.add_argument(
        "--house_filename",
        type=str,
        default="multi_appliance_house{house}_lf.csv",
        help="Per-house filename for --split_houses. Use {house} placeholder.",
    )
    parser.add_argument(
        "--trim_to_common_start",
        action="store_true",
        default=True,
        help="Drop leading rows until every appliance has non-zero power (default: on).",
    )
    parser.add_argument(
        "--no_trim_to_common_start",
        action="store_false",
        dest="trim_to_common_start",
        help="Keep leading rows where some appliances are still all-zero.",
    )
    return parser.parse_args()


def resolve_paths(config: dict, config_path: str) -> tuple[dict, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    paths = config["paths"]
    for key in ["data_dir", "save_path"]:
        if not os.path.isabs(paths[key]):
            paths[key] = os.path.normpath(os.path.join(project_root, paths[key]))
    return paths, project_root


def selected_appliances(config: dict, appliances_arg: str | None) -> list[str]:
    if appliances_arg:
        appliances = [item.strip() for item in appliances_arg.split(",") if item.strip()]
    else:
        appliances = list(config["global_params"].get("appliances_to_process", []))
    if not appliances:
        appliances = ["kettle", "fridge", "microwave", "dishwasher", "washingmachine"]

    unknown = [name for name in appliances if name not in config["appliances"]]
    if unknown:
        raise ValueError(f"Unknown appliance(s): {unknown}")
    return appliances


def selected_houses(config: dict, args: argparse.Namespace) -> list[int]:
    if args.houses:
        return [int(item.strip()) for item in args.houses.split(",") if item.strip()]
    if args.house is not None:
        return [int(args.house)]
    return [int(config["global_params"].get("houses", [2])[0])]


def last_timestamp_from_dat(path: str) -> float:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dat file: {path}")
    with open(path, "rb") as handle:
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
    raise ValueError(f"Could not read final timestamp from {path}")


def first_timestamp_from_dat(path: str) -> float:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dat file: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return float(line.split()[0])
    raise ValueError(f"Could not read first timestamp from {path}")


def overlap_files(data_dir: str, house: int, appliances: list[str], config: dict) -> list[str]:
    paths = [os.path.join(data_dir, f"house_{house}", "mains.dat")]
    for app in appliances:
        channel_id = config["appliances"][app].get("channel_map", {}).get(house)
        if channel_id is None:
            raise ValueError(f"no channel_map entry for {app} in house {house}")
        paths.append(os.path.join(data_dir, f"house_{house}", f"channel_{channel_id}.dat"))
    return paths


def overlap_time_range(data_dir: str, house: int, appliances: list[str], config: dict) -> tuple[float, float]:
    """Common usable time range where mains and every selected appliance has data."""
    paths = overlap_files(data_dir, house, appliances, config)
    start_ts = max(first_timestamp_from_dat(path) for path in paths)
    end_ts = min(last_timestamp_from_dat(path) for path in paths)
    if start_ts >= end_ts:
        raise ValueError(f"No common overlap for house {house} across mains and appliances.")
    return start_ts, end_ts


def time_bounds(
    global_params: dict,
    start_override: str | None,
    end_override: str | None,
    *,
    full_range: bool = False,
) -> tuple[float | None, float | None]:
    if full_range:
        return None, None
    tz = global_params.get("timezone", "UTC")
    start_t = start_override or global_params.get("start_time")
    end_t = end_override or global_params.get("end_time")
    start_ts = pd.to_datetime(start_t).tz_localize(tz).timestamp() if start_t else None
    end_ts = pd.to_datetime(end_t).tz_localize(tz).timestamp() if end_t else None
    return start_ts, end_ts


def load_mains(data_dir: str, house: int, start_ts: float | None, end_ts: float | None, tz: str, sample_period: str) -> pd.DataFrame:
    mains_path = os.path.join(data_dir, f"house_{house}", "mains.dat")
    if not os.path.exists(mains_path):
        raise FileNotFoundError(f"Missing mains file: {mains_path}")

    print(f"[1/3] Loading mains: {mains_path}")
    chunks = []
    for chunk in pd.read_csv(mains_path, sep=r"\s+", header=None, engine="c", chunksize=1_000_000):
        if start_ts is not None:
            chunk = chunk[chunk[0] >= start_ts]
        if end_ts is not None:
            chunk = chunk[chunk[0] <= end_ts]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No mains rows found for house {house} in the selected time range.")
    mains_df = pd.concat(chunks, ignore_index=True)
    if house == 2:
        mains_df["aggregate"] = mains_df[1]
    elif mains_df.shape[1] >= 3:
        mains_df["aggregate"] = mains_df[1] + mains_df[2]
    else:
        mains_df["aggregate"] = mains_df[1]

    mains_df = mains_df[[0, "aggregate"]]
    mains_df.columns = ["time", "aggregate"]
    mains_df.drop_duplicates(subset=["time"], keep="first", inplace=True)

    mains_df["time"] = pd.to_datetime(mains_df["time"], unit="s", utc=True).dt.tz_convert(tz)
    mains_df.set_index("time", inplace=True)
    mains_df.sort_index(inplace=True)
    mains_resampled = mains_df.resample(sample_period).mean()
    print(f"      mains rows after resample: {len(mains_resampled):,}")
    return mains_resampled


def load_appliance(
    data_dir: str,
    house: int,
    appliance: str,
    channel_id: int,
    start_ts: float | None,
    end_ts: float | None,
    tz: str,
    sample_period: str,
) -> pd.DataFrame:
    app_path = os.path.join(data_dir, f"house_{house}", f"channel_{channel_id}.dat")
    if not os.path.exists(app_path):
        raise FileNotFoundError(f"Missing appliance file for {appliance}: {app_path}")

    chunks = []
    for chunk in pd.read_csv(
        app_path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        dtype={0: np.float64, 1: np.float32},
        engine="c",
        chunksize=1_000_000,
    ):
        if start_ts is not None:
            chunk = chunk[chunk[0] >= start_ts]
        if end_ts is not None:
            chunk = chunk[chunk[0] <= end_ts]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No {appliance} rows found for house {house} in the selected time range.")
    app_df = pd.concat(chunks, ignore_index=True)
    app_df.columns = ["time", appliance]
    app_df.drop_duplicates(subset=["time"], keep="first", inplace=True)

    app_df["time"] = pd.to_datetime(app_df["time"], unit="s", utc=True).dt.tz_convert(tz)
    app_df.set_index("time", inplace=True)
    app_df.sort_index(inplace=True)
    return app_df.resample(sample_period).mean()


def trim_to_common_appliance_start(
    combined: pd.DataFrame,
    appliances: list[str],
    *,
    min_power_w: float = 0.0,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Drop leading timeline where any appliance is still all-zero (meter not active yet).

    UK-DALE mains often starts before submeters report real load. Those rows get
    false OFF labels and confuse training (e.g. fridge OFF while channel is dead).
    """
    starts: list[pd.Timestamp] = []
    for app in appliances:
        col = f"{app}_power"
        if col not in combined.columns:
            continue
        active = combined[col] > min_power_w
        if not active.any():
            raise ValueError(
                f"{app}_power has no values above {min_power_w} W; "
                "cannot determine common appliance start."
            )
        starts.append(combined.index[active][0])

    if not starts:
        return combined, None

    common_start = max(starts)
    trimmed = combined.loc[common_start:].copy()
    return trimmed, common_start


def fill_short_appliance_gaps(series: pd.Series, limit: int = 3) -> pd.Series:
    """Bridge brief NaN bins after 6 s resampling (UK-DALE packet gaps).

    Raw meters often skip one packet (~6–13 s). Without this, empty resample bins
    become 0 W and split otherwise-continuous ON labels (common on house-5 fridge).
    """
    if series.empty:
        return series
    filled = series.interpolate(method="linear", limit=limit, limit_area="inside")
    return filled.ffill(limit=1).bfill(limit=1)


def make_labels(
    power: np.ndarray,
    appliance_cfg: dict,
    algorithm_cfg: dict,
    house: int,
) -> np.ndarray:
    return apply_algorithm1_labeling(
        power,
        x_threshold=resolve_appliance_setting(appliance_cfg, "on_power_threshold", house, 50),
        l_window=algorithm_cfg.get("window_length", 0),
        x_noise=algorithm_cfg.get("x_noise", 0),
        remove_spikes=algorithm_cfg.get("remove_spikes", True),
        spike_window=algorithm_cfg.get("spike_window", 5),
        spike_threshold=algorithm_cfg.get("spike_threshold", 3.0),
        background_threshold=algorithm_cfg.get("background_threshold", 50),
        min_off_duration=resolve_appliance_setting(appliance_cfg, "min_off_duration", house, 1),
        min_on_duration=resolve_appliance_setting(appliance_cfg, "min_on_duration", house, 1),
    )


def add_zscore_columns(df: pd.DataFrame, config: dict, appliances: list[str]) -> pd.DataFrame:
    result = df.copy()
    global_params = config["global_params"]
    result["aggregate_zscore"] = (
        result["aggregate"] - global_params["aggregate_mean"]
    ) / global_params["aggregate_std"]

    for app in appliances:
        power_col = f"{app}_power"
        result[f"{app}_power_zscore"] = (
            result[power_col] - config["appliances"][app]["mean"]
        ) / config["appliances"][app]["std"]
    return result


def build_one_house_lf(config: dict, args: argparse.Namespace, house: int) -> tuple[pd.DataFrame, list[str]]:
    paths, _ = resolve_paths(config, args.config)
    global_params = config["global_params"]
    algorithm_cfg = config.get("algorithm1", {})
    appliances = selected_appliances(config, args.appliances)
    tz = global_params.get("timezone", "UTC")
    sample_seconds = int(global_params.get("sample_seconds", 6))
    sample_period = f"{sample_seconds}s"
    start_ts, end_ts = time_bounds(global_params, args.start, args.end, full_range=args.full_range)
    if args.full_range and args.start is None and args.end is None:
        overlap_start, overlap_end = overlap_time_range(paths["data_dir"], house, appliances, config)
        start_ts, end_ts = overlap_start, overlap_end
    if args.last_days is not None:
        overlap_start, overlap_end = overlap_time_range(paths["data_dir"], house, appliances, config)
        requested_seconds = float(args.last_days) * 86400.0
        if overlap_end - overlap_start < requested_seconds:
            available_days = (overlap_end - overlap_start) / 86400.0
            raise ValueError(
                f"House {house} has only {available_days:.2f} common days across mains and "
                f"all selected appliances; cannot make a {args.last_days:g}-day dataset."
            )
        end_ts = overlap_end
        start_ts = end_ts - requested_seconds

    print("=" * 72)
    print("LOW-FREQUENCY MULTI-APPLIANCE UK-DALE PREPROCESSING")
    print("=" * 72)
    print(f"house      : {house}")
    print(f"appliances : {appliances}")
    print(f"sample     : {sample_seconds} seconds")
    print(f"data_dir   : {paths['data_dir']}")
    if start_ts is not None and end_ts is not None:
        start_label = pd.to_datetime(start_ts, unit="s", utc=True).tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S")
        end_label = pd.to_datetime(end_ts, unit="s", utc=True).tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S")
        print(f"time range : {start_label} to {end_label}")

    mains = load_mains(paths["data_dir"], house, start_ts, end_ts, tz, sample_period)
    combined = mains.copy()

    print("[2/3] Loading appliance channels and aligning to mains")
    for app in appliances:
        app_cfg = config["appliances"][app]
        channel_id = app_cfg.get("channel_map", {}).get(house)
        if channel_id is None:
            message = f"no channel_map entry for {app} in house {house}"
            if not args.allow_missing_appliances:
                raise ValueError(
                    f"{message}. This would create false zero labels. "
                    "Use --allow_missing_appliances only for inspection, not paper-style training."
                )
            print(f"      skip {app}: {message}")
            combined[f"{app}_power"] = 0.0
            combined[f"{app}_on"] = 0
            continue

        app_resampled = load_appliance(
            paths["data_dir"],
            house,
            app,
            channel_id,
            start_ts,
            end_ts,
            tz,
            sample_period,
        )
        aligned = combined[["aggregate"]].join(app_resampled, how="left")
        gap_limit = int(
            resolve_appliance_setting(
                app_cfg,
                "resample_gap_fill",
                house,
                algorithm_cfg.get("resample_gap_fill", 3),
            )
        )
        aligned[app] = fill_short_appliance_gaps(aligned[app], limit=gap_limit)
        aligned = aligned.dropna(subset=["aggregate"]).copy()
        aligned[app] = aligned[app].fillna(0.0)

        power = np.minimum(aligned[app].to_numpy(dtype=np.float32), aligned["aggregate"].to_numpy(dtype=np.float32))
        threshold = resolve_appliance_setting(app_cfg, "on_power_threshold", house, 50)
        label = make_labels(power.copy(), app_cfg, algorithm_cfg, house)
        app_frame = pd.DataFrame(
            {
                f"{app}_power": power,
                f"{app}_on": label.astype(int),
            },
            index=aligned.index,
        )
        combined = combined.join(app_frame, how="left")
        print(
            f"      {app:<15} channel={channel_id:<3} thresh={threshold:<4}W "
            f"rows={len(app_frame):,} ON rows={int(label.sum()):,}"
        )

    combined = combined.dropna(subset=["aggregate"]).copy()
    for app in appliances:
        combined[f"{app}_power"] = pd.to_numeric(combined[f"{app}_power"], errors="coerce").fillna(0.0)
        combined[f"{app}_on"] = pd.to_numeric(combined[f"{app}_on"], errors="coerce").fillna(0).astype(int)

    if args.trim_to_common_start:
        before_rows = len(combined)
        combined, common_start = trim_to_common_appliance_start(combined, appliances)
        dropped = before_rows - len(combined)
        if common_start is not None and dropped > 0:
            print(
                f"[trim] dropped {dropped:,} leading rows before all appliances active "
                f"(common start {common_start.tz_convert(tz).strftime('%Y-%m-%d %H:%M:%S')})"
            )
        elif common_start is not None:
            print(f"[trim] common appliance start: {common_start.tz_convert(tz).strftime('%Y-%m-%d %H:%M:%S')}")

    combined.reset_index(inplace=True)
    combined["readable_time"] = combined["time"].dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M:%S")
    combined.drop(columns=["time"], inplace=True)
    combined.insert(1, "house", house)

    power_cols = [f"{app}_power" for app in appliances]
    on_cols = [f"{app}_on" for app in appliances]
    ordered = ["readable_time", "house", "aggregate", *power_cols, *on_cols]
    combined = combined[ordered]

    if args.output_mode == "zscore":
        z = add_zscore_columns(combined, config, appliances)
        keep = ["readable_time", "house", "aggregate_zscore", *[f"{app}_power_zscore" for app in appliances], *on_cols]
        combined = z[keep].rename(columns={"aggregate_zscore": "aggregate"})
        for app in appliances:
            combined.rename(columns={f"{app}_power_zscore": f"{app}_power"}, inplace=True)
    elif args.output_mode == "both":
        combined = add_zscore_columns(combined, config, appliances)

    return combined, appliances


def build_multi_appliance_lf(config: dict, args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], list[int]]:
    houses = selected_houses(config, args)
    frames = []
    appliances: list[str] = []
    for house in houses:
        frame, appliances = build_one_house_lf(config, args, house)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["house", "readable_time"]).reset_index(drop=True)
    return combined, appliances, houses


def default_output_path(config: dict, args: argparse.Namespace, houses: list[int]) -> str:
    paths, _ = resolve_paths(config, args.config)
    if args.output:
        return os.path.abspath(args.output)
    house_label = "_".join(str(house) for house in houses)
    return os.path.join(paths["save_path"], f"multi_appliance_house{house_label}_lf.csv")


def print_on_summary(df: pd.DataFrame, appliances: list[str]) -> None:
    print("\nON/OFF summary")
    print("appliance        ON rows      ON percent")
    for app in appliances:
        on_rows = int(df[f"{app}_on"].sum())
        on_pct = (on_rows / len(df) * 100.0) if len(df) else 0.0
        print(f"{app:<15} {on_rows:>8,} {on_pct:>13.3f}%")


def per_house_output_path(output_dir: str, house: int, filename_template: str) -> str:
    return os.path.join(output_dir, filename_template.format(house=house))


def main() -> None:
    start_time = time.time()
    args = get_arguments()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    paths, _ = resolve_paths(config, args.config)
    if args.data_dir:
        config["paths"]["data_dir"] = os.path.abspath(args.data_dir)
    if args.split_houses:
        houses = [int(item.strip()) for item in args.split_houses.split(",") if item.strip()]
        output_dir = os.path.abspath(args.output_dir or paths["save_path"])
        os.makedirs(output_dir, exist_ok=True)
        print(f"Writing one CSV per house to: {output_dir}")
        print(f"data_dir: {config['paths']['data_dir']}")
        for house in houses:
            house_start = time.time()
            df, appliances = build_one_house_lf(config, args, house)
            output_path = per_house_output_path(output_dir, house, args.house_filename)
            df.to_csv(output_path, index=False)
            print("[3/3] Saved low-frequency multi-appliance CSV")
            print(f"output : {output_path}")
            print(f"rows   : {len(df):,}")
            print(f"columns: {list(df.columns)}")
            print_on_summary(df, appliances)
            print(f"house {house} done in {(time.time() - house_start) / 60.0:.2f} min.\n")
        print(f"All houses done in {(time.time() - start_time) / 60.0:.2f} min.")
        return

    df, appliances, houses = build_multi_appliance_lf(config, args)
    output_path = default_output_path(config, args, houses)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("[3/3] Saved low-frequency multi-appliance CSV")
    print(f"output : {output_path}")
    print(f"rows   : {len(df):,}")
    print(f"columns: {list(df.columns)}")
    print_on_summary(df, appliances)
    print(f"\nDone in {(time.time() - start_time) / 60.0:.2f} min.")


if __name__ == "__main__":
    main()

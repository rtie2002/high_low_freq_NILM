"""
Low-frequency multi-appliance REDD preprocessing.

This script creates one aligned CSV containing:

    readable_time
    house
    aggregate
    <appliance>_power for each selected appliance
    <appliance>_on for each selected appliance

REDD normally uses four target appliances in this project:

    microwave, fridge, dishwasher, washingmachine

The output format intentionally matches ukdale_processing_multi_appliance.py
so the same downstream multi-appliance pipeline can consume UK-DALE and REDD
CSV files.
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
    default_config = os.path.join(project_root, "config", "preprocess", "redd.yaml")

    parser = argparse.ArgumentParser(
        description="Create one low-frequency multi-appliance REDD CSV."
    )
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--house", type=int, default=None, help="Single REDD house id.")
    parser.add_argument(
        "--houses",
        type=str,
        default=None,
        help="Comma list of houses to concatenate, e.g. 1,2,3. Overrides --house.",
    )
    parser.add_argument("--start", type=str, default=None, help="Override start date/time.")
    parser.add_argument("--end", type=str, default=None, help="Override end date/time.")
    parser.add_argument(
        "--appliances",
        type=str,
        default=None,
        help="Comma list, e.g. microwave,fridge,dishwasher,washingmachine.",
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
        "--data_dir",
        type=str,
        default=None,
        help="Override REDD root containing house_1/, house_2/, ...",
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
        help="Write one CSV per house instead of merging, e.g. 1,2,3.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for --split_houses. Default: paths.save_path from config.",
    )
    parser.add_argument(
        "--house_filename",
        type=str,
        default="multi_appliance_house{house}_lf.csv",
        help="Per-house filename for --split_houses. Use {house} placeholder.",
    )
    return parser.parse_args()


def resolve_paths(config: dict) -> tuple[dict, str]:
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
        appliances = ["microwave", "fridge", "dishwasher", "washingmachine"]

    unknown = [name for name in appliances if name not in config["appliances"]]
    if unknown:
        raise ValueError(f"Unknown appliance(s): {unknown}")
    return appliances


def selected_houses(config: dict, args: argparse.Namespace, appliances: list[str]) -> list[int]:
    if args.houses:
        return [int(item.strip()) for item in args.houses.split(",") if item.strip()]
    if args.house is not None:
        return [int(args.house)]

    houses: set[int] = set()
    for app in appliances:
        houses.update(int(house) for house in config["appliances"][app].get("houses", []))
    return sorted(houses)


def appliance_channel(config: dict, appliance: str, house: int) -> int | None:
    app_cfg = config["appliances"][appliance]
    houses = [int(item) for item in app_cfg.get("houses", [])]
    channels = [int(item) for item in app_cfg.get("channels", [])]
    if house not in houses:
        return None
    return channels[houses.index(house)]


def parse_time(value: str | None) -> float | None:
    if not value:
        return None
    return pd.to_datetime(value).timestamp()


def time_bounds(global_params: dict, start_override: str | None, end_override: str | None) -> tuple[float | None, float | None]:
    start_t = start_override or global_params.get("start_date")
    end_t = end_override or global_params.get("end_date")
    return parse_time(start_t), parse_time(end_t)


def read_dat(
    path: str,
    value_name: str,
    start_ts: float | None,
    end_ts: float | None,
    *,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dat file: {path}")

    chunks = []
    for chunk in pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["time", value_name],
        dtype={"time": np.float64, value_name: np.float32},
        engine="c",
        chunksize=chunksize,
    ):
        if start_ts is not None:
            chunk = chunk[chunk["time"] >= start_ts]
        if end_ts is not None:
            chunk = chunk[chunk["time"] <= end_ts]
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No rows found in selected time range: {path}")

    data = pd.concat(chunks, ignore_index=True)
    data.drop_duplicates(subset=["time"], keep="first", inplace=True)
    data["time"] = pd.to_datetime(data["time"], unit="s")
    data.set_index("time", inplace=True)
    data.sort_index(inplace=True)
    return data


def load_mains(
    data_dir: str,
    house: int,
    start_ts: float | None,
    end_ts: float | None,
    sample_period: str,
) -> pd.DataFrame:
    house_dir = os.path.join(data_dir, f"house_{house}")
    mains_1 = read_dat(os.path.join(house_dir, "channel_1.dat"), "mains_1", start_ts, end_ts)
    mains_2 = read_dat(os.path.join(house_dir, "channel_2.dat"), "mains_2", start_ts, end_ts)

    mains = mains_1.join(mains_2, how="outer")
    mains["aggregate"] = mains["mains_1"] + mains["mains_2"]
    mains = mains[["aggregate"]].resample(sample_period).mean().bfill(limit=1)
    mains.dropna(subset=["aggregate"], inplace=True)
    print(f"      mains rows after resample: {len(mains):,}")
    return mains


def load_appliance(
    data_dir: str,
    house: int,
    appliance: str,
    channel_id: int,
    start_ts: float | None,
    end_ts: float | None,
    sample_period: str,
) -> pd.DataFrame:
    app_path = os.path.join(data_dir, f"house_{house}", f"channel_{channel_id}.dat")
    app = read_dat(app_path, appliance, start_ts, end_ts)
    return app.resample(sample_period).mean().bfill(limit=1)


def make_labels(power: np.ndarray, appliance_cfg: dict, algorithm_cfg: dict, house: int) -> np.ndarray:
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
        result[f"{app}_power_zscore"] = (
            result[f"{app}_power"] - config["appliances"][app]["mean"]
        ) / config["appliances"][app]["std"]
    return result


def build_one_house_lf(config: dict, args: argparse.Namespace, house: int) -> tuple[pd.DataFrame, list[str]]:
    paths, _ = resolve_paths(config)
    if args.data_dir:
        paths["data_dir"] = os.path.abspath(args.data_dir)

    global_params = config["global_params"]
    algorithm_cfg = config.get("algorithm1", {})
    appliances = selected_appliances(config, args.appliances)
    sample_seconds = int(global_params.get("sample_seconds", 1))
    sample_period = f"{sample_seconds}s"
    start_ts, end_ts = time_bounds(global_params, args.start, args.end)

    print("=" * 72)
    print("LOW-FREQUENCY MULTI-APPLIANCE REDD PREPROCESSING")
    print("=" * 72)
    print(f"house      : {house}")
    print(f"appliances : {appliances}")
    print(f"sample     : {sample_seconds} seconds")
    print(f"data_dir   : {paths['data_dir']}")
    if start_ts is not None or end_ts is not None:
        print(f"time range : {args.start or global_params.get('start_date')} to {args.end or global_params.get('end_date')}")

    print("[1/3] Loading mains")
    combined = load_mains(paths["data_dir"], house, start_ts, end_ts, sample_period)

    print("[2/3] Loading appliance channels and aligning to mains")
    for app in appliances:
        app_cfg = config["appliances"][app]
        channel_id = appliance_channel(config, app, house)
        if channel_id is None:
            message = f"no channel entry for {app} in house {house}"
            if not args.allow_missing_appliances:
                raise ValueError(f"{message}. Use --allow_missing_appliances only for inspection.")
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
            sample_period,
        )
        aligned = combined[["aggregate"]].join(app_resampled, how="left")
        aligned[app] = aligned[app].fillna(0.0)

        power = np.minimum(
            aligned[app].to_numpy(dtype=np.float32),
            aligned["aggregate"].to_numpy(dtype=np.float32),
        )
        label = make_labels(power.copy(), app_cfg, algorithm_cfg, house)
        combined[f"{app}_power"] = power
        combined[f"{app}_on"] = label.astype(int)
        threshold = resolve_appliance_setting(app_cfg, "on_power_threshold", house, 50)
        print(
            f"      {app:<15} channel={channel_id:<3} thresh={threshold:<4}W "
            f"rows={len(aligned):,} ON rows={int(label.sum()):,}"
        )

    combined = combined.dropna(subset=["aggregate"]).copy()
    for app in appliances:
        combined[f"{app}_power"] = pd.to_numeric(combined[f"{app}_power"], errors="coerce").fillna(0.0)
        combined[f"{app}_on"] = pd.to_numeric(combined[f"{app}_on"], errors="coerce").fillna(0).astype(int)

    combined.reset_index(inplace=True)
    combined["readable_time"] = combined["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    combined.drop(columns=["time"], inplace=True)
    combined.insert(1, "house", house)

    power_cols = [f"{app}_power" for app in appliances]
    on_cols = [f"{app}_on" for app in appliances]
    combined = combined[["readable_time", "house", "aggregate", *power_cols, *on_cols]]

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
    appliances = selected_appliances(config, args.appliances)
    houses = selected_houses(config, args, appliances)
    frames = []
    for house in houses:
        frame, appliances = build_one_house_lf(config, args, house)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["house", "readable_time"]).reset_index(drop=True)
    return combined, appliances, houses


def default_output_path(config: dict, args: argparse.Namespace, houses: list[int]) -> str:
    paths, _ = resolve_paths(config)
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

    paths, _ = resolve_paths(config)
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

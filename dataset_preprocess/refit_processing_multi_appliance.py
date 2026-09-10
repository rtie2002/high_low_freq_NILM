"""
Low-frequency multi-appliance REFIT preprocessing.

Same idea as redd_processing_multi_appliance.py / ukdale_processing_multi_appliance.py:
build one aligned whole-house CSV with

    readable_time
    unix_time, house, sequence_id
    aggregate
    <appliance>_power
    aggregate_observed, <appliance>_observed
    <appliance>_on

Input:
    dataset_preprocess/REFIT/House_{N}.csv
    columns: Time, Unix, Aggregate, Appliance1..Appliance9

Native REFIT is approximately 8 s. UK-DALE and REFIT are both placed on an
8 s grid so cross-domain training uses one physical sampling interval.
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
import yaml

from ukdale_processing import (
    apply_algorithm1_labeling,
    contiguous_segment_ids,
    fill_complete_short_gaps,
    label_power_by_segments,
    resolve_appliance_setting,
    resolve_time_samples,
    write_dataframe_csv,
)


def get_arguments() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, "config", "preprocess", "refit.yaml")

    parser = argparse.ArgumentParser(
        description="Create low-frequency multi-appliance REFIT CSV (UK-DALE-style)."
    )
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--house", type=int, default=None, help="Single REFIT house id.")
    parser.add_argument(
        "--houses",
        type=str,
        default=None,
        help="Comma list of houses to concatenate, e.g. 2,3,5. Overrides --house.",
    )
    parser.add_argument("--start", type=str, default=None, help="Override start date/time.")
    parser.add_argument("--end", type=str, default=None, help="Override end date/time.")
    parser.add_argument(
        "--full_range",
        action="store_true",
        help="Ignore start/end from config and use each house's full time range.",
    )
    parser.add_argument(
        "--last_days",
        type=float,
        default=None,
        help="Use the last N days of each house.",
    )
    parser.add_argument(
        "--appliances",
        type=str,
        default=None,
        help="Comma list, e.g. kettle,fridge,dishwasher,washingmachine,microwave.",
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
        help="Override REFIT root containing House_*.csv.",
    )
    parser.add_argument(
        "--allow_missing_appliances",
        action="store_true",
        help="Fill missing appliance IAMs with zeros instead of failing.",
    )
    parser.add_argument(
        "--split_houses",
        type=str,
        default=None,
        help="Write one CSV per house instead of merging, e.g. 2,3,5,9,11,20.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for --split_houses. Default: paths.save_path.",
    )
    parser.add_argument(
        "--house_filename",
        type=str,
        default="refit_house{house}_lf_8s.csv",
        help="Per-house filename for --split_houses. Use {house} placeholder.",
    )
    parser.add_argument(
        "--trim_to_common_start",
        action="store_true",
        default=True,
        help="Drop leading rows until every available meter has reported (default: on).",
    )
    parser.add_argument(
        "--no_trim_to_common_start",
        action="store_false",
        dest="trim_to_common_start",
        help="Keep leading rows before every available meter has reported.",
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
        appliances = ["kettle", "fridge", "dishwasher", "washingmachine", "microwave"]

    unknown = [name for name in appliances if name not in config["appliances"]]
    if unknown:
        raise ValueError(f"Unknown appliance(s): {unknown}")
    return appliances


def selected_houses(config: dict, args: argparse.Namespace, appliances: list[str]) -> list[int]:
    if args.houses:
        return [int(item.strip()) for item in args.houses.split(",") if item.strip()]
    if args.house is not None:
        return [int(args.house)]

    global_houses = config["global_params"].get("houses")
    if global_houses:
        return [int(h) for h in global_houses]

    houses: set[int] = set()
    for app in appliances:
        houses.update(int(h) for h in config["appliances"][app].get("houses", []))
        houses.update(int(h) for h in config["appliances"][app].get("channel_map", {}).keys())
    return sorted(houses)


def appliance_iam(config: dict, appliance: str, house: int) -> int | None:
    """Return Appliance{N} IAM index for one appliance in one house."""
    app_cfg = config["appliances"][appliance]
    channel_map = app_cfg.get("channel_map") or {}
    if house in channel_map:
        return int(channel_map[house])
    if str(house) in channel_map:
        return int(channel_map[str(house)])
    return None


def parse_time(value: str | None, tz: str) -> float | None:
    if not value:
        return None
    ts = pd.to_datetime(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    return float(ts.timestamp())


def time_bounds(
    global_params: dict,
    start_override: str | None,
    end_override: str | None,
    *,
    full_range: bool,
) -> tuple[float | None, float | None]:
    if full_range:
        return None, None
    tz = global_params.get("timezone", "Europe/London")
    start_t = start_override or global_params.get("start_date")
    end_t = end_override or global_params.get("end_date")
    return parse_time(start_t, tz), parse_time(end_t, tz)


def house_csv_path(data_dir: str, house: int, config: dict) -> str:
    template = config["paths"].get("house_filename", "House_{house}.csv")
    path = os.path.join(data_dir, template.format(house=house))
    if not os.path.isfile(path):
        # Accept CLEAN_HouseN.csv naming as fallback
        alt = os.path.join(data_dir, f"CLEAN_House{house}.csv")
        if os.path.isfile(alt):
            return alt
        raise FileNotFoundError(f"Missing REFIT house CSV: {path}")
    return path


def trim_to_common_appliance_start(
    combined: pd.DataFrame,
    appliances: list[str],
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    starts: list[pd.Timestamp] = []
    for app in appliances:
        col = f"{app}_power"
        if col not in combined.columns:
            continue
        observed_col = f"{app}_observed"
        observed = (
            combined[observed_col].astype(bool)
            if observed_col in combined.columns
            else combined[col].notna()
        )
        if not observed.any():
            continue
        starts.append(combined.index[observed][0])
    if not starts:
        return combined, None
    common_start = max(starts)
    return combined.loc[common_start:].copy(), common_start


def make_labels(
    power: np.ndarray,
    appliance_cfg: dict,
    algorithm_cfg: dict,
    house: int,
    sample_seconds: int,
) -> np.ndarray:
    # Per-appliance overrides matter for short pulsed loads (e.g. REFIT microwave):
    # global remove_spikes=True deletes those bursts before thresholding.
    remove_spikes = bool(
        resolve_appliance_setting(
            appliance_cfg,
            "remove_spikes",
            house,
            algorithm_cfg.get("remove_spikes", True),
        )
    )
    return apply_algorithm1_labeling(
        power,
        x_threshold=resolve_appliance_setting(appliance_cfg, "on_power_threshold", house, 50),
        l_window=algorithm_cfg.get("window_length", 0),
        x_noise=algorithm_cfg.get("x_noise", 0),
        remove_spikes=remove_spikes,
        spike_window=int(
            resolve_appliance_setting(
                appliance_cfg,
                "spike_window",
                house,
                algorithm_cfg.get("spike_window", 5),
            )
        ),
        spike_threshold=float(
            resolve_appliance_setting(
                appliance_cfg,
                "spike_threshold",
                house,
                algorithm_cfg.get("spike_threshold", 3.0),
            )
        ),
        background_threshold=algorithm_cfg.get("background_threshold", 50),
        min_off_duration=resolve_time_samples(
            appliance_cfg, "min_off_duration", house, sample_seconds, 1
        ),
        min_on_duration=resolve_time_samples(
            appliance_cfg, "min_on_duration", house, sample_seconds, 1
        ),
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


def load_house_raw(
    path: str,
    appliances: list[str],
    config: dict,
    house: int,
    *,
    start_ts: float | None,
    end_ts: float | None,
    tz: str,
    allow_missing: bool,
) -> pd.DataFrame:
    """Load Aggregate + needed Appliance{N} columns; return timed DataFrame."""
    usecols = ["Time", "Unix", "Aggregate"]
    rename = {"Aggregate": "aggregate"}
    for app in appliances:
        iam = appliance_iam(config, app, house)
        if iam is None:
            if not allow_missing:
                raise ValueError(f"no channel_map entry for {app} in house {house}")
            continue
        col = f"Appliance{iam}"
        usecols.append(col)
        rename[col] = app

    # Deduplicate usecols while preserving order
    seen = set()
    usecols_unique = []
    for c in usecols:
        if c not in seen:
            seen.add(c)
            usecols_unique.append(c)

    # Prefer typed numeric read; filter on Unix before building DatetimeIndex.
    dtype_map: dict[str, str] = {"Unix": "float64", "Aggregate": "float32"}
    for col in usecols_unique:
        if col.startswith("Appliance"):
            dtype_map[col] = "float32"

    df = pd.read_csv(path, usecols=usecols_unique, dtype=dtype_map, engine="c")
    if "Unix" in df.columns and df["Unix"].notna().any():
        if start_ts is not None:
            df = df[df["Unix"] >= start_ts]
        if end_ts is not None:
            df = df[df["Unix"] <= end_ts]
        ts = pd.to_datetime(df["Unix"], unit="s", utc=True).dt.tz_convert(tz)
    else:
        ts = pd.to_datetime(df["Time"], errors="coerce")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        else:
            ts = ts.dt.tz_convert(tz)
        if start_ts is not None:
            start = pd.to_datetime(start_ts, unit="s", utc=True).tz_convert(tz)
            keep = ts >= start
            df = df.loc[keep]
            ts = ts.loc[keep]
        if end_ts is not None:
            end = pd.to_datetime(end_ts, unit="s", utc=True).tz_convert(tz)
            keep = ts <= end
            df = df.loc[keep]
            ts = ts.loc[keep]

    out = df.rename(columns=rename)
    out.index = ts
    app_cols = [c for c in rename.values() if c != "aggregate" and c in out.columns]
    # Keep unique app columns only (IAM collisions would already have overwritten in rename).
    seen_apps: list[str] = []
    for c in app_cols:
        if c not in seen_apps:
            seen_apps.append(c)
    out = out[["aggregate", *seen_apps]]
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="first")]
    # Force single Series columns (guard against accidental duplicate names).
    if isinstance(out["aggregate"], pd.DataFrame):
        out["aggregate"] = out["aggregate"].iloc[:, 0]

    if out.empty:
        raise ValueError(f"No rows in selected time range for {path}")
    return out


def house_time_span(path: str, tz: str) -> tuple[float, float]:
    """First/last unix seconds from a REFIT house CSV (same semantics, lower peak RAM)."""
    first = None
    last = None
    for chunk in pd.read_csv(path, usecols=["Unix"], chunksize=1_000_000, engine="c"):
        unix = pd.to_numeric(chunk["Unix"], errors="coerce").dropna()
        if unix.empty:
            continue
        if first is None:
            first = float(unix.iloc[0])
        last = float(unix.iloc[-1])
    if first is None or last is None:
        raise ValueError(f"No Unix timestamps in {path}")
    return first, last


def build_one_house_lf(config: dict, args: argparse.Namespace, house: int) -> tuple[pd.DataFrame, list[str]]:
    paths, _ = resolve_paths(config)
    if args.data_dir:
        paths["data_dir"] = os.path.abspath(args.data_dir)

    global_params = config["global_params"]
    algorithm_cfg = config.get("algorithm1", {})
    appliances = selected_appliances(config, args.appliances)
    tz = global_params.get("timezone", "Europe/London")
    sample_seconds = int(global_params.get("sample_seconds", 6))
    sample_period = f"{sample_seconds}s"
    allow_missing = bool(args.allow_missing_appliances)

    csv_path = house_csv_path(paths["data_dir"], house, config)
    start_ts, end_ts = time_bounds(
        global_params, args.start, args.end, full_range=args.full_range
    )

    if args.last_days is not None:
        span_start, span_end = house_time_span(csv_path, tz)
        requested = float(args.last_days) * 86400.0
        if span_end - span_start < requested:
            available_days = (span_end - span_start) / 86400.0
            raise ValueError(
                f"House {house} has only {available_days:.2f} days; "
                f"cannot make a {args.last_days:g}-day dataset."
            )
        end_ts = span_end
        start_ts = end_ts - requested
    elif args.full_range or (start_ts is None and end_ts is None):
        start_ts, end_ts = None, None

    print("=" * 72)
    print("LOW-FREQUENCY MULTI-APPLIANCE REFIT PREPROCESSING")
    print("=" * 72)
    print(f"house      : {house}")
    print(f"appliances : {appliances}")
    print(f"sample     : {sample_seconds} seconds (native REFIT ~8 s -> resample)")
    print(f"source     : {csv_path}")
    if start_ts is not None and end_ts is not None:
        start_label = (
            pd.to_datetime(start_ts, unit="s", utc=True).tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S")
        )
        end_label = (
            pd.to_datetime(end_ts, unit="s", utc=True).tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S")
        )
        print(f"time range : {start_label} to {end_label}")

    print("[1/3] Loading house CSV")
    raw = load_house_raw(
        csv_path,
        appliances,
        config,
        house,
        start_ts=start_ts,
        end_ts=end_ts,
        tz=tz,
        allow_missing=allow_missing,
    )
    print(f"      raw rows: {len(raw):,}")

    print(f"[2/3] Resample to {sample_seconds}s grid and label ON/OFF", flush=True)
    # One joint resample preserves the same mean bins as per-column resample.
    value_cols = ["aggregate", *[app for app in appliances if app in raw.columns]]
    resampled = raw[value_cols].resample(sample_period).mean()
    agg_gap_limit = resolve_time_samples(
        algorithm_cfg, "resample_gap_fill", house, sample_seconds, 3
    )
    combined = pd.DataFrame(index=resampled.index)
    combined["aggregate_observed"] = resampled["aggregate"].notna().astype(np.int8)
    combined["aggregate"] = fill_complete_short_gaps(resampled["aggregate"], agg_gap_limit)
    print(
        f"      aggregate bins={len(combined):,} "
        f"missing_after_short_fill={int(combined['aggregate'].isna().sum()):,}"
    )

    for app in appliances:
        app_cfg = config["appliances"][app]
        iam = appliance_iam(config, app, house)
        if iam is None or app not in resampled.columns:
            message = f"no channel_map / column for {app} in house {house}"
            if not allow_missing:
                raise ValueError(message)
            print(f"      skip {app}: {message}")
            combined[f"{app}_power"] = 0.0
            combined[f"{app}_observed"] = 0
            continue

        app_series = resampled[app]
        observed = app_series.notna()
        gap_limit = resolve_time_samples(
            app_cfg, "resample_gap_fill", house, sample_seconds, agg_gap_limit
        )
        combined[f"{app}_power"] = fill_complete_short_gaps(app_series, gap_limit)
        combined[f"{app}_observed"] = observed.astype(np.int8)
        print(
            f"      {app:<15} IAM={iam:<3} missing_after_short_fill="
            f"{int(combined[f'{app}_power'].isna().sum()):,}"
        )

    if args.trim_to_common_start:
        before_rows = len(combined)
        combined, common_start = trim_to_common_appliance_start(combined, appliances)
        dropped = before_rows - len(combined)
        if common_start is not None and dropped > 0:
            print(
                f"[trim] dropped {dropped:,} leading rows before all meters reported "
                f"(common start {common_start.tz_convert(tz).strftime('%Y-%m-%d %H:%M:%S')})"
            )

    power_cols = [f"{app}_power" for app in appliances]
    required = ["aggregate", *power_cols]
    before_valid = len(combined)
    combined = combined.dropna(subset=required).copy()
    print(
        f"[segments] dropped {before_valid - len(combined):,} rows with unresolved "
        "meter gaps"
    )
    if combined.empty:
        raise ValueError(f"No fully observed rows remain for REFIT house {house}.")

    combined["sequence_id"] = contiguous_segment_ids(combined.index, sample_period)
    sequence_ids = combined["sequence_id"].to_numpy()
    aggregate = combined["aggregate"].to_numpy(dtype=np.float32)
    for app in appliances:
        power = combined[f"{app}_power"].to_numpy(dtype=np.float32)
        labels = label_power_by_segments(
            power,
            sequence_ids,
            lambda segment, app=app: make_labels(
                segment, config["appliances"][app], algorithm_cfg, house, sample_seconds
            ),
        )
        combined[f"{app}_on"] = labels
        threshold = resolve_appliance_setting(
            config["appliances"][app], "on_power_threshold", house, 50
        )
        above_mains = int(np.sum(power > aggregate))
        print(
            f"      {app:<15} thresh={threshold:<4}W ON rows={int(labels.sum()):,} "
            f"power>aggregate rows={above_mains:,} (kept, not clipped)"
        )

    print(f"[segments] contiguous sequences={combined['sequence_id'].nunique():,}")
    time_index = combined.index
    combined = combined.reset_index(drop=True)
    combined.insert(0, "unix_time", (time_index.asi8 // 10**9).astype(np.int64))
    combined.insert(0, "readable_time", time_index.strftime("%Y-%m-%d %H:%M:%S"))
    combined.insert(2, "house", house)

    metadata_cols = ["readable_time", "unix_time", "house", "sequence_id"]
    observed_cols = ["aggregate_observed", *[f"{app}_observed" for app in appliances]]
    on_cols = [f"{app}_on" for app in appliances]
    combined = combined[[*metadata_cols, "aggregate", *power_cols, *observed_cols, *on_cols]]

    if args.output_mode == "zscore":
        z = add_zscore_columns(combined, config, appliances)
        keep = [
            *metadata_cols,
            "aggregate_zscore",
            *[f"{app}_power_zscore" for app in appliances],
            *observed_cols,
            *on_cols,
        ]
        combined = z[keep].rename(columns={"aggregate_zscore": "aggregate"})
        for app in appliances:
            combined.rename(columns={f"{app}_power_zscore": f"{app}_power"}, inplace=True)
    elif args.output_mode == "both":
        combined = add_zscore_columns(combined, config, appliances)

    return combined, appliances


def build_multi_appliance_lf(
    config: dict, args: argparse.Namespace
) -> tuple[pd.DataFrame, list[str], list[int]]:
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
            write_dataframe_csv(df, output_path)
            print("[3/3] Saved low-frequency multi-appliance CSV")
            print(f"output : {output_path}")
            print(f"rows   : {len(df):,}")
            print(f"columns: {list(df.columns)}")
            print_on_summary(df, appliances)
            print(f"house {house} done in {(time.time() - house_start) / 60.0:.2f} min.\n")
            del df
            gc.collect()
        print(f"All houses done in {(time.time() - start_time) / 60.0:.2f} min.")
        return

    df, appliances, houses = build_multi_appliance_lf(config, args)
    output_path = default_output_path(config, args, houses)
    write_dataframe_csv(df, output_path)

    print("[3/3] Saved low-frequency multi-appliance CSV")
    print(f"output : {output_path}")
    print(f"rows   : {len(df):,}")
    print(f"columns: {list(df.columns)}")
    print_on_summary(df, appliances)
    print(f"\nDone in {(time.time() - start_time) / 60.0:.2f} min.")


if __name__ == "__main__":
    main()

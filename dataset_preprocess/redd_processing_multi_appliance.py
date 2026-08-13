"""
Low-frequency multi-appliance REDD preprocessing.

Same idea as ukdale_processing_multi_appliance.py: build one aligned whole-house
CSV with

    readable_time
    house
    aggregate
    <appliance>_power
    <appliance>_on

Input sources (auto-detected):

    A) NILMTK HDF5  — dataset_preprocess/REDD/redd.h5  (preferred if present)
    B) Raw low_freq — <data_dir>/house_N/channel_*.dat

Resample mains + appliances onto a shared grid (default 6 s, UK-DALE-aligned),
join appliances to the mains timeline, then apply Algorithm-1 ON labels.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

from ukdale_processing import apply_algorithm1_labeling, resolve_appliance_setting

try:
    import tables as tb
except ImportError:  # pragma: no cover
    tb = None


def get_arguments() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, "config", "preprocess", "redd.yaml")

    parser = argparse.ArgumentParser(
        description="Create one low-frequency multi-appliance REDD CSV (UK-DALE-style)."
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
        "--full_range",
        action="store_true",
        help="Ignore start/end from config and use each house's full overlapping range.",
    )
    parser.add_argument(
        "--last_days",
        type=float,
        default=None,
        help="Use the last N days of common overlap per house.",
    )
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
        help="Override REDD root (folder with redd.h5 and/or house_1/).",
    )
    parser.add_argument(
        "--h5",
        type=str,
        default=None,
        help="Explicit path to redd.h5 (overrides paths.h5_file).",
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
        default="redd_house{house}_lf_6s.csv",
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

    global_houses = config["global_params"].get("houses")
    if global_houses:
        return [int(h) for h in global_houses]

    houses: set[int] = set()
    for app in appliances:
        houses.update(int(house) for house in config["appliances"][app].get("houses", []))
        houses.update(int(house) for house in config["appliances"][app].get("channel_map", {}).keys())
    return sorted(houses)


def normalize_channel_ids(raw) -> list[int] | None:
    """Accept int, [int,...], or None from channel_map / legacy channels list."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    return [int(raw)]


def appliance_channels(config: dict, appliance: str, house: int) -> list[int] | None:
    """Return meter/channel id list for one appliance in one house (sum if >1)."""
    app_cfg = config["appliances"][appliance]
    channel_map = app_cfg.get("channel_map")
    if channel_map is not None:
        # YAML may store int keys; accept both
        if house in channel_map:
            return normalize_channel_ids(channel_map[house])
        if str(house) in channel_map:
            return normalize_channel_ids(channel_map[str(house)])
        return None

    # Legacy: parallel houses[] / channels[] lists
    houses = [int(item) for item in app_cfg.get("houses", [])]
    channels = list(app_cfg.get("channels", []))
    if house not in houses:
        return None
    return normalize_channel_ids(channels[houses.index(house)])


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
    full_range: bool = False,
) -> tuple[float | None, float | None]:
    if full_range:
        return None, None
    tz = global_params.get("timezone", "US/Eastern")
    start_t = start_override or global_params.get("start_date") or global_params.get("start_time")
    end_t = end_override or global_params.get("end_date") or global_params.get("end_time")
    return parse_time(start_t, tz), parse_time(end_t, tz)


def resolve_h5_path(config: dict, args: argparse.Namespace, data_dir: str) -> str | None:
    if args.h5:
        path = os.path.abspath(args.h5)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--h5 not found: {path}")
        return path

    h5_name = config["paths"].get("h5_file")
    candidates = []
    if h5_name:
        if os.path.isabs(h5_name):
            candidates.append(h5_name)
        else:
            candidates.append(os.path.join(data_dir, h5_name))
    candidates.append(os.path.join(data_dir, "redd.h5"))

    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


def detect_source(config: dict, args: argparse.Namespace, data_dir: str) -> tuple[str, str | None]:
    """Return ('h5', path) or ('dat', None). Prefer HDF5 when available."""
    h5_path = resolve_h5_path(config, args, data_dir)
    if h5_path is not None:
        return "h5", h5_path
    house1 = os.path.join(data_dir, "house_1")
    if os.path.isdir(house1):
        return "dat", None
    raise FileNotFoundError(
        f"No REDD source under {data_dir}. Expected redd.h5 or house_1/channel_*.dat. "
        "Download Zenodo redd.h5 into dataset_preprocess/REDD/ or extract low_freq."
    )


def fill_short_appliance_gaps(series: pd.Series, limit: int = 3) -> pd.Series:
    if series.empty:
        return series
    filled = series.interpolate(method="linear", limit=limit, limit_area="inside")
    return filled.ffill(limit=1).bfill(limit=1)


def trim_to_common_appliance_start(
    combined: pd.DataFrame,
    appliances: list[str],
    *,
    min_power_w: float = 0.0,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    starts: list[pd.Timestamp] = []
    for app in appliances:
        col = f"{app}_power"
        if col not in combined.columns:
            continue
        active = combined[col] > min_power_w
        if not active.any():
            # Zero-filled missing appliance (--allow_missing_appliances); skip trim.
            continue
        starts.append(combined.index[active][0])
    if not starts:
        return combined, None
    common_start = max(starts)
    return combined.loc[common_start:].copy(), common_start


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


# ---------------------------------------------------------------------------
# Raw .dat readers (official low_freq layout)
# ---------------------------------------------------------------------------


def read_dat(
    path: str,
    value_name: str,
    start_ts: float | None,
    end_ts: float | None,
    tz: str,
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
    data["time"] = pd.to_datetime(data["time"], unit="s", utc=True).dt.tz_convert(tz)
    data.set_index("time", inplace=True)
    data.sort_index(inplace=True)
    return data


def first_last_dat(path: str) -> tuple[float, float]:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        first = None
        for line in handle:
            line = line.strip()
            if line:
                first = float(line.split()[0])
                break
    if first is None:
        raise ValueError(f"Empty dat file: {path}")
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
                    return first, float(line.split()[0])
                buffer.clear()
            else:
                buffer.append(char[0])
        line = bytes(reversed(buffer)).decode("utf-8", errors="ignore").strip()
        if line:
            return first, float(line.split()[0])
    raise ValueError(f"Could not read last timestamp from {path}")


# ---------------------------------------------------------------------------
# NILMTK redd.h5 readers (PyTables; no nilmtk package required)
# ---------------------------------------------------------------------------


def _h5_meter_path(house: int, meter_id: int) -> str:
    return f"/building{house}/elec/meter{meter_id}/table"


def read_h5_meter(
    h5_path: str,
    house: int,
    meter_id: int,
    value_name: str,
    start_ts: float | None,
    end_ts: float | None,
    tz: str,
) -> pd.DataFrame:
    if tb is None:
        raise ImportError("tables (PyTables) is required to read redd.h5. pip/conda install tables")

    node_path = _h5_meter_path(house, meter_id)
    with tb.open_file(h5_path, mode="r") as handle:
        if node_path not in handle:
            raise FileNotFoundError(f"Missing {node_path} in {h5_path}")
        table = handle.get_node(node_path)
        index_ns = table.col("index")
        values = np.asarray(table.col("values_block_0")).reshape(-1)

    # NILMTK stores timezone-aware ns timestamps (US/Eastern wall time as UTC ns)
    times = pd.to_datetime(index_ns, unit="ns", utc=True)
    if start_ts is not None:
        start_ns = int(pd.Timestamp(start_ts, unit="s", tz="UTC").value)
        mask_start = index_ns >= start_ns
    else:
        mask_start = np.ones(len(index_ns), dtype=bool)
    if end_ts is not None:
        end_ns = int(pd.Timestamp(end_ts, unit="s", tz="UTC").value)
        mask_end = index_ns <= end_ns
    else:
        mask_end = np.ones(len(index_ns), dtype=bool)
    mask = mask_start & mask_end
    if not mask.any():
        raise ValueError(f"No rows for house {house} meter {meter_id} in selected time range.")

    frame = pd.DataFrame({value_name: values[mask].astype(np.float32)}, index=times[mask])
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    frame.index = frame.index.tz_convert(tz)
    return frame


def first_last_h5_meter(h5_path: str, house: int, meter_id: int) -> tuple[float, float]:
    if tb is None:
        raise ImportError("tables (PyTables) is required to read redd.h5")
    node_path = _h5_meter_path(house, meter_id)
    with tb.open_file(h5_path, mode="r") as handle:
        table = handle.get_node(node_path)
        first_ns = int(table[0]["index"])
        last_ns = int(table[-1]["index"])
    first = pd.Timestamp(first_ns, unit="ns", tz="UTC").timestamp()
    last = pd.Timestamp(last_ns, unit="ns", tz="UTC").timestamp()
    return float(first), float(last)


def load_series_sum(
    *,
    source: str,
    data_dir: str,
    h5_path: str | None,
    house: int,
    meter_ids: Sequence[int],
    value_name: str,
    start_ts: float | None,
    end_ts: float | None,
    tz: str,
    sample_period: str,
) -> pd.DataFrame:
    """Load one or more meters/channels, sum on outer join, resample."""
    parts = []
    for mid in meter_ids:
        col = f"{value_name}_{mid}" if len(meter_ids) > 1 else value_name
        if source == "h5":
            assert h5_path is not None
            part = read_h5_meter(h5_path, house, mid, col, start_ts, end_ts, tz)
        else:
            path = os.path.join(data_dir, f"house_{house}", f"channel_{mid}.dat")
            part = read_dat(path, col, start_ts, end_ts, tz)
        parts.append(part)

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.join(part, how="outer")
    if len(meter_ids) > 1:
        merged[value_name] = merged.filter(like=f"{value_name}_").sum(axis=1, min_count=1)
        merged = merged[[value_name]]
    return merged.resample(sample_period).mean()


def load_mains(
    *,
    source: str,
    data_dir: str,
    h5_path: str | None,
    house: int,
    start_ts: float | None,
    end_ts: float | None,
    tz: str,
    sample_period: str,
) -> pd.DataFrame:
    print("[1/3] Loading mains (channel/meter 1 + 2)")
    mains = load_series_sum(
        source=source,
        data_dir=data_dir,
        h5_path=h5_path,
        house=house,
        meter_ids=[1, 2],
        value_name="aggregate",
        start_ts=start_ts,
        end_ts=end_ts,
        tz=tz,
        sample_period=sample_period,
    )
    mains = mains.dropna(subset=["aggregate"])
    print(f"      mains rows after resample: {len(mains):,}")
    return mains


def overlap_time_range(
    *,
    source: str,
    data_dir: str,
    h5_path: str | None,
    house: int,
    appliances: list[str],
    config: dict,
    allow_missing: bool = False,
) -> tuple[float, float]:
    """Common usable unix range where mains and every selected appliance has data."""
    meter_groups: list[list[int]] = [[1, 2]]
    for app in appliances:
        chans = appliance_channels(config, app, house)
        if chans is None:
            if allow_missing:
                continue
            raise ValueError(f"no channel_map entry for {app} in house {house}")
        meter_groups.append(chans)

    starts, ends = [], []
    for meters in meter_groups:
        for mid in meters:
            if source == "h5":
                assert h5_path is not None
                s, e = first_last_h5_meter(h5_path, house, mid)
            else:
                path = os.path.join(data_dir, f"house_{house}", f"channel_{mid}.dat")
                s, e = first_last_dat(path)
            starts.append(s)
            ends.append(e)
    start_ts, end_ts = max(starts), min(ends)
    if start_ts >= end_ts:
        raise ValueError(f"No common overlap for house {house} across mains and appliances.")
    return start_ts, end_ts


def build_one_house_lf(config: dict, args: argparse.Namespace, house: int) -> tuple[pd.DataFrame, list[str]]:
    paths, _ = resolve_paths(config)
    if args.data_dir:
        paths["data_dir"] = os.path.abspath(args.data_dir)

    global_params = config["global_params"]
    algorithm_cfg = config.get("algorithm1", {})
    appliances = selected_appliances(config, args.appliances)
    tz = global_params.get("timezone", "US/Eastern")
    sample_seconds = int(global_params.get("sample_seconds", 6))
    sample_period = f"{sample_seconds}s"

    source, h5_path = detect_source(config, args, paths["data_dir"])
    start_ts, end_ts = time_bounds(
        global_params, args.start, args.end, full_range=args.full_range
    )

    # Default: if no dates set, behave like UK-DALE --full_range (common overlap)
    allow_missing = bool(args.allow_missing_appliances)
    if (
        start_ts is None
        and end_ts is None
        and args.last_days is None
        and not args.full_range
        and not (args.start or args.end or global_params.get("start_date") or global_params.get("end_date"))
    ):
        start_ts, end_ts = overlap_time_range(
            source=source,
            data_dir=paths["data_dir"],
            h5_path=h5_path,
            house=house,
            appliances=appliances,
            config=config,
            allow_missing=allow_missing,
        )
    elif args.full_range and args.start is None and args.end is None:
        start_ts, end_ts = overlap_time_range(
            source=source,
            data_dir=paths["data_dir"],
            h5_path=h5_path,
            house=house,
            appliances=appliances,
            config=config,
            allow_missing=allow_missing,
        )

    if args.last_days is not None:
        overlap_start, overlap_end = overlap_time_range(
            source=source,
            data_dir=paths["data_dir"],
            h5_path=h5_path,
            house=house,
            appliances=appliances,
            config=config,
        )
        requested_seconds = float(args.last_days) * 86400.0
        if overlap_end - overlap_start < requested_seconds:
            available_days = (overlap_end - overlap_start) / 86400.0
            raise ValueError(
                f"House {house} has only {available_days:.2f} common days; "
                f"cannot make a {args.last_days:g}-day dataset."
            )
        end_ts = overlap_end
        start_ts = end_ts - requested_seconds

    print("=" * 72)
    print("LOW-FREQUENCY MULTI-APPLIANCE REDD PREPROCESSING")
    print("=" * 72)
    print(f"house      : {house}")
    print(f"appliances : {appliances}")
    print(f"sample     : {sample_seconds} seconds")
    print(f"source     : {source}" + (f" ({h5_path})" if h5_path else f" ({paths['data_dir']})"))
    if start_ts is not None and end_ts is not None:
        start_label = pd.to_datetime(start_ts, unit="s", utc=True).tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S")
        end_label = pd.to_datetime(end_ts, unit="s", utc=True).tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S")
        print(f"time range : {start_label} to {end_label}")

    combined = load_mains(
        source=source,
        data_dir=paths["data_dir"],
        h5_path=h5_path,
        house=house,
        start_ts=start_ts,
        end_ts=end_ts,
        tz=tz,
        sample_period=sample_period,
    )

    print("[2/3] Loading appliance channels and aligning to mains")
    for app in appliances:
        app_cfg = config["appliances"][app]
        channel_ids = appliance_channels(config, app, house)
        if channel_ids is None:
            message = f"no channel_map entry for {app} in house {house}"
            if not args.allow_missing_appliances:
                raise ValueError(
                    f"{message}. This would create false zero labels. "
                    "Use --allow_missing_appliances only for inspection."
                )
            print(f"      skip {app}: {message}")
            combined[f"{app}_power"] = 0.0
            combined[f"{app}_on"] = 0
            continue

        app_resampled = load_series_sum(
            source=source,
            data_dir=paths["data_dir"],
            h5_path=h5_path,
            house=house,
            meter_ids=channel_ids,
            value_name=app,
            start_ts=start_ts,
            end_ts=end_ts,
            tz=tz,
            sample_period=sample_period,
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

        power = np.minimum(
            aligned[app].to_numpy(dtype=np.float32),
            aligned["aggregate"].to_numpy(dtype=np.float32),
        )
        threshold = resolve_appliance_setting(app_cfg, "on_power_threshold", house, 50)
        label = make_labels(power.copy(), app_cfg, algorithm_cfg, house)
        app_frame = pd.DataFrame(
            {f"{app}_power": power, f"{app}_on": label.astype(int)},
            index=aligned.index,
        )
        combined = combined.join(app_frame, how="left")
        chan_label = "+".join(str(c) for c in channel_ids)
        print(
            f"      {app:<15} channel={chan_label:<8} thresh={threshold:<4}W "
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

    combined = combined.reset_index()
    time_col = combined.columns[0]
    if time_col != "time":
        combined.rename(columns={time_col: "time"}, inplace=True)
    combined["readable_time"] = combined["time"].dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M:%S")
    combined.drop(columns=["time"], inplace=True)
    combined.insert(1, "house", house)

    power_cols = [f"{app}_power" for app in appliances]
    on_cols = [f"{app}_on" for app in appliances]
    combined = combined[["readable_time", "house", "aggregate", *power_cols, *on_cols]]

    if args.output_mode == "zscore":
        z = add_zscore_columns(combined, config, appliances)
        keep = [
            "readable_time",
            "house",
            "aggregate_zscore",
            *[f"{app}_power_zscore" for app in appliances],
            *on_cols,
        ]
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
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    print("[3/3] Saved low-frequency multi-appliance CSV")
    print(f"output : {output_path}")
    print(f"rows   : {len(df):,}")
    print(f"columns: {list(df.columns)}")
    print_on_summary(df, appliances)
    print(f"\nDone in {(time.time() - start_time) / 60.0:.2f} min.")


if __name__ == "__main__":
    main()

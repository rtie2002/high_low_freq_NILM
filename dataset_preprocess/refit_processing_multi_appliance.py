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
import json
import math
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
    # --- REFIT alignment arguments ---
    parser.add_argument(
        "--resample_app_mode",
        choices=["mean", "max", "mean_max"],
        default="mean",
        help=(
            "Resampling aggregation for appliance columns. "
            "'mean' is the default and matches aggregate energy semantics. "
            "'max' is retained only for explicit diagnostics. "
            "'mean_max' uses mean for slow appliances and max for fast ones "
            "(fast = min_on_seconds < 120)."
        ),
    )
    parser.add_argument(
        "--alignment_mode",
        choices=["none", "audit", "event"],
        default="event",
        help=(
            "REFIT IAM timing treatment. 'event' shifts only complete appliance "
            "events whose ON and OFF aggregate edges independently support the "
            "same lag; aggregate power is never modified. 'audit' only reports."
        ),
    )
    parser.add_argument(
        "--alignment_max_lag_samples",
        type=int,
        default=2,
        help="Maximum event shift in 8-second samples (default: 2).",
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


def _wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Lower 95% Wilson bound for a binomial proportion."""
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return float((centre - spread) / denominator)


def audit_alignment(
    frame: pd.DataFrame,
    appliances: list[str],
    config: dict,
    house: int,
    *,
    max_lag_samples: int = 2,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Estimate stable IAM-to-aggregate offsets without moving single events.

    REFIT lacks the physical acquisition time of each meter, so this estimates an
    observed offset only. It rejects overlapping target edges, implausible mains
    edges, and ambiguous matches. A whole channel is eligible for shifting only
    when enough ON and OFF edges independently agree on the same offset.
    """
    if max_lag_samples < 0:
        raise ValueError("alignment_max_lag_samples must be non-negative")

    # Fixed measurement-quality gates (recorded in the report, not tuned by model).
    min_events = 30
    min_direction_events = 8
    min_appliance_edge_watts = 40.0
    other_appliance_edge_watts = 40.0
    min_aggregate_edge_watts = 20.0
    min_aggregate_fraction = 0.25
    max_relative_edge_error = 1.0
    min_score_margin = 0.10
    min_dominance = 0.70
    min_wilson_lower = 0.50

    sequence = frame["sequence_id"].to_numpy()
    aggregate_delta = (
        frame.groupby("sequence_id", sort=False)["aggregate"]
        .diff()
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    app_delta = {
        app: frame.groupby("sequence_id", sort=False)[f"{app}_power"]
        .diff()
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
        for app in appliances
    }
    lags = np.arange(-max_lag_samples, max_lag_samples + 1, dtype=np.int64)
    summaries: list[dict[str, object]] = []
    events: list[dict[str, object]] = []

    for app in appliances:
        threshold = float(
            resolve_appliance_setting(
                config["appliances"][app], "on_power_threshold", house, 50
            )
        )
        power = frame[f"{app}_power"].to_numpy(dtype=np.float64)
        state = power >= threshold
        previous_state = np.roll(state, 1)
        previous_state[0] = state[0]
        transitions = np.flatnonzero(
            (state != previous_state)
            & (np.abs(app_delta[app]) >= min_appliance_edge_watts)
            & (sequence == np.roll(sequence, 1))
        )

        accepted_lags: list[int] = []
        by_direction: dict[str, list[int]] = {"on": [], "off": []}
        rejected = {"boundary": 0, "other_target_edge": 0,
                    "no_plausible_aggregate_edge": 0, "ambiguous_lag": 0}
        for position_value in transitions:
            position = int(position_value)
            low, high = position - max_lag_samples, position + max_lag_samples
            if low < 1 or high >= len(frame) or np.any(
                sequence[low : high + 1] != sequence[position]
            ):
                rejected["boundary"] += 1
                continue
            if any(
                np.any(np.abs(app_delta[other][low : high + 1]) >= other_appliance_edge_watts)
                for other in appliances if other != app
            ):
                rejected["other_target_edge"] += 1
                continue

            target_edge = float(app_delta[app][position])
            direction = "on" if target_edge > 0 else "off"
            required_support = max(
                min_aggregate_edge_watts, min_aggregate_fraction * abs(target_edge)
            )
            candidates: list[tuple[float, int, float]] = []
            for lag_value in lags:
                lag = int(lag_value)
                aggregate_edge = float(aggregate_delta[position + lag])
                signed_support = aggregate_edge * (1.0 if target_edge > 0 else -1.0)
                if signed_support < required_support:
                    continue
                relative_error = abs(aggregate_edge - target_edge) / max(
                    abs(target_edge), min_appliance_edge_watts
                )
                if relative_error <= max_relative_edge_error:
                    candidates.append((relative_error, lag, aggregate_edge))

            if not candidates:
                rejected["no_plausible_aggregate_edge"] += 1
                continue
            candidates.sort(key=lambda item: (item[0], abs(item[1]), item[1]))
            best_error, best_lag, best_edge = candidates[0]
            second_error = candidates[1][0] if len(candidates) > 1 else float("inf")
            margin = second_error - best_error
            if margin < min_score_margin:
                rejected["ambiguous_lag"] += 1
                continue

            accepted_lags.append(best_lag)
            by_direction[direction].append(best_lag)
            events.append({
                "house": int(house), "appliance": app, "row": position,
                "time": str(frame.index[position]), "direction": direction,
                "appliance_edge_w": target_edge, "aggregate_edge_w": best_edge,
                "best_lag_samples": best_lag, "best_relative_error": best_error,
                "score_margin": margin if np.isfinite(margin) else None,
            })

        lag_counts = {int(lag): accepted_lags.count(int(lag)) for lag in lags}
        accepted = len(accepted_lags)
        dominant_lag = max(lag_counts, key=lag_counts.get) if accepted else 0
        dominant_count = lag_counts.get(dominant_lag, 0)
        dominance = dominant_count / accepted if accepted else 0.0
        wilson_lower = _wilson_lower_bound(dominant_count, accepted)
        direction_modes: dict[str, int | None] = {}
        direction_consistent = True
        for direction, direction_lags in by_direction.items():
            if len(direction_lags) < min_direction_events:
                direction_modes[direction] = None
                direction_consistent = False
                continue
            counts = {int(lag): direction_lags.count(int(lag)) for lag in lags}
            direction_modes[direction] = max(counts, key=counts.get)
            direction_consistent &= direction_modes[direction] == dominant_lag

        eligible = bool(
            accepted >= min_events and dominance >= min_dominance
            and wilson_lower >= min_wilson_lower and direction_consistent
        )
        summaries.append({
            "house": int(house), "appliance": app,
            "candidate_edges": int(len(transitions)), "accepted_edges": accepted,
            "accepted_on_edges": len(by_direction["on"]),
            "accepted_off_edges": len(by_direction["off"]),
            "dominant_lag_samples": int(dominant_lag), "dominance": float(dominance),
            "wilson_lower_95": float(wilson_lower),
            "on_lag_samples": direction_modes["on"],
            "off_lag_samples": direction_modes["off"],
            "eligible_for_fixed_shift": eligible,
            "recommended_shift_samples": int(dominant_lag) if eligible else 0,
            **{f"accepted_lag_{lag:+d}": count for lag, count in lag_counts.items()},
            **{f"rejected_{name}": count for name, count in rejected.items()},
        })
    return summaries, events


def apply_fixed_alignment_shifts(
    frame: pd.DataFrame,
    summaries: list[dict[str, object]],
    appliances: list[str],
    sample_period: str,
) -> pd.DataFrame:
    """Apply only audit-approved whole-channel shifts inside each sequence."""
    result = frame.copy()
    for summary in summaries:
        lag = int(summary["recommended_shift_samples"])
        app = str(summary["appliance"])
        summary["applied_shift_samples"] = 0
        if lag == 0 or app not in appliances:
            continue
        for suffix in ("power", "observed"):
            column = f"{app}_{suffix}"
            result[column] = result.groupby("sequence_id", sort=False)[column].shift(lag)
        summary["applied_shift_samples"] = lag

    required = ["aggregate", *[f"{app}_power" for app in appliances]]
    before = len(result)
    result = result.dropna(subset=required).copy()
    for app in appliances:
        result[f"{app}_observed"] = result[f"{app}_observed"].fillna(0).astype(np.int8)
    result["sequence_id"] = contiguous_segment_ids(result.index, sample_period)
    dropped = before - len(result)
    for summary in summaries:
        summary["rows_dropped_at_shifted_boundaries"] = dropped
    return result


def _true_runs(state: np.ndarray) -> list[tuple[int, int]]:
    """Return complete half-open ``[start, end)`` runs in a Boolean sequence."""
    padded = np.pad(state.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def realign_high_confidence_events(
    frame: pd.DataFrame,
    appliances: list[str],
    config: dict,
    house: int,
    *,
    max_lag_samples: int = 2,
    apply: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Realign only events supported by matching aggregate ON and OFF edges.

    REFIT does not retain the physical acquisition timestamp of each IAM reading.
    Consequently a single edge is insufficient evidence: an unrelated background
    load can produce a larger aggregate step nearby.  An event is shifted only
    when its rising and falling aggregate edges both select the same lag, the
    match is substantially better than every alternative, and no other monitored
    appliance changes near either boundary.

    The aggregate is immutable.  Appliance event energy above its local standby
    level is moved as a unit, so the appliance watt-sample sum is conserved.
    Ambiguous events remain unchanged.
    """
    if max_lag_samples < 0:
        raise ValueError("alignment_max_lag_samples must be non-negative")

    result = frame.copy()
    aggregate_before = frame["aggregate"].to_numpy(dtype=np.float64, copy=True)
    sequence = frame["sequence_id"].to_numpy()
    app_delta = {
        app: frame.groupby("sequence_id", sort=False)[f"{app}_power"]
        .diff().fillna(0.0).to_numpy(dtype=np.float64)
        for app in appliances
    }
    aggregate_delta = (
        frame.groupby("sequence_id", sort=False)["aggregate"]
        .diff().fillna(0.0).to_numpy(dtype=np.float64)
    )

    summaries: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    lags = range(-max_lag_samples, max_lag_samples + 1)

    for app in appliances:
        threshold = float(resolve_appliance_setting(
            config["appliances"][app], "on_power_threshold", house, 50
        ))
        original_power = frame[f"{app}_power"].to_numpy(dtype=np.float64)
        corrected_power = original_power.copy()
        state = original_power >= threshold
        accepted_by_lag = {lag: 0 for lag in lags}
        rejected = {
            "boundary": 0,
            "weak_appliance_edge": 0,
            "overlapping_target_edge": 0,
            "no_two_edge_match": 0,
            "ambiguous": 0,
            "destination_on": 0,
        }
        candidate_events = 0

        for seq_id in pd.unique(sequence):
            positions = np.flatnonzero(sequence == seq_id)
            if not len(positions):
                continue
            seq_start, seq_end = int(positions[0]), int(positions[-1]) + 1
            for start_local, end_local in _true_runs(state[seq_start:seq_end]):
                start = seq_start + start_local
                end = seq_start + end_local
                # Runs touching a sequence boundary have an unobserved edge.
                if start == seq_start or end == seq_end:
                    rejected["boundary"] += 1
                    continue
                candidate_events += 1
                rise = float(original_power[start] - original_power[start - 1])
                drop = float(original_power[end - 1] - original_power[end])
                minimum_app_edge = max(40.0, 0.30 * threshold)
                if rise < minimum_app_edge or drop < minimum_app_edge:
                    rejected["weak_appliance_edge"] += 1
                    continue

                low = start - max_lag_samples
                high = end + max_lag_samples
                if low < seq_start + 1 or high >= seq_end:
                    rejected["boundary"] += 1
                    continue

                # A nearby edge from another monitored appliance makes source
                # attribution impossible, so retain the recorded event unchanged.
                overlap = any(
                    np.any(np.abs(app_delta[other][start - max_lag_samples:
                                                   start + max_lag_samples + 1]) >= 40.0)
                    or np.any(np.abs(app_delta[other][end - max_lag_samples:
                                                     end + max_lag_samples + 1]) >= 40.0)
                    for other in appliances if other != app
                )
                if overlap:
                    rejected["overlapping_target_edge"] += 1
                    continue

                candidates: list[tuple[float, int, float, float]] = []
                for lag in lags:
                    mains_rise = float(aggregate_delta[start + lag])
                    mains_drop = float(-aggregate_delta[end + lag])
                    required_rise = max(20.0, 0.25 * rise)
                    required_drop = max(20.0, 0.25 * drop)
                    if mains_rise < required_rise or mains_drop < required_drop:
                        continue
                    rise_error = abs(mains_rise - rise) / max(rise, 40.0)
                    drop_error = abs(mains_drop - drop) / max(drop, 40.0)
                    score = 0.5 * (rise_error + drop_error)
                    if rise_error <= 1.0 and drop_error <= 1.0:
                        candidates.append((score, lag, mains_rise, mains_drop))

                if not candidates:
                    rejected["no_two_edge_match"] += 1
                    continue
                candidates.sort(key=lambda item: (item[0], abs(item[1]), item[1]))
                best = candidates[0]
                second_score = candidates[1][0] if len(candidates) > 1 else float("inf")
                if best[0] > 0.75 or second_score - best[0] < 0.20:
                    rejected["ambiguous"] += 1
                    continue

                lag = int(best[1])
                destination_start, destination_end = start + lag, end + lag
                added = (
                    range(end, destination_end) if lag > 0
                    else range(destination_start, start)
                )
                if any(state[index] for index in added):
                    rejected["destination_on"] += 1
                    continue

                accepted_by_lag[lag] += 1
                events.append({
                    "house": int(house),
                    "appliance": app,
                    "start": str(frame.index[start]),
                    "end": str(frame.index[end - 1]),
                    "lag_samples": lag,
                    "lag_seconds": lag * 8,
                    "score": float(best[0]),
                    "aggregate_rise_w": float(best[2]),
                    "aggregate_drop_w": float(best[3]),
                    "applied": bool(apply and lag != 0),
                })
                if not apply or lag == 0:
                    continue

                # Move only power above the local standby level. Subtracting and
                # re-adding the same vector preserves event energy exactly.
                neighbour = np.concatenate((
                    original_power[max(seq_start, start - 3):start],
                    original_power[end:min(seq_end, end + 3)],
                ))
                off_neighbour = neighbour[neighbour < threshold]
                standby = float(np.median(off_neighbour)) if len(off_neighbour) else 0.0
                excess = np.maximum(original_power[start:end] - standby, 0.0)
                corrected_power[start:end] -= excess
                corrected_power[destination_start:destination_end] += excess

        result[f"{app}_power"] = np.maximum(corrected_power, 0.0)
        summaries.append({
            "house": int(house),
            "appliance": app,
            "candidate_events": int(candidate_events),
            "accepted_events": int(sum(accepted_by_lag.values())),
            "shifted_events": int(sum(
                count for lag, count in accepted_by_lag.items() if lag != 0
            )) if apply else 0,
            **{f"accepted_lag_{lag:+d}": count for lag, count in accepted_by_lag.items()},
            **{f"rejected_{name}": count for name, count in rejected.items()},
        })

    if not np.array_equal(result["aggregate"].to_numpy(dtype=np.float64), aggregate_before):
        raise AssertionError("REFIT alignment must never modify aggregate power")
    return result, summaries, events


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

    # Aggregate: always mean-resampled (energy-conserving).
    # Appliances: max-resampled by default so short high-power pulses (e.g.
    # microwave bursts of 1–2 raw rows = 8–16 s) are not averaged below the
    # ON-power threshold when they straddle a bin boundary.
    resample_app_mode = getattr(args, "resample_app_mode", "max")

    def _is_fast_appliance(app_cfg: dict) -> bool:
        """Return True for appliances with a short on-cycle (< 120 s)."""
        return int(app_cfg.get("min_on_seconds", 9999)) < 120

    print(f"      resample_app_mode={resample_app_mode}")

    app_cols_present = [app for app in appliances if app in raw.columns]

    # Aggregate is always mean-resampled.
    agg_resampled_mean = raw["aggregate"].resample(sample_period).mean()

    # Appliances: resample according to mode.
    app_resampled: dict[str, pd.Series] = {}
    for app in app_cols_present:
        app_cfg = config["appliances"][app]
        if resample_app_mode == "mean":
            app_resampled[app] = raw[app].resample(sample_period).mean()
        elif resample_app_mode == "max":
            app_resampled[app] = raw[app].resample(sample_period).max()
        else:  # mean_max
            if _is_fast_appliance(app_cfg):
                app_resampled[app] = raw[app].resample(sample_period).max()
            else:
                app_resampled[app] = raw[app].resample(sample_period).mean()

    # Build resampled frame with a common DatetimeIndex.
    # All channels are resampled from the same raw index, so they share the same
    # 8 s bin boundaries. Use the aggregate index directly.
    resampled_index = agg_resampled_mean.index

    agg_gap_limit = resolve_time_samples(
        algorithm_cfg, "resample_gap_fill", house, sample_seconds, 3
    )
    combined = pd.DataFrame(index=resampled_index)
    combined["aggregate_observed"] = agg_resampled_mean.reindex(resampled_index).notna().astype(np.int8)
    combined["aggregate"] = fill_complete_short_gaps(
        agg_resampled_mean.reindex(resampled_index), agg_gap_limit
    )
    print(
        f"      aggregate bins={len(combined):,} "
        f"missing_after_short_fill={int(combined['aggregate'].isna().sum()):,}"
    )

    for app in appliances:
        app_cfg = config["appliances"][app]
        iam = appliance_iam(config, app, house)
        if iam is None or app not in app_resampled:
            message = f"no channel_map / column for {app} in house {house}"
            if not allow_missing:
                raise ValueError(message)
            print(f"      skip {app}: {message}")
            combined[f"{app}_power"] = 0.0
            combined[f"{app}_observed"] = 0
            continue

        app_series = app_resampled[app].reindex(resampled_index)
        # observed is based on the raw observation mask (pre-gap-fill).
        observed = app_resampled[app].reindex(resampled_index).notna()
        gap_limit = resolve_time_samples(
            app_cfg, "resample_gap_fill", house, sample_seconds, agg_gap_limit
        )
        combined[f"{app}_power"] = fill_complete_short_gaps(app_series, gap_limit)
        combined[f"{app}_observed"] = observed.astype(np.int8)
        print(
            f"      {app:<15} IAM={iam:<3} resample={'max' if resample_app_mode == 'max' or (resample_app_mode == 'mean_max' and _is_fast_appliance(app_cfg)) else 'mean'} "
            f"missing_after_short_fill={int(combined[f'{app}_power'].isna().sum()):,}"
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

    alignment_summaries: list[dict[str, object]] = []
    alignment_events: list[dict[str, object]] = []
    alignment_mode = str(getattr(args, "alignment_mode", "event"))
    if alignment_mode != "none":
        apply_alignment = alignment_mode == "event"
        print(
            "[3/3] Auditing REFIT event timing"
            + (" and applying high-confidence target shifts" if apply_alignment else "")
        )
        combined, alignment_summaries, alignment_events = realign_high_confidence_events(
            combined,
            appliances,
            config,
            house,
            max_lag_samples=int(getattr(args, "alignment_max_lag_samples", 2)),
            apply=apply_alignment,
        )
        for summary in alignment_summaries:
            print(
                f"      {summary['appliance']:<15} "
                f"accepted={summary['accepted_events']:,} "
                f"shifted={summary['shifted_events']:,}"
            )

    sequence_ids = combined["sequence_id"].to_numpy()
    aggregate = combined["aggregate"].to_numpy(dtype=np.float32)
    print("[labels] Computing ON/OFF labels from appliance power")
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
        # Report any bins where sub-meter power exceeds aggregate (should be
        # zero or near-zero after Algorithm 2 enforcement).
        above_mains = int(np.sum(power > aggregate))
        print(
            f"      {app:<15} thresh={threshold:<4}W "
            f"ON rows={int(labels.sum()):,}  "
            f"power>aggregate bins={above_mains:,}"
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

    combined.attrs["alignment_summary"] = alignment_summaries
    combined.attrs["alignment_events"] = alignment_events

    return combined, appliances


def build_multi_appliance_lf(
    config: dict, args: argparse.Namespace
) -> tuple[pd.DataFrame, list[str], list[int]]:
    appliances = selected_appliances(config, args.appliances)
    houses = selected_houses(config, args, appliances)
    frames = []
    alignment_summaries: list[dict[str, object]] = []
    alignment_events: list[dict[str, object]] = []
    for house in houses:
        frame, appliances = build_one_house_lf(config, args, house)
        alignment_summaries.extend(frame.attrs.get("alignment_summary", []))
        alignment_events.extend(frame.attrs.get("alignment_events", []))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["house", "readable_time"]).reset_index(drop=True)
    combined.attrs["alignment_summary"] = alignment_summaries
    combined.attrs["alignment_events"] = alignment_events
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


def write_alignment_reports(df: pd.DataFrame, output_path: str) -> None:
    """Write the accepted/rejected event audit beside a generated CSV."""
    stem, _ = os.path.splitext(output_path)
    summaries = df.attrs.get("alignment_summary", [])
    events = df.attrs.get("alignment_events", [])
    if summaries:
        pd.DataFrame(summaries).to_csv(f"{stem}.alignment_summary.csv", index=False)
    if events:
        pd.DataFrame(events).to_csv(f"{stem}.alignment_events.csv", index=False)


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
            write_alignment_reports(df, output_path)
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
    write_alignment_reports(df, output_path)

    print("[3/3] Saved low-frequency multi-appliance CSV")
    print(f"output : {output_path}")
    print(f"rows   : {len(df):,}")
    print(f"columns: {list(df.columns)}")
    print_on_summary(df, appliances)
    print(f"\nDone in {(time.time() - start_time) / 60.0:.2f} min.")


if __name__ == "__main__":
    main()

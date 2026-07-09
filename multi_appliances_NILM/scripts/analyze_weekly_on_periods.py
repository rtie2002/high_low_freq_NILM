#!/usr/bin/env python
"""Summarize weekly appliance ON activity from full-house CSV files.

Use this to answer:
    "Which week has the most ON activity across all appliances?"

The script prefers binary *_on columns. If they are missing, it can rebuild
ON/OFF labels from power columns using thresholds from an experiment yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.config import load_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank weeks by total appliance ON activity.")
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more full-house CSV files, e.g. house1/house2/house5.",
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=ROOT / "config" / "experiment_ukdale.yaml",
        help="Experiment yaml used for appliance names, state columns, and thresholds.",
    )
    parser.add_argument(
        "--timestamp-column",
        type=str,
        default=None,
        help="Timestamp column name. If omitted, auto-detect common names.",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=6.0,
        help="Used when no timestamp column exists; UK-DALE default is 6 s.",
    )
    parser.add_argument(
        "--week-start",
        type=str,
        default="MON",
        choices=["MON", "SUN"],
        help="How to define weekly buckets.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top weeks to print per file.",
    )
    parser.add_argument(
        "--block-weeks",
        type=int,
        nargs="*",
        default=[2, 4, 8],
        help="Continuous block lengths (in weeks) to rank.",
    )
    return parser.parse_args()


def _detect_timestamp_column(df: pd.DataFrame) -> str | None:
    for name in ("readable_time", "timestamp", "datetime", "time", "DateTime", "date", "Date"):
        if name in df.columns:
            return name
    return None


def _period_alias(week_start: str) -> str:
    return "W-MON" if week_start == "MON" else "W-SUN"


def _on_episode_count(state: pd.Series) -> int:
    state_i = state.astype("int8")
    return int(((state_i == 1) & (state_i.shift(fill_value=0) == 0)).sum())


def _house_label(full_df: pd.DataFrame, csv_path: Path) -> str:
    if "house" in full_df.columns and not full_df["house"].empty:
        return f"house {full_df['house'].iloc[0]}"
    return csv_path.stem


def _load_state_frame(csv_path: Path, experiment_cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    csv_cfg = experiment_cfg["csv"]
    app_cfg = csv_cfg["appliances"]
    appliances = list(app_cfg.keys())

    df = pd.read_csv(csv_path)
    state_columns = {app: app_cfg[app]["state"] for app in appliances}
    power_columns = {app: app_cfg[app]["power"] for app in appliances}
    thresholds = experiment_cfg.get("evaluation", {}).get("on_thresholds_watts", {})
    if not thresholds:
        raise ValueError(
            f"{csv_path}: experiment.evaluation.on_thresholds_watts is required"
        )

    state_df = pd.DataFrame(index=df.index)
    for app in appliances:
        state_col = state_columns[app]
        power_col = power_columns[app]
        if state_col in df.columns:
            state_df[app] = df[state_col].fillna(0).astype(int)
        elif power_col in df.columns:
            if app not in thresholds:
                raise ValueError(
                    f"Missing on_thresholds_watts for appliance '{app}' in experiment yaml"
                )
            thr = float(thresholds[app])
            state_df[app] = (df[power_col].fillna(0) > thr).astype(int)
        else:
            raise ValueError(f"Missing both state and power columns for appliance '{app}' in {csv_path}")
    return state_df, appliances


def _extract_on_events(
    full_df: pd.DataFrame,
    state_df: pd.DataFrame,
    appliances: list[str],
    *,
    timestamp_column: str | None,
    sample_seconds: float,
    week_start: str,
    house_label: str,
) -> pd.DataFrame:
    ts_col = timestamp_column or _detect_timestamp_column(full_df)
    if ts_col is not None:
        timestamp = pd.to_datetime(full_df[ts_col], errors="coerce")
    else:
        timestamp = pd.Series(pd.NaT, index=full_df.index)

    week_key = (
        timestamp.dt.to_period(_period_alias(week_start)).astype(str)
        if ts_col is not None
        else (pd.Series(range(len(full_df)), index=full_df.index) // int(round((7 * 24 * 3600) / sample_seconds))).astype(str)
    )

    rows: list[dict[str, object]] = []
    for app in appliances:
        state = state_df[app].fillna(0).astype(int)
        in_event = False
        start_idx = -1
        for idx, value in enumerate(state):
            if value == 1 and not in_event:
                in_event = True
                start_idx = idx
            end_now = in_event and (value == 0 or idx == len(state) - 1)
            if not end_now:
                continue

            end_idx = idx - 1 if value == 0 else idx
            steps = end_idx - start_idx + 1
            start_time = timestamp.iloc[start_idx] if ts_col is not None else pd.NaT
            end_time = timestamp.iloc[end_idx] if ts_col is not None else pd.NaT
            rows.append(
                {
                    "house": house_label,
                    "appliance": app,
                    "week": week_key.iloc[start_idx],
                    "start_index": int(start_idx),
                    "end_index": int(end_idx),
                    "timesteps": int(steps),
                    "duration_seconds": float(steps * sample_seconds),
                    "duration_minutes": float(steps * sample_seconds / 60.0),
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )
            in_event = False
            start_idx = -1

    return pd.DataFrame(rows)


def summarize_csv(
    csv_path: Path,
    experiment_cfg: dict[str, Any],
    timestamp_column: str | None,
    sample_seconds: float,
    week_start: str,
) -> pd.DataFrame:
    full_df = pd.read_csv(csv_path)
    ts_col = timestamp_column or _detect_timestamp_column(full_df)

    state_df, appliances = _load_state_frame(csv_path, experiment_cfg)

    if ts_col is not None:
        timestamp = pd.to_datetime(full_df[ts_col], errors="coerce")
        if timestamp.isna().all():
            raise ValueError(f"Could not parse timestamp column '{ts_col}' in {csv_path}")
        week_key = timestamp.dt.to_period(_period_alias(week_start))
    else:
        rows_per_week = int(round((7 * 24 * 3600) / sample_seconds))
        week_key = pd.Series(range(len(full_df)), index=full_df.index) // rows_per_week

    rows = []
    grouped = state_df.groupby(week_key)
    for week_id, group in grouped:
        if len(group) == 0:
            continue
        on_samples = group.sum(axis=0)
        on_hours = on_samples * sample_seconds / 3600.0
        on_episodes = {app: _on_episode_count(group[app]) for app in appliances}

        row = {
            "week": str(week_id),
            "rows": int(len(group)),
            "total_on_hours_all_appliances": float(on_hours.sum()),
            "total_on_episodes_all_appliances": int(sum(on_episodes.values())),
        }
        for app in appliances:
            row[f"{app}_on_hours"] = float(on_hours[app])
            row[f"{app}_on_episodes"] = int(on_episodes[app])
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["total_on_hours_all_appliances", "total_on_episodes_all_appliances"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _top_week_appliance_table(summary: pd.DataFrame, appliances: list[str], top_k: int) -> pd.DataFrame:
    rows = []
    for _, row in summary.head(top_k).iterrows():
        out: dict[str, object] = {
            "week": row["week"],
            "total_hours": round(float(row["total_on_hours_all_appliances"]), 2),
            "total_events": int(row["total_on_episodes_all_appliances"]),
        }
        for app in appliances:
            out[f"{app}_h"] = round(float(row[f"{app}_on_hours"]), 2)
            out[f"{app}_e"] = int(row[f"{app}_on_episodes"])
        rows.append(out)
    return pd.DataFrame(rows)


def _best_blocks(summary: pd.DataFrame, appliances: list[str], block_weeks: list[int]) -> dict[int, pd.DataFrame]:
    if summary.empty:
        return {}

    chrono = summary.copy()
    chrono["_order"] = pd.to_datetime(chrono["week"].str.split("/").str[0], errors="coerce")
    if chrono["_order"].isna().any():
        chrono["_order"] = range(len(chrono))
    chrono = chrono.sort_values("_order").reset_index(drop=True)

    numeric_cols = [c for c in chrono.columns if c not in ("week", "_order")]
    out: dict[int, pd.DataFrame] = {}
    for block_len in block_weeks:
        if block_len <= 0 or len(chrono) < block_len:
            continue
        rows = []
        for start in range(0, len(chrono) - block_len + 1):
            block = chrono.iloc[start : start + block_len]
            row: dict[str, object] = {
                "start_week": block.iloc[0]["week"],
                "end_week": block.iloc[-1]["week"],
                "weeks": int(block_len),
            }
            for col in numeric_cols:
                row[col] = float(block[col].sum()) if "hours" in col else int(block[col].sum())
            rows.append(row)
        block_df = pd.DataFrame(rows).sort_values(
            ["total_on_hours_all_appliances", "total_on_episodes_all_appliances"],
            ascending=[False, False],
        )
        out[block_len] = block_df.reset_index(drop=True)
    return out


def _compact_app_breakdown(summary: pd.DataFrame, appliances: list[str], top_k: int) -> pd.DataFrame:
    rows = []
    for _, row in summary.head(top_k).iterrows():
        app_parts = []
        for app in appliances:
            hours = float(row[f"{app}_on_hours"])
            events = int(row[f"{app}_on_episodes"])
            app_parts.append(f"{app}:{hours:.1f}h/{events}e")
        rows.append(
            {
                "week": row["week"],
                "total_h": round(float(row["total_on_hours_all_appliances"]), 2),
                "total_e": int(row["total_on_episodes_all_appliances"]),
                "per_appliance": " | ".join(app_parts),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    experiment_cfg = load_experiment(args.experiment)
    appliances = list(experiment_cfg["csv"]["appliances"].keys())

    for csv_path in args.csv:
        print("=" * 100)
        print(f"FILE: {csv_path}")
        if not csv_path.exists():
            print(f"Skipping missing file: {csv_path}")
            continue
        try:
            full_df = pd.read_csv(csv_path)
            house_label = _house_label(full_df, csv_path)
            state_df, _ = _load_state_frame(csv_path, experiment_cfg)
            summary = summarize_csv(
                csv_path=csv_path,
                experiment_cfg=experiment_cfg,
                timestamp_column=args.timestamp_column,
                sample_seconds=args.sample_seconds,
                week_start=args.week_start,
            )
        except Exception as exc:
            print(f"Failed to summarize {csv_path}: {exc}")
            continue
        if summary.empty:
            print("No weekly summary could be created.")
            continue

        print(f"\nDetected: {house_label}")
        print("\nTop weeks by total ON hours across all appliances:\n")
        cols = [
            "week",
            "rows",
            "total_on_hours_all_appliances",
            "total_on_episodes_all_appliances",
        ]
        print(summary[cols].head(args.top_k).to_string(index=False))

        print("\nPer-appliance breakdown for top weeks:\n")
        print(_compact_app_breakdown(summary, appliances, args.top_k).to_string(index=False))

        events = _extract_on_events(
            full_df,
            state_df,
            appliances,
            timestamp_column=args.timestamp_column,
            sample_seconds=args.sample_seconds,
            week_start=args.week_start,
            house_label=house_label,
        )
        if not events.empty:
            print("\nSample ON events (first 15 by longest duration):\n")
            event_preview = events.sort_values(["duration_seconds", "timesteps"], ascending=[False, False]).head(15)
            show_cols = [
                "house",
                "appliance",
                "week",
                "start_time",
                "end_time",
                "timesteps",
                "duration_minutes",
            ]
            print(event_preview[show_cols].to_string(index=False))

        blocks = _best_blocks(summary, appliances, args.block_weeks)
        for block_len, block_df in blocks.items():
            if block_df.empty:
                continue
            print(f"\nBest continuous {block_len}-week blocks:\n")
            display_cols = [
                "start_week",
                "end_week",
                "total_on_hours_all_appliances",
                "total_on_episodes_all_appliances",
            ]
            print(block_df[display_cols].head(min(5, args.top_k)).to_string(index=False))

        out_path = csv_path.with_name(csv_path.stem + "_weekly_on_summary.csv")
        summary.to_csv(out_path, index=False)
        print(f"\nSaved full weekly summary to: {out_path}")
        events_path = csv_path.with_name(csv_path.stem + "_on_events.csv")
        events.to_csv(events_path, index=False)
        print(f"Saved ON event table to: {events_path}")


if __name__ == "__main__":
    main()

"""
Multi-appliance version of high_frequency_data_extract.py.

This script follows the original extractor pipeline:

    raw HF .flac
    -> compute HF features with hf_feature.py
    -> load/resample LF aggregate + appliance channels from ukdale.yaml
    -> align LF labels to each HF timestamp
    -> save ONE multi-appliance CSV

Difference from high_frequency_data_extract.py:

    original output:
        one CSV per appliance, each with one appliance_power + on_off

    this output:
        one CSV containing all appliance_power and appliance_on columns together

Default use:
    python dataset_preprocess/high_frequency_data_extract/high_frequency_data_extract_multi_appliance.py

Optional:
    python dataset_preprocess/high_frequency_data_extract/high_frequency_data_extract_multi_appliance.py --weeks wk30,wk31
    python dataset_preprocess/high_frequency_data_extract/high_frequency_data_extract_multi_appliance.py --appliances kettle,fridge,microwave,dishwasher,washingmachine
    python dataset_preprocess/high_frequency_data_extract/high_frequency_data_extract_multi_appliance.py --no_plot
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional
    plt = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

# Import the original extractor and reuse its exact loading/extraction/fusion logic.
sys.path.insert(0, SCRIPT_DIR)
import high_frequency_data_extract as hfe  # noqa: E402


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-appliance NILM HF extractor + LF fusion"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(SCRIPT_DIR, "hf_config.yaml"),
        help="Path to hf_config.yaml",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Single .flac or folder. If omitted, uses batch section in hf_config.yaml.",
    )
    parser.add_argument("--lf_config", type=str, default=None)
    parser.add_argument("--weeks", type=str, default=None, help="Example: wk30,wk31")
    parser.add_argument(
        "--appliances",
        type=str,
        default=None,
        help="Example: kettle,fridge,microwave,dishwasher,washingmachine",
    )
    parser.add_argument("--verbose_windows", action="store_true")
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional output filename for single/folder mode",
    )
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument(
        "--no_save_plot",
        action="store_true",
        help="Do not save the ON/OFF Gantt chart PNG",
    )
    parser.add_argument(
        "--plot_png",
        type=str,
        default=None,
        help="Optional output PNG path for the ON/OFF Gantt chart",
    )
    parser.add_argument(
        "--plot_max_points",
        type=int,
        default=0,
        help="Maximum rows shown in ON/OFF Gantt plot. Use 0 for all rows.",
    )
    return parser.parse_args()


def app_power_col(app: str) -> str:
    return f"{app}_power"


def app_on_col(app: str) -> str:
    return f"{app}_on"


def combine_fused_dfs(fused_dfs: dict[str, pd.DataFrame], appliances: list[str]) -> pd.DataFrame:
    """
    Combine per-appliance fused outputs from the original extractor.

    Each original fused dataframe has:

        readable_time + HF features + aggregate + {appliance}_power + on_off

    This function keeps one copy of HF features/aggregate and appends all
    appliance power/on columns.
    """
    available = [app for app in appliances if app in fused_dfs and not fused_dfs[app].empty]
    if not available:
        return pd.DataFrame()

    first_df = fused_dfs[available[0]].copy()
    first_power_cols = {app_power_col(app) for app in appliances}
    drop_cols = first_power_cols | {"on_off"}
    base_cols = [col for col in first_df.columns if col not in drop_cols]
    combined = first_df[base_cols].copy()

    for app in available[1:]:
        df = fused_dfs[app].copy()
        app_base_cols = [col for col in base_cols if col in df.columns]
        if "readable_time" not in app_base_cols:
            continue

        combined = pd.merge(
            combined,
            df[app_base_cols],
            on="readable_time",
            how="outer",
            suffixes=("", "__new"),
        )
        for col in app_base_cols:
            if col == "readable_time":
                continue
            new_col = f"{col}__new"
            if new_col in combined.columns:
                combined[col] = combined[col].combine_first(combined[new_col])
                combined = combined.drop(columns=[new_col])

    for app in available:
        df = fused_dfs[app].copy()
        power_col = app_power_col(app)
        if power_col not in df.columns:
            continue

        label_cols = ["readable_time", power_col]
        if "on_off" in df.columns:
            df = df.rename(columns={"on_off": app_on_col(app)})
            label_cols.append(app_on_col(app))
        else:
            df[app_on_col(app)] = (pd.to_numeric(df[power_col], errors="coerce").fillna(0) > 0).astype(int)
            label_cols.append(app_on_col(app))

        combined = pd.merge(
            combined,
            df[label_cols],
            on="readable_time",
            how="outer",
        )

    combined = combined.sort_values("readable_time").reset_index(drop=True)

    for app in appliances:
        p_col = app_power_col(app)
        o_col = app_on_col(app)
        if p_col not in combined.columns:
            combined[p_col] = 0.0
        if o_col not in combined.columns:
            combined[o_col] = 0
        combined[p_col] = pd.to_numeric(combined[p_col], errors="coerce").fillna(0.0)
        combined[o_col] = pd.to_numeric(combined[o_col], errors="coerce").fillna(0).astype(int)

    power_cols = [app_power_col(app) for app in appliances]
    on_cols = [app_on_col(app) for app in appliances]
    tail = ["aggregate"] + power_cols + on_cols if "aggregate" in combined.columns else power_cols + on_cols
    front = ["readable_time"]
    middle = [col for col in combined.columns if col not in set(front + tail)]
    return combined[front + middle + tail]


def write_chunk(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = not os.path.exists(path)
    df.to_csv(path, index=False, mode="a", header=header)


def count_csv_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        return max(0, sum(1 for _ in f) - 1)


def gantt_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    clean = np.asarray(mask).astype(float)
    clean = np.nan_to_num(clean, nan=0.0)
    clean = (clean > 0).astype(int)
    diff = np.diff(np.concatenate([[0], clean, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def show_gantt(
    csv_path: str,
    appliances: list[str],
    max_points: int,
    png_path: str | None = None,
    show: bool = True,
) -> None:
    if plt is None:
        print("[PLOT] matplotlib unavailable; skip ON/OFF Gantt chart.")
        return
    if not os.path.exists(csv_path):
        return

    import matplotlib.dates as mdates

    df = pd.read_csv(csv_path)
    if max_points > 0 and len(df) > max_points:
        df = df.iloc[:max_points].copy()
        suffix = f" | first {max_points:,} rows"
    else:
        suffix = " | full timeline"

    x = pd.to_datetime(df["readable_time"], errors="coerce")
    if x.isna().all():
        x = pd.Series(np.arange(len(df)), index=df.index)
        time_axis = False
    else:
        time_axis = True

    fig, ax = plt.subplots(figsize=(17, max(5, len(appliances) * 0.9)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfb")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    yticks = []
    ylabels = []
    summary_rows = []
    if time_axis and len(x) > 1:
        step_seconds = x.sort_values().diff().dt.total_seconds().dropna()
        step_seconds = step_seconds[step_seconds > 0]
        sample_seconds = float(step_seconds.median()) if not step_seconds.empty else 6.0
        min_marker_seconds = max(sample_seconds, 20 * 60)
        x_num = mdates.date2num(x.dt.to_pydatetime())
        min_marker_width = min_marker_seconds / 86400.0
    else:
        sample_seconds = 1.0
        x_num = np.asarray(x)
        min_marker_width = 1.0

    for idx, app in enumerate(appliances):
        col = app_on_col(app)
        if col not in df.columns:
            continue
        y = len(appliances) - idx - 1
        yticks.append(y)
        ylabels.append(app)
        color = colors[idx % len(colors)]

        ax.axhspan(y - 0.42, y + 0.42, color="#f2f2f2" if idx % 2 else "#ffffff", zorder=0)
        segments = gantt_segments(df[col].values)
        for start, end in segments:
            if time_axis:
                x_start = x_num[start]
                if end < len(x_num):
                    width = max(x_num[end] - x_start, sample_seconds / 86400.0)
                else:
                    width = sample_seconds / 86400.0

                if width < min_marker_width:
                    ax.vlines(
                        x_start,
                        y - 0.34,
                        y + 0.34,
                        color=color,
                        linewidth=2.2,
                        alpha=0.95,
                        zorder=3,
                    )
                    ax.plot(
                        x_start,
                        y,
                        marker="o",
                        markersize=3.2,
                        color=color,
                        alpha=0.95,
                        zorder=4,
                    )
                else:
                    ax.broken_barh(
                        [(x_start, width)],
                        (y - 0.28, 0.56),
                        facecolors=color,
                        edgecolors=color,
                        linewidth=0.8,
                        alpha=0.9,
                        zorder=2,
                    )
            else:
                width = end - start
                if width <= 1:
                    ax.vlines(start, y - 0.34, y + 0.34, color=color, linewidth=2.2)
                else:
                    ax.broken_barh(
                        [(start, width)],
                        (y - 0.28, 0.56),
                        facecolors=color,
                        edgecolors=color,
                        linewidth=0.8,
                        alpha=0.9,
                    )

        on_rows = int(pd.to_numeric(df[col], errors="coerce").fillna(0).gt(0).sum())
        duration_seconds = on_rows * sample_seconds
        duration_hours = duration_seconds / 3600.0
        summary_rows.append((app, len(segments), on_rows, duration_hours))
        ax.text(
            1.005,
            y,
            f"{len(segments):>3} events | {duration_hours:>6.2f} h | {on_rows:,} rows",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=9.5,
            color="#444444",
            family="monospace",
        )

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(-0.6, len(appliances) - 0.4)
    ax.set_xlabel("Time" if time_axis else "Row index")
    title = f"Multi-appliance ON/OFF timeline: {os.path.basename(csv_path)}{suffix}"
    if time_axis and len(x) > 0:
        title += f"\n{x.iloc[0].strftime('%Y-%m-%d %H:%M')} to {x.iloc[-1].strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(title, fontsize=13, pad=14)
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.grid(axis="y", alpha=0.08, linewidth=0.8)
    ax.text(
        1.005,
        1.025,
        "events | ON duration | ON rows",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#333333",
        family="monospace",
    )
    if time_axis:
        locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate(rotation=0)
    fig.subplots_adjust(left=0.12, right=0.78, top=0.85, bottom=0.12)
    if png_path:
        os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Saved ON/OFF Gantt chart: {png_path}")
    print("\n[PLOT] ON/OFF event summary")
    print("  appliance        events    ON hours     ON rows")
    for app, event_count, on_rows, duration_hours in summary_rows:
        print(f"  {app:<15} {event_count:>6} {duration_hours:>11.2f} {on_rows:>11,}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def flac_files_from_input(input_path: str) -> list[str]:
    path = os.path.abspath(input_path)
    if os.path.isfile(path) and path.endswith(".flac"):
        return [path]
    if os.path.isdir(path):
        return sorted(os.path.join(path, name) for name in os.listdir(path) if name.endswith(".flac"))
    raise FileNotFoundError(f"Invalid --input_path: {input_path}")


def output_name_for_week(config: dict, week: str | None, weeks: list[str] | None = None) -> str:
    house_id = config["hyperparameters"].get("house_id", 2)
    if weeks and len(weeks) > 1:
        return f"multi_appliance_house{house_id}_{weeks[0]}_to_{weeks[-1]}_merged.csv"
    if week:
        return f"multi_appliance_house{house_id}_{week}.csv"
    return f"multi_appliance_house{house_id}.csv"


def run_multi_flac_pipeline(
    flac_files: list[str],
    config: dict,
    lf_config_path: str | None,
    appliances: list[str],
    output_csv: str,
    week_label: str | None,
    lf_start_ts: float | None = None,
    lf_end_ts: float | None = None,
) -> str:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if os.path.exists(output_csv):
        os.remove(output_csv)

    house_id = config["hyperparameters"].get("house_id", 2)
    lf_cache = None
    if lf_config_path and os.path.exists(lf_config_path):
        lf_cache = hfe.load_lf_data(
            lf_config_path,
            house_id,
            appliances,
            start_ts_override=lf_start_ts,
            end_ts_override=lf_end_ts,
        )
    else:
        raise FileNotFoundError(f"LF config not found: {lf_config_path}")

    for idx, flac_path in enumerate(flac_files):
        print(f"\n[{idx + 1}/{len(flac_files)}] Multi-appliance processing: {os.path.basename(flac_path)}")
        fused = hfe.process_file(
            flac_path,
            config,
            lf_cache=lf_cache,
            save_hf_csv=False,
            appliances_filter=appliances,
        )
        combined = combine_fused_dfs(fused, appliances)
        if combined.empty:
            print("  [MULTI] No rows fused for this FLAC.")
            continue
        write_chunk(output_csv, combined)
        print(f"  [MULTI] Appended {len(combined):,} rows -> {os.path.basename(output_csv)}")

    print("\n" + "=" * 72)
    print(f"MULTI-APPLIANCE OUTPUT COMPLETE {week_label or ''}")
    print("=" * 72)
    print(f"output : {output_csv}")
    print(f"rows   : {count_csv_rows(output_csv):,}")
    print("=" * 72)
    return output_csv


def run_batch(args: argparse.Namespace, config: dict, appliances: list[str], lf_path: str) -> list[str]:
    batch = config.get("batch", {})
    weeks = [
        w.strip()
        for w in (args.weeks or ",".join(batch.get("weeks", []))).split(",")
        if w.strip()
    ]
    if not weeks:
        raise ValueError("No weeks configured. Set batch.weeks or use --weeks wk30,wk31.")

    output_dir = config["paths"]["save_path"]
    outputs = []
    all_week_frames = []

    for week in weeks:
        week_dir = hfe.week_directory(config, week)
        if not os.path.isdir(week_dir):
            print(f"[SKIP] Week folder not found: {week_dir}")
            continue
        flacs = sorted(os.path.join(week_dir, name) for name in os.listdir(week_dir) if name.endswith(".flac"))
        if not flacs:
            print(f"[SKIP] No FLAC files in: {week_dir}")
            continue

        start_ts, end_ts = hfe.flac_time_range(flacs)
        out_csv = os.path.join(output_dir, output_name_for_week(config, week))
        run_multi_flac_pipeline(
            flacs,
            config,
            lf_path,
            appliances,
            out_csv,
            week_label=week,
            lf_start_ts=start_ts,
            lf_end_ts=end_ts,
        )
        outputs.append(out_csv)
        all_week_frames.append(pd.read_csv(out_csv))

    if len(all_week_frames) > 1:
        merged = pd.concat(all_week_frames, ignore_index=True)
        merged = merged.sort_values("readable_time").reset_index(drop=True)
        merged_csv = os.path.join(output_dir, output_name_for_week(config, None, weeks))
        merged.to_csv(merged_csv, index=False)
        outputs.append(merged_csv)
        print(f"\n[MERGED] {merged_csv} ({len(merged):,} rows)")

    return outputs


def main() -> None:
    args = get_arguments()
    config, _ = hfe.load_hf_config(args.config)
    if args.verbose_windows:
        config.setdefault("logging", {})["verbose_windows"] = True

    appliances = hfe.get_appliances_filter(config, args.appliances)
    if not appliances:
        appliances = ["kettle", "fridge", "microwave", "dishwasher", "washingmachine"]

    lf_path = args.lf_config or config["paths"].get("lf_config")

    if args.input_path:
        flacs = flac_files_from_input(args.input_path)
        start_ts, end_ts = hfe.flac_time_range(flacs)
        output_dir = config["paths"]["save_path"]
        out_name = args.output_name or output_name_for_week(config, None)
        out_csv = os.path.join(output_dir, out_name)
        outputs = [
            run_multi_flac_pipeline(
                flacs,
                config,
                lf_path,
                appliances,
                out_csv,
                week_label=None,
                lf_start_ts=start_ts,
                lf_end_ts=end_ts,
            )
        ]
    else:
        outputs = run_batch(args, config, appliances, lf_path)

    if outputs and (not args.no_plot or not args.no_save_plot):
        if args.no_save_plot:
            png_path = None
        else:
            png_path = args.plot_png
            if not png_path:
                base, _ = os.path.splitext(outputs[-1])
                png_path = f"{base}_on_gantt.png"
        show_gantt(
            outputs[-1],
            appliances,
            args.plot_max_points,
            png_path=png_path,
            show=not args.no_plot,
        )

    print("\n[DONE] Multi-appliance HF extraction finished.")


if __name__ == "__main__":
    main()

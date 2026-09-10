"""
Recompute ON/OFF labels in existing *_lf_8s.csv house files from power columns.

Does not re-read raw meters or change power / sequence_id / aggregate — only
`<appliance>_on` columns, using the current preprocess YAML thresholds.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ukdale_processing import (
    apply_algorithm1_labeling,
    label_power_by_segments,
    resolve_appliance_setting,
    resolve_time_samples,
    write_dataframe_csv,
)


APPS = ["washingmachine", "dishwasher", "fridge", "kettle", "microwave"]


def make_labels(
    power: np.ndarray,
    appliance_cfg: dict,
    algorithm_cfg: dict,
    house: int,
    sample_seconds: int,
) -> np.ndarray:
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
                appliance_cfg, "spike_window", house, algorithm_cfg.get("spike_window", 5)
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


def parse_house_csv(path: Path) -> tuple[str, int]:
    # ukdale_house5_lf_8s.csv / refit_house11_lf_8s.csv
    stem = path.name.replace("_lf_8s.csv", "")
    ds, house_s = stem.split("_house")
    return ds, int(house_s)


def relabel_file(path: Path, config: dict) -> dict[str, tuple[int, int]]:
    ds, house = parse_house_csv(path)
    apps_cfg = config.get("appliances", {})
    algorithm_cfg = config.get("algorithm1", {})
    sample_seconds = int(config["global_params"]["sample_seconds"])
    df = pd.read_csv(path)
    if "sequence_id" not in df.columns:
        raise ValueError(f"{path}: missing sequence_id")

    seq = df["sequence_id"].to_numpy()
    changes: dict[str, tuple[int, int]] = {}
    for app in APPS:
        power_col = f"{app}_power"
        on_col = f"{app}_on"
        if power_col not in df.columns or on_col not in df.columns:
            continue
        app_cfg = apps_cfg.get(app)
        if not app_cfg:
            continue
        old_on = int(df[on_col].sum())
        power = df[power_col].to_numpy(dtype=np.float64)
        labels = label_power_by_segments(
            power,
            seq,
            lambda p, _cfg=app_cfg: make_labels(
                p, _cfg, algorithm_cfg, house, sample_seconds
            ),
        )
        df[on_col] = labels.astype(np.int8)
        new_on = int(df[on_col].sum())
        changes[app] = (old_on, new_on)

    write_dataframe_csv(df, str(path))
    return changes


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ukdale-dir",
        type=Path,
        default=project_root / "multi_appliances_NILM" / "datasets" / "ukdale",
    )
    parser.add_argument(
        "--refit-dir",
        type=Path,
        default=project_root / "multi_appliances_NILM" / "datasets" / "refit",
    )
    parser.add_argument(
        "--ukdale-config",
        type=Path,
        default=project_root / "config" / "preprocess" / "ukdale.yaml",
    )
    parser.add_argument(
        "--refit-config",
        type=Path,
        default=project_root / "config" / "preprocess" / "refit.yaml",
    )
    parser.add_argument(
        "--datasets",
        default="ukdale,refit",
        help="Comma list: ukdale,refit",
    )
    args = parser.parse_args()

    jobs: list[tuple[str, Path, Path]] = []
    wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
    if "ukdale" in wanted:
        for p in sorted(args.ukdale_dir.glob("ukdale_house*_lf_8s.csv")):
            jobs.append(("ukdale", args.ukdale_config, p))
    if "refit" in wanted:
        for p in sorted(args.refit_dir.glob("refit_house*_lf_8s.csv")):
            jobs.append(("refit", args.refit_config, p))

    configs: dict[str, dict] = {}
    for ds, cfg_path, _ in jobs:
        if ds not in configs:
            with open(cfg_path, "r", encoding="utf-8") as f:
                configs[ds] = yaml.safe_load(f)

    print(f"Relabeling {len(jobs)} house CSV(s)...", flush=True)
    for ds, _, path in jobs:
        print(f"  {path.name}", flush=True)
        changes = relabel_file(path, configs[ds])
        for app, (old_on, new_on) in changes.items():
            delta = new_on - old_on
            print(f"    {app}: on {old_on} -> {new_on} ({delta:+d})", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()

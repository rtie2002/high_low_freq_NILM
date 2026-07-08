#!/usr/bin/env python
"""Export official UNETNiLM `data.zip` (.npy) into CSV splits.

Source zip (in this repo):
  NILM_model/baseline/UNETNILM/data.zip

The official zip stores *preprocessed* arrays and does not include timestamps.
So the exported CSVs use an integer `row` index column.

For each split, we export CSVs in the *same column style* as the rest of
`multi_appliances_NILM`:
  - readable_time  (string; here it's just the integer row index)
  - house          (int; official UNETNiLM zip is UK-DALE House 1)
  - aggregate      (recovered from official `noise_inputs.npy`)
  - sub_mains      (recovered from official `denoise_inputs.npy`)
  - {appliance}_power (recovered from official normalized targets)
  - {appliance}_on

These "recovered" power values are the official median-filtered watt signals after
inverse normalization. They are not the original timestamped raw UK-DALE CSV.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

APPLIANCES = ["kettle", "fridge", "dishwasher", "washingmachine", "microwave"]
ZIP_SPLITS = {"training": "training", "validating": "validation", "testing": "test"}
APP_STATS = {
    "kettle": {"mean": 700.0, "std": 1000.0},
    "fridge": {"mean": 200.0, "std": 400.0},
    "dishwasher": {"mean": 700.0, "std": 700.0},
    "washingmachine": {"mean": 400.0, "std": 700.0},
    "microwave": {"mean": 500.0, "std": 800.0},
}
NOISE_MEAN = 389.0
NOISE_STD = 445.0
DENOISE_MEAN = 123.0
DENOISE_STD = 369.0
HOUSE_ID = 1

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_ZIP = REPO_ROOT / "NILM_model" / "baseline" / "UNETNILM" / "data.zip"
DEFAULT_OUT = ROOT / "datasets" / "unetnilm_official_zip_csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def _load_arrays(extracted_root: Path, split: str) -> dict[str, np.ndarray]:
    d = extracted_root / "data" / "ukdale" / split
    return {
        "noise": np.load(d / "noise_inputs.npy"),
        "denoise": np.load(d / "denoise_inputs.npy"),
        "targets": np.load(d / "targets.npy"),
        "states": np.load(d / "states.npy"),
    }


def _to_df(arr: dict[str, np.ndarray]) -> pd.DataFrame:
    n = int(arr["noise"].shape[0])
    row = np.arange(n, dtype=np.int64)
    df = pd.DataFrame(
        {
            "readable_time": row.astype(str),
            "house": np.full(n, HOUSE_ID, dtype=np.int64),
        }
    )
    df["aggregate"] = (arr["noise"].astype(np.float32) * NOISE_STD) + NOISE_MEAN
    df["sub_mains"] = (arr["denoise"].astype(np.float32) * DENOISE_STD) + DENOISE_MEAN
    targets = arr["targets"].astype(np.float32)
    states = arr["states"].astype(np.int64)
    for i, app in enumerate(APPLIANCES):
        stats = APP_STATS[app]
        df[f"{app}_power"] = (targets[:, i] * stats["std"]) + stats["mean"]
        df[f"{app}_on"] = states[:, i]
    ordered = [
        "readable_time",
        "house",
        "aggregate",
        "sub_mains",
        *[f"{a}_power" for a in APPLIANCES],
        *[f"{a}_on" for a in APPLIANCES],
    ]
    return df[ordered]


def main() -> None:
    args = parse_args()
    zip_path: Path = args.zip_path
    out_dir: Path = args.out_dir

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    manifest: dict[str, object] = {
        "source_zip": str(zip_path.resolve()),
        "appliances": APPLIANCES,
        "note": (
            "Official UNETNiLM data.zip has no timestamps; CSV uses integer row index. "
            "aggregate/sub_mains/*_power are inverse-normalized filtered watts."
        ),
        "splits": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        for out_split, zip_split in ZIP_SPLITS.items():
            arrays = _load_arrays(tmp_dir, zip_split)
            df = _to_df(arrays)
            split_dir = out_dir / out_split
            split_dir.mkdir(parents=True, exist_ok=True)
            out_csv = split_dir / f"ukdale_unetnilm_{out_split}_official.csv"
            df.to_csv(out_csv, index=False)
            manifest["splits"][out_split] = {
                "zip_split": zip_split,
                "output_csv": str(out_csv.resolve()),
                "rows": int(len(df)),
                "columns": list(df.columns),
            }
            print(f"{out_split:10s} -> {out_csv} ({len(df):,} rows)")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {manifest_path}")


if __name__ == "__main__":
    main()


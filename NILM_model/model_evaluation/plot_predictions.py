from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .plots import plot_prediction_waveforms


def _parse_pair(text: str) -> tuple[str, tuple[str, str]]:
    parts = [part.strip() for part in text.split(":")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Use appliance:true_col:pred_col")
    return parts[0], (parts[1], parts[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot NILM aggregate, true appliance, and predicted waveforms.")
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction CSV file.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--time_col", default="readable_time")
    parser.add_argument("--aggregate_col", default="aggregate")
    parser.add_argument(
        "--pair",
        action="append",
        type=_parse_pair,
        required=True,
        help="Appliance plot definition: appliance:true_col:pred_col. Repeat for multiple appliances.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--title", default="NILM Prediction Waveforms")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    pairs = dict(args.pair)
    path = plot_prediction_waveforms(
        frame,
        args.output,
        time_col=args.time_col,
        aggregate_col=args.aggregate_col,
        true_pred_pairs=pairs,
        start=args.start,
        samples=args.samples,
        title=args.title,
    )
    print(f"Saved prediction plot: {path}")


if __name__ == "__main__":
    main()


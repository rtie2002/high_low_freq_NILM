from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .metrics import compute_metrics_table


def _parse_pair(text: str) -> tuple[str, tuple[str, str]]:
    parts = [part.strip() for part in text.split(":")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Use appliance:true_col:pred_col")
    return parts[0], (parts[1], parts[2])


def _parse_on_col(text: str) -> tuple[str, str]:
    parts = [part.strip() for part in text.split(":")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use appliance:on_col")
    return parts[0], parts[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute common NILM metrics from a prediction CSV.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pair",
        action="append",
        type=_parse_pair,
        required=True,
        help="Appliance definition: appliance:true_power_col:pred_power_col. Repeat for multiple appliances.",
    )
    parser.add_argument(
        "--true_on",
        action="append",
        type=_parse_on_col,
        default=[],
        help="Optional true ON column: appliance:true_on_col.",
    )
    parser.add_argument(
        "--pred_on",
        action="append",
        type=_parse_on_col,
        default=[],
        help="Optional predicted ON column/probability: appliance:pred_on_col.",
    )
    parser.add_argument("--on_threshold_watts", type=float, default=15.0)
    parser.add_argument("--sae_period", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    metrics = compute_metrics_table(
        frame,
        dict(args.pair),
        true_on_cols=dict(args.true_on),
        pred_on_cols=dict(args.pred_on),
        on_threshold_watts=args.on_threshold_watts,
        sae_period=args.sae_period,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)
    print(f"Saved metrics: {args.output}")


if __name__ == "__main__":
    main()

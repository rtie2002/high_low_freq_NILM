from __future__ import annotations

import argparse
from pathlib import Path

from .plots import plot_loss_details, plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot NILM training loss and metric history.")
    parser.add_argument("--history", type=Path, required=True, help="CSV with epoch/train_loss/val_loss columns.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", default="Training History")
    parser.add_argument("--loss_cols", nargs="*", default=["train_loss", "val_loss"])
    parser.add_argument("--metric_cols", nargs="*", default=["val_mae", "val_sae", "val_f1"])
    parser.add_argument(
        "--kind",
        choices=["history", "loss_detail"],
        default="history",
        help="Use loss_detail for loss_detail_*.csv files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kind == "loss_detail":
        path = plot_loss_details(args.history, args.output, title=args.title)
        print(f"Saved detailed loss plot: {path}")
    else:
        path = plot_training_history(
            args.history,
            args.output,
            loss_cols=args.loss_cols,
            metric_cols=args.metric_cols,
            title=args.title,
        )
        print(f"Saved training plot: {path}")


if __name__ == "__main__":
    main()

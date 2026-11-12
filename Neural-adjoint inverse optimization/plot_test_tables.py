"""Render side-by-side UK-DALE and REFIT test tables from a completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HOUSES = {"ukdale_house2": "UK-DALE · House 2", "refit_house20": "REFIT · House 20"}
APPLIANCES = {
    "kettle": "Kettle", "fridge": "Fridge", "dishwasher": "Dishwasher",
    "washingmachine": "Washer", "microwave": "Microwave",
}


def render_table(root: Path, stage: str, metrics: dict[str, str], title: str, note: str,
                 *, validation: bool = False) -> None:
    filename = "metrics.csv" if stage == "inverse" else "reconstruction_metrics.csv"
    houses = {"validation": "Validation · UK-DALE 5 + REFIT 2"} if validation else HOUSES
    stage_dir = root / stage
    if stage == "reconstruction" and (root / "forward" / stage).is_dir():
        stage_dir = root / "forward" / stage
    tables = [pd.read_csv(stage_dir / house / filename).set_index("appliance") for house in houses]
    manifests = [
        json.loads((stage_dir / house / "run_manifest.json").read_text(encoding="utf-8"))
        for house in houses
    ]
    coverage = "   ·   ".join(
        f"{name}: {m['covered_rows']:,}/{m['available_rows']:,} rows "
        f"({m['covered_rows'] / m['available_rows']:.1%})"
        for name, m in zip(houses.values(), manifests)
    )

    fig, ax = plt.subplots(figsize=(14 if len(houses) == 2 else 9, 2.1 + len(metrics) * 0.30))
    ax.set_axis_off()
    ax.set_position((0.025, 0.24, 0.95, 0.56))
    fig.text(0.025, 0.94, title, fontsize=17, fontweight="bold", va="top", color="#17263b")
    fig.text(0.025, 0.84, "Single run · five appliances · macro average", fontsize=10)
    labels = ["Metric", *([*APPLIANCES.values(), "Avg."] * len(houses))]
    rows = []
    for column, label in metrics.items():
        row = [label]
        for table in tables:
            values = table.loc[list(APPLIANCES), column]
            average = table.at["overall", column] if "overall" in table.index else values.mean()
            decimals = 2 if "(W)" in label else 3
            row.extend("—" if pd.isna(v) else f"{v:.{decimals}f}" for v in [*values, average])
        rows.append(row)

    metric_width = 0.12
    cells = 6 * len(houses)
    cell_width = (1 - metric_width) / cells
    table = ax.table(
        cellText=rows, colLabels=labels, cellLoc="center",
        colWidths=[metric_width] + [cell_width] * cells, bbox=(0, 0, 1, 0.87),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for i, name in enumerate(houses.values()):
        x = metric_width + (i * 6 + 3) * cell_width
        ax.text(x, 0.94, name, ha="center", va="center", fontsize=11, fontweight="bold")
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d8e1eb")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#23435e")
            cell.set_text_props(color="white", weight="bold")
        elif col in (6, 12):
            cell.set_facecolor("#dce9f4")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f7fa")

    fig.text(0.025, 0.13, coverage, fontsize=9)
    fig.text(0.025, 0.07, note, fontsize=9)
    split = "validation" if validation else "test"
    for extension in ("png", "pdf"):
        fig.savefig(stage_dir / f"{split}_metrics_table.{extension}", dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing reconstruction/ and inverse/.")
    parser.add_argument("--inverse-only", action="store_true")
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()
    render_table(
        args.run_dir, "inverse",
        {
            "mae": "MAE (W)", "sae": "SAE (W)", "on_mae": "ON MAE (W)",
            "off_mae": "OFF MAE (W)", "average_precision": "AP", "precision": "Precision",
            "recall": "Recall", "f1": "F1",
        },
        "Stage 2 · Aggregate-only disaggregation · " + ("validation" if args.validation else "held-out test"),
        "AP ranks by ungated predicted watts. Power metrics use the decoded state gate.",
        validation=args.validation,
    )
    if args.inverse_only or args.validation:
        return
    render_table(
        args.run_dir, "reconstruction",
        {"mae_watts": "MAE (W)", "rmse_watts": "RMSE (W)", "on_mae_watts": "ON MAE (W)", "off_mae_watts": "OFF MAE (W)"},
        "Stage 1 · Submeter input versus reconstructed power",
        "Each appliance trace is supplied to the CNN. Errors are measured on the raw decoder output.",
    )


if __name__ == "__main__":
    main()

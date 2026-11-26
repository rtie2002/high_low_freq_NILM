"""Draw a Stage 1 metric table and confusion matrices without importing PyTorch."""

import json
import sys

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


if __name__ == "__main__":
    results = json.load(sys.stdin)
    labels = [
        "Washer" if appliance == "washing_machine" else appliance.capitalize()
        for appliance in results["appliances"]
    ]
    classifier, embedding = results["metrics"]
    reference_method = results["reference_matching"]
    reference_label = "Mean reference" if reference_method == "centroid" else "Nearest reference"
    true_counts = np.asarray(classifier["matrix"]).sum(axis=1)

    figure = Figure(figsize=(12, 8), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1, 2.1))
    figure.suptitle(
        f"Stage 1 {results['split'].lower()} | {results['event_count']:,} events | Epoch {results['epoch']}",
        fontsize=17, fontweight="bold",
    )

    summary = figure.add_subplot(grid[0, :])
    summary.axis("off")
    summary.set_title(
        f"Classifier: {classifier['accuracy']:.1%} accuracy, {classifier['macro_f1']:.3f} macro-F1"
        f"   |   {reference_label}: {embedding['accuracy']:.1%} accuracy, {embedding['macro_f1']:.3f} macro-F1",
        fontsize=11, pad=12,
    )
    rows = [
        [
            label, f"{true_counts[i]:,}",
            f"{classifier['correct'][i]:,}", f"{classifier['recall'][i]:.1%}", f"{classifier['f1'][i]:.3f}",
            f"{embedding['correct'][i]:,}", f"{embedding['recall'][i]:.1%}", f"{embedding['f1'][i]:.3f}",
        ]
        for i, label in enumerate(labels)
    ]
    table = summary.table(
        cellText=rows,
        colLabels=[
            "Appliance", "True events", "Classifier correct", "Classifier recall", "Classifier F1",
            "Vector correct", "Vector recall", "Vector F1",
        ],
        colWidths=[0.13, 0.10, 0.15, 0.15, 0.115, 0.135, 0.115, 0.105],
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dfe8")
        if row == 0:
            cell.set_facecolor("#203b55")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eff4f8")

    # Colors use the fraction within each true class, while numbers are counts.
    matrices = []
    for index, (title, metrics) in enumerate((("Five-class classifier", classifier), (reference_label, embedding))):
        axes = figure.add_subplot(grid[1, index])
        counts = np.asarray(metrics["matrix"])
        rates = counts / np.maximum(true_counts[:, None], 1)
        heatmap = axes.imshow(rates, cmap="Blues", vmin=0, vmax=1)
        axes.set_title(title, fontsize=12, fontweight="bold")
        axes.set_xlabel("Predicted appliance")
        axes.set_ylabel("True appliance")
        axes.set_xticks(range(len(labels)), labels, rotation=26, ha="right")
        axes.set_yticks(range(len(labels)), labels)
        for (row, column), value in np.ndenumerate(counts):
            color = "white" if rates[row, column] > 0.55 else "#1b2d3d"
            axes.text(column, row, f"{value:,}", ha="center", va="center", color=color, fontsize=9)
        matrices.append(axes)

    figure.colorbar(heatmap, ax=matrices, shrink=0.85, format=PercentFormatter(1),
                    label="Share of events within each true class")
    figure.savefig(sys.argv[1], dpi=180)

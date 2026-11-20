"""Run held-out Stage-2 tests for configured houses and save one comparison table."""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


if __package__:
    from ..data_loader import APPLIANCES, load_windows
    from ..metrics import average_precision, f1, mae, off_mae, on_mae, precision, recall, sae
else:
    project_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_dir))
    from data_loader import APPLIANCES, load_windows
    from metrics import average_precision, f1, mae, off_mae, on_mae, precision, recall, sae


STAGE2_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STAGE2_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
STAGE2_SCRIPT = STAGE2_DIR / "inverse_optimization.py"


def new_result_directory(previous):
    """Return the timestamped Stage-2 directory created by one child process."""
    current = {path.resolve() for path in RESULTS_DIR.iterdir() if path.is_dir()}
    created = current - previous
    if not created:
        raise FileNotFoundError("Stage 2 did not create a new result directory.")
    return max(created, key=lambda path: path.stat().st_mtime)


def house_metrics(estimated, true_power, true_state, thresholds):
    """Compute all metrics once across every non-overlapping test sample."""
    estimated = torch.from_numpy(estimated)
    true_power = torch.from_numpy(true_power)
    true_state = torch.from_numpy(true_state)
    return {
        "mae": mae(estimated, true_power),
        "sae": sae(estimated, true_power),
        "on_mae": on_mae(estimated, true_power, true_state),
        "off_mae": off_mae(estimated, true_power, true_state),
        "average_precision": average_precision(estimated, true_state),
        "precision": precision(estimated, true_state, thresholds),
        "recall": recall(estimated, true_state, thresholds),
        "f1": f1(estimated, true_state, thresholds),
    }


def save_test_table(metrics_by_house, houses, output_path):
    """Save a two-house metric table in the same style as the requested figure."""
    metric_rows = [
        ("MAE (W)", "mae", 2),
        ("SAE", "sae", 3),
        ("ON MAE (W)", "on_mae", 2),
        ("OFF MAE (W)", "off_mae", 2),
        ("AP", "average_precision", 3),
        ("Precision", "precision", 3),
        ("Recall", "recall", 3),
        ("F1", "f1", 3),
    ]
    table_rows = []
    for label, name, decimals in metric_rows:
        row = [label]
        for house in houses:
            values = metrics_by_house[house["label"]][name].detach().cpu()
            row.extend([f"{value:.{decimals}f}" for value in values])
            row.append(f"{torch.nanmean(values):.{decimals}f}")
        table_rows.append(row)

    labels = ["Metric"] + [item.capitalize() for item in APPLIANCES] + ["Avg."]
    column_labels = labels + labels[1:]
    figure, axis = plt.subplots(figsize=(20, 6.5))
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=column_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.65)

    group_width = 6
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f3b63")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#edf2f7")
        if column in {group_width, group_width * 2} and row > 0:
            cell.set_facecolor("#dbe8f5")
            cell.set_text_props(weight="bold")

    axis.text(0.29, 0.96, houses[0]["label"], ha="center", va="bottom", transform=axis.transAxes, fontsize=15, weight="bold")
    axis.text(0.76, 0.96, houses[1]["label"], ha="center", va="bottom", transform=axis.transAxes, fontsize=15, weight="bold")
    axis.axvline(0.5, ymin=0, ymax=0.9, color="#9fb3c8", linewidth=1.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    with (STAGE2_DIR / "config.yaml").open(encoding="utf-8") as file:
        settings = yaml.safe_load(file)
    inverse = settings["inverse"]
    test = settings["test"]
    houses = test["houses"]
    if len(houses) != 2:
        raise ValueError("The comparison table requires exactly two configured test houses.")

    report_dir = RESULTS_DIR / f"stage2_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    report_dir.mkdir(parents=True)
    metrics_by_house = {}
    child_runs = []

    for house in houses:
        before = {path.resolve() for path in RESULTS_DIR.iterdir() if path.is_dir()}
        command = [
            sys.executable, str(STAGE2_SCRIPT), "--checkpoint", str(args.checkpoint.resolve()),
            "--split", test["split"], "--dataset", house["dataset"],
            "--house", str(house["house"]), "--window-stride", str(test["window_stride"]),
        ]
        if test["num_windows"] is not None:
            command.extend(["--num-windows", str(test["num_windows"])])
        print(f"\n========== TEST: {house['label']} ==========")
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
        child_run = new_result_directory(before)
        child_runs.append((house["label"], child_run))

        true_power, _, true_state = load_windows(
            test["split"], stride=test["window_stride"],
            dataset=house["dataset"], house=house["house"],
        )
        estimated = np.load(child_run / "estimated_appliances.npy")
        if test["num_windows"] is not None:
            true_power, true_state = true_power[:len(estimated)], true_state[:len(estimated)]
        metrics_by_house[house["label"]] = house_metrics(
            estimated, true_power, true_state, inverse["on_thresholds"]
        )

    table_path = report_dir / "figures" / "metrics" / "stage2_test_house_metrics.png"
    save_test_table(metrics_by_house, houses, table_path)

    csv_path = report_dir / "test_house_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["house", "metric", *APPLIANCES, "average"])
        for house in houses:
            for name, values in metrics_by_house[house["label"]].items():
                values = values.detach().cpu()
                writer.writerow([house["label"], name, *values.tolist(), torch.nanmean(values).item()])

    run_path = report_dir / "child_test_runs.txt"
    run_path.write_text("\n".join(f"{label}: {path}" for label, path in child_runs), encoding="utf-8")
    print(f"\nTest metrics table: {table_path}")
    print(f"Test metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()

from datetime import datetime
from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from time import perf_counter


# =============================================================================
# End-to-End ExtraTrees Wrapper Feature-Selection Pipeline
# =============================================================================
# This script runs steps 1-6 of the ExtraTrees wrapper NILM workflow:
#   1. Tune ON/OFF classifier hyperparameters.
#   2. Select classifier features.
#   3. Generate classifier ON/OFF predictions inside the regression wrapper.
#   4. Tune power regressor hyperparameters.
#   5. Select regression features using MAE/SAE/EA.
#   6. Evaluate once on the held-out test split and save final outputs.
#
# Step 7, deep-learning validation of the selected features, is intentionally
# left for a separate experiment.

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_SELECTION_DIR = SCRIPT_DIR.parent
REPO_ROOT = FEATURE_SELECTION_DIR.parent
BASE_RESULTS_DIR = FEATURE_SELECTION_DIR / "results"

DATASET_FILENAME = "multi_appliance_house2_wk24_to_wk31_merged.csv"
DATASET_STEM = Path(DATASET_FILENAME).stem
DATASET_PATH = FEATURE_SELECTION_DIR / "dataset" / DATASET_FILENAME

CLASSIFIER_TUNING_DIR = BASE_RESULTS_DIR / f"extratrees_hyperparameter_tuning_onoff_{DATASET_STEM}"
CLASSIFIER_SELECTION_DIR = BASE_RESULTS_DIR / f"extratrees_forward_selection_onoff_{DATASET_STEM}"
REGRESSOR_TUNING_DIR = BASE_RESULTS_DIR / f"extratrees_hyperparameter_tuning_regression_{DATASET_STEM}"
REGRESSION_SELECTION_DIR = BASE_RESULTS_DIR / f"extratrees_forward_selection_classification_regression_{DATASET_STEM}"


PIPELINE_STEPS = [
    {
        "id": "01_classifier_hyperparameter_tuning",
        "title": "Tune ExtraTreesClassifier hyperparameters",
        "script": SCRIPT_DIR / "extratrees_hyperparameter_tuning.py",
        "result_dir": CLASSIFIER_TUNING_DIR,
        "required_outputs": [
            CLASSIFIER_TUNING_DIR / "best_hyperparameters.json",
            CLASSIFIER_TUNING_DIR / "hyperparameter_trials.csv",
        ],
        "freshness_inputs": [
            DATASET_PATH,
        ],
    },
    {
        "id": "02_classifier_forward_selection",
        "title": "Select ON/OFF classifier features",
        "script": SCRIPT_DIR / "extratrees_nilm_forward_selection_classification.py",
        "result_dir": CLASSIFIER_SELECTION_DIR,
        "required_outputs": [
            CLASSIFIER_SELECTION_DIR / "forward_selection_log.csv",
            CLASSIFIER_SELECTION_DIR / "selected_features.txt",
        ],
        "freshness_inputs": [
            DATASET_PATH,
            CLASSIFIER_TUNING_DIR / "best_hyperparameters.json",
        ],
    },
    {
        "id": "03_regressor_hyperparameter_tuning",
        "title": "Tune ExtraTreesRegressor hyperparameters",
        "script": SCRIPT_DIR / "extratrees_regressor_hyperparameter_tuning.py",
        "result_dir": REGRESSOR_TUNING_DIR,
        "required_outputs": [
            REGRESSOR_TUNING_DIR / "best_regressor_hyperparameters.json",
            REGRESSOR_TUNING_DIR / "regressor_hyperparameter_trials.csv",
        ],
        "freshness_inputs": [
            DATASET_PATH,
        ],
    },
    {
        "id": "04_classification_assisted_regression_selection",
        "title": "Select regression features and run held-out final test",
        "script": SCRIPT_DIR / "extratrees_nilm_forward_selection_classification_regression.py",
        "result_dir": REGRESSION_SELECTION_DIR,
        "required_outputs": [
            REGRESSION_SELECTION_DIR / "classification_regression_forward_selection_log.csv",
            REGRESSION_SELECTION_DIR / "selected_regression_features.txt",
            REGRESSION_SELECTION_DIR / "final_regression_test_metrics_per_appliance.csv",
        ],
        "freshness_inputs": [
            DATASET_PATH,
            CLASSIFIER_TUNING_DIR / "best_hyperparameters.json",
            CLASSIFIER_SELECTION_DIR / "forward_selection_log.csv",
            REGRESSOR_TUNING_DIR / "best_regressor_hyperparameters.json",
        ],
    },
]


IMPORTANT_OUTPUTS = {
    "classifier_hyperparameter_tuning": [
        CLASSIFIER_TUNING_DIR / "best_hyperparameters.json",
        CLASSIFIER_TUNING_DIR / "hyperparameter_trials.csv",
        CLASSIFIER_TUNING_DIR / "optimization_history_macro_f1.png",
        CLASSIFIER_TUNING_DIR / "per_appliance_f1_by_trial.png",
        CLASSIFIER_TUNING_DIR / "runtime_vs_macro_f1.png",
        CLASSIFIER_TUNING_DIR / "hyperparameter_importance.png",
        CLASSIFIER_TUNING_DIR / "hyperparameter_slice_plots.png",
    ],
    "classifier_forward_selection": [
        CLASSIFIER_SELECTION_DIR / "forward_selection_log.csv",
        CLASSIFIER_SELECTION_DIR / "selected_features.txt",
        CLASSIFIER_SELECTION_DIR / "forward_selection_macro_micro_f1.png",
        CLASSIFIER_SELECTION_DIR / "forward_selection_per_appliance_f1.png",
        CLASSIFIER_SELECTION_DIR / "onoff_evidence_plots",
    ],
    "regressor_hyperparameter_tuning": [
        REGRESSOR_TUNING_DIR / "best_regressor_hyperparameters.json",
        REGRESSOR_TUNING_DIR / "regressor_hyperparameter_trials.csv",
        REGRESSOR_TUNING_DIR / "optimization_history_composite_score.png",
        REGRESSOR_TUNING_DIR / "regression_metrics_by_trial.png",
        REGRESSOR_TUNING_DIR / "runtime_vs_composite_score.png",
        REGRESSOR_TUNING_DIR / "hyperparameter_importance.png",
        REGRESSOR_TUNING_DIR / "hyperparameter_slice_plots.png",
    ],
    "classification_assisted_regression_selection": [
        REGRESSION_SELECTION_DIR / "classification_regression_forward_selection_log.csv",
        REGRESSION_SELECTION_DIR / "direct_regression_forward_selection_log.csv",
        REGRESSION_SELECTION_DIR / "selected_regression_features.txt",
        REGRESSION_SELECTION_DIR / "final_regression_test_metrics_per_appliance.csv",
        REGRESSION_SELECTION_DIR / "final_regression_test_predictions.csv",
        REGRESSION_SELECTION_DIR / "regression_forward_selection_mae_sae_ea.png",
        REGRESSION_SELECTION_DIR / "regression_forward_selection_composite_score.png",
        REGRESSION_SELECTION_DIR / "regression_per_appliance_ea.png",
        REGRESSION_SELECTION_DIR / "metric_curves",
        REGRESSION_SELECTION_DIR / "prediction_waveforms",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full ExtraTrees wrapper feature-selection NILM pipeline.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a step if its required output files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned steps without running them.",
    )
    return parser.parse_args()


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def best_row(rows, metric_name, *, higher_is_better=True):
    usable_rows = [
        row for row in rows
        if row.get(metric_name) not in (None, "")
    ]
    if not usable_rows:
        return None
    return max(
        usable_rows,
        key=lambda row: float(row[metric_name]),
    ) if higher_is_better else min(
        usable_rows,
        key=lambda row: float(row[metric_name]),
    )


def required_outputs_exist(step):
    return all(path.exists() for path in step["required_outputs"])


def outputs_are_current(step):
    if not required_outputs_exist(step):
        return False

    input_paths = [step["script"], *step.get("freshness_inputs", [])]
    existing_inputs = [path for path in input_paths if path.exists()]
    if not existing_inputs:
        return True

    newest_input_mtime = max(path.stat().st_mtime for path in existing_inputs)
    oldest_output_mtime = min(path.stat().st_mtime for path in step["required_outputs"])
    return oldest_output_mtime >= newest_input_mtime


def run_step(step, logs_dir, skip_existing=False):
    if skip_existing and outputs_are_current(step):
        return {
            "id": step["id"],
            "title": step["title"],
            "status": "skipped_current_outputs",
            "return_code": 0,
            "elapsed_seconds": 0.0,
            "log_path": None,
            "result_dir": str(step["result_dir"]),
        }

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{step['id']}.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    command = [
        sys.executable,
        "-u",
        str(step["script"]),
    ]

    print()
    print("=" * 96)
    print(f"Running {step['id']}: {step['title']}")
    print(f"Script: {step['script']}")
    print(f"Log: {log_path}")
    print("=" * 96, flush=True)

    start_time = perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        if process.stdout is None:
            raise RuntimeError("subprocess stdout unexpectedly None")
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return_code = process.wait()

    elapsed = perf_counter() - start_time
    status = "completed" if return_code == 0 else "failed"
    return {
        "id": step["id"],
        "title": step["title"],
        "status": status,
        "return_code": return_code,
        "elapsed_seconds": elapsed,
        "log_path": str(log_path),
        "result_dir": str(step["result_dir"]),
    }


def summarize_outputs():
    classifier_tuning = read_json(CLASSIFIER_TUNING_DIR / "best_hyperparameters.json") or {}
    regressor_tuning = read_json(REGRESSOR_TUNING_DIR / "best_regressor_hyperparameters.json") or {}

    classifier_selection_rows = read_csv_rows(CLASSIFIER_SELECTION_DIR / "forward_selection_log.csv")
    best_classifier_row = best_row(classifier_selection_rows, "macro_f1", higher_is_better=True)

    regression_selection_rows = read_csv_rows(
        REGRESSION_SELECTION_DIR / "classification_regression_forward_selection_log.csv"
    )
    best_regression_row = best_row(regression_selection_rows, "selection_score", higher_is_better=False)
    direct_selection_rows = read_csv_rows(REGRESSION_SELECTION_DIR / "direct_regression_forward_selection_log.csv")
    best_direct_row = best_row(direct_selection_rows, "selection_score", higher_is_better=False)

    final_metrics_rows = read_csv_rows(REGRESSION_SELECTION_DIR / "final_regression_test_metrics_per_appliance.csv")

    return {
        "classifier_best_macro_f1": classifier_tuning.get("best_macro_f1"),
        "classifier_best_params": classifier_tuning.get("best_params"),
        "classifier_selected_round": best_classifier_row.get("round") if best_classifier_row else None,
        "classifier_selected_feature_count": best_classifier_row.get("feature_count") if best_classifier_row else None,
        "classifier_selected_macro_f1": best_classifier_row.get("macro_f1") if best_classifier_row else None,
        "classifier_selected_features": best_classifier_row.get("selected_features") if best_classifier_row else None,
        "regressor_best_composite_score": regressor_tuning.get("best_composite_score"),
        "regressor_best_params": regressor_tuning.get("best_params"),
        "regression_selected_round": best_regression_row.get("round") if best_regression_row else None,
        "regression_selected_feature_count": best_regression_row.get("feature_count") if best_regression_row else None,
        "regression_selected_score": best_regression_row.get("selection_score") if best_regression_row else None,
        "regression_selected_features": best_regression_row.get("selected_features") if best_regression_row else None,
        "direct_selected_round": best_direct_row.get("round") if best_direct_row else None,
        "direct_selected_feature_count": best_direct_row.get("feature_count") if best_direct_row else None,
        "direct_selected_score": best_direct_row.get("selection_score") if best_direct_row else None,
        "direct_selected_features": best_direct_row.get("selected_features") if best_direct_row else None,
        "final_test_metrics": final_metrics_rows,
    }


def existing_paths(paths):
    return [str(path) for path in paths if path.exists()]


def snapshot_outputs(pipeline_dir):
    snapshot_root = pipeline_dir / "artifacts"
    snapshotted_outputs = {}

    for group_name, paths in IMPORTANT_OUTPUTS.items():
        group_dir = snapshot_root / group_name
        copied_paths = []
        for source_path in paths:
            if not source_path.exists():
                continue

            group_dir.mkdir(parents=True, exist_ok=True)
            destination_path = group_dir / source_path.name
            if source_path.is_dir():
                shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, destination_path)
            copied_paths.append(str(destination_path))

        snapshotted_outputs[group_name] = copied_paths

    return snapshotted_outputs


def write_markdown_summary(summary, step_results, pipeline_dir, snapshotted_outputs):
    summary_path = pipeline_dir / "pipeline_summary.md"
    lines = [
        "# ExtraTrees Wrapper Feature-Selection Pipeline",
        "",
        f"Dataset: `{DATASET_FILENAME}`",
        f"Pipeline folder: `{pipeline_dir}`",
        "",
        "## Step Status",
        "",
        "| Step | Status | Time (min) | Result folder |",
        "|---|---:|---:|---|",
    ]

    for step_result in step_results:
        elapsed_min = float(step_result["elapsed_seconds"]) / 60.0
        lines.append(
            f"| {step_result['id']} | {step_result['status']} | {elapsed_min:.1f} | "
            f"`{step_result['result_dir']}` |"
        )

    lines.extend([
        "",
        "## Selected Classifier Features",
        "",
        f"- Round: `{summary.get('classifier_selected_round')}`",
        f"- Feature count: `{summary.get('classifier_selected_feature_count')}`",
        f"- Validation Macro F1: `{summary.get('classifier_selected_macro_f1')}`",
        f"- Features: `{summary.get('classifier_selected_features')}`",
        "",
        "## Selected Assisted Regression Features",
        "",
        f"- Round: `{summary.get('regression_selected_round')}`",
        f"- Feature count: `{summary.get('regression_selected_feature_count')}`",
        f"- Validation selection score: `{summary.get('regression_selected_score')}`",
        f"- Features: `{summary.get('regression_selected_features')}`",
        "",
        "## Selected Direct Regression Features",
        "",
        f"- Round: `{summary.get('direct_selected_round')}`",
        f"- Feature count: `{summary.get('direct_selected_feature_count')}`",
        f"- Validation selection score: `{summary.get('direct_selected_score')}`",
        f"- Features: `{summary.get('direct_selected_features')}`",
        "",
        "## Final Held-Out Test Metrics",
        "",
    ])

    final_rows = summary.get("final_test_metrics") or []
    if final_rows:
        lines.extend([
            "| Model | Appliance | MAE (W) | SAE | EA |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in final_rows:
            lines.append(
                f"| {row.get('model')} | {row.get('appliance')} | "
                f"{row.get('mae_w')} | {row.get('sae')} | {row.get('ea')} |"
            )
    else:
        lines.append("Final test metrics were not found.")

    lines.extend([
        "",
        "## Important Outputs",
        "",
    ])
    for group_name, paths in IMPORTANT_OUTPUTS.items():
        lines.append(f"### {group_name}")
        group_paths = snapshotted_outputs.get(group_name) or existing_paths(paths)
        if group_paths:
            for path in group_paths:
                lines.append(f"- `{path}`")
        else:
            lines.append("- No outputs found.")
        lines.append("")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_dir = BASE_RESULTS_DIR / f"extratrees_wrapper_pipeline_{DATASET_STEM}_{timestamp}"
    logs_dir = pipeline_dir / "logs"

    print("ExtraTrees wrapper feature-selection pipeline")
    print(f"Dataset: {DATASET_FILENAME}")
    print(f"Pipeline output folder: {pipeline_dir}")
    print()
    print("Planned steps:")
    for step in PIPELINE_STEPS:
        print(f"  - {step['id']}: {step['script']}")

    if args.dry_run:
        return 0

    pipeline_dir.mkdir(parents=True, exist_ok=True)
    step_results = []
    pipeline_start = perf_counter()

    for step in PIPELINE_STEPS:
        step_result = run_step(step, logs_dir, skip_existing=args.skip_existing)
        step_results.append(step_result)
        if step_result["return_code"] != 0:
            break

    failed_steps = [step for step in step_results if step["return_code"] != 0]
    pipeline_status = "failed" if failed_steps else "completed"
    if len(step_results) < len(PIPELINE_STEPS) and not failed_steps:
        pipeline_status = "incomplete"

    summary = summarize_outputs()
    snapshotted_outputs = snapshot_outputs(pipeline_dir)
    manifest = {
        "dataset": DATASET_FILENAME,
        "pipeline_dir": str(pipeline_dir),
        "pipeline_status": pipeline_status,
        "elapsed_seconds": perf_counter() - pipeline_start,
        "steps": step_results,
        "summary": summary,
        "important_outputs": {
            group_name: existing_paths(paths)
            for group_name, paths in IMPORTANT_OUTPUTS.items()
        },
        "snapshotted_outputs": snapshotted_outputs,
    }
    manifest_path = pipeline_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path = write_markdown_summary(summary, step_results, pipeline_dir, snapshotted_outputs)

    print()
    print("=" * 96)
    print("Pipeline finished")
    print(f"Status: {pipeline_status}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Logs: {logs_dir}")
    print("=" * 96)

    return 1 if failed_steps else 0


if __name__ == "__main__":
    raise SystemExit(main())

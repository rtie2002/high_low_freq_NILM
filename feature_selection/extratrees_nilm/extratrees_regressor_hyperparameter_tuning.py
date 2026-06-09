from pathlib import Path
from time import perf_counter
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score


# =============================================================================
# 1. Configuration
# =============================================================================
# Run command:
#   python feature_selection/extratrees_nilm/extratrees_regressor_hyperparameter_tuning.py
#
# This script tunes ExtraTreesRegressor hyperparameters separately from the
# ON/OFF classifier. It optimizes a multi-metric regression objective on the
# validation split and saves params for the classification-regression forward
# selection script.
FEATURE_SELECTION_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = FEATURE_SELECTION_DIR / "dataset"
DATASET_FILENAME = "multi_appliance_house2_wk24_to_wk31_merged.csv"
DATASET_PATH = DATASET_DIR / DATASET_FILENAME

BASE_RESULTS_DIR = FEATURE_SELECTION_DIR / "results"
RUN_NAME = f"extratrees_hyperparameter_tuning_regression_{Path(DATASET_FILENAME).stem}"
RESULTS_DIR = BASE_RESULTS_DIR / RUN_NAME
BEST_PARAMS_PATH = RESULTS_DIR / "best_regressor_hyperparameters.json"
TRIALS_LOG_PATH = RESULTS_DIR / "regressor_hyperparameter_trials.csv"
OPTIMIZATION_HISTORY_PLOT = RESULTS_DIR / "optimization_history_composite_score.png"
METRIC_HISTORY_PLOT = RESULTS_DIR / "regression_metrics_by_trial.png"
RUNTIME_SCORE_PLOT = RESULTS_DIR / "runtime_vs_composite_score.png"
HYPERPARAMETER_IMPORTANCE_PLOT = RESULTS_DIR / "hyperparameter_importance.png"
HYPERPARAMETER_SLICE_PLOT = RESULTS_DIR / "hyperparameter_slice_plots.png"

TRAIN_SIZE = 0.6
VALIDATION_SIZE = 0.2
N_TRIALS = 100
RANDOM_STATE = 42
FAST_SEARCH_SPACE = True
EPSILON = 1e-6
SAE_WINDOW_POINTS = 600

OBJECTIVE_METRIC_WEIGHTS = {
    "avg_nmae": 0.35,
    "avg_nrmse": 0.25,
    "avg_relative_energy_error": 0.20,
    "avg_sae": 0.10,
    "avg_r2": 0.10,
}
OBJECTIVE_HIGHER_IS_BETTER = {
    "avg_r2",
}


# =============================================================================
# 2. Load Optuna
# =============================================================================
try:
    import optuna
except ImportError:
    print("Optuna is not installed in this Python environment.")
    print("Installing Optuna with:")
    print(f"  {sys.executable} -m pip install optuna")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna


# =============================================================================
# 3. Load Dataset
# =============================================================================
df = pd.read_csv(DATASET_PATH)


# =============================================================================
# 4. Preprocessing: Detect Feature Columns and Power Labels
# =============================================================================
APPLIANCE_NAMES = [
    "kettle",
    "fridge",
    "microwave",
    "dishwasher",
    "washingmachine",
]

TIME_COLUMNS = ["readable_time"]
POWER_LABEL_COLUMNS = [f"{appliance}_power" for appliance in APPLIANCE_NAMES]
ON_OFF_LABEL_COLUMNS = [f"{appliance}_on" for appliance in APPLIANCE_NAMES]
NON_FEATURE_COLUMNS = TIME_COLUMNS + POWER_LABEL_COLUMNS + ON_OFF_LABEL_COLUMNS
FEATURE_COLUMNS = [column for column in df.columns if column not in NON_FEATURE_COLUMNS]

# TUNING_FEATURES can be changed if you want to tune on an already selected
# subset. None means tune using all detected aggregate/HF input features.
TUNING_FEATURES = None
FEATURES_USED = FEATURE_COLUMNS if TUNING_FEATURES is None else TUNING_FEATURES

X = df[FEATURES_USED]
y_power = df[POWER_LABEL_COLUMNS]


# =============================================================================
# 5. Time-Based Train / Validation / Test Split
# =============================================================================
train_end = int(len(df) * TRAIN_SIZE)
validation_end = int(len(df) * (TRAIN_SIZE + VALIDATION_SIZE))

X_train = X.iloc[:train_end]
y_train = y_power.iloc[:train_end]

X_validation = X.iloc[train_end:validation_end]
y_validation = y_power.iloc[train_end:validation_end]
y_validation_array = y_validation.to_numpy(dtype=np.float64)

power_scale = np.maximum(
    y_train.max(axis=0).to_numpy(dtype=np.float64) - y_train.min(axis=0).to_numpy(dtype=np.float64),
    EPSILON,
)


# =============================================================================
# 6. Regression Metrics
# =============================================================================
def regression_scores(y_true, y_pred):
    error = y_true - y_pred
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    nmae = mae / power_scale
    nrmse = rmse / power_scale

    sae_values = []
    relative_energy_error_values = []
    for column_index in range(y_true.shape[1]):
        true_column = y_true[:, column_index]
        pred_column = y_pred[:, column_index]
        window_sae_values = []
        window_relative_values = []
        for start in range(0, len(true_column), SAE_WINDOW_POINTS):
            end = min(start + SAE_WINDOW_POINTS, len(true_column))
            true_energy = np.sum(true_column[start:end])
            predicted_energy = np.sum(pred_column[start:end])
            window_sae_values.append(np.abs(predicted_energy - true_energy))
            window_relative_values.append(
                np.abs(predicted_energy - true_energy) / np.maximum(np.abs(true_energy), EPSILON)
            )
        sae_values.append(np.mean(window_sae_values))
        relative_energy_error_values.append(np.mean(window_relative_values))

    sae = np.asarray(sae_values, dtype=np.float64)
    relative_energy_error = np.asarray(relative_energy_error_values, dtype=np.float64)
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")

    scores = {
        "avg_mae": float(np.mean(mae)),
        "avg_rmse": float(np.mean(rmse)),
        "avg_nmae": float(np.mean(nmae)),
        "avg_nrmse": float(np.mean(nrmse)),
        "avg_sae": float(np.mean(sae)),
        "avg_relative_energy_error": float(np.mean(relative_energy_error)),
        "avg_r2": float(np.mean(r2)),
    }
    for label, mae_value, rmse_value, nmae_value, nrmse_value, sae_value, relative_value, r2_value in zip(
        POWER_LABEL_COLUMNS,
        mae,
        rmse,
        nmae,
        nrmse,
        sae,
        relative_energy_error,
        r2,
    ):
        scores[f"{label}_mae"] = float(mae_value)
        scores[f"{label}_rmse"] = float(rmse_value)
        scores[f"{label}_nmae"] = float(nmae_value)
        scores[f"{label}_nrmse"] = float(nrmse_value)
        scores[f"{label}_sae"] = float(sae_value)
        scores[f"{label}_relative_energy_error"] = float(relative_value)
        scores[f"{label}_r2"] = float(r2_value)
    return scores


def objective_score(scores):
    score = 0.0
    for metric_name, weight in OBJECTIVE_METRIC_WEIGHTS.items():
        value = scores[metric_name]
        if metric_name in OBJECTIVE_HIGHER_IS_BETTER:
            value = -value
        score += weight * value
    return score / sum(OBJECTIVE_METRIC_WEIGHTS.values())


# =============================================================================
# 7. Surrogate Optimization Objective
# =============================================================================
def objective(trial):
    if FAST_SEARCH_SPACE:
        max_depth = trial.suggest_int("max_depth", 15, 80)
        n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        criterion = trial.suggest_categorical("criterion", ["squared_error", "absolute_error"])
    else:
        max_depth_choice = trial.suggest_categorical("max_depth_choice", ["bounded", "none"])
        max_depth = None
        if max_depth_choice == "bounded":
            max_depth = trial.suggest_int("max_depth", 8, 100)
        n_estimators = trial.suggest_int("n_estimators", 50, 600, step=25)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 15)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 30)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        criterion = trial.suggest_categorical("criterion", ["squared_error", "absolute_error", "friedman_mse"])

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "criterion": criterion,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    trial_start = perf_counter()
    print()
    print("=" * 88, flush=True)
    print(f"Regressor trial {trial.number + 1}/{N_TRIALS} started", flush=True)
    print(f"Parameters: {params}", flush=True)

    model = ExtraTreesRegressor(**params)
    model.fit(X_train, y_train)
    prediction = model.predict(X_validation)
    scores = regression_scores(y_validation_array, prediction)
    composite_score = objective_score(scores)
    trial_elapsed = perf_counter() - trial_start

    for key, value in scores.items():
        trial.set_user_attr(key, value)

    print(f"Trial {trial.number + 1}/{N_TRIALS} finished in {trial_elapsed:.1f}s", flush=True)
    print(
        f"Composite={composite_score:.4f} | "
        f"NMAE={scores['avg_nmae']:.4f} | "
        f"NRMSE={scores['avg_nrmse']:.4f} | "
        f"SAE={scores['avg_sae']:.2f} | "
        f"RelEnergy={scores['avg_relative_energy_error']:.4f} | "
        f"R2={scores['avg_r2']:.4f}",
        flush=True,
    )

    return composite_score


# =============================================================================
# 8. Run Hyperparameter Tuning
# =============================================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
start_time = perf_counter()

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)


# =============================================================================
# 9. Save Best Hyperparameters and Trial History
# =============================================================================
best_params = study.best_params.copy()

max_depth_choice = best_params.pop("max_depth_choice", None)
if max_depth_choice == "none":
    best_params["max_depth"] = None

best_params.update({
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
})

result = {
    "best_composite_score": study.best_value,
    "best_params": best_params,
    "objective_metric_weights": OBJECTIVE_METRIC_WEIGHTS,
    "objective_higher_is_better": sorted(OBJECTIVE_HIGHER_IS_BETTER),
    "features_used": FEATURES_USED,
    "dataset": str(DATASET_PATH),
    "train_rows": len(X_train),
    "validation_rows": len(X_validation),
    "n_trials": N_TRIALS,
}

BEST_PARAMS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

trials_df = study.trials_dataframe()
trials_df.to_csv(TRIALS_LOG_PATH, index=False)


# =============================================================================
# 10. Save Tuning Visualizations
# =============================================================================
complete_trials_df = trials_df[trials_df["state"] == "COMPLETE"].copy()
complete_trials_df = complete_trials_df.sort_values("number").reset_index(drop=True)

if not complete_trials_df.empty:
    plt.style.use("seaborn-v0_8-whitegrid")
    trial_numbers = complete_trials_df["number"] + 1
    composite_scores = complete_trials_df["value"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trial_numbers, composite_scores, marker="o", linewidth=1.8, markersize=4, label="Trial score")
    ax.plot(trial_numbers, composite_scores.cummin(), marker="s", linewidth=2.4, markersize=4, label="Best score so far")
    ax.set_title("Regressor Hyperparameter Optimization History", fontsize=14, weight="bold")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation composite score (lower is better)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="major", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OPTIMIZATION_HISTORY_PLOT, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    metric_columns = {
        "user_attrs_avg_nmae": "NMAE",
        "user_attrs_avg_nrmse": "NRMSE",
        "user_attrs_avg_relative_energy_error": "Relative energy error",
        "user_attrs_avg_r2": "R2",
    }
    for column, label in metric_columns.items():
        if column in complete_trials_df.columns:
            ax.plot(trial_numbers, complete_trials_df[column], marker="o", markersize=3.5, linewidth=1.8, label=label)
    ax.set_title("Regression Metrics Across Hyperparameter Trials", fontsize=14, weight="bold")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation metric value")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="major", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(METRIC_HISTORY_PLOT, dpi=220)
    plt.close(fig)

    duration_seconds = complete_trials_df["duration"].dt.total_seconds()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(duration_seconds, composite_scores, s=42, alpha=0.8)
    best_idx = int(composite_scores.idxmin())
    ax.scatter(
        duration_seconds.loc[best_idx],
        composite_scores.loc[best_idx],
        s=90,
        marker="*",
        color="crimson",
        label="Best trial",
    )
    ax.set_title("Runtime vs Validation Composite Score", fontsize=14, weight="bold")
    ax.set_xlabel("Trial Runtime (seconds)")
    ax.set_ylabel("Validation composite score")
    ax.grid(True, which="major", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(RUNTIME_SCORE_PLOT, dpi=220)
    plt.close(fig)

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as exc:
        print(f"[warning] Could not compute hyperparameter importance: {exc}")
        importances = {}

    if importances:
        importance_items = sorted(importances.items(), key=lambda item: item[1], reverse=True)
        names = [item[0] for item in importance_items]
        values = [item[1] for item in importance_items]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(names[::-1], values[::-1], color="#4c78a8")
        ax.set_title("Regressor Hyperparameter Importance", fontsize=14, weight="bold")
        ax.set_xlabel("Importance")
        ax.grid(True, axis="x", alpha=0.30)
        fig.tight_layout()
        fig.savefig(HYPERPARAMETER_IMPORTANCE_PLOT, dpi=220)
        plt.close(fig)

    param_columns = [column for column in complete_trials_df.columns if column.startswith("params_")]
    if param_columns:
        n_cols = 3
        n_rows = int((len(param_columns) + n_cols - 1) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(4, 3.5 * n_rows)))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax, column in zip(axes, param_columns):
            param_name = column.replace("params_", "")
            values = complete_trials_df[column]
            if values.dtype == "object":
                categories = list(dict.fromkeys(values.astype(str)))
                x_values = values.astype(str).map({name: idx for idx, name in enumerate(categories)})
                ax.scatter(x_values, composite_scores, s=36, alpha=0.8)
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, rotation=25, ha="right")
            else:
                ax.scatter(values, composite_scores, s=36, alpha=0.8)
            ax.set_title(param_name)
            ax.set_ylabel("Composite score")
            ax.grid(True, alpha=0.30)

        for ax in axes[len(param_columns):]:
            ax.axis("off")

        fig.suptitle("Regressor Hyperparameter Slice Plots", fontsize=15, weight="bold")
        fig.tight_layout()
        fig.savefig(HYPERPARAMETER_SLICE_PLOT, dpi=220)
        plt.close(fig)


# =============================================================================
# 11. Console Report
# =============================================================================
elapsed = perf_counter() - start_time

print()
print("ExtraTreesRegressor hyperparameter tuning completed.")
print(f"Dataset: {DATASET_PATH}")
print(f"Features used: {len(FEATURES_USED)}")
print(f"Train rows: {len(X_train):,}")
print(f"Validation rows: {len(X_validation):,}")
print(f"Trials: {N_TRIALS}")
print(f"Best validation composite score: {study.best_value:.4f}")
print(f"Best parameters: {best_params}")
print(f"Saved best parameters: {BEST_PARAMS_PATH}")
print(f"Saved trial log: {TRIALS_LOG_PATH}")
print(f"Saved optimization history: {OPTIMIZATION_HISTORY_PLOT}")
print(f"Saved metric history: {METRIC_HISTORY_PLOT}")
print(f"Saved runtime-score plot: {RUNTIME_SCORE_PLOT}")
print(f"Saved hyperparameter importance: {HYPERPARAMETER_IMPORTANCE_PLOT}")
print(f"Saved hyperparameter slice plots: {HYPERPARAMETER_SLICE_PLOT}")
print(f"Elapsed time: {elapsed / 60:.1f} min")

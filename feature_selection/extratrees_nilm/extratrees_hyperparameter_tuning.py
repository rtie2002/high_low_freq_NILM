from pathlib import Path
from time import perf_counter
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score


# =============================================================================
# 1. Configuration
# =============================================================================
# Run command:
#   python feature_selection/extratrees_nilm/extratrees_hyperparameter_tuning.py
#
# This script tunes ExtraTrees hyperparameters using surrogate optimization
# through Optuna. It does NOT perform forward feature selection. Its only job is
# to find a strong model configuration, save it, and let the forward-selection
# script reuse it later.
FEATURE_SELECTION_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = FEATURE_SELECTION_DIR / "dataset"
DATASET_FILENAME = "multi_appliance_house2_wk24_to_wk31_merged.csv"
DATASET_PATH = DATASET_DIR / DATASET_FILENAME

BASE_RESULTS_DIR = FEATURE_SELECTION_DIR / "results"
RUN_NAME = f"extratrees_hyperparameter_tuning_onoff_{Path(DATASET_FILENAME).stem}"
RESULTS_DIR = BASE_RESULTS_DIR / RUN_NAME
BEST_PARAMS_PATH = RESULTS_DIR / "best_hyperparameters.json"
TRIALS_LOG_PATH = RESULTS_DIR / "hyperparameter_trials.csv"
OPTIMIZATION_HISTORY_PLOT = RESULTS_DIR / "optimization_history_macro_f1.png"
PER_APPLIANCE_F1_PLOT = RESULTS_DIR / "per_appliance_f1_by_trial.png"
RUNTIME_SCORE_PLOT = RESULTS_DIR / "runtime_vs_macro_f1.png"
HYPERPARAMETER_IMPORTANCE_PLOT = RESULTS_DIR / "hyperparameter_importance.png"
HYPERPARAMETER_SLICE_PLOT = RESULTS_DIR / "hyperparameter_slice_plots.png"

TRAIN_SIZE = 0.6
VALIDATION_SIZE = 0.2
N_TRIALS = 100
RANDOM_STATE = 42
FAST_SEARCH_SPACE = True


# =============================================================================
# 2. Load Optuna
# =============================================================================
# Optuna is used here as the surrogate optimizer. If it is not installed in the
# active Python environment, this script installs it automatically using the same
# Python executable that is running this file.
try:
    import optuna
except ImportError as exc:
    print("Optuna is not installed in this Python environment.")
    print("Installing Optuna with:")
    print(f"  {sys.executable} -m pip install optuna")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna


# =============================================================================
# 3. Load Dataset
# =============================================================================
# The dataset contains aligned aggregate features and appliance labels.
df = pd.read_csv(DATASET_PATH)


# =============================================================================
# 4. Preprocessing: Detect Feature Columns and ON/OFF Labels
# =============================================================================
APPLIANCE_NAMES = [
    "kettle",
    "fridge",
    "microwave",
    "dishwasher",
    "washingmachine",
]

TIME_COLUMNS = [
    "readable_time",
]

POWER_LABEL_COLUMNS = [
    f"{appliance}_power" for appliance in APPLIANCE_NAMES
]

ON_OFF_LABEL_COLUMNS = [
    f"{appliance}_on" for appliance in APPLIANCE_NAMES
]

NON_FEATURE_COLUMNS = TIME_COLUMNS + POWER_LABEL_COLUMNS + ON_OFF_LABEL_COLUMNS

FEATURE_COLUMNS = [
    column for column in df.columns
    if column not in NON_FEATURE_COLUMNS
]


# =============================================================================
# 5. Build Model Matrices
# =============================================================================
# TUNING_FEATURES can be changed later if we want to tune on a selected subset.
# None means tune the model using all detected input features.
TUNING_FEATURES = None
FEATURES_USED = FEATURE_COLUMNS if TUNING_FEATURES is None else TUNING_FEATURES

X = df[FEATURES_USED]
y_on = df[ON_OFF_LABEL_COLUMNS]


# =============================================================================
# 6. Time-Based Train / Validation / Test Split
# =============================================================================
# Hyperparameters are selected using the validation set only.
# The final test set is not used in this script.
train_end = int(len(df) * TRAIN_SIZE)
validation_end = int(len(df) * (TRAIN_SIZE + VALIDATION_SIZE))

X_train = X.iloc[:train_end]
y_train = y_on.iloc[:train_end]

X_validation = X.iloc[train_end:validation_end]
y_validation = y_on.iloc[train_end:validation_end]


# =============================================================================
# 7. Surrogate Optimization Objective
# =============================================================================
# Each Optuna trial:
#   1. Suggests one ExtraTrees hyperparameter set.
#   2. Trains a fresh ExtraTrees model.
#   3. Predicts appliance ON/OFF labels on the validation set.
#   4. Returns Macro F1 as the score to maximize.
def objective(trial):
    if FAST_SEARCH_SPACE:
        max_depth_choice = "bounded"
        max_depth = trial.suggest_int("max_depth", 15, 60)
        n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    else:
        max_depth_choice = trial.suggest_categorical(
            "max_depth_choice",
            ["bounded", "none"],
        )
        max_depth = None
        if max_depth_choice == "bounded":
            max_depth = trial.suggest_int("max_depth", 8, 40)
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=25)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "criterion": criterion,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    trial_start = perf_counter()
    print()
    print("=" * 88, flush=True)
    print(f"Trial {trial.number + 1}/{N_TRIALS} started", flush=True)
    print(f"Parameters: {params}", flush=True)

    model = ExtraTreesClassifier(**params)
    model.fit(X_train, y_train)

    prediction = model.predict(X_validation)
    macro_f1 = f1_score(
        y_validation,
        prediction,
        average="macro",
        zero_division=0,
    )
    micro_f1 = f1_score(
        y_validation,
        prediction,
        average="micro",
        zero_division=0,
    )
    per_appliance_f1 = f1_score(
        y_validation,
        prediction,
        average=None,
        zero_division=0,
    )
    trial_elapsed = perf_counter() - trial_start

    trial.set_user_attr("micro_f1", micro_f1)
    for label, score in zip(ON_OFF_LABEL_COLUMNS, per_appliance_f1):
        trial.set_user_attr(f"{label}_f1", float(score))

    print(f"Trial {trial.number + 1}/{N_TRIALS} finished in {trial_elapsed:.1f}s", flush=True)
    print(f"Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}", flush=True)
    print("Per-appliance F1:", flush=True)
    for label, score in zip(ON_OFF_LABEL_COLUMNS, per_appliance_f1):
        print(f"  {label}: {score:.4f}", flush=True)

    return macro_f1


# =============================================================================
# 8. Run Hyperparameter Tuning
# =============================================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
start_time = perf_counter()

study = optuna.create_study(
    direction="maximize",
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
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
})

result = {
    "best_macro_f1": study.best_value,
    "best_params": best_params,
    "features_used": FEATURES_USED,
    "dataset": str(DATASET_PATH),
    "train_rows": len(X_train),
    "validation_rows": len(X_validation),
    "n_trials": N_TRIALS,
}

BEST_PARAMS_PATH.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

trials_df = study.trials_dataframe()
trials_df.to_csv(TRIALS_LOG_PATH, index=False)


# =============================================================================
# 10. Save Tuning Visualizations
# =============================================================================
# These figures validate whether surrogate optimization is moving toward better
# hyperparameter settings and whether improvements help all appliances or only
# the frequent/easy ones.
complete_trials_df = trials_df[trials_df["state"] == "COMPLETE"].copy()
complete_trials_df = complete_trials_df.sort_values("number").reset_index(drop=True)

if not complete_trials_df.empty:
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Optimization history: trial score and best-so-far score.
    fig, ax = plt.subplots(figsize=(10, 5))
    trial_numbers = complete_trials_df["number"] + 1
    macro_scores = complete_trials_df["value"]
    ax.plot(
        trial_numbers,
        macro_scores,
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="Trial Macro F1",
    )
    ax.plot(
        trial_numbers,
        macro_scores.cummax(),
        marker="s",
        linewidth=2.4,
        markersize=4,
        label="Best Macro F1 so far",
    )
    ax.set_title("Optuna/TPE Hyperparameter Optimization History", fontsize=14, weight="bold")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation Macro F1")
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="major", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OPTIMIZATION_HISTORY_PLOT, dpi=220)
    plt.close(fig)

    # 2. Per-appliance F1 by trial.
    fig, ax = plt.subplots(figsize=(10, 5))
    for label in ON_OFF_LABEL_COLUMNS:
        column = f"user_attrs_{label}_f1"
        if column in complete_trials_df.columns:
            ax.plot(
                trial_numbers,
                complete_trials_df[column],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=label,
            )
    ax.set_title("Per-Appliance F1 Across Hyperparameter Trials", fontsize=14, weight="bold")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation F1")
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="major", alpha=0.35)
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(PER_APPLIANCE_F1_PLOT, dpi=220)
    plt.close(fig)

    # 3. Runtime vs score: checks whether slower settings are worth it.
    duration_seconds = complete_trials_df["duration"].dt.total_seconds()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(duration_seconds, macro_scores, s=42, alpha=0.8)
    best_idx = int(macro_scores.idxmax())
    ax.scatter(
        duration_seconds.loc[best_idx],
        macro_scores.loc[best_idx],
        s=90,
        marker="*",
        color="crimson",
        label="Best trial",
    )
    ax.set_title("Runtime vs Validation Macro F1", fontsize=14, weight="bold")
    ax.set_xlabel("Trial Runtime (seconds)")
    ax.set_ylabel("Validation Macro F1")
    ax.set_ylim(0, 1.02)
    ax.grid(True, which="major", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RUNTIME_SCORE_PLOT, dpi=220)
    plt.close(fig)

    # 4. Hyperparameter importance estimated from completed Optuna trials.
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
        ax.set_title("Hyperparameter Importance", fontsize=14, weight="bold")
        ax.set_xlabel("Importance")
        ax.grid(True, axis="x", alpha=0.30)
        fig.tight_layout()
        fig.savefig(HYPERPARAMETER_IMPORTANCE_PLOT, dpi=220)
        plt.close(fig)

    # 5. Slice plots: each tuned parameter against validation Macro F1.
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
                ax.scatter(x_values, macro_scores, s=36, alpha=0.8)
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, rotation=25, ha="right")
            else:
                ax.scatter(values, macro_scores, s=36, alpha=0.8)
            ax.set_title(param_name)
            ax.set_ylabel("Macro F1")
            ax.grid(True, alpha=0.30)

        for ax in axes[len(param_columns):]:
            ax.axis("off")

        fig.suptitle("Hyperparameter Slice Plots", fontsize=15, weight="bold")
        fig.tight_layout()
        fig.savefig(HYPERPARAMETER_SLICE_PLOT, dpi=220)
        plt.close(fig)


# =============================================================================
# 11. Console Report
# =============================================================================
elapsed = perf_counter() - start_time

print()
print("ExtraTrees surrogate hyperparameter tuning completed.")
print(f"Dataset: {DATASET_PATH}")
print(f"Features used: {len(FEATURES_USED)}")
print(f"Train rows: {len(X_train):,}")
print(f"Validation rows: {len(X_validation):,}")
print(f"Trials: {N_TRIALS}")
print(f"Best validation Macro F1: {study.best_value:.4f}")
print(f"Best parameters: {best_params}")
print(f"Saved best parameters: {BEST_PARAMS_PATH}")
print(f"Saved trial log: {TRIALS_LOG_PATH}")
print(f"Saved optimization history: {OPTIMIZATION_HISTORY_PLOT}")
print(f"Saved per-appliance F1 plot: {PER_APPLIANCE_F1_PLOT}")
print(f"Saved runtime-score plot: {RUNTIME_SCORE_PLOT}")
print(f"Saved hyperparameter importance: {HYPERPARAMETER_IMPORTANCE_PLOT}")
print(f"Saved hyperparameter slice plots: {HYPERPARAMETER_SLICE_PLOT}")
print(f"Elapsed time: {elapsed / 60:.1f} min")

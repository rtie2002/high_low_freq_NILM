from pathlib import Path
from time import perf_counter
import json

import pandas as pd
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
DATASET_FILENAME = "multi_appliance_house2_wk30_to_wk31_merged.csv"
DATASET_PATH = DATASET_DIR / DATASET_FILENAME

RESULTS_DIR = FEATURE_SELECTION_DIR / "results"
BEST_PARAMS_PATH = RESULTS_DIR / "extratrees_best_hyperparameters.json"
TRIALS_LOG_PATH = RESULTS_DIR / "extratrees_hyperparameter_trials.csv"

TRAIN_SIZE = 0.6
VALIDATION_SIZE = 0.2
N_TRIALS = 50
RANDOM_STATE = 42


# =============================================================================
# 2. Load Optuna
# =============================================================================
# Optuna is used here as the surrogate optimizer. If it is not installed, install
# it in the active Python environment before running this script:
#   pip install optuna
try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "Optuna is required for surrogate hyperparameter tuning. "
        "Install it with: pip install optuna"
    ) from exc


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
    max_depth_choice = trial.suggest_categorical(
        "max_depth_choice",
        ["bounded", "none"],
    )
    max_depth = None
    if max_depth_choice == "bounded":
        max_depth = trial.suggest_int("max_depth", 8, 40)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
        "max_depth": max_depth,
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", None],
        ),
        "criterion": trial.suggest_categorical(
            "criterion",
            ["gini", "entropy"],
        ),
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    model = ExtraTreesClassifier(**params)
    model.fit(X_train, y_train)

    prediction = model.predict(X_validation)
    macro_f1 = f1_score(
        y_validation,
        prediction,
        average="macro",
        zero_division=0,
    )

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

if best_params.pop("max_depth_choice") == "none":
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
# 10. Console Report
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
print(f"Elapsed time: {elapsed / 60:.1f} min")

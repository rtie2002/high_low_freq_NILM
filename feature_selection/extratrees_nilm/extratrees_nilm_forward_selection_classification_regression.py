from pathlib import Path
from time import perf_counter
import gc
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold


# =============================================================================
# 1. Configuration
# =============================================================================
FEATURE_SELECTION_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = FEATURE_SELECTION_DIR / "dataset"
DATASET_FILENAME = "multi_appliance_house2_wk24_to_wk31_merged.csv"
DATASET_PATH = DATASET_DIR / DATASET_FILENAME

BASE_RESULTS_DIR = FEATURE_SELECTION_DIR / "results"
RUN_NAME = f"extratrees_forward_selection_classification_regression_{Path(DATASET_FILENAME).stem}"
RESULTS_DIR = BASE_RESULTS_DIR / RUN_NAME
CLASSIFIER_TUNING_RUN_NAME = f"extratrees_hyperparameter_tuning_onoff_{Path(DATASET_FILENAME).stem}"
CLASSIFIER_BEST_PARAMS_PATH = BASE_RESULTS_DIR / CLASSIFIER_TUNING_RUN_NAME / "best_hyperparameters.json"
CLASSIFIER_SELECTION_RUN_NAME = f"extratrees_forward_selection_onoff_{Path(DATASET_FILENAME).stem}"
CLASSIFIER_SELECTION_LOG_PATH = BASE_RESULTS_DIR / CLASSIFIER_SELECTION_RUN_NAME / "forward_selection_log.csv"
REGRESSOR_TUNING_RUN_NAME = f"extratrees_hyperparameter_tuning_regression_{Path(DATASET_FILENAME).stem}"
REGRESSOR_BEST_PARAMS_PATH = BASE_RESULTS_DIR / REGRESSOR_TUNING_RUN_NAME / "best_regressor_hyperparameters.json"
FORWARD_SELECTION_LOG = RESULTS_DIR / "classification_regression_forward_selection_log.csv"
DIRECT_FORWARD_SELECTION_LOG = RESULTS_DIR / "direct_regression_forward_selection_log.csv"
REGRESSION_PLOT = RESULTS_DIR / "regression_forward_selection_mae_sae_ea.png"
COMPOSITE_SELECTION_PLOT = RESULTS_DIR / "regression_forward_selection_composite_score.png"
PER_APPLIANCE_EA_PLOT = RESULTS_DIR / "regression_per_appliance_ea.png"
METRIC_PLOT_DIR = RESULTS_DIR / "metric_curves"
SELECTED_FEATURES_TXT = RESULTS_DIR / "selected_regression_features.txt"
PREDICTION_CSV = RESULTS_DIR / "final_regression_test_predictions.csv"
FINAL_TEST_METRICS_CSV = RESULTS_DIR / "final_regression_test_metrics_per_appliance.csv"
PREDICTION_PLOT_DIR = RESULTS_DIR / "prediction_waveforms"

CACHE_DIR = FEATURE_SELECTION_DIR / "cache" / f"{Path(DATASET_FILENAME).stem}_classification_regression"
CACHE_METADATA_PATH = CACHE_DIR / "metadata.json"
X_CACHE_PATH = CACHE_DIR / "X_features_float32.dat"
Y_ON_CACHE_PATH = CACHE_DIR / "y_on_uint8.dat"
Y_POWER_CACHE_PATH = CACHE_DIR / "y_power_float32.dat"

TRAIN_SIZE = 0.6
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42
CSV_CHUNKSIZE = 100_000
CACHE_FEATURE_DTYPE = "float32"
CACHE_LABEL_DTYPE = "uint8"
CACHE_POWER_DTYPE = "float32"
EPSILON = 1e-6

# Regression wrapper selection is much heavier than classification-only selection
# because each candidate trains direct and classifier-assisted regressors.
# Set to None if you really want all features.
MAX_SELECTED_FEATURES = 30

# Final prediction waveform plots use a window from the test set. Keep this
# moderate so aggregate/ground-truth/predicted curves are readable.
PLOT_START_INDEX = 0
PLOT_ROWS = 5000
CLASSIFIER_OOF_SPLITS = 5

# Candidate ranking uses a composite of three NILM regression metrics.
# Ranks are computed within each forward-selection round, then combined with
# these weights. Lower composite score is better.
SELECTION_METRIC_WEIGHTS = {
    "avg_mae": 0.40,
    "avg_sae": 0.30,
    "avg_ea": 0.30,
}
SELECTION_HIGHER_IS_BETTER = {
    "avg_ea",
}
SELECTION_WEIGHT_SUM = sum(SELECTION_METRIC_WEIGHTS.values())
if SELECTION_WEIGHT_SUM <= 0:
    raise ValueError("SELECTION_METRIC_WEIGHTS must contain at least one positive weight.")

# Fallback classifier feature subset. The normal pipeline path loads the
# validation-selected classifier subset from CLASSIFIER_SELECTION_LOG_PATH.
DEFAULT_CLASSIFIER_FEATURES = [
    "S_apparent",
    "PF",
    "I_skew",
    "I7",
    "I9",
    "V5",
    "THDI",
    "V3",
    "V9",
    "I_env_4",
    "aggregate",
    "I_kurt",
    "V_skew",
    "DWT_E0",
    "THDV",
    "P_active",
    "V_rms",
    "I1",
    "Fcv",
    "I_spec_entropy",
    "V13",
    "I5",
    "I3",
    "V7",
    "I_env_3",
    "I_rms",
]

DEFAULT_CLASSIFIER_PARAMS = {
    "n_estimators": 125,
    "max_depth": 28,
    "min_samples_leaf": 1,
    "min_samples_split": 3,
    "max_features": "sqrt",
    "criterion": "gini",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

DEFAULT_REGRESSOR_PARAMS = {
    "n_estimators": 125,
    "max_depth": 28,
    "min_samples_leaf": 1,
    "min_samples_split": 3,
    "max_features": "sqrt",
    "criterion": "squared_error",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def load_classifier_params():
    if not CLASSIFIER_BEST_PARAMS_PATH.exists():
        print(f"Classifier tuning file not found, using default classifier params: {CLASSIFIER_BEST_PARAMS_PATH}")
        return DEFAULT_CLASSIFIER_PARAMS.copy()

    result = json.loads(CLASSIFIER_BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    params = result.get("best_params")
    if not isinstance(params, dict):
        raise ValueError(f"Missing best_params in {CLASSIFIER_BEST_PARAMS_PATH}")

    loaded_params = DEFAULT_CLASSIFIER_PARAMS.copy()
    loaded_params.update(params)
    loaded_params["random_state"] = RANDOM_STATE
    loaded_params["n_jobs"] = -1
    print(f"Loaded tuned classifier params from: {CLASSIFIER_BEST_PARAMS_PATH}")
    print(f"Classifier params: {loaded_params}")
    return loaded_params


def load_regressor_params():
    if not REGRESSOR_BEST_PARAMS_PATH.exists():
        print(f"Regressor tuning file not found, using default regressor params: {REGRESSOR_BEST_PARAMS_PATH}")
        return DEFAULT_REGRESSOR_PARAMS.copy()

    result = json.loads(REGRESSOR_BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    params = result.get("best_params")
    if not isinstance(params, dict):
        raise ValueError(f"Missing best_params in {REGRESSOR_BEST_PARAMS_PATH}")

    params = params.copy()
    params["random_state"] = RANDOM_STATE
    params["n_jobs"] = -1
    print(f"Loaded tuned regressor params from: {REGRESSOR_BEST_PARAMS_PATH}")
    print(f"Regressor params: {params}")
    return params


CLASSIFIER_PARAMS = load_classifier_params()
REGRESSOR_PARAMS = load_regressor_params()


# =============================================================================
# 2. Dataset Schema
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

CSV_COLUMNS = pd.read_csv(DATASET_PATH, nrows=0).columns.tolist()
FEATURE_COLUMNS = [column for column in CSV_COLUMNS if column not in NON_FEATURE_COLUMNS]


def load_classifier_features():
    if not CLASSIFIER_SELECTION_LOG_PATH.exists():
        print(f"Classifier selection log not found, using fallback classifier features: {CLASSIFIER_SELECTION_LOG_PATH}")
        return DEFAULT_CLASSIFIER_FEATURES.copy(), "fallback hard-coded classifier features"

    selection_log_df = pd.read_csv(CLASSIFIER_SELECTION_LOG_PATH)
    if selection_log_df.empty or "selected_features" not in selection_log_df.columns:
        raise ValueError(f"Classifier selection log is empty or missing selected_features: {CLASSIFIER_SELECTION_LOG_PATH}")

    if "macro_f1" in selection_log_df.columns:
        best_row = selection_log_df.loc[selection_log_df["macro_f1"].idxmax()]
        source = (
            f"validation Macro-F1 classifier subset from round {int(best_row['round'])} "
            f"in {CLASSIFIER_SELECTION_LOG_PATH}"
        )
    else:
        best_row = selection_log_df.iloc[-1]
        source = f"last classifier subset in {CLASSIFIER_SELECTION_LOG_PATH}"

    selected_features = [
        feature for feature in str(best_row["selected_features"]).split(",")
        if feature
    ]
    if not selected_features:
        raise ValueError(f"No classifier features found in {CLASSIFIER_SELECTION_LOG_PATH}")

    print(f"Loaded classifier features from: {CLASSIFIER_SELECTION_LOG_PATH}")
    print(f"Classifier feature source: {source}")
    print(f"Classifier feature count: {len(selected_features)}")
    return selected_features, source


CLASSIFIER_FEATURES, CLASSIFIER_FEATURE_SOURCE = load_classifier_features()
missing_classifier_features = [feature for feature in CLASSIFIER_FEATURES if feature not in FEATURE_COLUMNS]
if missing_classifier_features:
    raise ValueError(f"Classifier feature(s) missing from dataset: {missing_classifier_features}")


# =============================================================================
# 3. Disk-Backed Dataset Cache
# =============================================================================
def count_csv_rows(csv_path):
    with csv_path.open("rb") as file:
        return max(sum(1 for _ in file) - 1, 0)


def cache_is_valid(row_count):
    if not CACHE_METADATA_PATH.exists():
        return False
    if not X_CACHE_PATH.exists() or not Y_ON_CACHE_PATH.exists() or not Y_POWER_CACHE_PATH.exists():
        return False

    metadata = json.loads(CACHE_METADATA_PATH.read_text(encoding="utf-8"))
    source_stat = DATASET_PATH.stat()
    expected = {
        "source_path": str(DATASET_PATH),
        "source_size": source_stat.st_size,
        "source_mtime": source_stat.st_mtime,
        "row_count": row_count,
        "feature_columns": FEATURE_COLUMNS,
        "on_off_label_columns": ON_OFF_LABEL_COLUMNS,
        "power_label_columns": POWER_LABEL_COLUMNS,
        "feature_dtype": CACHE_FEATURE_DTYPE,
        "label_dtype": CACHE_LABEL_DTYPE,
        "power_dtype": CACHE_POWER_DTYPE,
    }
    return metadata == expected


def build_disk_cache(row_count):
    print("Building disk-backed classification-regression dataset cache...")
    print(f"Cache folder: {CACHE_DIR}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    x_cache = np.memmap(
        X_CACHE_PATH,
        dtype=CACHE_FEATURE_DTYPE,
        mode="w+",
        shape=(row_count, len(FEATURE_COLUMNS)),
    )
    y_on_cache = np.memmap(
        Y_ON_CACHE_PATH,
        dtype=CACHE_LABEL_DTYPE,
        mode="w+",
        shape=(row_count, len(ON_OFF_LABEL_COLUMNS)),
    )
    y_power_cache = np.memmap(
        Y_POWER_CACHE_PATH,
        dtype=CACHE_POWER_DTYPE,
        mode="w+",
        shape=(row_count, len(POWER_LABEL_COLUMNS)),
    )

    offset = 0
    use_columns = FEATURE_COLUMNS + ON_OFF_LABEL_COLUMNS + POWER_LABEL_COLUMNS
    for chunk_index, chunk in enumerate(
        pd.read_csv(DATASET_PATH, usecols=use_columns, chunksize=CSV_CHUNKSIZE),
        start=1,
    ):
        rows = len(chunk)
        row_slice = slice(offset, offset + rows)
        x_cache[row_slice, :] = chunk[FEATURE_COLUMNS].to_numpy(dtype=CACHE_FEATURE_DTYPE)
        y_on_cache[row_slice, :] = chunk[ON_OFF_LABEL_COLUMNS].to_numpy(dtype=CACHE_LABEL_DTYPE)
        y_power_cache[row_slice, :] = chunk[POWER_LABEL_COLUMNS].to_numpy(dtype=CACHE_POWER_DTYPE)
        offset += rows
        print(f"  Cached chunk {chunk_index}: {offset:,}/{row_count:,} rows", flush=True)

    x_cache.flush()
    y_on_cache.flush()
    y_power_cache.flush()
    del x_cache
    del y_on_cache
    del y_power_cache
    gc.collect()

    source_stat = DATASET_PATH.stat()
    metadata = {
        "source_path": str(DATASET_PATH),
        "source_size": source_stat.st_size,
        "source_mtime": source_stat.st_mtime,
        "row_count": row_count,
        "feature_columns": FEATURE_COLUMNS,
        "on_off_label_columns": ON_OFF_LABEL_COLUMNS,
        "power_label_columns": POWER_LABEL_COLUMNS,
        "feature_dtype": CACHE_FEATURE_DTYPE,
        "label_dtype": CACHE_LABEL_DTYPE,
        "power_dtype": CACHE_POWER_DTYPE,
    }
    CACHE_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


row_count = count_csv_rows(DATASET_PATH)
if not cache_is_valid(row_count):
    build_disk_cache(row_count)
else:
    print(f"Using existing disk-backed cache: {CACHE_DIR}")

X_all = np.memmap(
    X_CACHE_PATH,
    dtype=CACHE_FEATURE_DTYPE,
    mode="r",
    shape=(row_count, len(FEATURE_COLUMNS)),
)
y_on_all = np.memmap(
    Y_ON_CACHE_PATH,
    dtype=CACHE_LABEL_DTYPE,
    mode="r",
    shape=(row_count, len(ON_OFF_LABEL_COLUMNS)),
)
y_power_all = np.memmap(
    Y_POWER_CACHE_PATH,
    dtype=CACHE_POWER_DTYPE,
    mode="r",
    shape=(row_count, len(POWER_LABEL_COLUMNS)),
)


# =============================================================================
# 4. Time-Based Train/Validation/Test Split
# =============================================================================
train_end = int(row_count * TRAIN_SIZE)
validation_end = int(row_count * (TRAIN_SIZE + VALIDATION_SIZE))
if not (0 < train_end < validation_end < row_count):
    raise ValueError(
        "Invalid split sizes. Expected non-empty train/validation/test splits; "
        f"got row_count={row_count}, train_end={train_end}, validation_end={validation_end}."
    )

X_train = X_all[:train_end]
X_validation = X_all[train_end:validation_end]
X_test = X_all[validation_end:]
y_on_train = y_on_all[:train_end]
y_on_validation = y_on_all[train_end:validation_end]
y_on_test = y_on_all[validation_end:]
y_power_train = y_power_all[:train_end]
y_power_validation = y_power_all[train_end:validation_end]
y_power_test = y_power_all[validation_end:]

y_on_train_array = np.asarray(y_on_train)
y_on_validation_array = np.asarray(y_on_validation)
y_on_test_array = np.asarray(y_on_test)
y_power_validation_array = np.asarray(y_power_validation)
y_power_test_array = np.asarray(y_power_test)


# =============================================================================
# 5. Fixed Classifier Branch
# =============================================================================
def matrix_from_features(source_matrix, feature_subset):
    feature_indices = [FEATURE_COLUMNS.index(feature) for feature in feature_subset]
    return np.asarray(source_matrix[:, feature_indices])


def classification_scores(y_true, y_pred):
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        **{f"{label}_f1": float(score) for label, score in zip(ON_OFF_LABEL_COLUMNS, per_f1)},
    }


def out_of_fold_classifier_predictions(X_source, y_source):
    n_rows = len(X_source)
    if n_rows < 2:
        raise ValueError("Need at least two training rows to build out-of-fold classifier predictions.")

    n_splits = min(CLASSIFIER_OOF_SPLITS, n_rows)
    predictions = np.zeros((n_rows, y_source.shape[1]), dtype=CACHE_LABEL_DTYPE)
    fold_splitter = KFold(n_splits=n_splits, shuffle=False)

    print(
        f"Building classifier out-of-fold train predictions with {n_splits} contiguous folds...",
        flush=True,
    )
    for fold_index, (fold_train_indices, fold_holdout_indices) in enumerate(fold_splitter.split(X_source), start=1):
        fold_start = perf_counter()
        fold_classifier = ExtraTreesClassifier(**CLASSIFIER_PARAMS)
        fold_classifier.fit(X_source[fold_train_indices], y_source[fold_train_indices])
        predictions[fold_holdout_indices] = fold_classifier.predict(
            X_source[fold_holdout_indices]
        ).astype(CACHE_LABEL_DTYPE)
        print(
            f"  OOF fold {fold_index}/{n_splits}: "
            f"holdout rows={len(fold_holdout_indices):,} | "
            f"time={perf_counter() - fold_start:.1f}s",
            flush=True,
        )
        del fold_classifier
        gc.collect()

    return predictions


print("Training fixed classifier branch using classification-selected features...")
classifier_start = perf_counter()
X_classifier_train = matrix_from_features(X_train, CLASSIFIER_FEATURES)
X_classifier_validation = matrix_from_features(X_validation, CLASSIFIER_FEATURES)
X_classifier_test = matrix_from_features(X_test, CLASSIFIER_FEATURES)

predicted_on_train = out_of_fold_classifier_predictions(
    X_classifier_train,
    y_on_train_array,
)
classifier_train_oof_metrics = classification_scores(y_on_train_array, predicted_on_train)

classifier = ExtraTreesClassifier(**CLASSIFIER_PARAMS)
classifier.fit(X_classifier_train, y_on_train)
predicted_on_validation = classifier.predict(X_classifier_validation).astype(CACHE_LABEL_DTYPE)
predicted_on_test = classifier.predict(X_classifier_test).astype(CACHE_LABEL_DTYPE)
classifier_validation_metrics = classification_scores(y_on_validation_array, predicted_on_validation)

print(
    "Classifier branch ready | "
    f"OOF train Macro F1={classifier_train_oof_metrics['macro_f1']:.4f} | "
    f"Validation Macro F1={classifier_validation_metrics['macro_f1']:.4f} | "
    f"time={(perf_counter() - classifier_start) / 60:.1f} min",
    flush=True,
)
print("Fixed classifier branch out-of-fold train classification report:")
print(classification_report(
    y_on_train_array,
    predicted_on_train,
    target_names=ON_OFF_LABEL_COLUMNS,
    zero_division=0,
))
print("Fixed classifier branch validation classification report:")
print(classification_report(
    y_on_validation_array,
    predicted_on_validation,
    target_names=ON_OFF_LABEL_COLUMNS,
    zero_division=0,
))

del classifier
del X_classifier_train
del X_classifier_validation
del X_classifier_test
gc.collect()


# =============================================================================
# 6. Regression Scoring Helpers
# =============================================================================
def regression_scores(y_true, y_pred):
    error = y_true - y_pred
    mae = np.mean(np.abs(error), axis=0)

    true_energy = np.sum(y_true, axis=0)
    predicted_energy = np.sum(y_pred, axis=0)
    sae = np.abs(predicted_energy - true_energy) / np.maximum(np.abs(true_energy), EPSILON)
    ea = 1.0 - (
        np.sum(np.abs(error), axis=0)
        / (2.0 * np.maximum(np.sum(np.abs(y_true), axis=0), EPSILON))
    )
    overall_ea = 1.0 - (
        np.sum(np.abs(error))
        / (2.0 * np.maximum(np.sum(np.abs(y_true)), EPSILON))
    )

    scores = {
        "avg_mae": float(np.mean(mae)),
        "avg_sae": float(np.mean(sae)),
        "avg_ea": float(overall_ea),
    }
    for label, mae_value, sae_value, ea_value in zip(POWER_LABEL_COLUMNS, mae, sae, ea):
        scores[f"{label}_mae"] = float(mae_value)
        scores[f"{label}_sae"] = float(sae_value)
        scores[f"{label}_ea"] = float(ea_value)
    return scores


def train_regressor_and_score(feature_subset, target_matrix, target_power, extra_train=None, extra_target=None):
    X_train_subset = matrix_from_features(X_train, feature_subset)
    X_target_subset = matrix_from_features(target_matrix, feature_subset)

    if extra_train is not None and extra_target is not None:
        X_train_subset = np.hstack([X_train_subset, extra_train.astype(CACHE_FEATURE_DTYPE)])
        X_target_subset = np.hstack([X_target_subset, extra_target.astype(CACHE_FEATURE_DTYPE)])

    model = ExtraTreesRegressor(**REGRESSOR_PARAMS)
    try:
        model.fit(X_train_subset, y_power_train)
        prediction = model.predict(X_target_subset)
        return regression_scores(target_power, prediction)
    finally:
        del model
        del X_train_subset
        del X_target_subset
        if "prediction" in locals():
            del prediction
        gc.collect()


def train_regressor_and_predict(feature_subset, target_matrix, extra_train=None, extra_target=None):
    X_train_subset = matrix_from_features(X_train, feature_subset)
    X_target_subset = matrix_from_features(target_matrix, feature_subset)

    if extra_train is not None and extra_target is not None:
        X_train_subset = np.hstack([X_train_subset, extra_train.astype(CACHE_FEATURE_DTYPE)])
        X_target_subset = np.hstack([X_target_subset, extra_target.astype(CACHE_FEATURE_DTYPE)])

    model = ExtraTreesRegressor(**REGRESSOR_PARAMS)
    try:
        model.fit(X_train_subset, y_power_train)
        return model.predict(X_target_subset)
    finally:
        del model
        del X_train_subset
        del X_target_subset
        gc.collect()


def evaluate_candidate(feature_subset):
    direct_scores = train_regressor_and_score(
        feature_subset,
        target_matrix=X_validation,
        target_power=y_power_validation_array,
    )
    assisted_scores = train_regressor_and_score(
        feature_subset,
        target_matrix=X_validation,
        target_power=y_power_validation_array,
        extra_train=predicted_on_train,
        extra_target=predicted_on_validation,
    )
    return direct_scores, assisted_scores


def add_composite_selection_scores(round_results, score_key):
    metric_names = list(SELECTION_METRIC_WEIGHTS)

    for metric_name in metric_names:
        reverse = metric_name in SELECTION_HIGHER_IS_BETTER
        sorted_results = sorted(
            round_results,
            key=lambda item: item[score_key][metric_name],
            reverse=reverse,
        )
        previous_value = None
        previous_rank = None
        for rank, result in enumerate(sorted_results, start=1):
            value = result[score_key][metric_name]
            if previous_value is not None and np.isclose(value, previous_value):
                metric_rank = previous_rank
            else:
                metric_rank = rank
                previous_value = value
                previous_rank = rank
            result.setdefault("selection_metric_ranks", {})[metric_name] = metric_rank

    for result in round_results:
        weighted_rank_sum = sum(
            SELECTION_METRIC_WEIGHTS[metric_name] * result["selection_metric_ranks"][metric_name]
            for metric_name in metric_names
        )
        result["selection_score"] = weighted_rank_sum / SELECTION_WEIGHT_SUM


# =============================================================================
# 7. Wrapper Feature Selection: Regression Objective
# =============================================================================
# Selection decision uses a weighted rank composite of classifier-assisted
# regression metrics. Direct regression is also recorded for diagnosis.
selected_features = []
remaining_features = FEATURE_COLUMNS.copy()
selection_log = []
start_time = perf_counter()

best_selection_score = float("inf")
max_rounds = len(FEATURE_COLUMNS) if MAX_SELECTED_FEATURES is None else min(MAX_SELECTED_FEATURES, len(FEATURE_COLUMNS))

for round_number in range(1, max_rounds + 1):
    round_results = []
    total_candidates = len(remaining_features)

    print()
    print("=" * 88)
    print(f"Classification-regression forward selection round {round_number}/{max_rounds}")
    print(f"Currently selected regression features: {selected_features if selected_features else 'none'}")
    print(f"Testing {total_candidates} candidate feature(s)...")

    for candidate_index, candidate_feature in enumerate(remaining_features, start=1):
        candidate_subset = selected_features + [candidate_feature]
        candidate_start = perf_counter()
        print(
            f"  [{candidate_index:02d}/{total_candidates:02d}] "
            f"Testing add regression feature: {candidate_feature}",
            flush=True,
        )

        direct_scores, assisted_scores = evaluate_candidate(candidate_subset)
        candidate_elapsed = perf_counter() - candidate_start

        print("      Regression metrics, average over appliances:", flush=True)
        print(
            f"        direct   | "
            f"MAE={direct_scores['avg_mae']:.2f} W | "
            f"SAE={direct_scores['avg_sae']:.4f} | "
            f"EA={direct_scores['avg_ea']:.4f}",
            flush=True,
        )
        print(
            f"        assisted | "
            f"MAE={assisted_scores['avg_mae']:.2f} W | "
            f"SAE={assisted_scores['avg_sae']:.4f} | "
            f"EA={assisted_scores['avg_ea']:.4f} | "
            f"time={candidate_elapsed:.1f}s",
            flush=True,
        )

        round_results.append({
            "candidate_feature": candidate_feature,
            "feature_subset": candidate_subset,
            "direct_scores": direct_scores,
            "assisted_scores": assisted_scores,
        })

    add_composite_selection_scores(round_results, score_key="assisted_scores")
    best_candidate = min(round_results, key=lambda item: item["selection_score"])
    selected_features.append(best_candidate["candidate_feature"])
    remaining_features.remove(best_candidate["candidate_feature"])

    current_selection_score = best_candidate["selection_score"]
    improvement = best_selection_score - current_selection_score if np.isfinite(best_selection_score) else 0.0
    best_selection_score = current_selection_score

    log_row = {
        "round": round_number,
        "added_feature": best_candidate["candidate_feature"],
        "feature_count": len(selected_features),
        "selected_features": ",".join(selected_features),
        "selection_score": best_candidate["selection_score"],
        "assisted_avg_mae": best_candidate["assisted_scores"]["avg_mae"],
        "assisted_avg_sae": best_candidate["assisted_scores"]["avg_sae"],
        "assisted_avg_ea": best_candidate["assisted_scores"]["avg_ea"],
        "direct_avg_mae": best_candidate["direct_scores"]["avg_mae"],
        "direct_avg_sae": best_candidate["direct_scores"]["avg_sae"],
        "direct_avg_ea": best_candidate["direct_scores"]["avg_ea"],
        "improvement": improvement,
        "classifier_train_oof_macro_f1": classifier_train_oof_metrics["macro_f1"],
        "classifier_train_oof_micro_f1": classifier_train_oof_metrics["micro_f1"],
        "classifier_validation_macro_f1": classifier_validation_metrics["macro_f1"],
        "classifier_validation_micro_f1": classifier_validation_metrics["micro_f1"],
    }
    for metric_name, weight in SELECTION_METRIC_WEIGHTS.items():
        log_row[f"selection_weight_{metric_name}"] = weight
        log_row[f"selection_rank_{metric_name}"] = best_candidate["selection_metric_ranks"][metric_name]
    for label in POWER_LABEL_COLUMNS:
        log_row[f"assisted_{label}_mae"] = best_candidate["assisted_scores"][f"{label}_mae"]
        log_row[f"direct_{label}_mae"] = best_candidate["direct_scores"][f"{label}_mae"]
        log_row[f"assisted_{label}_sae"] = best_candidate["assisted_scores"][f"{label}_sae"]
        log_row[f"direct_{label}_sae"] = best_candidate["direct_scores"][f"{label}_sae"]
        log_row[f"assisted_{label}_ea"] = best_candidate["assisted_scores"][f"{label}_ea"]
        log_row[f"direct_{label}_ea"] = best_candidate["direct_scores"][f"{label}_ea"]

    selection_log.append(log_row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_log).to_csv(FORWARD_SELECTION_LOG, index=False)

    elapsed = perf_counter() - start_time
    print()
    print(f"Round {round_number} selected: {best_candidate['candidate_feature']}")
    print(f"Best composite selection score: {best_selection_score:.4f}")
    print(f"Composite score improvement this round: {improvement:.4f}")
    print("Selection metric ranks for chosen feature:")
    print(pd.DataFrame([
        {
            "metric": metric_name,
            "weight": weight,
            "rank": best_candidate["selection_metric_ranks"][metric_name],
            "value": best_candidate["assisted_scores"][metric_name],
        }
        for metric_name, weight in SELECTION_METRIC_WEIGHTS.items()
    ]).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Regression comparison for selected feature combination:")
    print(pd.DataFrame([
        {"model": "direct", **{key: best_candidate["direct_scores"][key] for key in ["avg_mae", "avg_sae", "avg_ea"]}},
        {"model": "classifier_assisted", **{key: best_candidate["assisted_scores"][key] for key in ["avg_mae", "avg_sae", "avg_ea"]}},
    ]).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Per-appliance regression metrics for this selected feature combination:")
    appliance_metric_rows = []
    for label in POWER_LABEL_COLUMNS:
        appliance = label.replace("_power", "")
        appliance_metric_rows.append({
            "model": "direct",
            "appliance": appliance,
            "mae_w": best_candidate["direct_scores"][f"{label}_mae"],
            "sae": best_candidate["direct_scores"][f"{label}_sae"],
            "ea": best_candidate["direct_scores"][f"{label}_ea"],
        })
        appliance_metric_rows.append({
            "model": "assisted",
            "appliance": label.replace("_power", ""),
            "mae_w": best_candidate["assisted_scores"][f"{label}_mae"],
            "sae": best_candidate["assisted_scores"][f"{label}_sae"],
            "ea": best_candidate["assisted_scores"][f"{label}_ea"],
        })
    print(pd.DataFrame(appliance_metric_rows).to_string(
        index=False,
        float_format=lambda value: f"{value:.2f}",
    ))
    print(f"Elapsed time: {elapsed / 60:.1f} min")
    gc.collect()


# =============================================================================
# 8. Direct-Only Wrapper Feature Selection
# =============================================================================
# This second forward selection gives the direct regressor its own selected
# feature subset. Without this, the final direct-vs-assisted comparison would
# use a subset optimized for the assisted model only.
direct_selected_features = []
direct_remaining_features = FEATURE_COLUMNS.copy()
direct_selection_log = []
direct_start_time = perf_counter()
direct_best_selection_score = float("inf")

for round_number in range(1, max_rounds + 1):
    round_results = []
    total_candidates = len(direct_remaining_features)

    print()
    print("=" * 88)
    print(f"Direct-regression forward selection round {round_number}/{max_rounds}")
    print(f"Currently selected direct features: {direct_selected_features if direct_selected_features else 'none'}")
    print(f"Testing {total_candidates} candidate feature(s)...")

    for candidate_index, candidate_feature in enumerate(direct_remaining_features, start=1):
        candidate_subset = direct_selected_features + [candidate_feature]
        candidate_start = perf_counter()
        print(
            f"  [{candidate_index:02d}/{total_candidates:02d}] "
            f"Testing add direct feature: {candidate_feature}",
            flush=True,
        )

        direct_scores = train_regressor_and_score(
            candidate_subset,
            target_matrix=X_validation,
            target_power=y_power_validation_array,
        )
        candidate_elapsed = perf_counter() - candidate_start
        print(
            f"      direct | "
            f"MAE={direct_scores['avg_mae']:.2f} W | "
            f"SAE={direct_scores['avg_sae']:.4f} | "
            f"EA={direct_scores['avg_ea']:.4f} | "
            f"time={candidate_elapsed:.1f}s",
            flush=True,
        )

        round_results.append({
            "candidate_feature": candidate_feature,
            "feature_subset": candidate_subset,
            "direct_scores": direct_scores,
        })

    add_composite_selection_scores(round_results, score_key="direct_scores")
    best_candidate = min(round_results, key=lambda item: item["selection_score"])
    direct_selected_features.append(best_candidate["candidate_feature"])
    direct_remaining_features.remove(best_candidate["candidate_feature"])

    current_selection_score = best_candidate["selection_score"]
    improvement = (
        direct_best_selection_score - current_selection_score
        if np.isfinite(direct_best_selection_score)
        else 0.0
    )
    direct_best_selection_score = current_selection_score

    log_row = {
        "round": round_number,
        "added_feature": best_candidate["candidate_feature"],
        "feature_count": len(direct_selected_features),
        "selected_features": ",".join(direct_selected_features),
        "selection_score": best_candidate["selection_score"],
        "direct_avg_mae": best_candidate["direct_scores"]["avg_mae"],
        "direct_avg_sae": best_candidate["direct_scores"]["avg_sae"],
        "direct_avg_ea": best_candidate["direct_scores"]["avg_ea"],
        "improvement": improvement,
    }
    for metric_name, weight in SELECTION_METRIC_WEIGHTS.items():
        log_row[f"selection_weight_{metric_name}"] = weight
        log_row[f"selection_rank_{metric_name}"] = best_candidate["selection_metric_ranks"][metric_name]
    for label in POWER_LABEL_COLUMNS:
        log_row[f"direct_{label}_mae"] = best_candidate["direct_scores"][f"{label}_mae"]
        log_row[f"direct_{label}_sae"] = best_candidate["direct_scores"][f"{label}_sae"]
        log_row[f"direct_{label}_ea"] = best_candidate["direct_scores"][f"{label}_ea"]

    direct_selection_log.append(log_row)
    pd.DataFrame(direct_selection_log).to_csv(DIRECT_FORWARD_SELECTION_LOG, index=False)

    elapsed = perf_counter() - direct_start_time
    print()
    print(f"Direct round {round_number} selected: {best_candidate['candidate_feature']}")
    print(f"Best direct composite selection score: {direct_best_selection_score:.4f}")
    print(f"Direct composite score improvement this round: {improvement:.4f}")
    print(f"Elapsed direct-selection time: {elapsed / 60:.1f} min")
    gc.collect()


# =============================================================================
# 9. Save Curves and Report
# =============================================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
selection_log_df = pd.DataFrame(selection_log)
selection_log_df.to_csv(FORWARD_SELECTION_LOG, index=False)
direct_selection_log_df = pd.DataFrame(direct_selection_log)
direct_selection_log_df.to_csv(DIRECT_FORWARD_SELECTION_LOG, index=False)

if not selection_log_df.empty:
    plt.style.use("seaborn-v0_8-whitegrid")
    METRIC_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    def style_metric_axis(ax):
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(True, which="major", color="#9ca3af", alpha=0.55, linewidth=0.95)
        ax.grid(True, which="minor", color="#d1d5db", alpha=0.45, linewidth=0.55)
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")
            spine.set_linewidth(0.9)

    def mark_best_assisted_point(ax, metric_name, lower_is_better=True):
        column = f"assisted_avg_{metric_name}"
        if column not in selection_log_df.columns:
            return
        best_index = selection_log_df[column].idxmin() if lower_is_better else selection_log_df[column].idxmax()
        best_row = selection_log_df.loc[best_index]
        x_value = best_row["feature_count"]
        y_value = best_row[column]
        ax.scatter(
            [x_value],
            [y_value],
            s=115,
            marker="*",
            color="#dc2626",
            edgecolor="#7f1d1d",
            linewidth=0.7,
            zorder=5,
            label=f"Best assisted ({int(x_value)} features)",
        )
        ax.axvline(x_value, color="#dc2626", alpha=0.25, linewidth=1.2, linestyle="--")

    def save_average_metric_plot(metric_name, ylabel, lower_is_better=True):
        direct_column = f"direct_avg_{metric_name}"
        assisted_column = f"assisted_avg_{metric_name}"
        if direct_column not in selection_log_df.columns or assisted_column not in selection_log_df.columns:
            return

        fig, ax = plt.subplots(figsize=(13, 5.8))
        ax.plot(
            selection_log_df["feature_count"],
            selection_log_df[direct_column],
            marker="o",
            linewidth=2.4,
            markersize=5.2,
            label="Direct regression",
        )
        ax.plot(
            selection_log_df["feature_count"],
            selection_log_df[assisted_column],
            marker="s",
            linewidth=2.4,
            markersize=5.2,
            label="Classifier-assisted regression",
        )
        mark_best_assisted_point(ax, metric_name, lower_is_better)
        direction_text = "lower is better" if lower_is_better else "higher is better"
        ax.set_title(f"Average {ylabel} During Forward Selection", fontsize=14, weight="bold")
        ax.set_xlabel("Number of Selected Regression Features")
        ax.set_ylabel(f"{ylabel} ({direction_text})")
        style_metric_axis(ax)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(METRIC_PLOT_DIR / f"average_{metric_name}_curve.png", dpi=220)
        plt.close(fig)

    def save_per_appliance_assisted_metric_plot(metric_name, ylabel, lower_is_better=True):
        fig, ax = plt.subplots(figsize=(13, 5.8))
        plotted = False
        for label in POWER_LABEL_COLUMNS:
            column = f"assisted_{label}_{metric_name}"
            if column in selection_log_df.columns:
                plotted = True
                ax.plot(
                    selection_log_df["feature_count"],
                    selection_log_df[column],
                    marker="o",
                    linewidth=2,
                    markersize=4.5,
                    label=label.replace("_power", ""),
                )
        if not plotted:
            plt.close(fig)
            return

        direction_text = "lower is better" if lower_is_better else "higher is better"
        ax.set_title(f"Per-Appliance Assisted {ylabel} During Forward Selection", fontsize=14, weight="bold")
        ax.set_xlabel("Number of Selected Regression Features")
        ax.set_ylabel(f"{ylabel} ({direction_text})")
        style_metric_axis(ax)
        ax.legend(loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(METRIC_PLOT_DIR / f"per_appliance_assisted_{metric_name}_curve.png", dpi=220)
        plt.close(fig)

    metric_specs = [
        ("mae", "MAE (W)", True),
        ("sae", "SAE", True),
        ("ea", "EA", False),
    ]
    for metric_name, ylabel, lower_is_better in metric_specs:
        save_average_metric_plot(metric_name, ylabel, lower_is_better)
        save_per_appliance_assisted_metric_plot(metric_name, ylabel, lower_is_better)

    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.plot(
        selection_log_df["feature_count"],
        selection_log_df["selection_score"],
        marker="o",
        linewidth=2.6,
        markersize=5.2,
        color="#1f77b4",
    )
    best_selection_index = selection_log_df["selection_score"].idxmin()
    best_selection_row = selection_log_df.loc[best_selection_index]
    ax.scatter(
        [best_selection_row["feature_count"]],
        [best_selection_row["selection_score"]],
        s=115,
        marker="*",
        color="#dc2626",
        edgecolor="#7f1d1d",
        linewidth=0.7,
        zorder=5,
        label=f"Best score ({int(best_selection_row['feature_count'])} features)",
    )
    ax.axvline(best_selection_row["feature_count"], color="#dc2626", alpha=0.25, linewidth=1.2, linestyle="--")
    ax.set_title("Composite Selection Score During Forward Selection", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Selected Regression Features")
    ax.set_ylabel("Weighted rank score (lower is better)")
    style_metric_axis(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(COMPOSITE_SELECTION_PLOT, dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 13.5), sharex=True)
    summary_specs = [
        ("mae", "Average MAE (W)", "lower is better"),
        ("sae", "Average SAE", "lower is better"),
        ("ea", "Average EA", "higher is better"),
    ]
    for ax, (metric_name, ylabel, direction_text) in zip(axes, summary_specs):
        lower_is_better = metric_name != "ea"
        ax.plot(selection_log_df["feature_count"], selection_log_df[f"direct_avg_{metric_name}"], marker="o", linewidth=2.4, markersize=5.0, label="Direct regression")
        ax.plot(selection_log_df["feature_count"], selection_log_df[f"assisted_avg_{metric_name}"], marker="s", linewidth=2.4, markersize=5.0, label="Classifier-assisted regression")
        mark_best_assisted_point(ax, metric_name, lower_is_better)
        ax.set_ylabel(f"{ylabel}\n({direction_text})")
        style_metric_axis(ax)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Number of Selected Regression Features")
    fig.suptitle("ExtraTrees Classification-Regression Forward Selection", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(REGRESSION_PLOT, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5.8))
    for label in POWER_LABEL_COLUMNS:
        column = f"assisted_{label}_ea"
        if column in selection_log_df.columns:
            ax.plot(
                selection_log_df["feature_count"],
                selection_log_df[column],
                marker="o",
                linewidth=2,
                markersize=4.5,
                label=label.replace("_power", ""),
            )
    ax.set_title("Per-Appliance EA During Assisted Regression Selection", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Selected Regression Features")
    ax.set_ylabel("EA (higher is better)")
    style_metric_axis(ax)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(PER_APPLIANCE_EA_PLOT, dpi=220)
    plt.close(fig)

    with SELECTED_FEATURES_TXT.open("w", encoding="utf-8") as file:
        file.write("ExtraTrees Classification-Regression Forward Feature Selection\n")
        file.write(f"Dataset: {DATASET_PATH}\n")
        file.write(f"Rows: {row_count:,}\n")
        file.write(f"Classifier features ({len(CLASSIFIER_FEATURES)}): {', '.join(CLASSIFIER_FEATURES)}\n")
        file.write(f"Classifier feature source: {CLASSIFIER_FEATURE_SOURCE}\n")
        file.write(f"Classifier train OOF Macro F1: {classifier_train_oof_metrics['macro_f1']:.4f}\n")
        file.write(f"Classifier validation Macro F1: {classifier_validation_metrics['macro_f1']:.4f}\n")
        file.write(f"Classifier train OOF Micro F1: {classifier_train_oof_metrics['micro_f1']:.4f}\n")
        file.write(f"Classifier validation Micro F1: {classifier_validation_metrics['micro_f1']:.4f}\n")
        file.write("\n")
        file.write("Regression selection objective: weighted assisted-metric rank composite\n")
        for metric_name, weight in SELECTION_METRIC_WEIGHTS.items():
            direction = "higher is better" if metric_name in SELECTION_HIGHER_IS_BETTER else "lower is better"
            file.write(f"  {metric_name}: weight={weight:.2f}, {direction}\n")
        file.write("\n")
        file.write("Regression-selected feature order:\n")
        for _, row in selection_log_df.iterrows():
            file.write(
                f"Round {int(row['round']):02d}: "
                f"{row['added_feature']} | "
                f"selection score={row['selection_score']:.4f} | "
                f"assisted avg MAE={row['assisted_avg_mae']:.2f} W | "
                f"direct avg MAE={row['direct_avg_mae']:.2f} W | "
                f"assisted avg SAE={row['assisted_avg_sae']:.4f} | "
                f"assisted avg EA={row['assisted_avg_ea']:.4f}\n"
            )
        file.write("\n")
        file.write("Direct-regression-selected feature order:\n")
        for _, row in direct_selection_log_df.iterrows():
            file.write(
                f"Round {int(row['round']):02d}: "
                f"{row['added_feature']} | "
                f"selection score={row['selection_score']:.4f} | "
                f"direct avg MAE={row['direct_avg_mae']:.2f} W | "
                f"direct avg SAE={row['direct_avg_sae']:.4f} | "
                f"direct avg EA={row['direct_avg_ea']:.4f}\n"
            )


# =============================================================================
# 9. Save Final Test Predictions and Waveform Visualizations
# =============================================================================
final_direct_prediction = None
final_direct_on_assisted_subset_prediction = None
final_assisted_prediction = None

best_final_row = selection_log_df.loc[selection_log_df["selection_score"].idxmin()] if not selection_log_df.empty else None
final_assisted_selected_features = (
    [feature for feature in str(best_final_row["selected_features"]).split(",") if feature]
    if best_final_row is not None
    else selected_features
)
best_direct_final_row = (
    direct_selection_log_df.loc[direct_selection_log_df["selection_score"].idxmin()]
    if not direct_selection_log_df.empty
    else None
)
final_direct_selected_features = (
    [feature for feature in str(best_direct_final_row["selected_features"]).split(",") if feature]
    if best_direct_final_row is not None
    else direct_selected_features
)

if final_assisted_selected_features and final_direct_selected_features:
    print("Training final direct and classifier-assisted regressors for held-out test evaluation...")
    if best_final_row is not None:
        print(
            f"Final assisted model uses assisted-validation-selected subset from round {int(best_final_row['round'])} "
            f"({len(final_assisted_selected_features)} features)."
        )
    if best_direct_final_row is not None:
        print(
            f"Final direct model uses direct-validation-selected subset from round {int(best_direct_final_row['round'])} "
            f"({len(final_direct_selected_features)} features)."
        )
    final_prediction_start = perf_counter()
    final_direct_prediction = train_regressor_and_predict(
        final_direct_selected_features,
        target_matrix=X_test,
    )
    final_direct_on_assisted_subset_prediction = train_regressor_and_predict(
        final_assisted_selected_features,
        target_matrix=X_test,
    )
    final_assisted_prediction = train_regressor_and_predict(
        final_assisted_selected_features,
        target_matrix=X_test,
        extra_train=predicted_on_train,
        extra_target=predicted_on_test,
    )
    final_direct_scores = regression_scores(y_power_test_array, final_direct_prediction)
    final_direct_on_assisted_subset_scores = regression_scores(
        y_power_test_array,
        final_direct_on_assisted_subset_prediction,
    )
    final_assisted_scores = regression_scores(y_power_test_array, final_assisted_prediction)

    print("Held-out test regression metrics:")
    print(pd.DataFrame([
        {"model": "direct_selected", **{key: final_direct_scores[key] for key in ["avg_mae", "avg_sae", "avg_ea"]}},
        {"model": "direct_on_assisted_selected", **{key: final_direct_on_assisted_subset_scores[key] for key in ["avg_mae", "avg_sae", "avg_ea"]}},
        {"model": "classifier_assisted_selected", **{key: final_assisted_scores[key] for key in ["avg_mae", "avg_sae", "avg_ea"]}},
    ]).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    final_metric_rows = []
    for label in POWER_LABEL_COLUMNS:
        appliance = label.replace("_power", "")
        final_metric_rows.append({
            "model": "direct_selected",
            "appliance": appliance,
            "mae_w": final_direct_scores[f"{label}_mae"],
            "sae": final_direct_scores[f"{label}_sae"],
            "ea": final_direct_scores[f"{label}_ea"],
        })
        final_metric_rows.append({
            "model": "direct_on_assisted_selected",
            "appliance": appliance,
            "mae_w": final_direct_on_assisted_subset_scores[f"{label}_mae"],
            "sae": final_direct_on_assisted_subset_scores[f"{label}_sae"],
            "ea": final_direct_on_assisted_subset_scores[f"{label}_ea"],
        })
        final_metric_rows.append({
            "model": "classifier_assisted_selected",
            "appliance": appliance,
            "mae_w": final_assisted_scores[f"{label}_mae"],
            "sae": final_assisted_scores[f"{label}_sae"],
            "ea": final_assisted_scores[f"{label}_ea"],
        })
    final_metric_rows.extend([
        {"model": "direct_selected", "appliance": "average", "mae_w": final_direct_scores["avg_mae"], "sae": final_direct_scores["avg_sae"], "ea": final_direct_scores["avg_ea"]},
        {"model": "direct_on_assisted_selected", "appliance": "average", "mae_w": final_direct_on_assisted_subset_scores["avg_mae"], "sae": final_direct_on_assisted_subset_scores["avg_sae"], "ea": final_direct_on_assisted_subset_scores["avg_ea"]},
        {"model": "classifier_assisted_selected", "appliance": "average", "mae_w": final_assisted_scores["avg_mae"], "sae": final_assisted_scores["avg_sae"], "ea": final_assisted_scores["avg_ea"]},
    ])
    final_metrics_df = pd.DataFrame(final_metric_rows)
    final_metrics_df.to_csv(FINAL_TEST_METRICS_CSV, index=False)
    print(f"Held-out per-appliance test metrics: {FINAL_TEST_METRICS_CSV}")

    readable_time = pd.read_csv(DATASET_PATH, usecols=TIME_COLUMNS).iloc[validation_end:].reset_index(drop=True)
    prediction_df = pd.DataFrame({
        "test_row_index": np.arange(len(y_power_test_array)),
        "readable_time": readable_time[TIME_COLUMNS[0]].to_numpy(),
    })

    aggregate_feature = None
    for candidate in ["aggregate", "P_active", "S_apparent"]:
        if candidate in FEATURE_COLUMNS:
            aggregate_feature = candidate
            break

    if aggregate_feature is not None:
        aggregate_index = FEATURE_COLUMNS.index(aggregate_feature)
        prediction_df[aggregate_feature] = np.asarray(X_test[:, aggregate_index])

    for appliance_index, label in enumerate(POWER_LABEL_COLUMNS):
        appliance = label.replace("_power", "")
        prediction_df[f"{appliance}_true_power"] = y_power_test_array[:, appliance_index]
        prediction_df[f"{appliance}_direct_pred_power"] = final_direct_prediction[:, appliance_index]
        prediction_df[f"{appliance}_direct_on_assisted_features_pred_power"] = final_direct_on_assisted_subset_prediction[:, appliance_index]
        prediction_df[f"{appliance}_assisted_pred_power"] = final_assisted_prediction[:, appliance_index]
        prediction_df[f"{appliance}_predicted_on"] = predicted_on_test[:, appliance_index]

    prediction_df.to_csv(PREDICTION_CSV, index=False)

    PREDICTION_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_start = min(PLOT_START_INDEX, max(len(prediction_df) - 1, 0))
    plot_end = min(plot_start + PLOT_ROWS, len(prediction_df))
    plot_df = prediction_df.iloc[plot_start:plot_end].copy()
    x_values = pd.to_datetime(plot_df["readable_time"], errors="coerce")
    if x_values.isna().all():
        x_values = plot_df["test_row_index"]

    for label in POWER_LABEL_COLUMNS:
        appliance = label.replace("_power", "")
        fig, ax = plt.subplots(figsize=(13, 4.8))
        if aggregate_feature is not None:
            ax_aggregate = ax.twinx()
            ax_aggregate.plot(
                x_values,
                plot_df[aggregate_feature],
                color="#9ca3af",
                linewidth=1.2,
                alpha=0.45,
                label=aggregate_feature,
            )
            ax_aggregate.set_ylabel(f"{aggregate_feature} (W)", color="#6b7280")
            ax_aggregate.tick_params(axis="y", labelcolor="#6b7280")

        ax.plot(
            x_values,
            plot_df[f"{appliance}_true_power"],
            color="#111827",
            linewidth=2.0,
            label="ground truth",
        )
        ax.plot(
            x_values,
            plot_df[f"{appliance}_direct_pred_power"],
            color="#2563eb",
            linewidth=1.7,
            alpha=0.85,
            label="direct prediction",
        )
        ax.plot(
            x_values,
            plot_df[f"{appliance}_direct_on_assisted_features_pred_power"],
            color="#7c3aed",
            linewidth=1.4,
            alpha=0.75,
            linestyle="--",
            label="direct on assisted-selected features",
        )
        ax.plot(
            x_values,
            plot_df[f"{appliance}_assisted_pred_power"],
            color="#dc2626",
            linewidth=1.7,
            alpha=0.85,
            label="classifier-assisted prediction",
        )
        ax.set_title(f"{appliance}: Aggregate, Ground Truth, and Predicted Power", fontsize=13, weight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Appliance power (W)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(PREDICTION_PLOT_DIR / f"{appliance}_prediction_waveform.png", dpi=220)
        plt.close(fig)

    print(
        "Final prediction outputs saved | "
        f"CSV: {PREDICTION_CSV} | "
        f"metrics: {FINAL_TEST_METRICS_CSV} | "
        f"plots: {PREDICTION_PLOT_DIR} | "
        f"time={(perf_counter() - final_prediction_start) / 60:.1f} min",
        flush=True,
    )


print()
print("=" * 88)
print("Classification-regression forward selection complete")
print(f"Dataset: {DATASET_PATH}")
print(f"Rows: {row_count:,}")
print(f"Train rows: {len(X_train):,}")
print(f"Validation rows: {len(X_validation):,}")
print(f"Test rows: {len(X_test):,}")
print(f"Classifier feature source: {CLASSIFIER_FEATURE_SOURCE}")
classifier_test_metrics = classification_scores(y_on_test_array, predicted_on_test)
print(f"Classifier validation Macro F1: {classifier_validation_metrics['macro_f1']:.4f}")
print(f"Classifier validation Micro F1: {classifier_validation_metrics['micro_f1']:.4f}")
print(f"Classifier test Macro F1: {classifier_test_metrics['macro_f1']:.4f}")
print(f"Classifier test Micro F1: {classifier_test_metrics['micro_f1']:.4f}")
print("Classifier test report:")
print(classification_report(
    y_on_test_array,
    predicted_on_test,
    target_names=ON_OFF_LABEL_COLUMNS,
    zero_division=0,
))
print("Direct-validation-selected regression features:")
for index, feature in enumerate(final_direct_selected_features, start=1):
    print(f"  {index:02d}. {feature}")
print("Assisted-validation-selected regression features:")
for index, feature in enumerate(final_assisted_selected_features, start=1):
    print(f"  {index:02d}. {feature}")
print(f"Assisted selection log: {FORWARD_SELECTION_LOG}")
print(f"Direct selection log: {DIRECT_FORWARD_SELECTION_LOG}")
print(f"Regression curve: {REGRESSION_PLOT}")
print(f"Per-appliance EA curve: {PER_APPLIANCE_EA_PLOT}")
print(f"All metric curves: {METRIC_PLOT_DIR}")
print(f"Selected feature order: {SELECTED_FEATURES_TXT}")
if final_assisted_prediction is not None:
    print(f"Final prediction CSV: {PREDICTION_CSV}")
    print(f"Prediction waveform plots: {PREDICTION_PLOT_DIR}")

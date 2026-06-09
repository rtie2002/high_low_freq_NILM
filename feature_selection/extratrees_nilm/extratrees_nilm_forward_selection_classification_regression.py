from pathlib import Path
from time import perf_counter
import gc
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, r2_score


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
FORWARD_SELECTION_LOG = RESULTS_DIR / "classification_regression_forward_selection_log.csv"
REGRESSION_PLOT = RESULTS_DIR / "regression_forward_selection_nmae.png"
PER_APPLIANCE_NMAE_PLOT = RESULTS_DIR / "regression_per_appliance_nmae.png"
METRIC_PLOT_DIR = RESULTS_DIR / "metric_curves"
SELECTED_FEATURES_TXT = RESULTS_DIR / "selected_regression_features.txt"
PREDICTION_CSV = RESULTS_DIR / "final_regression_test_predictions.csv"
PREDICTION_PLOT_DIR = RESULTS_DIR / "prediction_waveforms"

CACHE_DIR = FEATURE_SELECTION_DIR / "cache" / f"{Path(DATASET_FILENAME).stem}_classification_regression"
CACHE_METADATA_PATH = CACHE_DIR / "metadata.json"
X_CACHE_PATH = CACHE_DIR / "X_features_float32.dat"
Y_ON_CACHE_PATH = CACHE_DIR / "y_on_uint8.dat"
Y_POWER_CACHE_PATH = CACHE_DIR / "y_power_float32.dat"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CSV_CHUNKSIZE = 100_000
CACHE_FEATURE_DTYPE = "float32"
CACHE_LABEL_DTYPE = "uint8"
CACHE_POWER_DTYPE = "float32"
EPSILON = 1e-6
SAE_WINDOW_POINTS = 600

# Regression wrapper selection is much heavier than classification-only selection
# because each candidate trains direct and classifier-assisted regressors.
# Set to None if you really want all features.
MAX_SELECTED_FEATURES = 30

# Final prediction waveform plots use a window from the test set. Keep this
# moderate so aggregate/ground-truth/predicted curves are readable.
PLOT_START_INDEX = 0
PLOT_ROWS = 5000

# Best classification-selected feature subset from the wk24_to_wk31 classifier run.
# This branch produces the predicted ON/OFF inputs used by the assisted regressor.
CLASSIFIER_FEATURES = [
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

CLASSIFIER_PARAMS = {
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

REGRESSOR_PARAMS = {
    "n_estimators": 125,
    "max_depth": 28,
    "min_samples_leaf": 1,
    "min_samples_split": 3,
    "max_features": "sqrt",
    "criterion": "squared_error",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


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
# 4. Time-Based Train/Test Split
# =============================================================================
split_index = int(row_count * (1 - TEST_SIZE))

X_train = X_all[:split_index]
X_test = X_all[split_index:]
y_on_train = y_on_all[:split_index]
y_on_test = y_on_all[split_index:]
y_power_train = y_power_all[:split_index]
y_power_test = y_power_all[split_index:]

y_on_train_array = np.asarray(y_on_train)
y_on_test_array = np.asarray(y_on_test)
y_power_test_array = np.asarray(y_power_test)
power_scale = np.maximum(np.mean(np.abs(y_power_test_array), axis=0), EPSILON)


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


print("Training fixed classifier branch using classification-selected features...")
classifier_start = perf_counter()
X_classifier_train = matrix_from_features(X_train, CLASSIFIER_FEATURES)
X_classifier_test = matrix_from_features(X_test, CLASSIFIER_FEATURES)

classifier = ExtraTreesClassifier(**CLASSIFIER_PARAMS)
classifier.fit(X_classifier_train, y_on_train)
predicted_on_train = classifier.predict(X_classifier_train).astype(CACHE_LABEL_DTYPE)
predicted_on_test = classifier.predict(X_classifier_test).astype(CACHE_LABEL_DTYPE)
classifier_metrics = classification_scores(y_on_test_array, predicted_on_test)

print(
    "Classifier branch ready | "
    f"Macro F1={classifier_metrics['macro_f1']:.4f} | "
    f"Micro F1={classifier_metrics['micro_f1']:.4f} | "
    f"time={(perf_counter() - classifier_start) / 60:.1f} min",
    flush=True,
)
print("Fixed classifier branch classification report:")
print(classification_report(
    y_on_test_array,
    predicted_on_test,
    target_names=ON_OFF_LABEL_COLUMNS,
    zero_division=0,
))

del classifier
del X_classifier_train
del X_classifier_test
gc.collect()


# =============================================================================
# 6. Regression Scoring Helpers
# =============================================================================
def regression_scores(y_true, y_pred):
    error = y_true - y_pred
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    nmae = mae / power_scale
    nrmse = rmse / power_scale

    # SAE_delta follows the NILM literature definition:
    # for each fixed time window, compare the summed true/predicted energy and
    # divide by the number of points in that window. With 6-second samples,
    # 600 points correspond to one hour.
    usable_points = (len(y_true) // SAE_WINDOW_POINTS) * SAE_WINDOW_POINTS
    if usable_points == 0:
        usable_points = len(y_true)
        window_points = len(y_true)
    else:
        window_points = SAE_WINDOW_POINTS
    true_windows = y_true[:usable_points].reshape(-1, window_points, y_true.shape[1])
    pred_windows = y_pred[:usable_points].reshape(-1, window_points, y_pred.shape[1])
    sae = np.mean(
        np.abs(np.sum(true_windows, axis=1) - np.sum(pred_windows, axis=1)) / window_points,
        axis=0,
    )

    true_energy = np.sum(y_true, axis=0)
    predicted_energy = np.sum(y_pred, axis=0)
    relative_energy_error = (
        np.abs(predicted_energy - true_energy)
        / np.maximum(np.abs(true_energy), EPSILON)
    )
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
    for (
        label,
        mae_value,
        rmse_value,
        nmae_value,
        nrmse_value,
        sae_value,
        relative_energy_error_value,
        r2_value,
    ) in zip(
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
        scores[f"{label}_relative_energy_error"] = float(relative_energy_error_value)
        scores[f"{label}_r2"] = float(r2_value)
    return scores


def train_regressor_and_score(feature_subset, extra_train=None, extra_test=None):
    X_train_subset = matrix_from_features(X_train, feature_subset)
    X_test_subset = matrix_from_features(X_test, feature_subset)

    if extra_train is not None and extra_test is not None:
        X_train_subset = np.hstack([X_train_subset, extra_train.astype(CACHE_FEATURE_DTYPE)])
        X_test_subset = np.hstack([X_test_subset, extra_test.astype(CACHE_FEATURE_DTYPE)])

    model = ExtraTreesRegressor(**REGRESSOR_PARAMS)
    try:
        model.fit(X_train_subset, y_power_train)
        prediction = model.predict(X_test_subset)
        return regression_scores(y_power_test_array, prediction)
    finally:
        del model
        del X_train_subset
        del X_test_subset
        if "prediction" in locals():
            del prediction
        gc.collect()


def train_regressor_and_predict(feature_subset, extra_train=None, extra_test=None):
    X_train_subset = matrix_from_features(X_train, feature_subset)
    X_test_subset = matrix_from_features(X_test, feature_subset)

    if extra_train is not None and extra_test is not None:
        X_train_subset = np.hstack([X_train_subset, extra_train.astype(CACHE_FEATURE_DTYPE)])
        X_test_subset = np.hstack([X_test_subset, extra_test.astype(CACHE_FEATURE_DTYPE)])

    model = ExtraTreesRegressor(**REGRESSOR_PARAMS)
    try:
        model.fit(X_train_subset, y_power_train)
        return model.predict(X_test_subset)
    finally:
        del model
        del X_train_subset
        del X_test_subset
        gc.collect()


def evaluate_candidate(feature_subset):
    direct_scores = train_regressor_and_score(feature_subset)
    assisted_scores = train_regressor_and_score(
        feature_subset,
        extra_train=predicted_on_train,
        extra_test=predicted_on_test,
    )
    return direct_scores, assisted_scores


# =============================================================================
# 7. Wrapper Feature Selection: Regression Objective
# =============================================================================
# Selection decision uses classifier-assisted regression average normalized MAE.
# Direct regression is also recorded for diagnosis.
selected_features = []
remaining_features = FEATURE_COLUMNS.copy()
selection_log = []
start_time = perf_counter()

best_assisted_nmae = float("inf")
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
            f"NMAE={direct_scores['avg_nmae']:.4f} | "
            f"MAE={direct_scores['avg_mae']:.2f} W | "
            f"NRMSE={direct_scores['avg_nrmse']:.4f} | "
            f"SAE_delta={direct_scores['avg_sae']:.2f} W | "
            f"R2={direct_scores['avg_r2']:.4f}",
            flush=True,
        )
        print(
            f"        assisted | "
            f"NMAE={assisted_scores['avg_nmae']:.4f} | "
            f"MAE={assisted_scores['avg_mae']:.2f} W | "
            f"NRMSE={assisted_scores['avg_nrmse']:.4f} | "
            f"SAE_delta={assisted_scores['avg_sae']:.2f} W | "
            f"R2={assisted_scores['avg_r2']:.4f} | "
            f"time={candidate_elapsed:.1f}s",
            flush=True,
        )

        round_results.append({
            "candidate_feature": candidate_feature,
            "feature_subset": candidate_subset,
            "direct_scores": direct_scores,
            "assisted_scores": assisted_scores,
        })

    best_candidate = min(round_results, key=lambda item: item["assisted_scores"]["avg_nmae"])
    selected_features.append(best_candidate["candidate_feature"])
    remaining_features.remove(best_candidate["candidate_feature"])

    current_assisted_nmae = best_candidate["assisted_scores"]["avg_nmae"]
    improvement = best_assisted_nmae - current_assisted_nmae
    best_assisted_nmae = current_assisted_nmae

    log_row = {
        "round": round_number,
        "added_feature": best_candidate["candidate_feature"],
        "feature_count": len(selected_features),
        "selected_features": ",".join(selected_features),
        "assisted_avg_nmae": best_candidate["assisted_scores"]["avg_nmae"],
        "assisted_avg_nrmse": best_candidate["assisted_scores"]["avg_nrmse"],
        "assisted_avg_mae": best_candidate["assisted_scores"]["avg_mae"],
        "assisted_avg_rmse": best_candidate["assisted_scores"]["avg_rmse"],
        "assisted_avg_sae": best_candidate["assisted_scores"]["avg_sae"],
        "assisted_avg_relative_energy_error": best_candidate["assisted_scores"]["avg_relative_energy_error"],
        "assisted_avg_r2": best_candidate["assisted_scores"]["avg_r2"],
        "direct_avg_nmae": best_candidate["direct_scores"]["avg_nmae"],
        "direct_avg_nrmse": best_candidate["direct_scores"]["avg_nrmse"],
        "direct_avg_mae": best_candidate["direct_scores"]["avg_mae"],
        "direct_avg_rmse": best_candidate["direct_scores"]["avg_rmse"],
        "direct_avg_sae": best_candidate["direct_scores"]["avg_sae"],
        "direct_avg_relative_energy_error": best_candidate["direct_scores"]["avg_relative_energy_error"],
        "direct_avg_r2": best_candidate["direct_scores"]["avg_r2"],
        "improvement": improvement,
        "classifier_macro_f1": classifier_metrics["macro_f1"],
        "classifier_micro_f1": classifier_metrics["micro_f1"],
    }
    for label in POWER_LABEL_COLUMNS:
        log_row[f"assisted_{label}_nmae"] = best_candidate["assisted_scores"][f"{label}_nmae"]
        log_row[f"direct_{label}_nmae"] = best_candidate["direct_scores"][f"{label}_nmae"]
        log_row[f"assisted_{label}_nrmse"] = best_candidate["assisted_scores"][f"{label}_nrmse"]
        log_row[f"direct_{label}_nrmse"] = best_candidate["direct_scores"][f"{label}_nrmse"]
        log_row[f"assisted_{label}_mae"] = best_candidate["assisted_scores"][f"{label}_mae"]
        log_row[f"direct_{label}_mae"] = best_candidate["direct_scores"][f"{label}_mae"]
        log_row[f"assisted_{label}_rmse"] = best_candidate["assisted_scores"][f"{label}_rmse"]
        log_row[f"direct_{label}_rmse"] = best_candidate["direct_scores"][f"{label}_rmse"]
        log_row[f"assisted_{label}_sae"] = best_candidate["assisted_scores"][f"{label}_sae"]
        log_row[f"direct_{label}_sae"] = best_candidate["direct_scores"][f"{label}_sae"]
        log_row[f"assisted_{label}_relative_energy_error"] = best_candidate["assisted_scores"][f"{label}_relative_energy_error"]
        log_row[f"direct_{label}_relative_energy_error"] = best_candidate["direct_scores"][f"{label}_relative_energy_error"]
        log_row[f"assisted_{label}_r2"] = best_candidate["assisted_scores"][f"{label}_r2"]
        log_row[f"direct_{label}_r2"] = best_candidate["direct_scores"][f"{label}_r2"]

    selection_log.append(log_row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_log).to_csv(FORWARD_SELECTION_LOG, index=False)

    elapsed = perf_counter() - start_time
    print()
    print(f"Round {round_number} selected: {best_candidate['candidate_feature']}")
    print(f"Best assisted regression avg NMAE: {best_assisted_nmae:.4f}")
    print(f"Improvement this round: {improvement:.4f}")
    print("Regression comparison for selected feature combination:")
    print(pd.DataFrame([
        {"model": "direct", **{key: best_candidate["direct_scores"][key] for key in ["avg_nmae", "avg_nrmse", "avg_mae", "avg_rmse", "avg_sae", "avg_r2"]}},
        {"model": "classifier_assisted", **{key: best_candidate["assisted_scores"][key] for key in ["avg_nmae", "avg_nrmse", "avg_mae", "avg_rmse", "avg_sae", "avg_r2"]}},
    ]).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Per-appliance regression metrics for this selected feature combination:")
    appliance_metric_rows = []
    for label in POWER_LABEL_COLUMNS:
        appliance = label.replace("_power", "")
        appliance_metric_rows.append({
            "model": "direct",
            "appliance": appliance,
            "mae_w": best_candidate["direct_scores"][f"{label}_mae"],
            "rmse_w": best_candidate["direct_scores"][f"{label}_rmse"],
            "nmae": best_candidate["direct_scores"][f"{label}_nmae"],
            "nrmse": best_candidate["direct_scores"][f"{label}_nrmse"],
            "sae_delta_w": best_candidate["direct_scores"][f"{label}_sae"],
            "relative_energy_error": best_candidate["direct_scores"][f"{label}_relative_energy_error"],
            "r2": best_candidate["direct_scores"][f"{label}_r2"],
        })
        appliance_metric_rows.append({
            "model": "assisted",
            "appliance": label.replace("_power", ""),
            "mae_w": best_candidate["assisted_scores"][f"{label}_mae"],
            "rmse_w": best_candidate["assisted_scores"][f"{label}_rmse"],
            "nmae": best_candidate["assisted_scores"][f"{label}_nmae"],
            "nrmse": best_candidate["assisted_scores"][f"{label}_nrmse"],
            "sae_delta_w": best_candidate["assisted_scores"][f"{label}_sae"],
            "relative_energy_error": best_candidate["assisted_scores"][f"{label}_relative_energy_error"],
            "r2": best_candidate["assisted_scores"][f"{label}_r2"],
        })
    print(pd.DataFrame(appliance_metric_rows).to_string(
        index=False,
        float_format=lambda value: f"{value:.2f}",
    ))
    print(f"Elapsed time: {elapsed / 60:.1f} min")
    gc.collect()


# =============================================================================
# 8. Save Curves and Report
# =============================================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
selection_log_df = pd.DataFrame(selection_log)
selection_log_df.to_csv(FORWARD_SELECTION_LOG, index=False)

if not selection_log_df.empty:
    plt.style.use("seaborn-v0_8-whitegrid")
    METRIC_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    def save_average_metric_plot(metric_name, ylabel, lower_is_better=True):
        direct_column = f"direct_avg_{metric_name}"
        assisted_column = f"assisted_avg_{metric_name}"
        if direct_column not in selection_log_df.columns or assisted_column not in selection_log_df.columns:
            return

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(
            selection_log_df["feature_count"],
            selection_log_df[direct_column],
            marker="o",
            linewidth=2,
            label="Direct regression",
        )
        ax.plot(
            selection_log_df["feature_count"],
            selection_log_df[assisted_column],
            marker="s",
            linewidth=2,
            label="Classifier-assisted regression",
        )
        direction_text = "lower is better" if lower_is_better else "higher is better"
        ax.set_title(f"Average {ylabel} During Forward Selection", fontsize=14, weight="bold")
        ax.set_xlabel("Number of Selected Regression Features")
        ax.set_ylabel(f"{ylabel} ({direction_text})")
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
        ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(METRIC_PLOT_DIR / f"average_{metric_name}_curve.png", dpi=220)
        plt.close(fig)

    def save_per_appliance_assisted_metric_plot(metric_name, ylabel, lower_is_better=True):
        fig, ax = plt.subplots(figsize=(11, 5))
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
                    markersize=3.5,
                    label=label.replace("_power", ""),
                )
        if not plotted:
            plt.close(fig)
            return

        direction_text = "lower is better" if lower_is_better else "higher is better"
        ax.set_title(f"Per-Appliance Assisted {ylabel} During Forward Selection", fontsize=14, weight="bold")
        ax.set_xlabel("Number of Selected Regression Features")
        ax.set_ylabel(f"{ylabel} ({direction_text})")
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
        ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
        ax.legend(loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(METRIC_PLOT_DIR / f"per_appliance_assisted_{metric_name}_curve.png", dpi=220)
        plt.close(fig)

    metric_specs = [
        ("nmae", "Normalized MAE", True),
        ("nrmse", "Normalized RMSE", True),
        ("mae", "MAE (W)", True),
        ("rmse", "RMSE (W)", True),
        ("sae", "SAE_delta (W)", True),
        ("relative_energy_error", "Relative Energy Error", True),
        ("r2", "R2", False),
    ]
    for metric_name, ylabel, lower_is_better in metric_specs:
        save_average_metric_plot(metric_name, ylabel, lower_is_better)
        save_per_appliance_assisted_metric_plot(metric_name, ylabel, lower_is_better)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(selection_log_df["feature_count"], selection_log_df["direct_avg_nmae"], marker="o", linewidth=2, label="Direct regression")
    ax.plot(selection_log_df["feature_count"], selection_log_df["assisted_avg_nmae"], marker="s", linewidth=2, label="Classifier-assisted regression")
    ax.set_title("ExtraTrees Classification-Regression Forward Selection", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Selected Regression Features")
    ax.set_ylabel("Average Normalized MAE (lower is better)")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(REGRESSION_PLOT, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for label in POWER_LABEL_COLUMNS:
        column = f"assisted_{label}_nmae"
        if column in selection_log_df.columns:
            ax.plot(
                selection_log_df["feature_count"],
                selection_log_df[column],
                marker="o",
                linewidth=2,
                markersize=3.5,
                label=label.replace("_power", ""),
            )
    ax.set_title("Per-Appliance NMAE During Assisted Regression Selection", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Selected Regression Features")
    ax.set_ylabel("Normalized MAE (lower is better)")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(PER_APPLIANCE_NMAE_PLOT, dpi=220)
    plt.close(fig)

    with SELECTED_FEATURES_TXT.open("w", encoding="utf-8") as file:
        file.write("ExtraTrees Classification-Regression Forward Feature Selection\n")
        file.write(f"Dataset: {DATASET_PATH}\n")
        file.write(f"Rows: {row_count:,}\n")
        file.write(f"Classifier features ({len(CLASSIFIER_FEATURES)}): {', '.join(CLASSIFIER_FEATURES)}\n")
        file.write(f"Classifier Macro F1: {classifier_metrics['macro_f1']:.4f}\n")
        file.write(f"Classifier Micro F1: {classifier_metrics['micro_f1']:.4f}\n\n")
        file.write("Regression-selected feature order:\n")
        for _, row in selection_log_df.iterrows():
            file.write(
                f"Round {int(row['round']):02d}: "
                f"{row['added_feature']} | "
                f"assisted avg NMAE={row['assisted_avg_nmae']:.4f} | "
                f"direct avg NMAE={row['direct_avg_nmae']:.4f} | "
                f"assisted avg R2={row['assisted_avg_r2']:.4f} | "
                f"assisted avg SAE_delta={row['assisted_avg_sae']:.2f} W | "
                f"assisted relative energy error={row['assisted_avg_relative_energy_error']:.4f}\n"
            )


# =============================================================================
# 9. Save Final Test Predictions and Waveform Visualizations
# =============================================================================
final_direct_prediction = None
final_assisted_prediction = None

if selected_features:
    print("Training final direct and classifier-assisted regressors for visualization...")
    final_prediction_start = perf_counter()
    final_direct_prediction = train_regressor_and_predict(selected_features)
    final_assisted_prediction = train_regressor_and_predict(
        selected_features,
        extra_train=predicted_on_train,
        extra_test=predicted_on_test,
    )

    readable_time = pd.read_csv(DATASET_PATH, usecols=TIME_COLUMNS).iloc[split_index:].reset_index(drop=True)
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
print(f"Test rows: {len(X_test):,}")
print(f"Classifier Macro F1: {classifier_metrics['macro_f1']:.4f}")
print(f"Classifier Micro F1: {classifier_metrics['micro_f1']:.4f}")
print("Classifier report:")
print(classification_report(
    y_on_test_array,
    predicted_on_test,
    target_names=ON_OFF_LABEL_COLUMNS,
    zero_division=0,
))
print("Selected regression features:")
for index, feature in enumerate(selected_features, start=1):
    print(f"  {index:02d}. {feature}")
print(f"Selection log: {FORWARD_SELECTION_LOG}")
print(f"Regression curve: {REGRESSION_PLOT}")
print(f"Per-appliance NMAE curve: {PER_APPLIANCE_NMAE_PLOT}")
print(f"All metric curves: {METRIC_PLOT_DIR}")
print(f"Selected feature order: {SELECTED_FEATURES_TXT}")
if final_assisted_prediction is not None:
    print(f"Final prediction CSV: {PREDICTION_CSV}")
    print(f"Prediction waveform plots: {PREDICTION_PLOT_DIR}")

from pathlib import Path
from time import perf_counter
import gc
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# =============================================================================
# 1. Configuration
# =============================================================================
# Change these values to test another dataset or tune the baseline model.
FEATURE_SELECTION_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = FEATURE_SELECTION_DIR / "dataset"
DATASET_FILENAME = "multi_appliance_house2_wk24_to_wk31_merged.csv"
DATASET_PATH = DATASET_DIR / DATASET_FILENAME
BASE_RESULTS_DIR = FEATURE_SELECTION_DIR / "results"
RUN_NAME = f"extratrees_forward_selection_onoff_{Path(DATASET_FILENAME).stem}"
RESULTS_DIR = BASE_RESULTS_DIR / RUN_NAME
TUNING_RUN_NAME = f"extratrees_hyperparameter_tuning_onoff_{Path(DATASET_FILENAME).stem}"
BEST_PARAMS_PATH = BASE_RESULTS_DIR / TUNING_RUN_NAME / "best_hyperparameters.json"
LEGACY_BEST_PARAMS_PATH = BASE_RESULTS_DIR / "extratrees_best_hyperparameters.json"
FORWARD_SELECTION_LOG = RESULTS_DIR / "forward_selection_log.csv"
FORWARD_SELECTION_PLOT = RESULTS_DIR / "forward_selection_macro_micro_f1.png"
PER_APPLIANCE_PLOT = RESULTS_DIR / "forward_selection_per_appliance_f1.png"
SELECTED_FEATURES_TXT = RESULTS_DIR / "selected_features.txt"
EVIDENCE_PLOT_DIR = RESULTS_DIR / "onoff_evidence_plots"
CACHE_DIR = FEATURE_SELECTION_DIR / "cache" / Path(DATASET_FILENAME).stem
CACHE_METADATA_PATH = CACHE_DIR / "metadata.json"
X_CACHE_PATH = CACHE_DIR / "X_features_float32.dat"
Y_ON_CACHE_PATH = CACHE_DIR / "y_on_uint8.dat"

TRAIN_SIZE = 0.6
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42
CSV_CHUNKSIZE = 100_000
CACHE_FEATURE_DTYPE = "float32"
CACHE_LABEL_DTYPE = "uint8"
EVIDENCE_CONTEXT_POINTS = 60
EVIDENCE_MIN_ON_POINTS = 2
EVIDENCE_OFF_WINDOW_POINTS = 240
EVIDENCE_OFF_MAX_TRUE_ON_FRACTION = 0.0
EVIDENCE_OFF_MAX_PRED_ON_FRACTION = 0.0
EVIDENCE_MAX_WINDOWS_PER_TYPE = 3

DEFAULT_EXTRATREES_PARAMS = {
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

EXTRATREES_PARAMS = DEFAULT_EXTRATREES_PARAMS
HYPERPARAMETER_SOURCE = "manual tuned parameters"


# =============================================================================
# 2. Dataset Schema
# =============================================================================
# These appliance names are used to locate target columns such as:
#   kettle_power, fridge_power, ...
#   kettle_on, fridge_on, ...
APPLIANCE_NAMES = [
    "kettle",
    "fridge",
    "microwave",
    "dishwasher",
    "washingmachine",
]

# Metadata/time columns are not used as model input features.
TIME_COLUMNS = [
    "readable_time",
]

# Regression targets for the later power prediction stage.
POWER_LABEL_COLUMNS = [
    f"{appliance}_power" for appliance in APPLIANCE_NAMES
]

# Classification targets for the current ON/OFF mini NILM stage.
ON_OFF_LABEL_COLUMNS = [
    f"{appliance}_on" for appliance in APPLIANCE_NAMES
]

# Exclude metadata and target columns from the model input features.
NON_FEATURE_COLUMNS = TIME_COLUMNS + POWER_LABEL_COLUMNS + ON_OFF_LABEL_COLUMNS

CSV_COLUMNS = pd.read_csv(DATASET_PATH, nrows=0).columns.tolist()

# Input feature columns. This automatically keeps P_active, PF, harmonics,
# DWT energy, bandpower, entropy, aggregate, and other engineered features.
FEATURE_COLUMNS = [
    column for column in CSV_COLUMNS
    if column not in NON_FEATURE_COLUMNS
]


# =============================================================================
# 3. Disk-Backed Dataset Cache
# =============================================================================
# Long datasets should not be kept as a full pandas DataFrame. The cache below
# stores X and y_on as disk-backed arrays. The feature-selection logic is still
# the same; only data loading/storage changes.
def count_csv_rows(csv_path):
    with csv_path.open("rb") as file:
        return max(sum(1 for _ in file) - 1, 0)


def cache_is_valid(row_count):
    if not CACHE_METADATA_PATH.exists():
        return False
    if not X_CACHE_PATH.exists() or not Y_ON_CACHE_PATH.exists():
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
        "feature_dtype": CACHE_FEATURE_DTYPE,
        "label_dtype": CACHE_LABEL_DTYPE,
    }
    return metadata == expected


def build_disk_cache(row_count):
    print("Building disk-backed dataset cache...")
    print(f"Cache folder: {CACHE_DIR}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    x_cache = np.memmap(
        X_CACHE_PATH,
        dtype=CACHE_FEATURE_DTYPE,
        mode="w+",
        shape=(row_count, len(FEATURE_COLUMNS)),
    )
    y_cache = np.memmap(
        Y_ON_CACHE_PATH,
        dtype=CACHE_LABEL_DTYPE,
        mode="w+",
        shape=(row_count, len(ON_OFF_LABEL_COLUMNS)),
    )

    offset = 0
    use_columns = FEATURE_COLUMNS + ON_OFF_LABEL_COLUMNS
    for chunk_index, chunk in enumerate(
        pd.read_csv(DATASET_PATH, usecols=use_columns, chunksize=CSV_CHUNKSIZE),
        start=1,
    ):
        rows = len(chunk)
        row_slice = slice(offset, offset + rows)
        x_cache[row_slice, :] = chunk[FEATURE_COLUMNS].to_numpy(dtype=CACHE_FEATURE_DTYPE)
        y_cache[row_slice, :] = chunk[ON_OFF_LABEL_COLUMNS].to_numpy(dtype=CACHE_LABEL_DTYPE)
        offset += rows
        print(f"  Cached chunk {chunk_index}: {offset:,}/{row_count:,} rows", flush=True)

    x_cache.flush()
    y_cache.flush()
    del x_cache
    del y_cache
    gc.collect()

    source_stat = DATASET_PATH.stat()
    metadata = {
        "source_path": str(DATASET_PATH),
        "source_size": source_stat.st_size,
        "source_mtime": source_stat.st_mtime,
        "row_count": row_count,
        "feature_columns": FEATURE_COLUMNS,
        "on_off_label_columns": ON_OFF_LABEL_COLUMNS,
        "feature_dtype": CACHE_FEATURE_DTYPE,
        "label_dtype": CACHE_LABEL_DTYPE,
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


# =============================================================================
# 4. Time-Based Train/Validation/Test Split
# =============================================================================
# NILM data is time-series data, so we split by time instead of random shuffle:
# first 60% for training, next 20% for feature selection validation, final 20%
# for the held-out test report and evidence plots.
split_total = TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE
if abs(split_total - 1.0) > 1e-9:
    raise ValueError(
        "TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE must equal 1.0; "
        f"got {split_total:.6f}."
    )

train_end = int(row_count * TRAIN_SIZE)
validation_end = int(row_count * (TRAIN_SIZE + VALIDATION_SIZE))

if not (0 < train_end < validation_end < row_count):
    raise ValueError(
        "Invalid train/validation/test split. Check row_count, TRAIN_SIZE, "
        "VALIDATION_SIZE, and TEST_SIZE."
    )

X_train = X_all[:train_end]
X_validation = X_all[train_end:validation_end]
X_train_validation = X_all[:validation_end]
X_test = X_all[validation_end:]
y_on_train = y_on_all[:train_end]
y_on_validation = y_on_all[train_end:validation_end]
y_on_train_validation = y_on_all[:validation_end]
y_on_test = y_on_all[validation_end:]
y_on_validation_array = np.asarray(y_on_validation)
y_on_test_array = np.asarray(y_on_test)
test_metadata = pd.read_csv(
    DATASET_PATH,
    usecols=TIME_COLUMNS + POWER_LABEL_COLUMNS + ["aggregate"],
    skiprows=range(1, validation_end + 1),
)
test_time = pd.to_datetime(test_metadata[TIME_COLUMNS[0]])


# =============================================================================
# 5. Model Helper: ExtraTrees Multi-Appliance ON/OFF Classifier
# =============================================================================
# This function trains one ExtraTrees model for a selected feature subset and
# returns its validation scores. Forward selection will call this many times.
def train_and_score(feature_subset):
    feature_indices = [FEATURE_COLUMNS.index(feature) for feature in feature_subset]
    X_train_subset = np.asarray(X_train[:, feature_indices])
    X_validation_subset = np.asarray(X_validation[:, feature_indices])

    model = ExtraTreesClassifier(**EXTRATREES_PARAMS)

    try:
        model.fit(X_train_subset, y_on_train)
        prediction = model.predict(X_validation_subset)

        per_precision = precision_score(y_on_validation_array, prediction, average=None, zero_division=0)
        per_recall = recall_score(y_on_validation_array, prediction, average=None, zero_division=0)
        per_f1 = f1_score(y_on_validation_array, prediction, average=None, zero_division=0)

        per_accuracy = (y_on_validation_array == prediction).mean(axis=0)

        macro_precision = precision_score(y_on_validation_array, prediction, average="macro", zero_division=0)
        macro_recall = recall_score(y_on_validation_array, prediction, average="macro", zero_division=0)
        macro_f1 = f1_score(y_on_validation_array, prediction, average="macro", zero_division=0)
        macro_accuracy = float(per_accuracy.mean())

        micro_precision = precision_score(y_on_validation_array, prediction, average="micro", zero_division=0)
        micro_recall = recall_score(y_on_validation_array, prediction, average="micro", zero_division=0)
        micro_f1 = f1_score(y_on_validation_array, prediction, average="micro", zero_division=0)

        subset_accuracy = float((y_on_validation_array == prediction).all(axis=1).mean())

        per_appliance_scores = {}
        for label, precision, recall, f1, accuracy in zip(
            ON_OFF_LABEL_COLUMNS,
            per_precision,
            per_recall,
            per_f1,
            per_accuracy,
        ):
            per_appliance_scores[f"{label}_precision"] = float(precision)
            per_appliance_scores[f"{label}_recall"] = float(recall)
            per_appliance_scores[f"{label}_f1"] = float(f1)
            per_appliance_scores[f"{label}_accuracy"] = float(accuracy)

        average_scores = {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_accuracy": macro_accuracy,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "subset_accuracy": subset_accuracy,
        }

        return macro_f1, micro_f1, per_appliance_scores, average_scores
    finally:
        del model
        del X_train_subset
        del X_validation_subset
        if "prediction" in locals():
            del prediction
        gc.collect()


# =============================================================================
# 6. Wrapper Feature Selection: Forward Selection
# =============================================================================
# Start with no feature. At each round:
#   1. Try adding each remaining feature one by one.
#   2. Train an ExtraTrees NILM classifier for each candidate subset.
#   3. Keep the feature that gives the best Macro F1 improvement.
#   4. Continue until all features are selected so late improvements are visible.
selected_features = []
remaining_features = FEATURE_COLUMNS.copy()
selection_log = []
start_time = perf_counter()

best_macro_f1 = 0.0
best_micro_f1 = 0.0
best_prediction = None

for round_number in range(1, len(FEATURE_COLUMNS) + 1):
    round_results = []
    total_candidates = len(remaining_features)

    print()
    print("=" * 80)
    print(f"Forward selection round {round_number}/{len(FEATURE_COLUMNS)}")
    print(f"Currently selected features: {selected_features if selected_features else 'none'}")
    print(f"Testing {total_candidates} candidate feature(s)...")

    for candidate_index, candidate_feature in enumerate(remaining_features, start=1):
        candidate_subset = selected_features + [candidate_feature]
        candidate_start_time = perf_counter()

        print(
            f"  [{candidate_index:02d}/{total_candidates:02d}] "
            f"Testing add feature: {candidate_feature}",
            flush=True,
        )

        macro_f1, micro_f1, per_appliance_scores, average_scores = train_and_score(candidate_subset)
        candidate_elapsed = perf_counter() - candidate_start_time

        print(
            f"      Macro F1={macro_f1:.4f} | "
            f"Micro F1={micro_f1:.4f} | "
            f"time={candidate_elapsed:.1f}s",
            flush=True,
        )

        round_results.append({
            "round": round_number,
            "candidate_feature": candidate_feature,
            "feature_count": len(candidate_subset),
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "selected_features": ",".join(candidate_subset),
            "per_appliance_scores": per_appliance_scores,
            "average_scores": average_scores,
        })

    best_candidate = max(round_results, key=lambda item: item["macro_f1"])
    improvement = best_candidate["macro_f1"] - best_macro_f1

    selected_features.append(best_candidate["candidate_feature"])
    remaining_features.remove(best_candidate["candidate_feature"])

    best_macro_f1 = best_candidate["macro_f1"]
    best_micro_f1 = best_candidate["micro_f1"]

    selection_log.append({
        "round": round_number,
        "added_feature": best_candidate["candidate_feature"],
        "feature_count": len(selected_features),
        "macro_f1": best_macro_f1,
        "micro_f1": best_micro_f1,
        "improvement": improvement,
        "selected_features": ",".join(selected_features),
        **best_candidate["average_scores"],
        **best_candidate["per_appliance_scores"],
    })

    elapsed = perf_counter() - start_time
    print()
    print(f"Round {round_number} selected: {best_candidate['candidate_feature']}")
    print(f"Best Macro F1 so far: {best_macro_f1:.4f}")
    print(f"Best Micro F1 so far: {best_micro_f1:.4f}")
    print(f"Improvement this round: {improvement:.4f}")
    print("Validation classification metrics for this selected feature combination:")
    metric_rows = []
    for label in ON_OFF_LABEL_COLUMNS:
        scores = best_candidate["per_appliance_scores"]
        metric_rows.append({
            "label": label,
            "precision": scores[f"{label}_precision"],
            "recall": scores[f"{label}_recall"],
            "f1": scores[f"{label}_f1"],
            "accuracy": scores[f"{label}_accuracy"],
        })
    averages = best_candidate["average_scores"]
    metric_rows.append({
        "label": "macro_average",
        "precision": averages["macro_precision"],
        "recall": averages["macro_recall"],
        "f1": best_macro_f1,
        "accuracy": averages["macro_accuracy"],
    })
    metric_rows.append({
        "label": "micro_average",
        "precision": averages["micro_precision"],
        "recall": averages["micro_recall"],
        "f1": best_micro_f1,
        "accuracy": averages["subset_accuracy"],
    })
    metric_table = pd.DataFrame(metric_rows)
    print(metric_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Elapsed time: {elapsed / 60:.1f} min")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_log).to_csv(FORWARD_SELECTION_LOG, index=False)
    gc.collect()


# =============================================================================
# 7. Save Forward Selection Result and Score Curve
# =============================================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
selection_log_df = pd.DataFrame(selection_log)
selection_log_df.to_csv(FORWARD_SELECTION_LOG, index=False)

if not selection_log_df.empty:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        selection_log_df["feature_count"],
        selection_log_df["macro_f1"],
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Macro F1",
    )
    ax.plot(
        selection_log_df["feature_count"],
        selection_log_df["micro_f1"],
        color="#ff7f0e",
        linewidth=2.5,
        marker="s",
        markersize=4,
        label="Micro F1",
    )
    ax.set_title("ExtraTrees Forward Feature Selection on Validation Set", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Selected Features")
    ax.set_ylabel("Validation F1 Score")
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.legend(loc="lower right")
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FORWARD_SELECTION_PLOT, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for label in ON_OFF_LABEL_COLUMNS:
        score_column = f"{label}_f1"
        if score_column in selection_log_df.columns:
            ax.plot(
                selection_log_df["feature_count"],
                selection_log_df[score_column],
                linewidth=2,
                marker="o",
                markersize=3.5,
                label=label,
            )
    ax.set_title("Per-Appliance Validation F1 During Forward Selection", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Selected Features")
    ax.set_ylabel("Per-Appliance Validation F1 Score")
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.legend(loc="lower right", ncol=2)
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(PER_APPLIANCE_PLOT, dpi=220)
    plt.close(fig)

    with SELECTED_FEATURES_TXT.open("w", encoding="utf-8") as file:
        file.write("ExtraTrees Forward Feature Selection\n")
        file.write(f"Dataset: {DATASET_PATH}\n")
        file.write(f"Run folder: {RESULTS_DIR}\n\n")
        file.write("Selected feature order using validation F1:\n")
        for _, row in selection_log_df.iterrows():
            file.write(
                f"Round {int(row['round']):02d}: "
                f"{row['added_feature']} | "
                f"Validation Macro F1={row['macro_f1']:.4f} | "
                f"Validation Micro F1={row['micro_f1']:.4f}\n"
            )


def contiguous_true_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.r_[False, mask, False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts, ends))


def choose_on_windows(y_true_label):
    segments = [
        (start, end)
        for start, end in contiguous_true_segments(y_true_label == 1)
        if end - start >= EVIDENCE_MIN_ON_POINTS
    ]
    segments = sorted(segments, key=lambda item: item[1] - item[0], reverse=True)
    windows = []
    for start, end in segments[:EVIDENCE_MAX_WINDOWS_PER_TYPE]:
        window_start = max(0, start - EVIDENCE_CONTEXT_POINTS)
        window_end = min(len(y_true_label), end + EVIDENCE_CONTEXT_POINTS)
        windows.append((window_start, window_end, start, end))
    return windows


def choose_off_windows(y_true_label, y_pred_label, appliance_power):
    n_rows = len(y_true_label)
    if n_rows == 0:
        return []

    candidate_windows = []
    step = max(1, EVIDENCE_OFF_WINDOW_POINTS // 2)
    for start in range(0, max(1, n_rows - EVIDENCE_OFF_WINDOW_POINTS + 1), step):
        end = min(n_rows, start + EVIDENCE_OFF_WINDOW_POINTS)
        if end - start < max(12, EVIDENCE_OFF_WINDOW_POINTS // 4):
            continue
        true_on_fraction = float(np.mean(y_true_label[start:end]))
        pred_on_fraction = float(np.mean(y_pred_label[start:end]))
        if true_on_fraction > EVIDENCE_OFF_MAX_TRUE_ON_FRACTION:
            continue
        if pred_on_fraction > EVIDENCE_OFF_MAX_PRED_ON_FRACTION:
            continue

        aggregate_activity = float(np.mean(test_metadata["aggregate"].iloc[start:end]))
        appliance_activity = float(np.max(appliance_power[start:end]))
        candidate_windows.append((aggregate_activity, appliance_activity, start, end))

    candidate_windows = sorted(candidate_windows, reverse=True)
    return [(start, end, start, end) for _, _, start, end in candidate_windows[:EVIDENCE_MAX_WINDOWS_PER_TYPE]]


def plot_onoff_evidence_window(appliance, y_true_label, y_pred_label, window, kind, index):
    start, end, event_start, event_end = window
    time_values = test_time.iloc[start:end]
    power_column = f"{appliance}_power"
    appliance_power = test_metadata[power_column].iloc[start:end].to_numpy(dtype=float)
    aggregate_power = test_metadata["aggregate"].iloc[start:end].to_numpy(dtype=float)
    true_state = y_true_label[start:end].astype(float)
    pred_state = y_pred_label[start:end].astype(float)

    max_power = max(float(np.nanmax(appliance_power)) if len(appliance_power) else 0.0, 1.0)
    state_scale = max_power * 1.08

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(time_values, appliance_power, color="#1f2937", linewidth=2.2, label="ground-truth appliance power")
    ax.step(time_values, true_state * state_scale, where="post", color="#059669", linewidth=2.0, label="ground-truth ON state")
    ax.step(time_values, pred_state * state_scale, where="post", color="#dc2626", linewidth=1.8, linestyle="--", label="predicted ON state")

    if kind == "on":
        ax.axvspan(
            test_time.iloc[event_start],
            test_time.iloc[max(event_start, event_end - 1)],
            color="#bbf7d0",
            alpha=0.28,
            label="selected ON interval",
        )

    ax2 = ax.twinx()
    ax2.plot(time_values, aggregate_power, color="#94a3b8", linewidth=1.4, alpha=0.75, label="aggregate")
    ax2.set_ylabel("Aggregate power (W)", color="#64748b")
    ax2.tick_params(axis="y", colors="#64748b")

    title_kind = "ON-period zoom" if kind == "on" else "OFF-period false-positive check"
    ax.set_title(f"{appliance}: {title_kind}", fontsize=13, weight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Appliance power / state marker")
    ax.grid(True, which="major", alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper left", fontsize=8)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()

    out_path = EVIDENCE_PLOT_DIR / f"{appliance}_{kind}_evidence_{index:02d}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def save_onoff_evidence_plots(prediction):
    EVIDENCE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for label_index, label in enumerate(ON_OFF_LABEL_COLUMNS):
        appliance = label.replace("_on", "")
        power_column = f"{appliance}_power"
        y_true_label = y_on_test_array[:, label_index].astype(np.uint8)
        y_pred_label = prediction[:, label_index].astype(np.uint8)
        appliance_power = test_metadata[power_column].to_numpy(dtype=float)

        on_windows = choose_on_windows(y_true_label)
        for index, window in enumerate(on_windows, start=1):
            saved_paths.append(plot_onoff_evidence_window(appliance, y_true_label, y_pred_label, window, "on", index))

        off_windows = choose_off_windows(y_true_label, y_pred_label, appliance_power)
        for index, window in enumerate(off_windows, start=1):
            saved_paths.append(plot_onoff_evidence_window(appliance, y_true_label, y_pred_label, window, "off", index))

    return saved_paths


# =============================================================================
# 8. Console Report
# =============================================================================
print(f"Dataset: {DATASET_PATH}")
print(f"Rows: {row_count:,}")
print(f"Input features ({len(FEATURE_COLUMNS)}):")
for feature in FEATURE_COLUMNS:
    print(f"  - {feature}")

print(f"Power labels ({len(POWER_LABEL_COLUMNS)}): {POWER_LABEL_COLUMNS}")
print(f"ON/OFF labels ({len(ON_OFF_LABEL_COLUMNS)}): {ON_OFF_LABEL_COLUMNS}")
print()
print("ExtraTrees forward feature selection")
print(f"Hyperparameters: {HYPERPARAMETER_SOURCE}")
print(f"Train rows: {len(X_train):,}")
print(f"Validation rows: {len(X_validation):,}")
print(f"Final model train rows: {len(X_train_validation):,} (train + validation)")
print(f"Test rows: {len(X_test):,}")
print()
print("Selected features:")
for index, feature in enumerate(selected_features, start=1):
    print(f"  {index:02d}. {feature}")

print()
best_evidence_row = selection_log_df.loc[selection_log_df["macro_f1"].idxmax()] if not selection_log_df.empty else None
best_evidence_features = (
    [feature for feature in str(best_evidence_row["selected_features"]).split(",") if feature]
    if best_evidence_row is not None
    else selected_features
)
if best_evidence_row is not None:
    print(f"Peak validation Macro F1: {float(best_evidence_row['macro_f1']):.4f}")
    print(f"Peak validation Micro F1 at peak Macro round: {float(best_evidence_row['micro_f1']):.4f}")
    print(
        f"Final test/evidence model uses validation-selected Macro-F1 subset "
        f"from round {int(best_evidence_row['round'])} "
        f"({len(best_evidence_features)} features)."
    )
else:
    print(f"Final validation Macro F1: {best_macro_f1:.4f}")
    print(f"Final validation Micro F1: {best_micro_f1:.4f}")
print(f"Selection log: {FORWARD_SELECTION_LOG}")
print(f"Selection curve: {FORWARD_SELECTION_PLOT}")
print(f"Per-appliance curve: {PER_APPLIANCE_PLOT}")
print(f"Selected feature order: {SELECTED_FEATURES_TXT}")

if best_evidence_features:
    final_feature_indices = [FEATURE_COLUMNS.index(feature) for feature in best_evidence_features]
    final_X_train = np.asarray(X_train_validation[:, final_feature_indices])
    final_X_test = np.asarray(X_test[:, final_feature_indices])
    final_model = ExtraTreesClassifier(**EXTRATREES_PARAMS)
    final_model.fit(final_X_train, y_on_train_validation)
    best_prediction = final_model.predict(final_X_test)

    print()
    print("Held-out test classification report:")
    print(classification_report(
        y_on_test_array,
        best_prediction,
        target_names=ON_OFF_LABEL_COLUMNS,
        zero_division=0,
    ))

    evidence_paths = save_onoff_evidence_plots(best_prediction)
    print(f"ON/OFF evidence plots: {EVIDENCE_PLOT_DIR}")
    print(f"Saved evidence plot count: {len(evidence_paths)}")

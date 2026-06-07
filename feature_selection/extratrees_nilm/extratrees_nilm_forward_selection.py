from pathlib import Path
from time import perf_counter
import json

import matplotlib.pyplot as plt
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

TEST_SIZE = 0.2
RANDOM_STATE = 42

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
# 2. Load Dataset
# =============================================================================
# This CSV already contains aligned timesteps:
#   - aggregate-derived input features
#   - appliance power labels
#   - appliance ON/OFF labels
df = pd.read_csv(DATASET_PATH)


# =============================================================================
# 3. Preprocessing: Detect Feature Columns and Label Columns
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

# Input feature columns. This automatically keeps P_active, PF, harmonics,
# DWT energy, bandpower, entropy, aggregate, and other engineered features.
FEATURE_COLUMNS = [
    column for column in df.columns
    if column not in NON_FEATURE_COLUMNS
]


# =============================================================================
# 4. Build Model Matrices
# =============================================================================
# X is the input feature matrix.
# y_on is the multi-appliance ON/OFF classification label matrix.
# y_power is prepared for the later regression model but is not trained yet.
X = df[FEATURE_COLUMNS]
y_on = df[ON_OFF_LABEL_COLUMNS]
y_power = df[POWER_LABEL_COLUMNS]


# =============================================================================
# 5. Time-Based Train/Test Split
# =============================================================================
# NILM data is time-series data, so we split by time instead of random shuffle:
# first 80% for training, final 20% for testing.
split_index = int(len(df) * (1 - TEST_SIZE))

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_on_train = y_on.iloc[:split_index]
y_on_test = y_on.iloc[split_index:]


# =============================================================================
# 6. Model Helper: ExtraTrees Multi-Appliance ON/OFF Classifier
# =============================================================================
# This function trains one ExtraTrees model for a selected feature subset and
# returns its validation scores. Forward selection will call this many times.
def train_and_score(feature_subset):
    model = ExtraTreesClassifier(**EXTRATREES_PARAMS)

    model.fit(X_train[feature_subset], y_on_train)
    prediction = model.predict(X_test[feature_subset])

    per_precision = precision_score(y_on_test, prediction, average=None, zero_division=0)
    per_recall = recall_score(y_on_test, prediction, average=None, zero_division=0)
    per_f1 = f1_score(y_on_test, prediction, average=None, zero_division=0)

    y_true_array = y_on_test.to_numpy()
    per_accuracy = (y_true_array == prediction).mean(axis=0)

    macro_precision = precision_score(y_on_test, prediction, average="macro", zero_division=0)
    macro_recall = recall_score(y_on_test, prediction, average="macro", zero_division=0)
    macro_f1 = f1_score(y_on_test, prediction, average="macro", zero_division=0)
    macro_accuracy = float(per_accuracy.mean())

    micro_precision = precision_score(y_on_test, prediction, average="micro", zero_division=0)
    micro_recall = recall_score(y_on_test, prediction, average="micro", zero_division=0)
    micro_f1 = f1_score(y_on_test, prediction, average="micro", zero_division=0)

    subset_accuracy = float((y_true_array == prediction).all(axis=1).mean())

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

    return model, prediction, macro_f1, micro_f1, per_appliance_scores, average_scores


# =============================================================================
# 7. Wrapper Feature Selection: Forward Selection
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
best_model = None
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

        (
            candidate_model,
            candidate_prediction,
            macro_f1,
            micro_f1,
            per_appliance_scores,
            average_scores,
        ) = train_and_score(candidate_subset)
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
            "model": candidate_model,
            "prediction": candidate_prediction,
            "per_appliance_scores": per_appliance_scores,
            "average_scores": average_scores,
        })

    best_candidate = max(round_results, key=lambda item: item["macro_f1"])
    improvement = best_candidate["macro_f1"] - best_macro_f1

    selected_features.append(best_candidate["candidate_feature"])
    remaining_features.remove(best_candidate["candidate_feature"])

    best_macro_f1 = best_candidate["macro_f1"]
    best_micro_f1 = best_candidate["micro_f1"]
    best_model = best_candidate["model"]
    best_prediction = best_candidate["prediction"]

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
    print("Classification metrics for this selected feature combination:")
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


# =============================================================================
# 8. Save Forward Selection Result and Score Curve
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
    ax.set_title("ExtraTrees Forward Feature Selection", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Selected Features")
    ax.set_ylabel("Classification F1 Score")
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
    ax.set_title("Per-Appliance F1 During Forward Selection", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Selected Features")
    ax.set_ylabel("Per-Appliance F1 Score")
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
        file.write("Selected feature order:\n")
        for _, row in selection_log_df.iterrows():
            file.write(
                f"Round {int(row['round']):02d}: "
                f"{row['added_feature']} | "
                f"Macro F1={row['macro_f1']:.4f} | "
                f"Micro F1={row['micro_f1']:.4f}\n"
            )


# =============================================================================
# 9. Console Report
# =============================================================================
print(f"Dataset: {DATASET_PATH}")
print(f"Rows: {len(df):,}")
print(f"Input features ({len(FEATURE_COLUMNS)}):")
for feature in FEATURE_COLUMNS:
    print(f"  - {feature}")

print(f"Power labels ({len(POWER_LABEL_COLUMNS)}): {POWER_LABEL_COLUMNS}")
print(f"ON/OFF labels ({len(ON_OFF_LABEL_COLUMNS)}): {ON_OFF_LABEL_COLUMNS}")
print()
print("ExtraTrees forward feature selection")
print(f"Hyperparameters: {HYPERPARAMETER_SOURCE}")
print(f"Train rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")
print()
print("Selected features:")
for index, feature in enumerate(selected_features, start=1):
    print(f"  {index:02d}. {feature}")

print()
print(f"Best Macro F1: {best_macro_f1:.4f}")
print(f"Best Micro F1: {best_micro_f1:.4f}")
print(f"Selection log: {FORWARD_SELECTION_LOG}")
print(f"Selection curve: {FORWARD_SELECTION_PLOT}")
print(f"Per-appliance curve: {PER_APPLIANCE_PLOT}")
print(f"Selected feature order: {SELECTED_FEATURES_TXT}")

if best_prediction is not None:
    print()
    print(classification_report(
        y_on_test,
        best_prediction,
        target_names=ON_OFF_LABEL_COLUMNS,
        zero_division=0,
    ))

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report

# =============================================================================
# 1. Configuration
# =============================================================================
# Change these values to test another dataset or tune the baseline model.
DATASET_DIR = Path(__file__).parent / "dataset"
DATASET_FILENAME = "multi_appliance_house2_wk30_to_wk31_merged.csv"
DATASET_PATH = DATASET_DIR / DATASET_FILENAME
RESULTS_DIR = Path(__file__).parent / "results"
FORWARD_SELECTION_LOG = RESULTS_DIR / "extratrees_forward_selection_log.csv"
FORWARD_SELECTION_PLOT = RESULTS_DIR / "extratrees_forward_selection_curve.png"
PER_APPLIANCE_PLOT = RESULTS_DIR / "extratrees_forward_selection_per_appliance.png"
SELECTED_FEATURES_TXT = RESULTS_DIR / "extratrees_selected_features.txt"

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 50
MAX_DEPTH = 20


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
    model = ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train[feature_subset], y_on_train)
    prediction = model.predict(X_test[feature_subset])

    macro = f1_score(y_on_test, prediction, average="macro", zero_division=0)
    micro = f1_score(y_on_test, prediction, average="micro", zero_division=0)
    per_appliance = f1_score(y_on_test, prediction, average=None, zero_division=0)
    per_appliance_scores = {
        f"{label}_f1": score
        for label, score in zip(ON_OFF_LABEL_COLUMNS, per_appliance)
    }

    return model, prediction, macro, micro, per_appliance_scores


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
        **best_candidate["per_appliance_scores"],
    })

    elapsed = perf_counter() - start_time
    print()
    print(f"Round {round_number} selected: {best_candidate['candidate_feature']}")
    print(f"Best Macro F1 so far: {best_macro_f1:.4f}")
    print(f"Best Micro F1 so far: {best_micro_f1:.4f}")
    print(f"Improvement this round: {improvement:.4f}")
    print("Per-appliance F1 for this selected feature combination:")
    for label in ON_OFF_LABEL_COLUMNS:
        score = best_candidate["per_appliance_scores"][f"{label}_f1"]
        print(f"  {label}: {score:.4f}")
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
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
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
                label=label,
            )
    ax.set_title("Per-Appliance F1 During Forward Selection", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Selected Features")
    ax.set_ylabel("Per-Appliance F1 Score")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PER_APPLIANCE_PLOT, dpi=220)
    plt.close(fig)

    with SELECTED_FEATURES_TXT.open("w", encoding="utf-8") as file:
        file.write("ExtraTrees Forward Feature Selection\n")
        file.write(f"Dataset: {DATASET_PATH}\n\n")
        file.write("Selected feature order:\n")
        for _, row in selection_log_df.iterrows():
            file.write(
                f"Round {int(row['round']):02d}: "
                f"{row['added_feature']} | "
                f"Macro F1={row['macro_f1']:.4f} | "
                f"Micro F1={row['micro_f1']:.4f}\n"
            )

    """
    Old plot style with labels on every point. Kept disabled because the
    annotations overlap heavily once many features are selected.
    for _, row in selection_log_df.iterrows():
        plt.annotate(
            row["added_feature"],
            (row["feature_count"], row["macro_f1"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            rotation=25,
        )

    plt.title("ExtraTrees Forward Feature Selection")
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Classification F1 Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FORWARD_SELECTION_PLOT, dpi=200)
    plt.close()
    """


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

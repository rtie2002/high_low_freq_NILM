from pathlib import Path

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

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
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
# 6. Model: ExtraTrees Multi-Appliance ON/OFF Classifier
# =============================================================================
# ExtraTrees accepts a multi-output binary target matrix, so one model object
# predicts all appliance ON/OFF labels:
#   X -> [kettle_on, fridge_on, microwave_on, dishwasher_on, washingmachine_on]
on_off_model = ExtraTreesClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

on_off_model.fit(X_train, y_on_train)
y_on_pred = on_off_model.predict(X_test)


# =============================================================================
# 7. Evaluation
# =============================================================================
# Macro F1 treats each appliance equally.
# Micro F1 aggregates all appliance decisions and is influenced by frequent loads.
macro_f1 = f1_score(y_on_test, y_on_pred, average="macro", zero_division=0)
micro_f1 = f1_score(y_on_test, y_on_pred, average="micro", zero_division=0)


# =============================================================================
# 8. Console Report
# =============================================================================
print(f"Dataset: {DATASET_PATH}")
print(f"Rows: {len(df):,}")
print(f"Input features ({len(FEATURE_COLUMNS)}):")
for feature in FEATURE_COLUMNS:
    print(f"  - {feature}")

print(f"Power labels ({len(POWER_LABEL_COLUMNS)}): {POWER_LABEL_COLUMNS}")
print(f"ON/OFF labels ({len(ON_OFF_LABEL_COLUMNS)}): {ON_OFF_LABEL_COLUMNS}")
print()
print("ExtraTrees ON/OFF classification")
print(f"Train rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Micro F1: {micro_f1:.4f}")
print()
print(classification_report(
    y_on_test,
    y_on_pred,
    target_names=ON_OFF_LABEL_COLUMNS,
    zero_division=0,
))

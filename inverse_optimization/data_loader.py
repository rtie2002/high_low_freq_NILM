from pathlib import Path
import pandas as pd

# ============================================================
# 1. Find the dataset
# ============================================================

DATA_ROOT = Path(
    "dataset/mixed_ukdale_refit_5w_house_split"
)

train_csv = DATA_ROOT / "training" / "multi_appliance_training.csv"
valid_csv = DATA_ROOT / "validating" / "multi_appliance_validating.csv"
test_csv = DATA_ROOT / "testing" / "multi_appliance_testing.csv"

# ============================================================
# 2. Read the three splits
# ============================================================

train_data = pd.read_csv(train_csv)
valid_data = pd.read_csv(valid_csv)
test_data = pd.read_csv(test_csv)

# print(train_data.shape)
# print(valid_data.shape)
# print(test_data.shape)
print(train_data.columns.tolist()) #print all the column name

# ============================================================
# 3. Define one fixed appliance order
#
# Every future model, metric, plot, and prediction must use
# this exact same order.
# ============================================================

appliance_columns = [
    "kettle_power",
    "fridge_power",
    "dishwasher_power",
    "washingmachine_power",
    "microwave_power",
]

# Stage 1 forward-model data
X_train = train_data[appliance_columns].values
y_train = train_data["aggregate"].values

X_valid = valid_data[appliance_columns].values
y_valid = valid_data["aggregate"].values

X_test = test_data[appliance_columns].values
y_test = test_data["aggregate"].values




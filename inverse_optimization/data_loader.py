"""Load one NILM split and cut its continuous recordings into windows.

Stage 1 forward learning:
    five appliance powers [windows, time, 5] -> aggregate [windows, time]

The ON/OFF labels are returned for evaluation, not used as model inputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Settings and fixed column order
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
FORWARD_CONFIG_PATH = PROJECT_DIR / "stage1_forward" / "config.yaml"

# The window settings belong to the Stage-1 forward-model experiment.
with FORWARD_CONFIG_PATH.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

WINDOW_LENGTH = config["data"]["window_length"]
STRIDE = config["data"]["stride"]

DATA_ROOT = PROJECT_DIR / "dataset" / "mixed_ukdale_refit_5w_house_split"

SPLIT_FILES = {
    "training": DATA_ROOT / "training" / "multi_appliance_training.csv",
    "validating": DATA_ROOT / "validating" / "multi_appliance_validating.csv",
    "testing": DATA_ROOT / "testing" / "multi_appliance_testing.csv",
}

APPLIANCES = ["kettle", "fridge", "dishwasher", "washingmachine", "microwave"]

POWER_COLUMNS = [
    "kettle_power",
    "fridge_power",
    "dishwasher_power",
    "washingmachine_power",
    "microwave_power",
]

STATE_COLUMNS = [
    "kettle_on",
    "fridge_on",
    "dishwasher_on",
    "washingmachine_on",
    "microwave_on",
]


def load_windows(split_name: str):
    """Load only one split and return aligned power, aggregate, and state windows.

    Parameters
    ----------
    split_name:
        "training", "validating", or "testing".

    Returns
    -------
    appliance_power:
        float32 array [number_of_windows, WINDOW_LENGTH, 5].
        This is the Stage-1 GRU/LSTM input.
    aggregate_power:
        float32 array [number_of_windows, WINDOW_LENGTH].
        This is the Stage-1 GRU/LSTM target.
    appliance_state:
        int64 array [number_of_windows, WINDOW_LENGTH, 5].
        This is retained for later F1 evaluation only.
    """
    # Only the requested CSV enters memory. Training never loads the test CSV.
    data = pd.read_csv(SPLIT_FILES[split_name])

    # Ensure each group is in real time order before making windows.
    data = data.sort_values(["dataset", "house", "sequence_id", "unix_time"])

    power_windows = []
    aggregate_windows = []
    state_windows = []

    # A window stays within one continuous recording and never crosses a house
    # or recording boundary.
    groups = data.groupby(["dataset", "house", "sequence_id"], sort=False)
    for _, one_sequence in groups:
        sequence_power = one_sequence[POWER_COLUMNS].to_numpy(dtype=np.float32)
        sequence_aggregate = one_sequence["aggregate"].to_numpy(dtype=np.float32)
        sequence_state = one_sequence[STATE_COLUMNS].to_numpy(dtype=np.int64)

        for start in range(0, len(one_sequence) - WINDOW_LENGTH + 1, STRIDE):
            end = start + WINDOW_LENGTH
            power_windows.append(sequence_power[start:end])
            aggregate_windows.append(sequence_aggregate[start:end])
            state_windows.append(sequence_state[start:end])

    return (
        np.asarray(power_windows, dtype=np.float32),
        np.asarray(aggregate_windows, dtype=np.float32),
        np.asarray(state_windows, dtype=np.int64),
    )


if __name__ == "__main__":
    # This block runs only for: python data_loader.py
    # It does not run when train_forward.py imports load_windows.
    X_train, y_train, state_train = load_windows("training")

    print("Appliance order:", APPLIANCES)
    print("Stage-1 input:", X_train.shape, "= [windows, time, 5]")
    print("Stage-1 target:", y_train.shape, "= [windows, time]")
    print("Evaluation labels:", state_train.shape, "= [windows, time, 5]")

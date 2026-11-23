"""Load one NILM split and cut its continuous recordings into windows.

Stage 1 forward learning:
    five appliance powers plus optional calendar inputs -> aggregate [windows, time]

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

# These are known at inference time. They describe the local UK calendar, not
# appliance power, so Stage 2 must never optimise them.
TIME_FEATURE_COLUMNS = ["day_sin", "day_cos", "week_sin", "week_cos"]
INPUT_COLUMNS = POWER_COLUMNS + TIME_FEATURE_COLUMNS

STATE_COLUMNS = [
    "kettle_on",
    "fridge_on",
    "dishwasher_on",
    "washingmachine_on",
    "microwave_on",
]


def load_windows(
    split_name: str,
    stride: int | None = None,
    dataset: str | None = None,
    house: int | None = None,
    include_calendar_features: bool = False,
):
    """Load only one split and return aligned power, aggregate, and state windows.

    Parameters
    ----------
    split_name:
        "training", "validating", or "testing".
    stride:
        Number of samples between consecutive window starts. When omitted,
        use the Stage-1 training stride from ``stage1_forward/config.yaml``.
        Stage 2 may provide a different value because the frozen forward model
        can evaluate any window with the same fixed ``WINDOW_LENGTH``.
    dataset, house:
        Optional held-out dataset and house filter. Stage 2 uses these only
        for reporting one test house at a time.
    include_calendar_features:
        When true, append four known local-time features to the five appliance
        powers, producing a nine-feature forward-model input.

    Returns
    -------
    appliance_power:
        float32 array [number_of_windows, WINDOW_LENGTH, 5 or 9].
        This is the Stage-1 LSTM input. The optional four extra columns are
        fixed calendar features, never appliance labels.
    aggregate_power:
        float32 array [number_of_windows, WINDOW_LENGTH].
        This is the Stage-1 GRU/LSTM target.
    appliance_state:
        int64 array [number_of_windows, WINDOW_LENGTH, 5].
        This is retained for later F1 evaluation only.
    """
    # Only the requested CSV enters memory. Training never loads the test CSV.
    window_stride = STRIDE if stride is None else stride
    data = pd.read_csv(SPLIT_FILES[split_name])
    if dataset is not None:
        data = data[data["dataset"] == dataset]
    if house is not None:
        data = data[data["house"] == house]

    # Ensure each group is in real time order before making windows.
    data = data.sort_values(["dataset", "house", "sequence_id", "unix_time"])
    if include_calendar_features:
        data = add_calendar_features(data)
    input_columns = INPUT_COLUMNS if include_calendar_features else POWER_COLUMNS

    power_windows = []
    aggregate_windows = []
    state_windows = []

    # A window stays within one continuous recording and never crosses a house
    # or recording boundary.
    groups = data.groupby(["dataset", "house", "sequence_id"], sort=False)
    for _, one_sequence in groups:
        sequence_power = one_sequence[input_columns].to_numpy(dtype=np.float32)
        sequence_aggregate = one_sequence["aggregate"].to_numpy(dtype=np.float32)
        sequence_state = one_sequence[STATE_COLUMNS].to_numpy(dtype=np.int64)

        for start in range(0, len(one_sequence) - WINDOW_LENGTH + 1, window_stride):
            end = start + WINDOW_LENGTH
            power_windows.append(sequence_power[start:end])
            aggregate_windows.append(sequence_aggregate[start:end])
            state_windows.append(sequence_state[start:end])

    return (
        np.asarray(power_windows, dtype=np.float32),
        np.asarray(aggregate_windows, dtype=np.float32),
        np.asarray(state_windows, dtype=np.int64),
    )


def add_calendar_features(data):
    """Add fixed cyclic local-time inputs for the UK NILM datasets."""
    local_time = pd.to_datetime(data["readable_time"], utc=True).dt.tz_convert(
        "Europe/London"
    )
    seconds_in_day = 24 * 60 * 60
    time_of_day = (
        local_time.dt.hour * 3600
        + local_time.dt.minute * 60
        + local_time.dt.second
    )
    day_of_week = local_time.dt.dayofweek
    data = data.copy()
    data["day_sin"] = np.sin(2 * np.pi * time_of_day / seconds_in_day)
    data["day_cos"] = np.cos(2 * np.pi * time_of_day / seconds_in_day)
    data["week_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    data["week_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    return data


def load_random_continuous_interval(
    split_name: str, length: int, seed: int, include_calendar_features: bool = False
):
    """Return one reproducibly random interval that has no time gaps.

    The interval is selected from one dataset/house/sequence_id group, so it
    can safely be used for long waveform plots such as a day or a week.
    """
    data = pd.read_csv(SPLIT_FILES[split_name]).sort_values(
        ["dataset", "house", "sequence_id", "unix_time"]
    )
    if include_calendar_features:
        data = add_calendar_features(data)
    input_columns = INPUT_COLUMNS if include_calendar_features else POWER_COLUMNS
    groups = data.groupby(["dataset", "house", "sequence_id"], sort=False)
    eligible_groups = groups.size()
    eligible_groups = eligible_groups[eligible_groups >= length]
    if eligible_groups.empty:
        raise ValueError(
            f"No continuous {split_name} sequence contains {length} samples."
        )

    rng = np.random.default_rng(seed)
    group_name = eligible_groups.index[rng.integers(len(eligible_groups))]
    one_sequence = groups.get_group(group_name)
    start = rng.integers(len(one_sequence) - length + 1)
    interval = one_sequence.iloc[start:start + length]
    return (
        interval[input_columns].to_numpy(dtype=np.float32),
        interval["aggregate"].to_numpy(dtype=np.float32),
    )


def load_random_appliance_event_windows(
    split_name: str, length: int, seed: int, include_calendar_features: bool = False
):
    """Return one continuous ON-event interval and CSV ON/OFF labels per appliance."""
    data = pd.read_csv(SPLIT_FILES[split_name]).sort_values(
        ["dataset", "house", "sequence_id", "unix_time"]
    )
    if include_calendar_features:
        data = add_calendar_features(data)
    input_columns = INPUT_COLUMNS if include_calendar_features else POWER_COLUMNS
    groups = data.groupby(["dataset", "house", "sequence_id"], sort=False)
    rng = np.random.default_rng(seed)
    examples = {}

    for appliance, state_column in zip(APPLIANCES, STATE_COLUMNS):
        eligible_names = [
            group_name
            for group_name, one_sequence in groups
            if len(one_sequence) >= length and one_sequence[state_column].any()
        ]
        if not eligible_names:
            raise ValueError(f"No {split_name} window contains an ON event for {appliance}.")

        one_sequence = groups.get_group(eligible_names[rng.integers(len(eligible_names))])
        state = one_sequence[state_column].to_numpy(dtype=np.int64)
        starts = np.flatnonzero(state.astype(bool) & np.r_[True, ~state[:-1].astype(bool)])
        ends = np.flatnonzero(state.astype(bool) & np.r_[~state[1:].astype(bool), True])
        event_start, event_end = list(zip(starts, ends))[rng.integers(len(starts))]
        center = (int(event_start) + int(event_end)) // 2
        start = min(max(center - length // 2, 0), len(one_sequence) - length)
        interval = one_sequence.iloc[start:start + length]
        examples[appliance] = (
            interval[input_columns].to_numpy(dtype=np.float32),
            interval["aggregate"].to_numpy(dtype=np.float32),
            interval[STATE_COLUMNS].to_numpy(dtype=np.int64),
        )

    return examples


if __name__ == "__main__":
    # This block runs only for: python data_loader.py
    # It does not run when train_forward.py imports load_windows.
    X_train, y_train, state_train = load_windows("training")

    print("Appliance order:", APPLIANCES)
    print("Stage-1 input:", X_train.shape, "= [windows, time, 5]")
    print("Stage-1 target:", y_train.shape, "= [windows, time]")
    print("Evaluation labels:", state_train.shape, "= [windows, time, 5]")

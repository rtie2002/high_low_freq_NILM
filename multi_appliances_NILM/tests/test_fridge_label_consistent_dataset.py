from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.prepare_fridge_label_consistent_dataset import relabel_refit11_fridge


def test_relabels_only_refit11_and_zeros_newly_off_power() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["refit"] * 12 + ["refit"] * 4,
            "house": [11] * 12 + [3] * 4,
            "sequence_id": [0] * 6 + [1] * 6 + [0] * 4,
            "fridge_power": [
                0, 60, 60, 0, 0, 0,       # two-sample ON: removed (< 8)
                60, 60, 60, 60, 60, 60,   # six-sample ON: removed (< 8)
                35, 35, 35, 35,            # another house: untouched
            ],
            "fridge_on": [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )

    audit = relabel_refit11_fridge(frame)

    np.testing.assert_array_equal(frame.loc[:11, "fridge_on"], np.zeros(12))
    np.testing.assert_array_equal(frame.loc[:11, "fridge_power"], np.zeros(12))
    np.testing.assert_array_equal(frame.loc[12:, "fridge_on"], np.ones(4))
    np.testing.assert_array_equal(frame.loc[12:, "fridge_power"], np.full(4, 35))
    assert audit["newly_off_samples"] == 8


def test_short_off_gap_is_closed_inside_one_sequence() -> None:
    power = [60.0] * 4 + [0.0] * 2 + [60.0] * 4
    frame = pd.DataFrame(
        {
            "dataset": ["refit"] * 10,
            "house": [11] * 10,
            "sequence_id": [0] * 10,
            "fridge_power": power,
            "fridge_on": [1] * 10,
        }
    )

    relabel_refit11_fridge(frame)

    np.testing.assert_array_equal(frame["fridge_on"], np.ones(10))
    # Gap-closing keeps the state ON but does not invent target watts.
    np.testing.assert_array_equal(frame["fridge_power"], np.asarray(power))

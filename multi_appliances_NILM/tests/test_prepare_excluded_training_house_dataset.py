from __future__ import annotations

import pandas as pd

from scripts.prepare_excluded_training_house_dataset import exclude_training_house


def test_exclude_training_house_only_removes_selected_source() -> None:
    frame = pd.DataFrame({
        "dataset": ["refit", "refit", "ukdale"],
        "house": [11, 3, 1],
        "aggregate": [100.0, 200.0, 300.0],
    })

    kept, audit = exclude_training_house(frame, dataset="refit", house=11)

    assert list(zip(kept["dataset"], kept["house"])) == [("refit", 3), ("ukdale", 1)]
    assert audit["removed_training_rows"] == 1
    assert audit["remaining_training_rows"] == 2

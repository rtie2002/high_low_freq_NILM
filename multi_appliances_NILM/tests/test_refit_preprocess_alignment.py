import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset_preprocess"))

from refit_processing_multi_appliance import (  # noqa: E402
    apply_fixed_alignment_shifts,
    audit_alignment,
    processed_data_quality_report,
    realign_high_confidence_events,
)


def _config() -> dict:
    return {"appliances": {"microwave": {"on_power_threshold": 200}}}


def _regular_events(event_count: int, lag: int) -> pd.DataFrame:
    event_width = 4
    spacing = 10
    length = event_count * spacing + 8
    appliance = np.zeros(length, dtype=np.float64)
    for event in range(event_count):
        start = 3 + event * spacing
        appliance[start : start + event_width] = 800.0

    aggregate = np.full(length, 100.0, dtype=np.float64)
    if lag >= 0:
        aggregate[lag:] += appliance[: length - lag]
    else:
        aggregate[:lag] += appliance[-lag:]
    index = pd.date_range("2020-01-01", periods=length, freq="8s", tz="UTC")
    return pd.DataFrame(
        {
            "aggregate": aggregate,
            "microwave_power": appliance,
            "microwave_on": (appliance >= 200.0).astype(np.int8),
            "microwave_observed": np.ones(length, dtype=np.int8),
            "sequence_id": np.zeros(length, dtype=np.int64),
        },
        index=index,
    )


def test_audit_recommends_only_stable_whole_channel_lag() -> None:
    frame = _regular_events(event_count=40, lag=1)

    summaries, events = audit_alignment(
        frame, ["microwave"], _config(), house=3, max_lag_samples=2
    )

    summary = summaries[0]
    assert summary["accepted_edges"] >= 60
    assert summary["on_lag_samples"] == 1
    assert summary["off_lag_samples"] == 1
    assert summary["eligible_for_fixed_shift"] is True
    assert summary["recommended_shift_samples"] == 1
    assert {event["best_lag_samples"] for event in events} == {1}


def test_audit_does_not_shift_from_too_few_events() -> None:
    frame = _regular_events(event_count=4, lag=2)

    summaries, _ = audit_alignment(
        frame, ["microwave"], _config(), house=3, max_lag_samples=2
    )

    assert summaries[0]["dominant_lag_samples"] == 2
    assert summaries[0]["eligible_for_fixed_shift"] is False
    assert summaries[0]["recommended_shift_samples"] == 0


def test_correction_shifts_channel_without_moving_aggregate() -> None:
    frame = _regular_events(event_count=40, lag=1)
    summaries, _ = audit_alignment(
        frame, ["microwave"], _config(), house=3, max_lag_samples=2
    )
    aggregate_before = frame["aggregate"].copy()

    corrected = apply_fixed_alignment_shifts(
        frame, summaries, ["microwave"], sample_period="8s"
    )

    np.testing.assert_allclose(
        corrected["aggregate"].to_numpy(), aggregate_before.iloc[1:].to_numpy()
    )
    np.testing.assert_allclose(
        corrected["microwave_power"].to_numpy(),
        frame["microwave_power"].shift(1).iloc[1:].to_numpy(),
    )
    assert summaries[0]["applied_shift_samples"] == 1


def test_event_realigner_moves_target_not_aggregate_and_conserves_energy() -> None:
    frame = _regular_events(event_count=4, lag=1)
    aggregate_before = frame["aggregate"].copy()
    power_before = frame["microwave_power"].copy()
    state_before = frame["microwave_on"].copy()

    corrected, summaries, events = realign_high_confidence_events(
        frame,
        ["microwave"],
        _config(),
        house=3,
        max_lag_samples=2,
        apply=True,
    )

    np.testing.assert_array_equal(corrected["aggregate"], aggregate_before)
    np.testing.assert_allclose(
        corrected["microwave_power"].sum(), power_before.sum(), rtol=0.0, atol=1e-8
    )
    assert summaries[0]["shifted_events"] == 4
    assert abs(summaries[0]["energy_error_watt_samples"]) < 1e-8
    assert {event["lag_samples"] for event in events} == {1}
    np.testing.assert_array_equal(
        corrected["microwave_on"].to_numpy(),
        state_before.shift(1, fill_value=0).to_numpy(),
    )


def test_event_realigner_rejects_a_single_unsupported_boundary() -> None:
    frame = _regular_events(event_count=1, lag=1)
    # Remove the matching aggregate OFF edge. A single ON edge is not enough
    # evidence to rewrite an appliance event.
    frame.loc[frame.index[8:], "aggregate"] = frame["aggregate"].iloc[7]
    power_before = frame["microwave_power"].copy()

    corrected, summaries, _ = realign_high_confidence_events(
        frame,
        ["microwave"],
        _config(),
        house=3,
        max_lag_samples=2,
        apply=True,
    )

    np.testing.assert_array_equal(corrected["microwave_power"], power_before)
    assert summaries[0]["shifted_events"] == 0


def test_quality_report_separates_integrity_from_unknown_physical_time() -> None:
    frame = _regular_events(event_count=4, lag=1).reset_index()
    frame["unix_time"] = frame["index"].astype("int64") // 10**9
    frame["house"] = 3
    frame.attrs["alignment_summary"] = [{
        "candidate_events": 4,
        "accepted_events": 3,
        "shifted_events": 2,
        "energy_error_watt_samples": 0.0,
    }]

    report = processed_data_quality_report(frame, ["microwave"], 8)

    assert report["structural_checks_passed"] is True
    assert report["alignment_unresolved_events"] == 1
    assert report["all_events_physically_synchronised"] is None


if __name__ == "__main__":
    tests = sorted(name for name in globals() if name.startswith("test_"))
    for test_name in tests:
        globals()[test_name]()
    print(f"PASS: {len(tests)} REFIT alignment tests")

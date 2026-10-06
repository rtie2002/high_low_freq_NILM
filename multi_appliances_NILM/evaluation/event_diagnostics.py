"""Event-level waveform diagnostics for aligned NILM predictions.

These diagnostics complement sample-wise MAE/F1.  They measure whether a
complete appliance event was found, whether its boundaries are correct, and
whether its power waveform is faithful under changing interference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SNR_BIN_EDGES_DB = (-np.inf, -20.0, -10.0, 0.0, np.inf)
SNR_BIN_LABELS = ("<-20 dB", "-20 to -10 dB", "-10 to 0 dB", ">=0 dB")


def _runs(mask: np.ndarray, segment_ids: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive contiguous true runs without crossing sequence boundaries."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    segment_ids = np.asarray(segment_ids).reshape(-1)
    if len(mask) != len(segment_ids):
        raise ValueError("mask and segment_ids must have the same length")
    if not mask.any():
        return []
    boundary = np.r_[True, segment_ids[1:] != segment_ids[:-1]]
    starts = np.flatnonzero(mask & (boundary | ~np.r_[False, mask[:-1]]))
    ends = np.flatnonzero(mask & (np.r_[mask[1:] == 0, True] | np.r_[segment_ids[1:] != segment_ids[:-1], True]))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _best_overlapping_run(
    true_start: int,
    true_end: int,
    predicted_runs: Iterable[tuple[int, int]],
) -> tuple[int, int, float] | None:
    """Match a true event to the predicted event with the highest temporal IoU."""
    best: tuple[int, int, float] | None = None
    for pred_start, pred_end in predicted_runs:
        overlap = max(0, min(true_end, pred_end) - max(true_start, pred_start) + 1)
        if overlap == 0:
            continue
        union = max(true_end, pred_end) - min(true_start, pred_start) + 1
        iou = overlap / union
        if best is None or iou > best[2]:
            best = (pred_start, pred_end, float(iou))
    return best


def _delta_snr_db(
    aggregate: np.ndarray,
    target: np.ndarray,
    start: int,
    end: int,
    segment_ids: np.ndarray,
    epsilon: float = 1e-9,
) -> float:
    """Change-domain target-to-interference ratio including both event edges."""
    segment = segment_ids[start]
    left = start - 1 if start > 0 and segment_ids[start - 1] == segment else start
    right = end + 1 if end + 1 < len(target) and segment_ids[end + 1] == segment else end
    target_delta = np.diff(target[left : right + 1].astype(np.float64))
    interference = aggregate - target
    interference_delta = np.diff(interference[left : right + 1].astype(np.float64))
    target_energy = float(np.sum(target_delta**2))
    interference_energy = float(np.sum(interference_delta**2))
    if target_energy <= epsilon:
        return float("-inf")
    return float(10.0 * np.log10((target_energy + epsilon) / (interference_energy + epsilon)))


def _event_correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 2 or np.std(target) <= 1e-9 or np.std(prediction) <= 1e-9:
        return float("nan")
    return float(np.corrcoef(target, prediction)[0, 1])


def event_waveform_table(
    bundle: Any,
    *,
    run_label: str,
    aggregate_watts: np.ndarray,
    true_appliance_watts: np.ndarray,
    true_on: np.ndarray,
    segment_ids: np.ndarray,
    sample_seconds: float,
) -> pd.DataFrame:
    """Build one row per true appliance event for one prediction bundle.

    Event NRMSE is RMSE divided by true RMS power, which stays meaningful for
    near-constant loads such as a fridge.  Energy error is evaluated over the
    true event interval; timing errors and IoU separately measure event extent.
    """
    aggregate = np.asarray(aggregate_watts, dtype=np.float64).reshape(-1)
    truth = np.asarray(true_appliance_watts, dtype=np.float64)
    labels = np.asarray(true_on, dtype=bool)
    predicted = np.asarray(bundle.y_pred_watts, dtype=np.float64)
    predicted_on = np.asarray(bundle.y_pred_on, dtype=bool)
    segments = np.asarray(segment_ids).reshape(-1)
    n_samples = len(aggregate)
    expected = (n_samples, len(bundle.appliances))
    for name, values in {
        "true_appliance_watts": truth,
        "true_on": labels,
        "y_pred_watts": predicted,
        "y_pred_on": predicted_on,
    }.items():
        if values.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {values.shape}")
    if len(segments) != n_samples:
        raise ValueError("segment_ids length does not match predictions")

    residual = aggregate - truth.sum(axis=1)
    rows: list[dict[str, object]] = []
    for app_i, appliance in enumerate(bundle.appliances):
        true_runs = _runs(labels[:, app_i], segments)
        predicted_runs = _runs(predicted_on[:, app_i], segments)
        predicted_by_segment: dict[int, list[tuple[int, int]]] = {}
        for pred_start, pred_end in predicted_runs:
            predicted_by_segment.setdefault(int(segments[pred_start]), []).append(
                (pred_start, pred_end)
            )

        for event_id, (start, end) in enumerate(true_runs, start=1):
            event_true = truth[start : end + 1, app_i]
            event_pred = predicted[start : end + 1, app_i]
            error = event_pred - event_true
            rmse = float(np.sqrt(np.mean(error**2)))
            true_rms = float(np.sqrt(np.mean(event_true**2)))
            true_energy_wh = float(event_true.sum() * sample_seconds / 3600.0)
            pred_energy_wh = float(event_pred.sum() * sample_seconds / 3600.0)
            energy_error_pct = (
                100.0 * (pred_energy_wh - true_energy_wh) / true_energy_wh
                if true_energy_wh > 1e-9
                else float("nan")
            )
            match = _best_overlapping_run(
                start,
                end,
                predicted_by_segment.get(int(segments[start]), []),
            )
            pred_start = match[0] if match is not None else None
            pred_end = match[1] if match is not None else None
            iou = match[2] if match is not None else 0.0
            delta_snr = _delta_snr_db(
                aggregate,
                truth[:, app_i],
                start,
                end,
                segments,
            )
            rows.append({
                "run": run_label,
                "experiment_id": bundle.experiment_id,
                "split": bundle.split,
                "appliance": appliance,
                "event_id": event_id,
                "segment_id": int(segments[start]),
                "start_index": start,
                "end_index": end,
                "start_csv_row": int(bundle.csv_timesteps[start]),
                "end_csv_row": int(bundle.csv_timesteps[end]),
                "duration_samples": end - start + 1,
                "duration_seconds": (end - start + 1) * sample_seconds,
                "delta_snr_db": delta_snr,
                "median_aggregate_w": float(np.median(aggregate[start : end + 1])),
                "median_residual_background_w": float(np.median(residual[start : end + 1])),
                "target_aggregate_energy_ratio": float(
                    event_true.sum() / max(aggregate[start : end + 1].sum(), 1e-9)
                ),
                "detected": int(match is not None),
                "event_iou": iou,
                "start_error_seconds": (
                    (pred_start - start) * sample_seconds if pred_start is not None else np.nan
                ),
                "end_error_seconds": (
                    (pred_end - end) * sample_seconds if pred_end is not None else np.nan
                ),
                "event_rmse_w": rmse,
                "event_nrmse": rmse / max(true_rms, 1e-9),
                "waveform_correlation": _event_correlation(event_true, event_pred),
                "true_energy_wh": true_energy_wh,
                "pred_energy_wh": pred_energy_wh,
                "energy_error_pct": energy_error_pct,
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["snr_bin"] = pd.cut(
            table["delta_snr_db"],
            bins=SNR_BIN_EDGES_DB,
            labels=SNR_BIN_LABELS,
            right=False,
        )
    return table


def snr_summary_table(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event quality by run, split, appliance and delta-SNR range."""
    if events.empty:
        return pd.DataFrame()
    grouped = events.groupby(
        ["run", "split", "appliance", "snr_bin"],
        observed=False,
        dropna=False,
    )
    return grouped.agg(
        event_count=("event_id", "size"),
        detection_rate=("detected", "mean"),
        mean_event_iou=("event_iou", "mean"),
        median_event_nrmse=("event_nrmse", "median"),
        mean_waveform_correlation=("waveform_correlation", "mean"),
        median_abs_energy_error_pct=("energy_error_pct", lambda x: np.nanmedian(np.abs(x))),
        median_abs_start_error_seconds=("start_error_seconds", lambda x: np.nanmedian(np.abs(x))),
        median_abs_end_error_seconds=("end_error_seconds", lambda x: np.nanmedian(np.abs(x))),
        median_residual_background_w=("median_residual_background_w", "median"),
    ).reset_index()


def run_event_summary(
    events: pd.DataFrame,
    bundles: dict[str, Any],
    *,
    true_on: np.ndarray,
    segment_ids: np.ndarray,
    sample_seconds: float,
) -> pd.DataFrame:
    """Run-level event metrics including false events per observed hour."""
    duration_hours = len(true_on) * sample_seconds / 3600.0
    rows: list[dict[str, object]] = []
    for label, bundle in bundles.items():
        predicted_on = np.asarray(bundle.y_pred_on, dtype=bool)
        for app_i, appliance in enumerate(bundle.appliances):
            false_count = 0
            for start, end in _runs(predicted_on[:, app_i], segment_ids):
                if not np.any(true_on[start : end + 1, app_i]):
                    false_count += 1
            subset = events[(events["run"] == label) & (events["appliance"] == appliance)]
            rows.append({
                "run": label,
                "split": bundle.split,
                "appliance": appliance,
                "event_count": len(subset),
                "detection_rate": subset["detected"].mean(),
                "mean_event_iou": subset["event_iou"].mean(),
                "median_event_nrmse": subset["event_nrmse"].median(),
                "mean_waveform_correlation": subset["waveform_correlation"].mean(),
                "median_abs_energy_error_pct": np.nanmedian(np.abs(subset["energy_error_pct"])),
                "false_event_count": false_count,
                "false_events_per_hour": false_count / max(duration_hours, 1e-9),
            })
    return pd.DataFrame(rows)


def paired_event_differences(events: pd.DataFrame, run_a: str, run_b: str) -> pd.DataFrame:
    """Return run-B minus run-A event differences on exactly matched true events."""
    keys = ["split", "appliance", "segment_id", "start_csv_row", "end_csv_row"]
    metrics = [
        "detected", "event_iou", "event_nrmse", "waveform_correlation",
        "start_error_seconds", "end_error_seconds", "energy_error_pct",
    ]
    left = events[events["run"] == run_a][keys + metrics]
    right = events[events["run"] == run_b][keys + metrics]
    paired = left.merge(right, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    for metric in metrics:
        paired[f"delta_{metric}_b_minus_a"] = paired[f"{metric}_b"] - paired[f"{metric}_a"]
    return paired


def _select_events(app_events: pd.DataFrame, max_events: int = 4) -> list[pd.Series]:
    """Select low/median/high-SNR and worst-average-NRMSE true events."""
    one_per_event = app_events.drop_duplicates(
        ["segment_id", "start_csv_row", "end_csv_row"]
    ).copy()
    finite = one_per_event[np.isfinite(one_per_event["delta_snr_db"])]
    selected: list[pd.Series] = []
    if not finite.empty:
        for quantile in (0.1, 0.5, 0.9):
            target = finite["delta_snr_db"].quantile(quantile)
            selected.append(finite.loc[(finite["delta_snr_db"] - target).abs().idxmin()])
    worst_key = (
        app_events.groupby(["segment_id", "start_csv_row", "end_csv_row"])["event_nrmse"]
        .mean()
        .idxmax()
    )
    worst = one_per_event[
        (one_per_event["segment_id"] == worst_key[0])
        & (one_per_event["start_csv_row"] == worst_key[1])
        & (one_per_event["end_csv_row"] == worst_key[2])
    ].iloc[0]
    selected.append(worst)
    unique: list[pd.Series] = []
    seen: set[tuple[int, int, int]] = set()
    for row in selected:
        key = (int(row["segment_id"]), int(row["start_csv_row"]), int(row["end_csv_row"]))
        if key not in seen:
            unique.append(row)
            seen.add(key)
    return unique[:max_events]


def save_paired_waveform_plots(
    events: pd.DataFrame,
    bundles: dict[str, Any],
    *,
    aggregate_watts: np.ndarray,
    true_appliance_watts: np.ndarray,
    true_on: np.ndarray,
    segment_ids: np.ndarray,
    sample_seconds: float,
    output_dir: Path,
    dpi: int = 220,
) -> list[Path]:
    """Save one three-row figure per appliance using the same true events."""
    if len(bundles) != 2:
        raise ValueError("paired waveform plots require exactly two runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_labels = list(bundles)
    appliances = next(iter(bundles.values())).appliances
    saved: list[Path] = []
    for app_i, appliance in enumerate(appliances):
        app_events = events[events["appliance"] == appliance]
        selections = _select_events(app_events)
        if not selections:
            continue
        fig, axes = plt.subplots(
            3,
            len(selections),
            figsize=(5.0 * len(selections), 9.0),
            squeeze=False,
        )
        for col, event in enumerate(selections):
            start, end = int(event["start_index"]), int(event["end_index"])
            segment = segment_ids[start]
            duration = end - start + 1
            margin = min(200, max(20, int(0.1 * duration)))
            crop_start = start
            while crop_start > max(0, start - margin) and segment_ids[crop_start - 1] == segment:
                crop_start -= 1
            crop_end = end
            while crop_end + 1 < min(len(segment_ids), end + margin + 1) and segment_ids[crop_end + 1] == segment:
                crop_end += 1
            sl = slice(crop_start, crop_end + 1)
            time_min = np.arange(crop_end - crop_start + 1) * sample_seconds / 60.0
            truth = true_appliance_watts[sl, app_i]
            predictions = [bundles[label].y_pred_watts[sl, app_i] for label in run_labels]
            ymax = max(
                1.0,
                float(np.nanmax(np.concatenate([truth, *predictions]))),
            ) * 1.08

            ax = axes[0, col]
            ax.plot(time_min, truth, color="#1565c0", linewidth=1.8, label="Target true")
            ax.set_ylim(0, ymax)
            ax.set_title(
                f"Event {int(event['event_id'])} | ΔSNR={event['delta_snr_db']:.1f} dB\n"
                f"median residual={event['median_residual_background_w']:.0f} W"
            )
            context = ax.twinx()
            context.plot(time_min, aggregate_watts[sl], color="0.65", linewidth=0.8, alpha=0.55)
            context.set_ylabel("Aggregate (W)", color="0.45", fontsize=8)
            context.tick_params(axis="y", labelsize=7, colors="0.45")

            for row_i, label in enumerate(run_labels, start=1):
                ax = axes[row_i, col]
                ax.plot(time_min, truth, color="#1565c0", linewidth=1.5, label="Target true")
                ax.plot(
                    time_min,
                    bundles[label].y_pred_watts[sl, app_i],
                    color="#d32f2f",
                    linewidth=1.4,
                    label="Predicted",
                )
                on_mask = np.asarray(bundles[label].y_pred_on[sl, app_i], dtype=bool)
                ax.fill_between(time_min, 0, ymax, where=on_mask, color="#ff9800", alpha=0.08)
                ax.set_ylim(0, ymax)
                row = app_events[
                    (app_events["run"] == label)
                    & (app_events["segment_id"] == event["segment_id"])
                    & (app_events["start_csv_row"] == event["start_csv_row"])
                ].iloc[0]
                ax.text(
                    0.01,
                    0.97,
                    f"IoU={row['event_iou']:.2f}  NRMSE={row['event_nrmse']:.2f}  "
                    f"corr={row['waveform_correlation']:.2f}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
                if col == 0:
                    ax.set_ylabel(f"{label}\nPower (W)")
                ax.set_xlabel("Relative time (min)")
                ax.grid(alpha=0.2)
            axes[0, col].grid(alpha=0.2)
            axes[0, col].set_xlabel("Relative time (min)")
            if col == 0:
                axes[0, col].set_ylabel("Ground truth\nPower (W)")
                axes[1, col].legend(loc="upper right", fontsize=8)

        fig.suptitle(
            f"{next(iter(bundles.values())).split} — {appliance}: paired best-checkpoint waveforms",
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = output_dir / f"{appliance}_paired_waveforms.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved

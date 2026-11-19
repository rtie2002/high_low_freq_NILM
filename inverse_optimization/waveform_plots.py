"""MART-NILM-style ON-event figures shared by both experiment stages."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


WAVEFORM_DPI = 300
FULL_CYCLE_APPLIANCES = {"dishwasher", "washingmachine"}


def _on_events(state, min_duration):
    """Return inclusive start/end positions for complete ON intervals."""
    state = np.asarray(state, dtype=bool)
    starts = np.flatnonzero(state & np.r_[True, ~state[:-1]])
    ends = np.flatnonzero(state & np.r_[~state[1:], True])
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start + 1 >= min_duration
    ]


def _event_crop(event_start, event_end, total_steps, margin_min, margin_fraction, max_samples):
    """Keep the whole ON event plus OFF context, matching MART-NILM cropping."""
    event_length = event_end - event_start + 1
    margin = max(margin_min, int(margin_fraction * event_length))
    start = max(0, event_start - margin)
    end = min(total_steps, event_end + margin + 1)
    if max_samples is not None and end - start > max_samples:
        end = min(total_steps, event_end + margin + 1)
        start = max(0, end - max_samples)
    return start, end


def _context_crop(event_start, event_end, focus_start, focus_end, total_steps, scale):
    """Return a context crop that is ``scale`` times wider than the focus crop."""
    focus_length = max(1, focus_end - focus_start)
    target_length = max(focus_length + 1, int(round(focus_length * scale)))
    center = (event_start + event_end) // 2
    start = max(0, center - target_length // 2)
    end = min(total_steps, start + target_length)
    return max(0, end - target_length), end


def _shade_runs(axis, x, state, color, alpha):
    """Shade every contiguous True interval without bridging an OFF interval."""
    for start, end in _on_events(state, min_duration=1):
        axis.axvspan(x[start], x[end], color=color, alpha=alpha, linewidth=0)


def _select_event(appliance, state, power, min_duration, rng):
    """Use MART-NILM event selection: long cycles favour their longest event."""
    events = _on_events(state, min_duration)
    if not events:
        return None
    if appliance in FULL_CYCLE_APPLIANCES:
        return max(events, key=lambda event: event[1] - event[0])
    return events[rng.integers(len(events))]


def _plot_event(
    *,
    output_path,
    title,
    appliance,
    true_values,
    predicted_values,
    true_on,
    predicted_on=None,
    aggregate=None,
    start,
    end,
    long_figure,
    true_label,
    predicted_label,
):
    """Draw one focused or context waveform with MART-NILM colours and sizing."""
    time = np.arange(start, end)
    true_view = np.asarray(true_values, dtype=float)[start:end]
    predicted_view = np.maximum(np.asarray(predicted_values, dtype=float)[start:end], 0.0)
    true_on_view = np.asarray(true_on, dtype=bool)[start:end]
    predicted_on_view = (
        None if predicted_on is None else np.asarray(predicted_on, dtype=bool)[start:end]
    )
    aggregate_view = None if aggregate is None else np.asarray(aggregate, dtype=float)[start:end]

    points = max(1, end - start)
    if long_figure:
        width = min(28.0, max(8.8, 5.5 * (points / 400) ** 0.55))
        figure, axis = plt.subplots(figsize=(width, 3.1))
    else:
        side = min(13.75, max(5.5, 5.5 * (points / 500) ** 0.45))
        figure, axis = plt.subplots(figsize=(side, side))
        axis.set_box_aspect(1)

    if aggregate_view is not None:
        axis.plot(time, aggregate_view, color="#9a9a9a", alpha=0.55, linewidth=1.0, label="aggregate (W)")

    if predicted_on_view is None:
        _shade_runs(axis, time, true_on_view, "#6baed6", 0.20)
    else:
        _shade_runs(axis, time, true_on_view & ~predicted_on_view, "#6baed6", 0.20)
        _shade_runs(axis, time, predicted_on_view & ~true_on_view, "#7ad66d", 0.20)
        _shade_runs(axis, time, true_on_view & predicted_on_view, "#9b59b6", 0.28)

    axis.plot(time, true_view, color="#1f77b4", linewidth=1.8, label=true_label)
    axis.plot(time, predicted_view, color="#d62728", linewidth=1.5, alpha=0.92, label=predicted_label)
    axis.set_title(title)
    axis.set_xlabel("Sample index (8 seconds per sample)")
    axis.set_ylabel("Power (W)")
    axis.grid(True, alpha=0.25)

    handles, labels = axis.get_legend_handles_labels()
    handles.append(Patch(facecolor="#6baed6", alpha=0.20, label="true ON (CSV)"))
    labels.append("true ON (CSV)")
    if predicted_on_view is not None:
        handles.extend([
            Patch(facecolor="#7ad66d", alpha=0.20, label="predicted ON"),
            Patch(facecolor="#9b59b6", alpha=0.28, label="ON overlap"),
        ])
        labels.extend(["predicted ON", "ON overlap"])
    axis.legend(handles=handles, labels=labels, loc="upper right", fontsize=8 if long_figure else 9)

    maximum = max(1.0, float(np.nanmax(true_view)), float(np.nanmax(predicted_view)))
    if aggregate_view is not None and np.isfinite(aggregate_view).any():
        aggregate_maximum = float(np.nanmax(aggregate_view))
        maximum = max(maximum, aggregate_maximum) if aggregate_maximum <= maximum * 1.5 else maximum * 1.25
    minimum = min(0.0, float(np.nanmin(true_view)), float(np.nanmin(predicted_view)))
    padding = 0.1 * max(maximum - minimum, 1.0)
    axis.set_ylim(minimum - padding, maximum + padding)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=WAVEFORM_DPI, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_appliance_event_waveforms(
    output_dir,
    *,
    appliances,
    true_power,
    predicted_power,
    true_state,
    thresholds,
    aggregate,
    seed,
    file_prefix,
    title_prefix,
    min_duration=2,
    margin_min=40,
    margin_fraction=0.08,
    context_scale=10.0,
):
    """Save one focused ON event and one 10x context plot per appliance."""
    output_dir = Path(output_dir)
    rng = np.random.default_rng(seed)
    saved = []
    total_steps = len(true_power)

    for appliance_index, appliance in enumerate(appliances):
        event = _select_event(
            appliance,
            true_state[:, appliance_index],
            true_power[:, appliance_index],
            max(min_duration, 60) if appliance in FULL_CYCLE_APPLIANCES else min_duration,
            rng,
        )
        if event is None:
            continue
        event_start, event_end = event
        maximum = None if appliance in FULL_CYCLE_APPLIANCES else 1200
        focus_start, focus_end = _event_crop(
            event_start, event_end, total_steps, margin_min, margin_fraction, maximum
        )
        context_start, context_end = _context_crop(
            event_start, event_end, focus_start, focus_end, total_steps, context_scale
        )
        predicted_on = predicted_power[:, appliance_index] >= thresholds[appliance_index]
        app_dir = output_dir / appliance
        center = (event_start + event_end) // 2
        common = {
            "appliance": appliance,
            "true_values": true_power[:, appliance_index],
            "predicted_values": predicted_power[:, appliance_index],
            "true_on": true_state[:, appliance_index],
            "predicted_on": predicted_on,
            "aggregate": aggregate,
            "true_label": f"{appliance} true",
            "predicted_label": f"{appliance} pred",
        }
        saved.append(_plot_event(
            output_path=app_dir / f"{file_prefix}_01_t{center}.png",
            title=f"{title_prefix}{appliance} ON period 1".strip(),
            start=focus_start, end=focus_end, long_figure=False, **common,
        ))
        saved.append(_plot_event(
            output_path=app_dir / f"{file_prefix}_01_t{center}_context10x.png",
            title=f"{title_prefix}{appliance} ON period 1 (10x context)".strip(),
            start=context_start, end=context_end, long_figure=True, **common,
        ))
    return saved


def save_forward_event_waveforms(output_dir, examples, predictions, seed):
    """Use the same event selection for Stage 1 aggregate reconstructions."""
    output_dir = Path(output_dir)
    rng = np.random.default_rng(seed)
    saved = []
    for appliance_index, (appliance, (power, aggregate, state)) in enumerate(examples.items()):
        min_duration = 60 if appliance in FULL_CYCLE_APPLIANCES else 2
        event = _select_event(
            appliance, state[:, appliance_index], power[:, appliance_index], min_duration, rng
        )
        if event is None:
            continue
        event_start, event_end = event
        focus_start, focus_end = _event_crop(event_start, event_end, len(aggregate), 40, 0.08, 1200)
        context_start, context_end = _context_crop(
            event_start, event_end, focus_start, focus_end, len(aggregate), 10.0
        )
        app_dir = output_dir / appliance
        center = (event_start + event_end) // 2
        common = {
            "appliance": appliance,
            "true_values": aggregate,
            "predicted_values": predictions[appliance],
            "true_on": state[:, appliance_index],
            "aggregate": None,
            "true_label": "aggregate true",
            "predicted_label": "aggregate pred",
        }
        saved.append(_plot_event(
            output_path=app_dir / f"forward_01_t{center}.png",
            title=f"Forward model | {appliance} ON period 1",
            start=focus_start, end=focus_end, long_figure=False, **common,
        ))
        saved.append(_plot_event(
            output_path=app_dir / f"forward_01_t{center}_context10x.png",
            title=f"Forward model | {appliance} ON period 1 (10x context)",
            start=context_start, end=context_end, long_figure=True, **common,
        ))
    return saved

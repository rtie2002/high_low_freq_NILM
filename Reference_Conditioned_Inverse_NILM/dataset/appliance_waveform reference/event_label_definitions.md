# Reference event labels and timing

## Measurements

Both datasets are placed on the same UTC eight-second grid before event detection.
Each bin [t,t+8) takes the arithmetic mean of measurements in that bin. Peaks can
decrease after averaging; the result is a resampled power trace in watts, not a
claim to preserve every raw measurement or exact energy on irregular inputs.
No aggregate/appliance lag correction is applied.

A completely empty bin may hold the previous value for at most one bin, only
when bracketing raw timestamps are no more than 16 seconds apart. Long outages
and unbounded edges remain missing. There is no linear interpolation or zero
padding. This replaces the earlier UK-DALE two-bin interpolation policy.

Channel/aggregate pairs drop invalid bins before detection. Any resulting time
step larger than eight seconds splits recording sequences: a saved reference
never crosses a missing bin. Every saved waveform has exactly eight seconds
between samples. Manifest `sample_period_sec` records this once per event.

## Event rules

| Appliance | Threshold (W) | Minimum event duration (s) | Maximum internal OFF gap (s) | Maximum pre/post context (samples) |
|---|---:|---:|---:|---:|
| Kettle | 200 | 12 | 0 | 4 / 4 |
| Microwave | 200 | 6 | 30 | 4 / 4 |
| Fridge | 50 | 60 | 12 | 8 / 8 |
| Dishwasher | 50 | 960 | 1800 | 16 / 16 |
| Washing machine | 20 | 960 | 960 | 16 / 16 |

Physical duration values use the existing project preprocessing settings rather
than converting irregular REFIT row counts into seconds. Thresholds are unchanged.

`above_threshold` is 1 when measured appliance power is at least the threshold.
`event_on` is 1 throughout a retained operating event, including internal OFF
gaps joined by the rule above. The redundant `is_on` alias is no longer written.

## Compact output schema

Each `events.csv` contains only `event_id`, `dataset`, `house`, `channel`,
`event_on`, `above_threshold`, `source_idx`, `timestamp`, `unix_time`,
`appliance_power_w`, and `aggregate_power_w`.

Event boundaries, OFF context counts, duration, peak/mean power, boundary
completeness, `on_end_unix`, and file row ranges are stored once per event in
`manifest.csv`. Filtering diagnostics (`refit_quality_checked`,
`longest_severe_mismatch_sec`, `severe_active_fraction`, `refit_issue_fraction`)
are not persisted. Filtering decisions and progress remain in the extractor.

## Time and index definitions

All `*_idx` columns are zero-based inclusive positions in the source table
(the valid aligned/resampled channel table for both datasets). They are not
original raw-file row numbers or positions in events.csv.

Duration is the number of event bins times eight seconds. It includes joined
internal OFF phases and excludes context. It is an estimate, not a measurement
of the exact physical switching time.

Sample right boundaries are calculated as timestamp + eight seconds, since each
sample now represents one explicit bin. The event right boundary is saved as `on_end_unix` in
the manifest. `duration_censored` marks events touching a recording boundary;
their duration may be incomplete. Minimum duration tests use these same edges.

An internal OFF gap is measured from its first below-threshold timestamp to the
next above-threshold timestamp. Events and context require consecutive
eight-second timestamps. For dishwasher and washing machine references,
exclude events with a continuous near-idle interval (at most 5 W for at least
600 seconds) between two active spans that each meet the appliance's minimum
event duration. This flags an ambiguous program boundary without assigning a
new start or end time. The thresholds are conservative screening parameters.

OFF context stops at any above-threshold or invalid appliance sample, even when
that sample was removed from the retained event mask as a short activation.
All references need at least one actually measured OFF bin on each side.
Fridge and dishwasher references also require at least eight and sixteen
contiguous OFF-context bins per side, respectively. A forward-held bin can be
part of this context but cannot provide the only evidence of an OFF boundary.
Missing context is not padded. Older exports can still contain censored events.

`peak_power_w` and `mean_on_power_w` retain their existing meaning: maximum and
arithmetic sample mean over the complete event, including joined internal OFF
phases. The mean is not a time-weighted mean on irregular data.

The viewer displays sample markers with sample-and-hold steps, actual timestamp
boundaries, and separate event/threshold state traces. It does not time-shift data.

## Verification

Run `python test_event_selection.py` beside the extractor. The checks cover
irregular event duration, short activations, gap closing in seconds, recording
gaps, OFF context blocked by rejected high-power pulses, regular-grid timing,
and unchanged saved power values and timestamps.

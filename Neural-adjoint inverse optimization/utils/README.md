# Reusable NILM utilities

This directory contains the actual code used by the new inverse-NILM project.
It does not import Python code from the older direct-model project.

| File | Responsibility | Important equations / behaviour |
| --- | --- | --- |
| `data_loader.py` | CSV reading, fixed house splits, normalisation, contiguous sliding windows | Prevents a window from crossing a dataset, house, sequence, or timestamp gap. |
| `prediction.py` | Common `PredictionBundle` format | Stores reconstructed true/predicted power in watts as `(time, appliance)`. |
| `metrics.py` | Power, state, energy, event, and background diagnostics | MAE, SAE, F1, average precision, false-event count, and background false-positive rate. |
| `waveform_capture.py` | Aggregate-aligned ON-period and background-range figures | Uses the same `csv_timesteps` and `segment_ids` as metrics. |
| `state_postprocess.py` | Validation-only state calibration and duration/gap cleanup | Never fits thresholds on held-out test labels. |
| `output_format.py` | Stable CSV and JSON output | Text results are rounded; raw NPZ predictions are not. |
| `protocol.py` | One evaluation call for every inverse model | Constructs a `PredictionBundle`, then saves metrics and waveforms. |

## Main metric equations

For appliance \(i\), at \(N\) reconstructed timeline points, power mean
absolute error is implemented by `metrics.mae`:

\[
\operatorname{MAE}_i = \frac{1}{N}\sum_{t=1}^{N}
\left|\hat p_i(t)-p_i(t)\right|.
\]

The signal aggregate error (`metrics.sae`) splits the timeline into periods of
length \(L\) samples, compares total energy within each period, then divides by
\(L\):

\[
\operatorname{SAE}_i = \frac{1}{K L}\sum_{k=1}^{K}
\left|
\sum_{t \in k}\hat p_i(t)-\sum_{t \in k}p_i(t)
\right|.
\]

For binary ON/OFF labels, `metrics.per_appliance_f1` uses:

\[
F1_i=\frac{2TP_i}{2TP_i+FP_i+FN_i}.
\]

`metrics.evaluate_bundle` combines these with event-onset F1, ON/OFF MAE,
false-positive energy, and background-binned false-positive rate.  Read its
function docstring and implementation before changing any evaluation rule.

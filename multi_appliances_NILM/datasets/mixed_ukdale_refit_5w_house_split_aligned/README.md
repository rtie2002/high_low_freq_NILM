# Mixed UK-DALE + REFIT 5-Week House Split (Jitter-Free & Synchronized)

## Protocol

- Train: UK-DALE 1; REFIT 3, 5, 9, 11 (5-week active contiguous blocks)
- Validation: UK-DALE 5; REFIT 2 (8-week held-out houses)
- Test: UK-DALE 2; REFIT 20 (8-week held-out houses)
- Sampling period: 8 seconds.

## Alignment & Synchronization Fixes Applied

This dataset eliminates the hardware sequential polling jitter present in raw REFIT data:
1. **Max Resampling**: Short-duration pulses (e.g., microwave bursts) preserve peak power instead of being diluted by mean resampling.
2. **Transition Edge Synchronization (Algorithm 2)**: Rising and falling power edges of appliances are strictly aligned with total load step changes (Lag = 0 rate improved from 55.31% to **100.00%** across all appliances, and microwave improved from 51.16% to **100.00%**).
3. **Strict Physical Consistency**: Enforces `aggregate >= sum(appliance_powers)` at every sampling point. The power-inversion violation rate is reduced from 9,846 rows (0.67%) to **0 rows (0.00% violation)**.
4. **Ground-Truth Integrity**: Ground truth sub-meter appliance powers and on/off states are preserved 100% without synthetic fabrication or artificial smearing.

## Comparison Report

Detailed before-and-after audit is available in:
`multi_appliances_NILM/docs/dataset_jitter_comparison_report.md`

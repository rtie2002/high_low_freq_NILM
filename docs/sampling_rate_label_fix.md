# Sampling-rate label fix notes

## 1. What problem did we solve?

The old preprocessing code treated label-cleaning parameters as **sample counts**.
For example:

```yaml
min_on_duration: 300
resample_gap_fill: 3
```

This is dangerous because the physical meaning changes when the sampling rate
changes.

At 6 seconds:

```text
300 samples = 300 * 6  = 1800 seconds
3 samples   = 3 * 6    = 18 seconds
```

At 8 seconds:

```text
300 samples = 300 * 8  = 2400 seconds
3 samples   = 3 * 8    = 24 seconds
```

So the same YAML value silently becomes a different physical rule. This is why
the 8s version could mark appliance ON/OFF periods differently from the 6s
version, especially for washing machine and dishwasher where `min_on_duration`
and `min_off_duration` are large.

The scientific issue is: NILM state labels should represent appliance behavior
in real time, not arbitrary sample counts. If the sampling rate changes, the
duration rule should stay physically stable.

## 2. Main idea

Use seconds as the source of truth:

```yaml
min_on_seconds: 960
min_off_seconds: 960
resample_gap_fill_seconds: 18
```

Then convert seconds to samples inside preprocessing:

```text
samples = round(duration_seconds / sample_seconds)
```

With this rule:

```text
960 seconds at 6s -> 160 samples
960 seconds at 8s -> 120 samples
18 seconds at 6s  -> 3 samples
18 seconds at 8s  -> 2 samples
```

The model still receives sampled data, but the label-cleaning rule is now tied
to real physical time.

## 3. Before

Old behavior:

```python
min_on_duration = appliance_cfg.get("min_on_duration", 0)
min_off_duration = appliance_cfg.get("min_off_duration", 0)
gap_limit = appliance_cfg.get("resample_gap_fill", 0)
```

Problem:

```text
The config value means "number of rows".
If sample_seconds changes, the real duration changes.
```

Example:

```text
min_on_duration = 300

6s data: 300 rows = 30 minutes
8s data: 300 rows = 40 minutes
```

That is not a fair cross-domain or cross-sampling comparison.

## 4. After

New behavior:

```python
min_on_duration = resolve_time_samples(
    appliance_cfg, "min_on_duration", house, sample_seconds
)
```

The helper checks seconds-based keys first:

```python
min_on_seconds
min_off_seconds
resample_gap_fill_seconds
```

Then it converts to the correct sample count for the current sampling rate.

Legacy sample-count keys are still kept as fallback, so old config files do not
break immediately. But for new experiments, use seconds-based keys.

## 5. Why not force 8s labels to exactly match 6s labels?

Exact equality is not always possible.

When resampling from 6s to 8s, the bins no longer align perfectly:

```text
6s grid: 0, 6, 12, 18, 24, 30, ...
8s grid: 0, 8, 16, 24, 32, ...
```

Some short appliance events fall across different bin boundaries. After
averaging/interpolation, the apparent power shape can change. This is especially
visible for sparse appliances such as microwave and kettle.

So the goal is not "make every sample identical". The goal is:

```text
Keep the physical labeling rule stable across sampling rates.
Avoid changing ON/OFF meaning just because sample_seconds changed.
```

## 6. Boundary-censored short ON runs

Another bug risk is removing a short ON run at the start or end of a continuous
segment.

Example:

```text
segment starts here
ON ON ON OFF OFF OFF ...
```

This ON run looks short, but we do not know whether the appliance was already ON
before the segment started. The true ON duration may be longer than the visible
part.

So the updated logic only removes short ON runs when the run is fully inside the
segment:

```text
OFF ON ON OFF
```

A boundary-touching short run is kept:

```text
ON ON OFF ...
... OFF ON ON
```

This avoids deleting valid appliance states simply because the meter data was
cut at a gap or house boundary.

## 7. Pseudocode

```text
for each appliance:
    read appliance config
    read current sample_seconds

    min_on_samples  = resolve_time_samples(config, "min_on_duration")
    min_off_samples = resolve_time_samples(config, "min_off_duration")
    gap_samples     = resolve_time_samples(config, "resample_gap_fill")

    raw_state = power > on_power_threshold

    state = fill_short_off_gaps(raw_state, max_gap=min_off_samples)

    for each ON run in state:
        run_length = end_index - start_index
        touches_left_boundary = start_index == 0
        touches_right_boundary = end_index == sequence_length

        if run_length < min_on_samples:
            if not touches_left_boundary and not touches_right_boundary:
                set this ON run to OFF
            else:
                keep it because true duration is censored

    return state
```

## 8. What changed in the code?

The shared conversion helper lives in:

```text
dataset_preprocess/ukdale_processing.py
```

The multi-appliance REFIT and UK-DALE preprocessing now pass `sample_seconds`
into label creation:

```text
dataset_preprocess/refit_processing_multi_appliance.py
dataset_preprocess/ukdale_processing_multi_appliance.py
```

The config files now contain physical duration keys:

```text
config/preprocess/refit.yaml
config/preprocess/ukdale.yaml
```

Tests were added/updated in:

```text
multi_appliances_NILM/tests/test_sequence_integrity.py
```

They check:

```text
1800 seconds -> 300 samples at 6s
1800 seconds -> 225 samples at 8s
18 seconds   -> 3 samples at 6s
18 seconds   -> 2 samples at 8s
```

They also check that boundary-censored short ON runs are kept, while complete
interior short ON runs are removed.

## 9. Important warning

This fix improves label consistency, but it does not solve every 6s-vs-8s
difference.

For microwave and kettle, which have short and sharp events, a better next step
is:

```text
1. Detect appliance event intervals at the native resolution.
2. Store ON intervals in real time.
3. Project those intervals onto 6s or 8s grids by overlap ratio.
4. Keep mean power as the regression target.
```

That is a larger protocol change, so it should be done as a separate ablation.
Do not mix it with model architecture changes, otherwise it becomes hard to know
which change improved the result.


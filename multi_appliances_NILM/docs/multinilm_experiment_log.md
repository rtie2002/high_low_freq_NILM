# MultiNILM experiment log

This file records one controlled change per experiment. Validation results are
used for model selection; UK-DALE house 2 and REFIT house 20 remain report-only
test houses.

## Recovery baseline — 2026-10-08

Experiment: `multinilm_early_relation_full_mix_house_split`

- Dataset: `mixed_ukdale_refit_5w_house_split`
- Architecture: restored early relational MultiNILM
- Augmentation: original 50:50 real/full-random mix
- Selected checkpoint: epoch 138
- Validation: MAE 14.761 W, macro-F1 0.766, AP 0.787
- REFIT house 20: MAE 10.615 W, macro-F1 0.729, AP 0.708
- UK-DALE house 2: MAE 7.321 W, macro-F1 0.867, AP 0.928

The recovery baseline outperformed the paired-background and private-expert
variants on overall REFIT macro-F1. The remaining failures are specific:

- REFIT fridge: FPR 0.463; high residual loads are frequently assigned to the
  fridge.
- REFIT microwave: F1 0.432; real events around 302 W and 557 W residual
  background are missed, while other background ranges are detected.
- Minimum validation loss occurs at epoch 57, maximum AP at epoch 127, and
  minimum validation MAE at epoch 138. Therefore the current loss is not a
  reliable standalone checkpoint criterion.

## Focal-event stratified full mix — completed 2026-10-08

Experiment: `multinilm_focal_event_stratified_mix_house_split`

Single change: improve sampling inside the existing synthetic half.

1. Select one of the five appliances uniformly as the focal appliance.
2. Draw that appliance from a real training window containing an ON sample.
3. Draw residual background uniformly across the non-empty ranges
   `[0,100)`, `[100,200)`, `[200,400)`, `[400,800)`, and `[800,infinity)` W.
4. Draw the other four appliance traces with the original uniform rule.
5. Keep the other 50% of training windows fully real.

Everything else remains identical to the recovery baseline. This experiment
tests whether the failure comes from insufficient event/background coverage,
without adding model parameters or inference-time components.

Keep the change only if validation house-macro AP improves and the fridge
high-background FPR decreases without collapsing microwave AP/F1. Test-house
metrics are reported after this validation decision, not used to tune it.

Result at the selected checkpoint (epoch 84):

- Validation: MAE 14.424 W, macro-F1 0.765, AP 0.795.
- REFIT house 20: MAE 10.254 W, macro-F1 0.714, AP 0.735.
- UK-DALE house 2: MAE 7.535 W, macro-F1 0.871, AP 0.931.
- REFIT fridge FPR improved from 0.463 to 0.396.
- REFIT microwave AP improved from 0.336 to 0.456, but F1 changed from
  0.432 to 0.420.

Conclusion: partially effective. It improved threshold-independent ranking,
fridge background robustness, and overall REFIT MAE/AP. However, forcing a
focal ON event in every synthetic window shifted the effective ON prior on top
of the existing positive BCE weight. Early checkpoints produced many microwave
false positives, and the final model still missed the representative 200–800 W
background microwave events after calibrated hard gating.

## Partial focal-event mix (probability 0.25) — completed 2026-10-08

Experiment: `multinilm_focal_event_p025_stratified_mix_house_split`

Single change relative to the completed focal-event run: apply focal-event and
background-bin sampling to 25% of synthetic windows instead of 100%. The other
75% use the original full-mix sampling. Since synthetic windows remain 50% of
training, only 12.5% of all training windows receive forced focal sampling.

This retains explicit event/background coverage while reducing the double
positive-prior shift caused by focal oversampling plus BCE `pos_weight`.

Result at the selected checkpoint (epoch 141):

- Validation: MAE 14.716 W, macro-F1 0.758, AP 0.784.
- REFIT house 20: MAE 10.596 W, macro-F1 0.732, AP 0.725.
- UK-DALE house 2: MAE 7.297 W, macro-F1 0.858, AP 0.925.
- Validation fridge FPR was 0.205, compared with 0.171 for `p=1.0` and
  0.209 for the recovery baseline.
- REFIT microwave F1 improved to 0.502 and AP to 0.428, but validation
  microwave AP decreased to 0.340.

Conclusion: reject `p=0.25` as the main configuration. It improves the
thresholded REFIT microwave result, but this advantage is not supported by the
held-out validation house. It also gives back the validation AP and fridge-FPR
gains obtained with `p=1.0`, and reduces UK-DALE macro-F1. The main
configuration therefore returns to `p=1.0`, selected using validation data.

## Current decision

Keep the focal-event stratified full mix with `prob: 1.0`. This is a useful
data-sampling improvement rather than a complete solution: it adds no model
parameters, improves validation AP from 0.787 to 0.795, lowers validation
fridge FPR from 0.209 to 0.171, and improves REFIT AP from 0.708 to 0.735.
The remaining microwave hard-gating failures must be treated separately; the
test-house F1 gain from `p=0.25` is insufficient evidence for selecting it.

## Remove the second evaluation-time power gate — rejected 2026-10-08

Experiment output: `multinilm_focal_event_no_eval_power_gate`

No retraining and no architecture change. Evaluate the selected `p=1.0`
checkpoint again with `state_calibration.apply_to_power: false`. The network
power output is already multiplied by the soft state probability during its
forward pass. The normal evaluation path then multiplies that result by a
second, calibrated binary state mask. The validation-selected microwave
threshold is 0.96, so the second multiplication may create missed or truncated
power events even when the regression output contains useful evidence.

This diagnostic isolates postprocessing from representation learning. Keep the
change only if validation power metrics and waveform continuity improve without
unacceptable OFF-state leakage. State AP, F1, precision, and recall must remain
identical because their probability and binary-state paths are unchanged.

Result using the same selected checkpoint:

- Validation MAE worsened from 14.424 W to 23.709 W.
- REFIT house 20 MAE worsened from 10.254 W to 17.195 W.
- UK-DALE house 2 MAE worsened from 7.535 W to 9.405 W.
- Classification metrics were effectively unchanged, as expected.

Conclusion: reject the change and restore `apply_to_power: true`. The hard mask
is not the cause of the state-classification failure; it is currently needed to
suppress substantial OFF-state leakage from the softly gated regression output.

## Validation-selected microwave duration decoding — 2026-10-09

The selected model produces too many short microwave fragments. On validation,
the true microwave event has a median duration of 80 s, whereas predicted events
have a median duration of 40 s. A validation-only grid search selected a 32 s
minimum ON duration and a 24 s maximum gap to merge. No model was retrained.

- Validation microwave F1: 0.427 to 0.439.
- REFIT house 20 microwave F1: 0.420 to 0.447.
- REFIT microwave precision: 0.351 to 0.396.

The same search did not find a fridge hysteresis/duration rule that transferred
reliably. Therefore only the microwave duration settings are retained. This is a
small sequence-cleanup improvement, not a solution to weak state ranking.

## Source-aware REFIT alignment jitter — rejected 2026-10-09

Experiment: `multinilm_refit_alignment_jitter_house_split`

The previous generic microwave augmentation delayed 50% of all synthetic
microwave traces by one or two samples. This run replaced it with offsets drawn
only for REFIT traces from the REFIT training-house distribution
`P(-1,0,+1,+2) = (0.149,0.571,0.211,0.069)`. Architecture, loss, dataset split,
and random-mix sampling were unchanged.

Result at the validation-selected checkpoint (epoch 125):

- Validation: MAE 14.777 W, macro-F1 0.770, AP 0.798.
- REFIT house 20: MAE 11.067 W, macro-F1 0.724, AP 0.742.
- UK-DALE house 2: MAE 8.121 W, macro-F1 0.871, AP 0.932.
- REFIT fridge: F1 0.709, FPR 0.429.
- REFIT microwave: F1 0.403, AP 0.449, and ±16 s event-onset F1 0.276.

Conclusion: reject. The tolerant event score shows that the microwave failure
is not merely a one- or two-sample scoring offset. The source-aware empirical
distribution also weakened the stronger positive-delay regularisation that had
produced REFIT microwave F1 0.521 in `multinilm_meter_lag_house_split`.

## EMA-residual input ablation — rejected 2026-10-09

Experiment: `multinilm_ema_residual_house_split`

Return to the strongest completed meter-lag baseline and add one fixed causal
feature only:

`residual[t] = x[t] - EMA_45(x)[t]`.

The raw aggregate, signed delta, rolling statistics, GL features, architecture,
loss, sampling, and checkpoint rule remain unchanged. Unlike the rejected
local-contrast channel, this feature does not divide by a small local scale and
does not clip values. It tests whether slow-background removal exposes weak
fridge and microwave evidence without amplifying noise into artificial pulses.

Result at the validation-selected checkpoint (epoch 146):

- Validation: MAE 14.916 W, macro-F1 0.772, AP 0.786.
- REFIT house 20: MAE 10.557 W, macro-F1 0.735, AP 0.741.
- UK-DALE house 2: MAE 7.983 W, macro-F1 0.859, AP 0.917.
- REFIT fridge F1 changed from 0.723 to 0.720.
- REFIT microwave F1 changed from 0.521 to 0.527, but UK-DALE microwave F1
  fell from 0.714 to 0.642.
- The inspected REFIT microwave waveform gained a false approximately 1 kW
  pulse immediately before the true event; the noisy-background fridge false
  activations remained.

Conclusion: reject and remove the feature implementation. The small REFIT
microwave gain does not compensate for the cross-domain regression or the
non-physical extra pulse.

## Domain-agnostic normalization ablation — started 2026-10-09

Experiment: `multinilm_groupnorm_house_split`

Return to the 13-channel meter-lag baseline. Replace BatchNorm only in the
shared temporal encoder and appliance heads with GroupNorm. The IBN stem,
convolutions, attention, loss, sampling, and checkpoint rule remain unchanged.
This tests whether shared running statistics from mixed UK-DALE, REFIT, and
synthetic samples cause the observed domain-dependent feature behaviour.

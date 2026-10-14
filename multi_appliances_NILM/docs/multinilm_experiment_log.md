# MART-NILM experiment log

This file records one controlled change per experiment. Validation results are
used for model selection; UK-DALE house 2 and REFIT house 20 remain report-only
test houses.

## Paper Tables V--VII all-5-week protocol — launched 2026-10-14

The final paper protocol uses one continuous five-week block for every house.
The training rows and training-only normalisation are bitwise identical to the
established five-week baseline. Validation and test are deterministic centred
five-week crops of their former eight-week intervals; neither event counts nor
model performance determine the crop boundaries.

The first batch trains MART-NILM only with seeds 2026, 2027, and 2028. Its
three complete runs provide the MART-NILM row of Table V, the complete-model
row of Table VII, and the parameter count, checkpoint size, and training time
for Table VI. UNet-NILM and MAT-Conv are intentionally deferred to a later
batch. Checkpoint size is the actual `best.pt` file size in MiB
(`bytes / 1024^2`).

Table VII evaluates the complete model plus six joint five-appliance
MART-NILM ablations with the same three seeds. The complete row reuses the
three Table V runs, avoiding duplicate training; the other six rows require 18
additional runs. Each row removes only one architecture component, and only
the two validation houses are evaluated. Metrics are computed per validation
house and then averaged with equal house weight. Test houses are not used for
architecture selection. The persistent remote task was
`MARTNILM_Paper5W_3Seed_20261014`. All 21 runs completed successfully with
exit code 0.

### Table V: MART-NILM test results

Three-seed means and sample standard deviations are reported below. These test
results were generated only after the configuration had been frozen; they were
not used to select the model.

| Test house | MAE (W) | SAE (W) | ON MAE (W) | OFF MAE (W) | Precision | Recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| UK-DALE 2 | 7.451 +/- 0.091 | 5.275 +/- 0.100 | 176.669 +/- 7.675 | 4.726 +/- 0.105 | 0.815 +/- 0.011 | 0.960 +/- 0.004 | 0.876 +/- 0.010 |
| REFIT 20 | 10.798 +/- 0.355 | 5.893 +/- 0.390 | 309.207 +/- 4.577 | 7.216 +/- 0.430 | 0.661 +/- 0.009 | 0.797 +/- 0.029 | 0.716 +/- 0.016 |

### Table VI: computational cost

- Trainable parameters: 1.373 M.
- Checkpoint size: 5.335 MiB (the three checkpoint files have the same size to
  the reported precision).
- Training time: 485.209 +/- 0.183 s.
- Mean selected epoch: 128.33; every run completed 150 epochs.

### Table VII: architecture ablation

Results are equal-house averages over UK-DALE 5 and REFIT 2. Each value is the
mean +/- sample standard deviation over seeds 2026, 2027, and 2028.

| Variant | Params (M) | MAE (W) | ON MAE (W) | OFF MAE (W) | SAE (W) | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| MART-NILM | 1.373 | 20.791 +/- 0.175 | 315.408 +/- 10.405 | 13.666 +/- 0.182 | 15.016 +/- 0.385 | 0.687 +/- 0.003 |
| Single-scale stem | 1.371 | 20.647 +/- 0.388 | 341.673 +/- 19.265 | 13.453 +/- 0.730 | **14.442 +/- 0.646** | 0.686 +/- 0.009 |
| TCN without dilation | 1.373 | 21.563 +/- 0.344 | 327.525 +/- 7.113 | 14.261 +/- 0.370 | 15.433 +/- 0.079 | 0.673 +/- 0.015 |
| w/o channel gate | 1.331 | **20.343 +/- 0.739** | 345.352 +/- 8.210 | **13.302 +/- 0.528** | 14.448 +/- 0.366 | **0.697 +/- 0.015** |
| w/o local decoder | 0.961 | 21.344 +/- 0.279 | 340.848 +/- 21.447 | 14.082 +/- 0.224 | 15.202 +/- 0.352 | 0.675 +/- 0.012 |
| w/o relation attention | 1.332 | 20.780 +/- 0.339 | 304.224 +/- 14.495 | 13.966 +/- 0.266 | 14.748 +/- 0.304 | 0.675 +/- 0.009 |
| w/o state--power gate | 1.373 | 26.764 +/- 0.657 | **254.833 +/- 1.473** | 21.526 +/- 1.334 | 18.839 +/- 0.762 | 0.684 +/- 0.012 |

ON MAE reveals a different trade-off from overall MAE. The multi-scale stem,
channel gate, and local decoder reduce ON MAE by 26.27, 29.94, and 25.44 W,
respectively, on the validation houses. Their apparent redundancy under overall
MAE was caused by the much larger number of OFF samples. The state--power gate
also trades higher ON MAE for much lower OFF error. Consequently, no component
is removed using the aggregate metrics alone; waveform quality and both state-
conditioned errors must be considered together.

### Frozen test-house audit of Table VII — 2026-10-14

At the user's request, the seven already-trained architecture variants were
evaluated on UK-DALE house 2 and REFIT house 20. No model was retrained and no
test metric selected a checkpoint. Metrics were first calculated independently
for each house and then averaged with equal house weight. The table below is
therefore a report-only generalisation audit; using it to redesign MART-NILM
would make these houses development data and require new untouched test houses.

| Variant | MAE (W) | ON MAE (W) | OFF MAE (W) | SAE (W) | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| MART-NILM | 9.125 +/- 0.140 | 242.938 +/- 5.348 | 5.971 +/- 0.168 | 5.584 +/- 0.174 | 0.796 +/- 0.008 |
| Single-scale stem | 9.142 +/- 0.232 | 266.108 +/- 29.491 | 5.817 +/- 0.450 | 5.575 +/- 0.174 | 0.789 +/- 0.004 |
| TCN without dilation | 10.597 +/- 0.632 | 264.661 +/- 17.181 | 7.236 +/- 0.646 | 6.935 +/- 0.546 | 0.734 +/- 0.010 |
| w/o channel gate | **8.922 +/- 0.394** | 264.554 +/- 9.890 | **5.582 +/- 0.179** | **5.403 +/- 0.464** | **0.807 +/- 0.012** |
| w/o local decoder | 9.118 +/- 0.157 | 256.070 +/- 14.515 | 5.789 +/- 0.087 | 5.633 +/- 0.226 | 0.792 +/- 0.006 |
| w/o relation attention | 9.181 +/- 0.088 | 230.841 +/- 14.692 | 6.200 +/- 0.185 | 5.546 +/- 0.044 | 0.790 +/- 0.015 |
| w/o state--power gate | 12.823 +/- 0.428 | **214.022 +/- 5.690** | 10.795 +/- 0.536 | 8.128 +/- 0.556 | 0.795 +/- 0.011 |

Conclusions:

- The channel gate is a Pareto trade-off, not a useless component. Removing it
  improves MAE, SAE, OFF MAE, and macro-F1, but worsens ON MAE from 242.938 to
  264.554 W. It must not be removed solely from the OFF-dominated metrics.
- Keep temporal dilation. Removing it damages both houses and reduces the
  house-macro F1 from 0.796 to 0.734.
- Keep the state--power gate. Removing it increases house-macro MAE from 9.125
  to 12.823 W; its main contribution is physically consistent power output,
  not classification F1.
- The multi-scale stem reduces ON MAE by 23.170 W and slightly improves F1,
  although its effect on overall MAE and SAE is negligible.
- The local decoder reduces ON MAE by 13.132 W and slightly improves F1, but
  adds about 0.412 M parameters. This is a real accuracy--complexity trade-off,
  not evidence that the decoder is useless.
- Relation attention has a small but domain-dependent benefit. Removing it
  improves UK-DALE but worsens REFIT (MAE 11.205 versus 10.798 W; F1 0.699
  versus 0.716). It remains defensible for the cross-domain story, but the
  evidence is much weaker than for dilation and gating.
- Removing the state--power gate lowers ON MAE to 214.022 W, but raises OFF MAE
  from 5.971 to 10.795 W and overall MAE from 9.125 to 12.823 W. The lower ON
  error does not compensate for physically implausible power during OFF periods.

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

## Domain-agnostic normalization ablation — rejected 2026-10-09

Experiment: `multinilm_groupnorm_house_split`

Return to the 13-channel meter-lag baseline. Replace BatchNorm only in the
shared temporal encoder and appliance heads with GroupNorm. The IBN stem,
convolutions, attention, loss, sampling, and checkpoint rule remain unchanged.
This tests whether shared running statistics from mixed UK-DALE, REFIT, and
synthetic samples cause the observed domain-dependent feature behaviour.

Result at the validation-selected checkpoint (epoch 134):

- Validation: MAE 14.324 W, macro-F1 0.768.
- REFIT house 20: MAE 11.022 W, macro-F1 0.726, AP 0.716.
- UK-DALE house 2: MAE 8.472 W, macro-F1 0.870, AP 0.927.
- REFIT fridge F1 was 0.711 and microwave F1 was 0.434, both below the
  BatchNorm baseline (0.723 and 0.521).

Conclusion: reject and restore BatchNorm in the temporal encoder and heads.
Batch-statistic mixing is not the dominant failure mechanism, and GroupNorm
removes useful amplitude-distribution information without improving REFIT
robustness.

## Per-appliance power/state balancing — partially successful 2026-10-09

Experiment: `multinilm_per_appliance_balance_house_split`

The selected meter-lag baseline is restored. No architecture, feature,
augmentation, or individual loss component changes. Only the dynamic balance
is moved inside each appliance:

`L = sum_i [P_i + lambda * S_i * stopgrad(P_i / S_i)]`.

At epoch 150 of the baseline, microwave contributed 52.4% of the total power
loss while fridge contributed 43.9% of the state loss. The previous global
ratio therefore coupled microwave regression to fridge classification. The new
formula preserves the original power/state scale for every appliance but stops
one appliance from setting another appliance's state weight.

Result at the validation-selected checkpoint (epoch 150):

- Validation: MAE 15.369 W, macro-F1 0.779, AP 0.800.
- REFIT house 20: MAE 10.182 W, macro-F1 0.731, AP 0.746.
- UK-DALE house 2: MAE 8.971 W, macro-F1 0.867, AP 0.925.
- REFIT fridge AP improved from 0.713 to 0.773, FPR fell from 0.468 to
  0.403, and MAE fell from 31.836 W to 30.294 W.
- Validation microwave AP/F1 improved from 0.349/0.449 to 0.456/0.493;
  UK-DALE microwave AP/F1 improved from 0.730/0.714 to 0.756/0.751.
- REFIT microwave F1 nevertheless fell from 0.521 to 0.489, although ON-MAE
  improved from 478.462 W to 419.183 W.

The waveform audit explains the mixed microwave result. At the baseline's
epoch-150 losses, the local power/state ratios were approximately 177 for
microwave and 3.3 for fridge, compared with a global ratio near 17. Full local
balancing therefore amplified the microwave state gradient by roughly 10x and
reduced the fridge state gradient to roughly one fifth of the global scale.
The REFIT microwave example gained two high-power activations immediately
before the labelled event. The fridge removed one late false tail, consistent
with its lower aggregate FPR, but retained long false-ON plateaus under changing
background load.

Conclusion: the experiment confirms that global cross-appliance loss coupling
was harmful, but unrestricted local equality is too aggressive. Retain the
idea, not the exact rule. The next controlled run clips every local ratio to a
factor of three around the batch-global ratio. No architecture, input feature,
sampling, or individual loss component is changed.

## Clipped per-appliance power/state balancing — retained 2026-10-09

Experiment: `multinilm_clipped_per_appliance_balance_house_split`

Use

`r = L_power / L_state`,

`r_i = clip(P_i / S_i, r / 3, 3r)`,

`L = L_power + lambda * sum_i r_i S_i`.

This is a bounded gradient-normalisation experiment. It preserves partial
task separation while preventing the microwave state objective from receiving
the approximately 10x jump observed above. All other settings are identical to
the completed per-appliance run.

Result at the validation-selected checkpoint (epoch 125, process exit 0):

- Validation: MAE 14.884 W, macro-F1 0.782, AP 0.809.
- REFIT house 20: MAE 9.840 W, macro-F1 0.744, AP 0.771.
- UK-DALE house 2: MAE 8.402 W, macro-F1 0.850, AP 0.922.
- REFIT fridge: AP 0.754, F1 0.712, FPR 0.396. In the difficult residual
  ranges, FPR remains 0.674 at 200--400 W and 0.836 at 400--800 W.
- REFIT microwave: AP 0.570 and F1 0.561, exceeding the meter-lag baseline's
  0.455 and 0.521. ON-MAE worsened from 478.462 W to 543.416 W.
- UK-DALE microwave: AP 0.722 and F1 0.682, below the baseline's 0.730 and
  0.714.

Waveform inspection shows that clipping reduced the two premature REFIT
microwave pulses of unrestricted local balancing to one, but did not restore
the clean single event of the global-balance baseline. The representative
fridge trace still contains long false-ON plateaus at 30--80 W during true-OFF
periods, despite the lower aggregate FPR.

Decision: retain clipped balancing as the working validation-selected loss.
It gives the strongest validation AP/F1 and strongest REFIT AP obtained in this
controlled sequence, without adding inference-time components. Do not claim
that it solves fridge identifiability: high-background false activation remains
the dominant failure. The next investigation must target the information or
sampling available for fridge-like OFF confusers rather than add another loss
multiplier.

## Fridge-confuser coverage audit — 2026-10-09

Before changing the sampler, count 40--150 W positive and negative residual
edges while the fridge label is OFF. These are simple proxies for unmonitored
loads that can resemble a fridge transition. The training houses contain
2,264--6,690 such edges per 100,000 OFF samples, while REFIT house 20 contains
1,857 and UK-DALE house 2 contains 1,137. Training also includes REFIT houses 3
and 5, where respectively 85.3% and 98.1% of fridge-OFF samples have residual
background above 200 W.

Conclusion: insufficient high-background or edge-negative coverage is not the
main limitation. Do not add another hard-negative sampler; it would duplicate
examples already abundant in training and risks trading recall for lower FPR.
The next experiment instead tests whether the fixed 33-minute TCN receptive
field fails to use the 136-minute input's repeated fridge-cycle context.

## Fridge-only pooled global context — rejected 2026-10-09

Experiment: `multinilm_fridge_global_context_house_split`

Keep the selected clipped loss, 13 input features, TCN, task attention,
cross-appliance relation attention, and all data augmentation unchanged. Add
one low-resolution full-window branch after the shared TCN:

1. average-pool 1024 steps to 128 tokens;
2. apply one 4-head self-attention encoder layer;
3. linearly upsample to 1024 steps;
4. add through a 0.1 residual only to the fridge head.

The output projection is initialized to zero, so epoch zero is exactly the
selected TCN rather than an abruptly perturbed model. Microwave and the other
three appliance paths do not receive this branch. This is a controlled test of
global temporal information, motivated by dual-path single-channel source
separation: local convolutions retain edge detail while a compressed global
path can compare repeated patterns across the full sequence.

Result at the validation-selected checkpoint (epoch 100, exit 0):

- Validation: MAE 16.342 W, macro-F1 0.770, AP 0.791; checkpoint score 0.386,
  worse than 0.364 without the branch.
- REFIT house 20: MAE 10.214 W, macro-F1 0.710, AP 0.707.
- UK-DALE house 2: MAE 8.251 W, macro-F1 0.858, AP 0.918.
- REFIT fridge FPR worsened from 0.396 to 0.417. The 200--400 W bin worsened
  from 0.674 to 0.725 and the 400--800 W bin from 0.836 to 0.838.
- REFIT microwave F1 fell from 0.561 to 0.447 even though the new path was
  routed only to fridge.

The last observation exposes an additional reproducibility issue: constructing
the extra module consumes random numbers before the appliance heads are
initialized, so a fixed global seed does not preserve the original head
initialization or subsequent stochastic training path. The experiment therefore
does not isolate the branch perfectly. It nevertheless provides no validation
or fridge-FPR evidence for retaining another 0.15M parameters.

Conclusion: reject and remove the complete global-context implementation. The
active model returns to the 1.38M clipped-balance configuration. Both explicit
long-dilation context and pooled attention context have now failed to remove the
same false fridge plateaus; longer context alone is not the missing information.

## Aggregate transition-support audit — 2026-10-09

The saved best-checkpoint predictions were audited without retraining. For
each predicted ON transition, the audit measured the strongest positive mains
edge within +/-2 samples.

- REFIT20 true fridge onsets: median support 83 W; 89.8% had at least 50 W.
- REFIT20 false fridge onsets: median support 29 W; 43.4% had at least 50 W.
- UK-DALE2 true fridge onsets: median support 215 W; 99.1% had at least 50 W.
- UK-DALE2 false fridge onsets: median support 23 W; 45.3% had at least 50 W.

The edge is informative, but a validation-only hard event gate failed. The
validation-optimal threshold was 0 W (no gate). At 25 W, validation fridge F1
fell from 0.813 to 0.686 because long predictions containing true-ON samples
were deleted together with their unsupported onset. Therefore aggregate edge
support must not be used as a hard whole-event filter. It remains suitable as
a diagnostic or a soft boundary cue.

Microwave false onsets were different: their median edge support was 618 W on
REFIT20 and 859 W on UK-DALE2. Most microwave false detections are therefore
supported by a real high-power aggregate event; a generic edge constraint
cannot distinguish the target microwave from another high-power appliance.

## Event-level REFIT microwave target alignment — rejected 2026-10-09

Experiment: `multinilm_refit_microwave_aligned_house_split`

A secondary dataset was created without overwriting the benchmark. Complete
REFIT microwave target events were shifted by at most +/-3 samples to the
strongest physically plausible mains rise. Aggregate power, UK-DALE, all other
targets, houses, and time ranges were unchanged. The same deterministic rule
was used for train, validation, and test. Synthetic microwave lag augmentation
was disabled because the corrected protocol was intended to remove that lag.

The alignment audit improved mechanically: on REFIT20, 131/145 audited starts
had their strongest mains rise at lag zero, and aggregate-below-microwave at
the labelled onset fell from about 59% to 10.3%. This did not improve learning.
The validation-selected checkpoint was epoch 118 (exit 0):

- Validation: MAE 15.438 W, macro-F1 0.765; microwave F1 0.443.
- REFIT20: MAE 10.190 W, macro-F1 0.718, AP 0.717; microwave F1 0.491;
  fridge F1 0.718.
- UK-DALE2: MAE 8.369 W, macro-F1 0.868, AP 0.927; microwave F1 0.698.

The retained original-label run achieved validation MAE/F1/AP of
14.884/0.782/0.809 and REFIT20 microwave F1 0.561. The aligned protocol is
therefore rejected for model training. REFIT's lag is event- and period-
dependent, and choosing the largest local mains edge can attach a microwave
label to an unrelated simultaneous load. Training-house corrections also do
not match the severe lag distribution in the selected REFIT20 period. The
probabilistic synthetic lag augmentation is more robust than rewriting the
ground truth.

Decision: restore `multinilm_clipped_per_appliance_balance_house_split` and
its [0, +1, +2] microwave lag augmentation. Keep the aligned dataset only as a
diagnostic protocol. Do not report its improved alignment audit as improved
NILM performance.

## Fixed-feature-bank simplification sweep — completed 2026-10-09

Starting point: `multinilm_clipped_per_appliance_balance_house_split` at its
validation-selected checkpoint (MAE 14.884 W, macro-F1 0.782, AP 0.809). The
house split, architecture, loss, augmentation, optimizer, checkpoint rule,
calibration, and postprocessing were fixed. Only the 13-channel input bank was
replaced. All four candidates trained for 150 epochs with seed 2026 and were
evaluated on the held-out validation houses only; no test scenario was run.

| Candidate | Input channels | Validation MAE (W) | macro-F1 | AP |
|---|---|---:|---:|---:|
| retained starting point | 13: raw, delta, abs-delta, six rolling statistics, four GL | 14.884 | 0.782 | 0.809 |
| `multinilm_simplify_feature_1_raw` | raw | **13.836** | 0.775 | **0.804** |
| `multinilm_simplify_feature_2_raw_delta` | raw, signed delta | 14.406 | **0.776** | 0.793 |
| `multinilm_simplify_feature_3_raw_delta_mean` | raw, delta, rolling mean (45) | 15.514 | 0.768 | 0.789 |
| `multinilm_simplify_feature_4_raw_delta_mean_std` | raw, delta, rolling mean/std (45) | 15.339 | 0.759 | 0.781 |
| `multinilm_simplify_feature_2_raw_mean` | raw, rolling mean (45) | 15.571 | 0.766 | 0.795 |

The derived channels are not collectively justified. Raw-only removes twelve
fixed channels and improves MAE by 1.048 W, while losing only 0.007 macro-F1
and 0.005 AP. Signed delta does not recover that small gap, and the added slow
mean and variability channels worsen all three metrics when delta is present.
The raw-only candidate is therefore the leading simplification, but it is not
yet retained under the strict no-regression rule. The final replacement test
paired raw aggregate directly with a 45-sample rolling mean and was worse than
both the retained model and raw-only. Its main failure was microwave ranking
(validation AP 0.410 and F1 0.444). No tested handcrafted feature is justified
beside the raw aggregate. The next loss experiment therefore uses raw-only and
tests whether the 0.007 macro-F1 and 0.005 AP gap can be recovered while also
removing redundant objective terms. Test houses remain unopened for selection.

## Core-loss simplification on raw aggregate — completed 2026-10-09

Experiment: `multinilm_simplify_loss_core_raw`

Starting from the raw-only feature candidate, remove OFF-MSE, on-only delta
MSE, relative window-energy error, and the additional false-positive penalty.
Retain all-sample MSE, ON-MSE, positive-weighted BCE, and the already selected
bounded per-appliance power/state balancing. Architecture, augmentation,
optimizer, checkpoint rule, calibration, and postprocessing are unchanged.

Validation-selected result (150 epochs, seed 2026, process exit 0):

| Configuration | MAE (W) | macro-F1 | AP |
|---|---:|---:|---:|
| retained 13-feature starting point | 14.884 | 0.782 | 0.809 |
| raw-only with full loss | **13.836** | 0.775 | 0.804 |
| raw-only with core loss | 14.248 | 0.775 | 0.802 |

The core loss is much shorter, but it does not recover the small classification
gap and is also worse than raw-only with the full loss on MAE and AP. Its main
per-appliance weaknesses remain dishwasher AP 0.761 and microwave AP 0.465/F1
0.461. It is therefore not yet retained. Before adding any loss term back, run
the same core objective once with checkpoint selection based only on validation
AP. This removes the bespoke MAE-plus-one-minus-AP monitor and tests whether the
choice of epoch, rather than a missing auxiliary penalty, explains the gap.

### AP-monitor implementation audit — invalid run, 2026-10-10

The first `val_ap` monitor attempt is not a model result. It exposed a runner
bug: `_epoch_score` could read `val_ap`, but `_resolve_checkpoint_monitor`
treated every metric except F1 as a minimization target. Consequently, the run
saved the *lowest*-AP early checkpoint and evaluated at MAE 36.611 W, macro-F1
0.500, and AP 0.412. These numbers are invalid for the ablation and must not be
compared with any model.

The resolver is corrected so `val_ap` and `val_average_precision` both maximize
AP, with a unit test for direction and score comparison. The invalid run folder
is preserved under an explicit diagnostic name, and the intended AP-selected
experiment is rerun from scratch. No test house was evaluated.

### Seed-order audit — prior simplification comparisons are exploratory

The corrected AP retry exposed a larger reproducibility defect. In
`train_model`, `seed_everything` was called only after `_build_model`; the
configured seed therefore controlled later sampling but not neural-network
initialization. Repeated nominally identical configurations followed materially
different validation trajectories. The feature and core-loss results above are
useful screening evidence, but they are not accepted as final controlled
ablations.

The runner now resolves and applies the seed before model construction. The
interrupted AP retry is preserved as
`multinilm_simplify_loss_core_ap_monitor_raw_invalid_unseeded_init`.

An initial strict deterministic-kernel check was stopped after confirming that
it increased epoch time from about 4 s to about 25 s. Its partial folder is
preserved as `multinilm_simplify_deterministic_baseline_invalid_slow_kernel_mode`.
Strict kernels are not a model contribution and are not retained; the corrected
seed order is used with the original CUDA speed settings.

To avoid repeating a large sweep, only two seeded references are run:
the complete 13-feature starting configuration and raw-only with every other
setting fixed. Subsequent loss and architecture experiments inherit the same
pre-construction seed. The original reported validation result
(14.884 W, 0.782 macro-F1, 0.809 AP) remains the absolute no-regression target.

## Seeded feature decision — completed 2026-10-10

The complete starting configuration and raw-only were rerun after moving the
seed before model construction. Both used seed 2026 and the original loss,
architecture, augmentation, optimizer, composite checkpoint rule, calibration,
and postprocessing. No test house was evaluated.

| Seeded configuration | Input channels | MAE (W) | macro-F1 | AP |
|---|---:|---:|---:|---:|
| complete starting model | 13 | 15.586 | 0.772 | 0.795 |
| raw-only | **1** | **14.590** | 0.771 | 0.794 |

Raw-only removes twelve fixed transforms, improves MAE by 0.996 W, and changes
macro-F1/AP by only -0.001/-0.001 relative to the controlled baseline. The
handcrafted feature bank therefore has no stable classification contribution
and is rejected. This conclusion is stronger than the earlier exploratory
sweep because both candidates now start from the same seeded initialization.

The seeded raw history reaches its highest validation AP at epoch 137, whereas
the bespoke composite monitor saved epoch 126. One final raw-only run uses the
corrected `val_ap` monitor. It retains the full loss and changes only checkpoint
selection. Keep it only if formal calibrated validation also satisfies the
absolute original target; otherwise raw-only remains promising but cannot yet
become the retained final model.

## Raw-only AP checkpoint and waveform audit — completed 2026-10-10

Experiment: `multinilm_simplify_seeded_raw_ap_monitor`

Changing only checkpoint selection from the original composite score to maximum
validation AP selected epoch 143. It did not improve the final calibrated
validation result:

| Raw-only checkpoint rule | MAE (W) | macro-F1 | AP | Event-F1 | False events |
|---|---:|---:|---:|---:|---:|
| original composite, epoch 126 | **14.590** | **0.771** | 0.794 | **0.392** | **381** |
| maximum AP, epoch 143 | 15.576 | 0.770 | **0.801** | 0.365 | 403 |

The AP-only checkpoint gains 0.007 AP but loses 0.986 W MAE, 0.001 macro-F1,
0.027 event-F1, and creates 22 additional false events. It is rejected. This
also shows why a ranking metric alone is insufficient for checkpoint selection
in a joint detection-and-regression model.

A fixed-event waveform audit used the identically selected 50-plot validation
sets from the seeded 13-channel reference and seeded raw-only composite
checkpoint. Representative matched fridge, microwave, dishwasher, and
washing-machine events were inspected directly. The raw-only model introduced
no new giant pulses or power/state contradictions in those matched plots.
Fridge cycles were at least as continuous on the inspected noisy segment;
microwave pulses retained sharp boundaries; dishwasher and washing-machine
waveform quality was broadly comparable. This agrees with the aggregate metrics: raw-only
improves MAE, off-MAE, event-F1, and false-event count relative to the seeded
13-channel reference, while macro-F1/AP change by only -0.001/-0.001.

Decision: delete the AP-only checkpoint candidate and retain the original
composite checkpoint rule for the next controlled loss ablation. Raw-only is
the working simplification, but the original reported result remains the
absolute target until the remaining controlled ablations are complete. Test
houses remain unopened.

## Seeded core-loss ablation — completed 2026-10-10

Experiment: `multinilm_simplify_seeded_loss_core_raw`

On the seeded raw-only model, the core objective removed OFF-MSE, power-delta
MSE, relative energy error, and the extra state false-positive penalty together.
It retained all-sample MSE, ON-MSE, weighted BCE, and clipped per-appliance
power/state balancing.

| Loss | MAE (W) | macro-F1 | AP | Event-F1 | False events |
|---|---:|---:|---:|---:|---:|
| full retained loss | **14.590** | **0.771** | **0.794** | **0.392** | **381** |
| core loss | 15.134 | 0.762 | 0.790 | 0.350 | 391 |

The core loss is rejected. Its matched-event waveform comparison shows a real
trade-off rather than a harmless simplification. Fridge median event NRMSE
improves from 0.384 to 0.294, but microwave detection rate falls from 0.637 to
0.568, median event NRMSE worsens from 0.579 to 0.778, waveform correlation
falls from 0.377 to 0.322, and microwave false events rise from 93 to 101.
Kettle and washing-machine event IoU also decline.

The next and final loss simplification is narrower: remove only OFF-MSE and
relative energy, which overlap most directly with all-sample MSE, while keeping
the edge-shape and false-positive terms implicated by the failed core ablation.
No test house was evaluated.

## Seeded compact-loss ablation — completed 2026-10-10

Experiment: `multinilm_simplify_seeded_loss_compact_raw`

This narrower candidate removed only OFF-MSE and relative-energy error, while
retaining power-delta MSE and the state false-positive penalty.

| Loss | MAE (W) | macro-F1 | AP | Microwave F1 | Microwave event NRMSE |
|---|---:|---:|---:|---:|---:|
| full retained loss | **14.590** | **0.771** | **0.794** | **0.474** | **0.579** |
| compact loss | 15.110 | 0.763 | 0.788 | 0.445 | 0.789 |

The compact objective is also rejected. Although fridge event NRMSE improves,
microwave event detection falls from 0.637 to 0.589, event IoU from 0.460 to
0.423, and median event NRMSE worsens sharply. Kettle event detection/IoU and
the overall calibrated metrics also decline. Therefore the existing auxiliary
losses cannot be removed without a measurable validation and waveform cost in
the present model. The full loss is retained, and simplification proceeds to
the architecture. No test house was evaluated.

## Seeded task-attention ablation — completed 2026-10-10

Experiment: `multinilm_simplify_seeded_raw_no_task_attention`

Only the channel-wise task-attention module inside each appliance head was
disabled. Cross-appliance relation attention and every other setting remained
unchanged. Parameter count fell from 1.373 M to 1.33 M.

| Architecture | MAE (W) | macro-F1 | AP | Event-F1 | Microwave F1 |
|---|---:|---:|---:|---:|---:|
| task attention | 14.590 | **0.771** | **0.794** | **0.392** | **0.474** |
| no task attention | **14.558** | 0.763 | 0.779 | 0.357 | 0.442 |

The 0.032 W MAE change is negligible and does not compensate for the 0.015 AP
loss or the state/event degradation. On matched microwave events, detection
falls from 0.637 to 0.589, IoU from 0.460 to 0.421, median NRMSE worsens from
0.579 to 0.776, and false events rise from 93 to 112. Kettle and dishwasher
event quality also declines. The module is retained.

The next architecture test preserves both attention mechanisms and changes
only the number of appliance-local residual blocks from two to one. No test
house was evaluated.

## Seeded appliance-head depth ablation — completed 2026-10-10

Experiment: `multinilm_simplify_seeded_raw_one_head_block`

Only the number of residual local blocks in each appliance head changed from
two to one. Parameter count fell from 1.373 M to 1.13 M.

| Head depth | MAE (W) | macro-F1 | AP | Event-F1 | Microwave event NRMSE |
|---|---:|---:|---:|---:|---:|
| two blocks | **14.590** | **0.771** | 0.794 | **0.392** | **0.579** |
| one block | 15.084 | 0.770 | **0.802** | 0.368 | 0.812 |

The shallower decoder improves AP by 0.008 but fails the joint power/waveform
criterion. Microwave event detection drops from 0.637 to 0.589, IoU from 0.460
to 0.408, correlation from 0.377 to 0.350, and median NRMSE worsens to 0.812.
Overall MAE also worsens by 0.494 W. The second local block is retained.

The next run changes no component. It tests the single round state-task weight
`lambda_state=1.0` on raw input, instead of 0.8, to determine whether the
compact input can recover the small AP/F1 gap without adding complexity. No
test house is evaluated.

## Seeded state-weight check — completed 2026-10-10

Experiment: `multinilm_simplify_seeded_raw_lambda_state_1`

Changing only `lambda_state` from 0.8 to 1.0 improves overall MAE from 14.590
to 14.520 W and AP from 0.794 to 0.799, but macro-F1 falls from 0.771 to 0.768.
The event audit rejects the apparent power improvement: microwave detection
falls from 0.637 to 0.514, IoU from 0.460 to 0.384, median event NRMSE worsens
from 0.579 to 0.919, and median energy error rises from 34.9% to 73.0%.
`lambda_state=0.8` is retained; further weight tuning is stopped.

The next architecture test replaces only stem IBN with ordinary BatchNorm. It
directly tests whether the hardest-to-explain normalization is necessary while
preserving the multiscale convolutions, TCN, appliance heads, and relation
attention. No test house is evaluated.

## Seeded stem-normalization ablation — completed 2026-10-10

Experiment: `multinilm_simplify_seeded_raw_batch_stem_norm`

Replacing only stem IBN with BatchNorm is decisively rejected. Overall MAE
worsens from 14.590 to 15.894 W, macro-F1 from 0.771 to 0.758, and AP from
0.794 to 0.753. Fridge is the main failure: MAE rises from 20.399 to 26.304 W,
F1 falls from 0.837 to 0.780, event NRMSE worsens from 0.384 to 0.502, false
events rise from 228 to 372, and event IoU falls from 0.786 to 0.706. IBN is
retained as a supported cross-house normalization component.

The final architecture-depth check reduces the shared TCN from five dilated
blocks to three while keeping the stem, both attention modules, and appliance
heads fixed. This tests a substantial shared-encoder simplification without
confounding it with normalization. No test house is evaluated.

## Seeded shared-TCN depth ablation - completed 2026-10-10

Experiment: `multinilm_simplify_seeded_raw_three_tcn_blocks`

Reducing the shared TCN from five to three dilated blocks lowers the parameter
count from 1.373 M to 1.08 M, but is decisively rejected by both aggregate and
event-level validation results.

| TCN depth | MAE (W) | macro-F1 | AP | Microwave detection | Microwave event NRMSE |
|---|---:|---:|---:|---:|---:|
| five blocks | **14.590** | **0.771** | **0.794** | **0.637** | **0.579** |
| three blocks | 16.595 | 0.757 | 0.749 | 0.514 | 0.882 |

The shorter encoder also lowers microwave event IoU from 0.460 to 0.373 and
raises median absolute microwave energy error from 34.9% to 73.5%. Fridge
event IoU falls from 0.786 to 0.766 and waveform correlation from 0.106 to
0.055. The fifth-block receptive field and capacity therefore contribute to
event separation rather than merely increasing model size. The five-block TCN
is retained. No test house was evaluated.

## Seeded dropout check - completed 2026-10-10

Experiment: `multinilm_simplify_seeded_raw_dropout_035`

Increasing the single shared dropout value from 0.25 to 0.35 is rejected.
Validation MAE worsens from 14.590 to 15.345 W, macro-F1 from 0.771 to
0.769, and AP from 0.794 to 0.780. The event audit agrees: microwave median
event NRMSE rises from 0.579 to 0.628 and correlation falls from 0.377 to
0.359; fridge false events rise from 228 to 233 and correlation falls from
0.106 to 0.070. The original 0.25 dropout is retained. No test house was
evaluated.

## Seeded training-gate ablation - completed 2026-10-10

Experiment: `multinilm_simplify_seeded_raw_no_train_power_gate`

Removing the soft state gate from the training-time power output is rejected.
Validation MAE changes from 14.590 to 14.666 W, macro-F1 from 0.771 to
0.767, and AP from 0.794 to 0.772. Microwave F1 rises slightly from 0.474 to
0.487, but its event NRMSE worsens from 0.579 to 0.649 and correlation falls
from 0.377 to 0.353. Fridge F1 falls from 0.837 to 0.823, false events rise
from 228 to 259, and event NRMSE worsens from 0.384 to 0.403. Direct visual
inspection shows the difficult high-background fridge event growing from
roughly 1.2 kW repeated false pulses to roughly 2.3 kW pulses. The soft
training gate is therefore retained. No test house was evaluated.

## Seeded relation residual-scale check - completed 2026-10-10

Experiment: `multinilm_simplify_seeded_raw_relation_scale_05`

Increasing the relation-attention residual scale from 0.25 to 0.50 is
rejected. Validation MAE worsens from 14.590 to 15.565 W, macro-F1 from
0.771 to 0.769, and AP from 0.794 to 0.779. Microwave detection rises from
0.637 to 0.658, but median event NRMSE worsens from 0.579 to 0.640 and false
events rise from 93 to 97. Fridge false events also rise from 228 to 236.
Visual inspection confirms that repeated high-background fridge pulses and
truncated microwave plateaus remain. The 0.25 residual scale is retained and
no fine-grained scale sweep is performed. No test house was evaluated.

## Seeded detached soft-gate ablation - completed 2026-10-10

Experiment: `multinilm_simplify_seeded_raw_detached_train_power_gate`

The forward gate remains numerically identical, but the power loss is stopped
from back-propagating through state probability. This global decoupling is
rejected as the retained model, but it exposes a real task conflict. Fridge
improves from 20.399 to 18.540 W MAE and from 0.837 to 0.840 F1; its median
event NRMSE improves from 0.384 to 0.360 and the extreme repeated-pulse case
falls from roughly 1.2 kW to roughly 100--150 W. In contrast, microwave F1
falls from 0.474 to 0.431, median event NRMSE worsens from 0.579 to 0.682,
and false events rise from 93 to 136. Overall MAE/F1/AP are 15.139/0.763/0.773,
all below the retained raw model. A single midpoint gradient-scale experiment
is justified by these opposing endpoint results; appliance-specific exceptions
are not introduced. No test house was evaluated.

## Seeded half-gradient soft-gate check - completed 2026-10-10

Experiment: `multinilm_simplify_seeded_raw_half_gradient_train_power_gate`

Scaling the power-to-state gate gradient to 0.5 is rejected after the required
metric-plus-waveform audit. Aggregate validation metrics initially appear
promising: MAE improves from 14.590 to 13.579 W and macro-F1 from 0.771 to
0.776, while AP falls from 0.794 to 0.785. Microwave event detection and
median NRMSE improve from 0.637/0.579 to 0.712/0.550. However, false events
rise from 93 to 110. More importantly, the difficult high-background fridge
waveform becomes physically worse: repeated false pulses grow from roughly
1.2 kW to above 2 kW, its event NRMSE rises from 3.73 to 8.15 in the inspected
case, and total fridge false events rise from 228 to 246. The lower average
MAE therefore hides a severe waveform regression. Gradient scaling is not
retained and no further interpolation grid is run. No test house was evaluated.

## Shared BiGRU context - rejected 2026-10-10

Experiment: `multinilm_seeded_raw_shared_bigru_context`

A single bidirectional GRU (64 units per direction) was added as a 0.1-scale
residual after the shared TCN. It increased the model from 1.373 M to 1.447 M
parameters and exposed every output position to the full 1024-sample window.
The local TCN path, heads, loss, sampling, and evaluation remained unchanged.

Validation MAE/F1/AP changed from 14.590/0.771/0.794 to
16.887/0.741/0.789. Fridge event NRMSE worsened from 0.384 to 0.411 and its
false-event count increased from 228 to 2,461. Microwave detection fell from
0.637 to 0.610 and event NRMSE worsened from 0.579 to 0.727. Dishwasher and
washing-machine false events also rose from 16/7 to 402/242.

Conclusion: reject and remove the recurrent code. Full-window recurrent mixing
propagates ambiguous background evidence instead of resolving appliance
identity. The next experiment adds no component: reduce the common width from
128 to 96 channels to test whether lower capacity improves cross-house
generalisation while making the retained model smaller. No test house was
evaluated.

## Reduced hidden width - rejected 2026-10-10

Experiment: `multinilm_seeded_raw_hidden_96`

The shared representation and appliance heads were reduced from 128 to 96
channels. This added no component and reduced trainable parameters from
1.373 M to 0.789 M (43%). Validation MAE/F1/AP nevertheless changed from
14.590/0.771/0.794 to 15.857/0.746/0.776. Fridge false events increased from
228 to 2,309. Microwave detection fell from 0.637 to 0.486, event IoU from
0.460 to 0.364, and median event NRMSE worsened from 0.579 to 1.000.

Conclusion: reject. The 128-channel representation is not removable excess
capacity under this protocol; reducing it disproportionately harms sparse
events and temporal state stability. Do not continue a width sweep. The next
candidate keeps all retained features and changes only the state output kernel
from one to five samples, giving the classifier 40 seconds of direct local
context without adding another feature extractor. No test house was evaluated.

## Temporal state-head kernel - rejected 2026-10-10

Experiment: `multinilm_seeded_raw_state_head_kernel_5`

Only the final state classifier changed from a one-sample to a five-sample
convolution. This added about 2.6 K parameters (less than 0.2%) and left the
power head, feature extractor, loss, and data pipeline unchanged. Validation
MAE/F1/AP changed from 14.590/0.771/0.794 to 16.980/0.754/0.798. Microwave
F1 was effectively unchanged (0.474 to 0.473), but event NRMSE worsened from
0.579 to 0.773. Fridge F1 fell from 0.837 to 0.825, event NRMSE worsened from
0.384 to 0.429, and false events increased from 228 to 1,901.

Conclusion: reject and remove the wider state head. A short output filter does
not resolve appliance identity under confounding background and its gradients
disturb the soft-gated regression. The next run changes no architecture: it
tests stronger weight decay as a direct regularisation check. No test house was
evaluated.

## Stronger weight decay - rejected 2026-10-10

Experiment: `multinilm_seeded_raw_weight_decay_5e4`

Only Adam's weight decay changed from 0.0001 to 0.0005; architecture, loss,
sampling, seed, and checkpoint criterion were fixed. Validation MAE/F1/AP
changed from 14.590/0.771/0.794 to 16.549/0.753/0.796. Fridge event NRMSE
worsened from 0.384 to 0.404 and false events increased from 228 to 1,989.
Microwave detection fell from 0.637 to 0.548, event IoU from 0.460 to 0.404,
and event NRMSE worsened from 0.579 to 0.843.

Conclusion: reject the stronger regularisation and retain weight decay 0.0001.
The four completed checks in this round do not support adding a recurrent
context layer, narrowing the network, widening the state output kernel, or
increasing regularisation. The raw-only model remains the validation-selected
architecture. Training-time waveform rendering is changed to end-only because
it affects neither optimization nor checkpoint selection and removes repeated
multi-minute CPU stalls. No test house was evaluated.
## High-confidence REFIT two-edge event alignment — rejected 2026-10-11

Experiment: `k4_alpha1_refit_hc_event_alignment`

Goal: test whether REFIT aggregate–IAM timestamp mismatch can be corrected
without changing the model, split, aggregate input, or training protocol.  The
earlier aggregate-rewriting dataset was invalid because target appliance edges
were used to alter the model input and the repeated edits caused cumulative
aggregate drift.  It was not used for this experiment.

The replacement preprocessing keeps aggregate immutable and moves an existing
complete appliance ON interval by at most two 8-second samples only when its ON
and OFF aggregate edges independently support the same lag, no other monitored
appliance changes near either boundary, and the winning lag is separated from
the alternatives by a fixed margin. Ambiguous events remain unchanged. Event
power above local standby is subtracted and re-added, conserving appliance
watt-samples. The controlled dataset is
`mixed_ukdale_refit_5w_house_split_high_confidence_aligned_v2`.

Integrity checks passed for every CSV: row count, timestamp, house,
`sequence_id`, and split membership were identical to the retained baseline;
the maximum aggregate difference was exactly 0 W; UK-DALE was unchanged. In
REFIT20, 30/134 complete microwave events were shifted, changing 106 state
rows. Only 2 fridge events were shifted, changing 4 state rows.

The same `multinilm_k4.yaml`, seed 2026, augmentation, architecture, loss, and
150-epoch schedule were used. Training completed successfully on the RTX 4090
in 9m12s; the validation-selected checkpoint was epoch 147. Validation-only
evaluation produced:

- overall MAE 16.875 W, macro-F1 0.750, micro-F1 0.816, AP 0.790;
- microwave F1 0.462, AP 0.429, event-F1 0.458;
- fridge F1 0.831 and AP 0.906.

This is worse than the retained baseline validation MAE/F1/AP of
14.884/0.782/0.809. AP also decreased, so the result is not explained by state
threshold calibration. Representative microwave waveforms show multiple
submeter bursts and gaps inside one processed ON interval while the aggregate
supports only some bursts; a whole-event temporal shift cannot remove that
ambiguity. The method is rejected, and the test houses were not evaluated for
model selection.

Decision: retain the original house-split dataset and the existing training-only
microwave lag augmentation. Treat REFIT's missing per-IAM acquisition times as
label-time uncertainty; do not claim that cleaned REFIT timestamps can be
deterministically reconstructed from aggregate edges.

## REFIT alignment without training jitter A/B - completed 2026-10-11

This experiment removed the remaining confound from the preceding alignment
test. The same `multinilm_k4.yaml`, seed 2026, architecture, loss, sampler,
schedule, and checkpoint rule were used for both runs, while
`alignment_jitter.enabled=false`. Run A used the original house-split data;
run B used `mixed_ukdale_refit_5w_house_split_high_confidence_aligned_v2`.
Only validation results were used for this decision.

| Data, with jitter disabled | Best epoch | MAE (W) | macro-F1 | AP | Microwave F1 | Microwave AP |
|---|---:|---:|---:|---:|---:|---:|
| Original | 147 | 16.915 | 0.760 | **0.798** | 0.488 | **0.455** |
| High-confidence aligned | 137 | **16.436** | 0.760 | 0.797 | **0.500** | 0.444 |
| Retained original + training jitter | - | **14.884** | **0.782** | **0.809** | - | - |

The aligned data improves overall MAE by 0.479 W but does not improve macro-F1
and lowers AP by 0.001. Fridge MAE/F1 change from 22.165/0.832 to
21.490/0.835. Microwave F1 improves by 0.012, but its AP decreases by 0.011,
so the apparent thresholded gain is not a general ranking improvement.

A focused audit of the validation events actually moved by the deterministic
rule gives a more precise interpretation. On 20 shifted microwave events,
mean local IoU improves from 0.488 to 0.560 and event detection from 0.65 to
0.70. On 35 shifted fridge events, IoU is unchanged (0.929 to 0.928). Fridge
FPR improves from 0.538 to 0.501 in the 400--800 W residual bin and from 0.609
to 0.559 above 800 W. The five fixed validation waveform panels show no new
catastrophic output, but also no systematic correction outside those few
events.

The edge audit directly rejects a claim of 100% zero jitter. For REFIT house 2
microwave starts, the number whose strongest aggregate edge is at lag zero is
140/298 before and 139/298 after; in REFIT house 20 it is 28/145 before and
44/145 after. This statistic is not itself ground truth, but it demonstrates
that the conservative rule leaves substantial unresolved timing uncertainty.
The method cannot recover missing physical IAM acquisition timestamps.

Preprocessing was hardened independently of the model result. Mean resampling
is now the command-line and programmatic default for every power channel,
preventing the old hidden mean-aggregate/max-appliance mismatch. Each shifted
event has an enforced watt-sample energy-conservation assertion. Every direct
REFIT preprocessing output also writes a `.quality.json` report covering
timestamps, finite/non-negative power, the 8 s grid, binary labels, energy
error, and unresolved alignment events. Six alignment tests and a real
six-hour REFIT house 20 smoke test pass.

Decision: the correction is locally useful but is rejected as a replacement
for the established data treatment. Restore `alignment_jitter.enabled=true`
in the retained model because disabling it causes a material validation drop.
Do not claim complete REFIT synchronization; describe the event alignment, if
reported, as a high-confidence diagnostic/secondary protocol.

## Original data plus training jitter reproducibility audit - completed 2026-10-11

The historical validation result of 14.884 W MAE, 0.782 macro-F1, and 0.809
AP was rerun to determine whether it is reproducible. Two distinct model
configurations had previously been described as "original + jitter", so both
were evaluated on validation only with seed 2026. No test house was opened.

| Configuration | Input | Best epoch | MAE (W) | macro-F1 | AP | Microwave F1 | Microwave AP |
|---|---|---:|---:|---:|---:|---:|---:|
| current `multinilm_k4.yaml` + original data + jitter | raw only | 147 | 16.721 | 0.751 | 0.794 | 0.465 | 0.436 |
| historical saved config + original data + jitter | 13 channels | 133 | **14.301** | **0.771** | 0.792 | 0.460 | 0.420 |
| historical single run (reference only) | 13 channels | 125 | 14.884 | 0.782 | **0.809** | 0.500 | 0.478 |

The first run is not a strict reproduction of the 14.884 W result: the
current YAML has since been simplified from 13 fixed input channels to raw
aggregate only. It is nevertheless a controlled comparison with the recent
raw-only no-jitter run (16.915 W, 0.760 macro-F1, 0.798 AP). Under the current
model, jitter improves MAE by 0.194 W but reduces macro-F1 by 0.009 and AP by
0.004. Training jitter therefore does not provide a reproducible
classification improvement in this comparison.

The second run used the `model` section extracted directly from the historical
run's saved `config_merged.yaml`. It restored raw, signed and absolute delta,
six rolling statistics, and four GL channels, as well as the saved
postprocessing and jitter settings. Its parameter count (1,376,574) matches
the historical run. The old point estimate was not reproduced: MAE improved
by 0.583 W, but macro-F1 fell by 0.011 and AP by 0.017. Fridge F1/AP were
0.810/0.859; microwave F1/AP were 0.460/0.420. Fridge false-positive rate
still increased sharply with residual background: 0.434 at 200--400 W, 0.583
at 400--800 W, and 0.727 above 800 W. Long-context plots show false fridge
cycles in true-OFF regions and missed microwave bursts in dense intermittent
events, although clean isolated microwave pulses remain well reconstructed.

The saved configuration also matches the later seeded 13-channel baseline
except for `experiment_id`, yet that earlier run reported
15.586/0.772/0.795 and selected epoch 80. The original 14.884 run was produced
before the seed was moved ahead of model construction; its manifest says seed
2026, but the initial weights were not actually fixed by that seed. In
addition, CUDA kernels are not configured for strict determinism. Therefore
14.884/0.782/0.809 must be treated as a historical single-run observation,
not an exact reproducible baseline. Future comparisons should use controlled
seeded reruns and report variation across seeds rather than require equality
to that point estimate.

## Aligned data without jitter: held-out test report - completed 2026-10-12

After the validation-only 2-by-2 data/jitter comparison, the selected
high-confidence-aligned, no-jitter checkpoint (best epoch 137) was evaluated
once on the two untouched test houses. Validation calibration thresholds were
reused without test-time fitting. These results are for reporting only and
were not used to change the model.

| Test house | MAE (W) | macro-F1 | micro-F1 | AP |
|---|---:|---:|---:|---:|
| REFIT house 20 | 11.385 | 0.684 | 0.692 | 0.740 |
| UK-DALE house 2 | **8.485** | **0.868** | **0.874** | **0.938** |

| Appliance | REFIT20 MAE | REFIT20 F1 | REFIT20 AP | UK-DALE2 MAE | UK-DALE2 F1 | UK-DALE2 AP |
|---|---:|---:|---:|---:|---:|---:|
| kettle | 5.157 | 0.813 | 0.879 | 4.586 | 0.968 | 0.996 |
| fridge | 34.354 | 0.691 | 0.711 | 23.042 | 0.882 | 0.954 |
| dishwasher | 4.504 | 0.685 | 0.752 | 5.012 | 0.756 | 0.960 |
| washing machine | 4.221 | 0.770 | 0.858 | 6.181 | 0.937 | 0.984 |
| microwave | 8.686 | 0.459 | 0.501 | 3.606 | 0.797 | 0.798 |

The cross-house gap remains large. In REFIT20, fridge FPR is 0.706 at a
200--400 W residual and 0.832 at 400--800 W. Long-context waveforms show long
40--100 W false fridge cycles during true-OFF periods. Isolated microwave
events can be reconstructed accurately, but a representative 1.3 kW event is
missed completely and pulsed high-background segments can trigger false
microwave outputs. Alignment therefore provides a small validation benefit but
does not solve background-dependent fridge activation or REFIT microwave
generalisation.

## Full 2-by-2 cross-house audit - completed 2026-10-12

The validation result alone selected aligned data without jitter, but its
small 0.479 W validation-MAE gain did not transfer consistently. The remaining
three checkpoints were therefore evaluated on the same UK-DALE2 and REFIT20
cross-house development scenarios, using only thresholds calibrated on their
own validation predictions. All evaluations exited successfully. Because
these houses now inform method retention, they are development/generalisation
houses rather than an untouched final test set.

| Data / augmentation | Validation MAE/F1/AP | REFIT20 MAE/F1/AP | UK-DALE2 MAE/F1/AP |
|---|---|---|---|
| original / no jitter | 16.915 / **0.760** / **0.798** | 11.275 / **0.703** / 0.737 | **8.373** / 0.861 / **0.938** |
| aligned / no jitter | **16.436** / **0.760** / 0.797 | 11.385 / 0.684 / 0.740 | 8.485 / **0.868** / **0.938** |
| original / jitter | 16.721 / 0.751 / 0.794 | 11.388 / 0.693 / 0.733 | 8.379 / 0.862 / 0.924 |
| aligned / jitter | 16.875 / 0.750 / 0.790 | **11.121** / 0.699 / **0.746** | 8.522 / 0.858 / 0.926 |

Across the two development houses, the simple unaligned/no-jitter method has
the highest house-mean F1 (0.782) and nearly the lowest mean MAE (9.824 W;
aligned/jitter is 9.822 W). Aligned/no-jitter has the highest mean AP by only
0.001, but loses 0.006 mean F1. No alignment/jitter option dominates both
houses. REFIT microwave benefits from aligned/jitter (F1/AP 0.550/0.554), but
UK-DALE microwave falls to 0.731/0.734, compared with 0.798/0.803 for
original/no-jitter.

Waveform inspection agrees with the metrics. All four methods retain long
false fridge plateaus under high residual background. Alignment and jitter can
reduce timing error for selected microwave events, but also retain large false
microwave activations in confounding background segments. They do not solve
the underlying identification problem.

Decision: restore the original house-split data and disable alignment jitter
in the working configuration. This is the most stable cross-house option and
removes two difficult-to-defend processing steps. Future component changes are
first checkpointed on validation, then audited on both cross-house development
houses with metrics and waveforms. A separate unseen house split is required
for the final paper test.

## Controlled 5-week versus 10-week training duration - completed 2026-10-12

This experiment tested whether simply exposing the unchanged model to more
training data improves generalisation. Commit `ca86b30` introduced a
duration-controlled builder and the dataset config
`experiment_mixed_ukdale_refit_10w_house_split.yaml`. The exact established
5-week rows were retained as the centre of each training interval, and only
the preceding/following rows were added. Validation and both cross-house
development scenarios were copied exactly from the 5-week dataset. Thus the
held-out houses, timestamps, labels, evaluation, seed, model, loss, sampling,
augmentation, and checkpoint rule were unchanged. Normalisation was correctly
re-fitted on the longer training split only.

The training split grew from 1,857,045 to 3,677,016 rows (1.98 times). Fridge
events increased from 6,062 to 11,382, and microwave events from 780 to 1,322.
The run used `multinilm_k4.yaml`, original data, no alignment jitter, seed
2026, 150 epochs, and finished successfully on the RTX 4090. Best epoch was
111; total training time was 15 minutes 26 seconds. The result directory is
`runs/k4_original_no_jitter_10w/multinilm_fractional`.

| Training duration | Validation MAE/F1/AP | REFIT20 MAE/F1/AP | UK-DALE2 MAE/F1/AP |
|---|---|---|---|
| 5 weeks | 16.915 / **0.760** / 0.798 | **11.275** / **0.703** / **0.737** | 8.373 / 0.861 / **0.938** |
| 10 weeks | **16.747** / 0.757 / **0.806** | 11.734 / 0.697 / 0.736 | **8.055** / **0.863** / 0.934 |

The validation change is mixed rather than a clear improvement: MAE improves
by 0.168 W and AP by 0.008, while macro-F1 falls by 0.003. The gain does not
transfer consistently. REFIT20 becomes worse in all three overall metrics;
UK-DALE2 gains 0.318 W MAE and 0.002 F1 but loses 0.004 AP.

Critical-appliance results also remain mixed. Validation fridge improves from
22.165 to 20.993 W MAE and from 0.832 to 0.834 F1, but its overall FPR rises
from 0.161 to 0.176. REFIT20 fridge F1 rises from 0.689 to 0.708, whereas its
MAE and FPR worsen. REFIT20 microwave degrades from 8.315/0.470/0.460 to
8.875/0.450/0.421 MAE/F1/AP. UK-DALE2 microwave also falls from 0.798/0.803
to 0.780/0.787 F1/AP.

The background-conditioned audit gives the clearest rejection evidence:

| Split | Fridge residual bin | 5-week FPR | 10-week FPR |
|---|---:|---:|---:|
| Validation | 200--400 W | 0.348 | 0.390 |
| Validation | 400--800 W | 0.538 | 0.588 |
| Validation | above 800 W | 0.609 | 0.742 |
| REFIT20 | 200--400 W | 0.685 | 0.719 |
| REFIT20 | 400--800 W | 0.830 | 0.852 |
| REFIT20 | above 800 W | 0.482 | 0.692 |
| UK-DALE2 | 200--400 W | 0.332 | 0.383 |
| UK-DALE2 | 400--800 W | 0.829 | 0.946 |
| UK-DALE2 | above 800 W | 0.251 | 0.415 |

Waveforms agree with these statistics. Several true fridge cycles and
microwave plateaus have lower amplitude error with 10 weeks, especially clean
UK-DALE events and some medium-background REFIT microwave events. However,
false fridge plateaus become more frequent or stronger in every high-background
bin. REFIT microwave retains large false activations, and its lower F1/AP shows
that improved reconstruction of selected true events does not compensate for
the classification failures.

Decision: reject 10 weeks as the default training duration. More rows alone do
not solve the noisy-background identification problem and do not produce a
consistent cross-house improvement. Retain the controlled 10-week dataset and
run as negative ablation evidence, but keep the 5-week original/no-jitter split
for subsequent model work.

## Top-level MART-NILM module ablation — completed 2026-10-14

The earlier paper ablation mixed top-level architecture blocks with internal
choices such as dilation, channel attention, and decoder depth. It was replaced
with a module-level design that follows the architecture figure. Every variant
remains one joint five-appliance model and retains power and state outputs of
shape `(B, T, 5)`. Removed modules use only an identity mapping or the
pointwise channel projection needed to preserve the downstream interface.

The controlled variants were:

- multi-scale CNN reduced from kernels `[3, 5, 9]` to the central `[5]`
  branch, retaining the residual path, width, normalization, and feature
  encoder;
- feature encoder removed and replaced by a temporal-context-free `1x1`
  projection from 32 to 128 channels;
- shared temporal encoder replaced by identity;
- all five appliance decoders removed together, retaining only five independent
  `1x1` power/state readouts;
- cross-appliance relation attention replaced by identity.

This also corrected the semantics of the old decoder ablation:
`head_local_layers=0` still constructed a learnable `1x1` decoder block, so it
did not remove the decoder. The new `appliance_decoder_enabled=false` path
uses identity and disables both decoder channel attention and local residual
convolutions.

All experiments used the original mean-resampled 5-week all-house split,
`alignment_jitter.enabled=false`, seeds 2026/2027/2028, the same loss and
training protocol, and validation-selected checkpoints. The test houses were
evaluated only after each checkpoint was frozen. The remote Scheduled Task
`MARTNILM_Module_Ablation_20261014` completed with exit code 0. Runs are under
`runs/paper_5w/table_vii`; the complete row reuses
`runs/paper_5w/table_v/mart_nilm`.

Validation-house equal-weight results:

| Variant | Params (M) | MAE (W) | ON MAE (W) | SAE (W) | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Complete | 1.373 | 20.791 +/- 0.175 | 315.408 +/- 10.405 | 15.016 +/- 0.385 | 0.687 +/- 0.003 |
| w/o multi-scale CNN | 1.372 | 20.273 +/- 0.447 | 317.615 +/- 14.001 | 14.485 +/- 0.442 | 0.693 +/- 0.010 |
| w/o feature encoder | 1.326 | 20.247 +/- 0.199 | 337.029 +/- 15.645 | 14.354 +/- 0.311 | 0.682 +/- 0.002 |
| w/o temporal encoder | 0.634 | 22.118 +/- 0.645 | 373.535 +/- 12.810 | 15.748 +/- 0.401 | 0.638 +/- 0.006 |
| w/o appliance decoders | 0.836 | 21.149 +/- 0.610 | 348.586 +/- 8.891 | 14.510 +/- 0.535 | 0.673 +/- 0.012 |
| w/o relation attention | 1.332 | 20.780 +/- 0.339 | 304.224 +/- 14.495 | 14.748 +/- 0.304 | 0.675 +/- 0.009 |

Frozen two-test-house equal-weight results:

| Variant | Params (M) | MAE (W) | ON MAE (W) | SAE (W) | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Complete | 1.373 | 9.125 +/- 0.140 | 242.938 +/- 5.348 | 5.584 +/- 0.174 | 0.796 +/- 0.008 |
| w/o multi-scale CNN | 1.372 | 9.222 +/- 0.180 | 241.755 +/- 8.760 | 5.660 +/- 0.173 | 0.795 +/- 0.001 |
| w/o feature encoder | 1.326 | 9.157 +/- 0.057 | 252.014 +/- 2.211 | 5.588 +/- 0.069 | 0.790 +/- 0.006 |
| w/o temporal encoder | 0.634 | 10.565 +/- 0.333 | 288.068 +/- 10.253 | 6.558 +/- 0.156 | 0.710 +/- 0.013 |
| w/o appliance decoders | 0.836 | 9.221 +/- 0.662 | 265.839 +/- 4.296 | 5.633 +/- 0.633 | 0.783 +/- 0.015 |
| w/o relation attention | 1.332 | 9.181 +/- 0.088 | 230.841 +/- 14.692 | 5.546 +/- 0.044 | 0.790 +/- 0.015 |

Interpretation: the temporal encoder is indispensable, and the complete
appliance decoders provide a clear ON-state and classification benefit. The
feature encoder has a smaller but repeatable ON-state benefit. Relation
attention trades a lower ON-MAE for higher macro-F1 when retained; critically,
REFIT20 microwave F1 is 0.549 with it and 0.496 without it. The multi-scale CNN
is nearly neutral in the aggregate summary, but its value is concentrated on
microwave generalisation: removing it increases UK-DALE5 validation microwave
ON-MAE from 589.491 to 696.445 W and decreases REFIT20 microwave F1 from 0.549
to 0.512.

Waveform audit: the full and single-scale models retain the known noisy-REFIT
fridge false plateaus, so multi-scale processing does not solve that failure.
However, removing the temporal encoder produces more prolonged false fridge
cycles and substantially delayed or missed microwave plateaus. Removing the
appliance decoders increases fridge fragmentation and microwave amplitude
error. The module evidence therefore supports retaining the five displayed
blocks, while describing the multi-scale and relational gains as
appliance-specific rather than universal.

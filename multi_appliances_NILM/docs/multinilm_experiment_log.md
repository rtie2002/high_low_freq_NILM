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

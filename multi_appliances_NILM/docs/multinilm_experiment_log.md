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

## Partial focal-event mix (probability 0.25) — planned

Experiment: `multinilm_focal_event_p025_stratified_mix_house_split`

Single change relative to the completed focal-event run: apply focal-event and
background-bin sampling to 25% of synthetic windows instead of 100%. The other
75% use the original full-mix sampling. Since synthetic windows remain 50% of
training, only 12.5% of all training windows receive forced focal sampling.

This retains explicit event/background coverage while reducing the double
positive-prior shift caused by focal oversampling plus BCE `pos_weight`.

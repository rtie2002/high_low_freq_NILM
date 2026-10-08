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

## Focal-event stratified full mix — planned

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

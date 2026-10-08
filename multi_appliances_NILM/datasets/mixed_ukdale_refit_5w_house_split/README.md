# Mixed UK-DALE + REFIT 5-week house split

## Protocol

- Train, active contiguous 5-week sub-blocks selected inside the established
  8-week intervals: UK-DALE 1; REFIT 3, 5, 9, 11.
- Validation, held-out complete 8-week blocks: UK-DALE 5; REFIT 2.
- Test, complete 8-week blocks: UK-DALE 2; REFIT 20.
- Sampling period: 8 seconds.
- Normalization: fitted on the five training houses only.

Train, validation, and test houses are disjoint. All house bounds come from
`mixed_ukdale_refit_8w/selection_summary.csv`; the training search is restricted
to those established intervals before choosing five weeks. All rows are then
extracted again from the current corrected original house CSVs. Therefore the
rows should not be assumed byte-identical to an older prepared 8-week dataset
if the original CSVs have since been corrected.

## Files

- `training/multi_appliance_training.csv`
- `validating/multi_appliance_validating.csv`
- `testing/multi_appliance_testing.csv`
- `testing/ukdale_house2/multi_appliance_testing.csv`
- `testing/refit_house20/multi_appliance_testing.csv`
- `normalization_stats.json`
- `selection_summary.csv`: per-house event and residual-background audit
- `selection_meta.json`: split protocol and source selection metadata

## Rebuild

```powershell
python scripts/prepare_mixed_ukdale_refit_5week_house_split.py
```

Use `config/experiment_mixed_ukdale_refit_5w_house_split.yaml` for training and
evaluation.

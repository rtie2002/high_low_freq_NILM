# Mixed UK-DALE + REFIT 10-week training split

This is a controlled training-duration ablation of
`mixed_ukdale_refit_5w_house_split`.

- Training houses: UK-DALE 1 and REFIT 3, 5, 9, 11.
- Each 10-week training interval contains the exact established 5-week core
  and adds approximately 2.5 weeks before and after it.
- Validation houses remain UK-DALE 5 and REFIT 2, using the exact baseline
  rows.
- Cross-house development scenarios remain UK-DALE 2 and REFIT 20, using the
  exact baseline rows.
- Normalization is re-fitted on the 10-week training rows only.

Generate the dataset from the project directory with:

```powershell
python scripts/prepare_mixed_ukdale_refit_5week_house_split.py `
  --train-weeks 10 `
  --out-dir datasets/mixed_ukdale_refit_10w_house_split
```

The generated CSV files are intentionally excluded from Git. Use
`config/experiment_mixed_ukdale_refit_10w_house_split.yaml` for training.

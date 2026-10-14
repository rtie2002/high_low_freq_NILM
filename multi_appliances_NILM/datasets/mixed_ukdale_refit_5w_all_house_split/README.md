# Mixed UK-DALE/REFIT all-5-week house split

Every house contributes one continuous five-week calendar block.

- Train: UK-DALE 1; REFIT 3, 5, 9, 11
- Validation: UK-DALE 5; REFIT 2
- Test: UK-DALE 2; REFIT 20

The training rows are bitwise identical to the established five-week baseline.
Validation and test are deterministic centred five-week crops of their earlier
eight-week intervals; event counts and model performance are not used to choose
the boundaries. Normalisation is fitted on training houses only.

Rebuild from the project directory with:

```powershell
python scripts/prepare_mixed_ukdale_refit_5week_house_split.py
```

Large CSV files are ignored by Git. `selection_summary.csv`,
`selection_meta.json`, and `normalization_stats.json` record the protocol and
must accompany a generated dataset.

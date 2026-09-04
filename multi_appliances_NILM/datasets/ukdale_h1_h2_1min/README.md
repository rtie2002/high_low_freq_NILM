# UK-DALE H1 to H2 1-Minute Scenario

This folder is for cross-house validation at 1-minute resolution.

- Training source: UK-DALE house 1
- Validation source: last `--val-fraction` of house 1
- Testing source: UK-DALE house 2

Build the CSVs from the repository root:

```powershell
python .\multi_appliances_NILM\scripts\prepare_ukdale_h1_train_h2_test_1min.py
```

Train MultiNILM:

```powershell
python .\multi_appliances_NILM\main.py --model multinilm --experiment .\multi_appliances_NILM\config\experiment_ukdale_h1_h2_1min.yaml --model-config .\multi_appliances_NILM\config\models\multinilm_ukdale_h1_h2_1min.yaml
```

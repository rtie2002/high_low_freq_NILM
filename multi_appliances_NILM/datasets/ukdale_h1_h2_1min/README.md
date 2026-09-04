# UK-DALE H1 to H2 1-Minute Scenario

This folder is for cross-house validation at 1-minute resolution. The CSVs are
created directly from raw UK-DALE `.dat` files, not from the existing 6-second
CSV scenario.

- Training source: UK-DALE house 1
- Validation source: last `--val-fraction` of house 1
- Testing source: UK-DALE house 2

Build the CSVs from the repository root:

```powershell
python .\multi_appliances_NILM\scripts\prepare_ukdale_h1_train_h2_test_1min.py
```

On the training device, if your repo is at `D:\Raymond\high_low_freq_NILM`, use:

```powershell
cd "D:\Raymond\high_low_freq_NILM"
python .\multi_appliances_NILM\scripts\prepare_ukdale_h1_train_h2_test_1min.py --raw-dir .\dataset_preprocess\UK_DALE\UKDALE2017
```

Train MultiNILM:

```powershell
python .\multi_appliances_NILM\main.py --model multinilm --experiment .\multi_appliances_NILM\config\experiment_ukdale_h1_h2_1min.yaml --model-config .\multi_appliances_NILM\config\models\multinilm_ukdale_h1_h2_1min.yaml
```

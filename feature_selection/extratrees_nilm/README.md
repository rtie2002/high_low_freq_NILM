# ExtraTrees NILM Wrapper Feature Selection

This folder contains the ExtraTrees-based wrapper feature-selection workflow for
multi-appliance NILM.

The experiment is designed to answer:

```text
Which aggregate low/high-frequency features are useful for appliance ON/OFF
classification and final appliance-power regression?
```

## One-Command Pipeline

Run steps 1-6 automatically:

```bash
python feature_selection/extratrees_nilm/run_extratrees_wrapper_pipeline.py
```

Dry-run the planned steps without starting the expensive experiments:

```bash
python feature_selection/extratrees_nilm/run_extratrees_wrapper_pipeline.py --dry-run
```

Reuse existing completed step outputs:

```bash
python feature_selection/extratrees_nilm/run_extratrees_wrapper_pipeline.py --skip-existing
```

The pipeline writes a timestamped folder under:

```text
feature_selection/results/extratrees_wrapper_pipeline_<dataset>_<timestamp>/
```

That folder contains logs, a `pipeline_manifest.json`, and a
`pipeline_summary.md`. The important graphs, CSV files, tuned parameters,
selected features, and final held-out test metrics are snapshotted into the
pipeline folder so later runs do not overwrite the experiment record.

## Manual Step Order

### 1. Classifier Hyperparameter Tuning

```bash
python feature_selection/extratrees_nilm/extratrees_hyperparameter_tuning.py
```

Objective:

```text
maximize validation Macro F1 for appliance ON/OFF prediction
```

### 2. Classifier Forward Feature Selection

```bash
python feature_selection/extratrees_nilm/extratrees_nilm_forward_selection_classification.py
```

This script loads the tuned classifier parameters when available and selects
features using validation Macro F1.

For the full pipeline, held-out classifier test reporting is disabled by default
to avoid exposing test results before the final regression stage. To run a
standalone classifier-only test report and ON/OFF evidence plots:

```bash
$env:EXTRATREES_ENABLE_CLASSIFIER_TEST_REPORT="1"
python feature_selection/extratrees_nilm/extratrees_nilm_forward_selection_classification.py
```

### 3. Regressor Hyperparameter Tuning

```bash
python feature_selection/extratrees_nilm/extratrees_regressor_hyperparameter_tuning.py
```

Objective:

```text
weighted composite of MAE, SAE, and EA
```

### 4. Classification-Assisted Regression Forward Selection

```bash
python feature_selection/extratrees_nilm/extratrees_nilm_forward_selection_classification_regression.py
```

This script automatically loads:

```text
classifier tuned parameters
classifier validation-selected feature subset
regressor tuned parameters
```

It compares:

```text
direct regression with direct-selected features:
    features -> appliance power

direct regression on assisted-selected features:
    features -> appliance power

classifier-assisted regression with assisted-selected features:
    features + predicted ON/OFF -> appliance power
```

Training ON/OFF inputs for the assisted regressor use out-of-fold classifier
predictions to avoid same-row stacking leakage.

## Split Discipline

The workflow uses:

```text
train 60%
validation 20%
test 20%
```

Feature selection and hyperparameter tuning use validation only. The held-out
test split is used only for final reporting, evidence plots, waveform plots, and
the final per-appliance MAE/SAE/EA table.

## Deep Learning Validation

Deep learning validation is step 7 and is intentionally separate. The selected
features from this wrapper pipeline should later be tested in the deep-learning
model against:

```text
low-frequency only
all features
selected features
selected high+low features
```

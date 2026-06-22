# ExtraTrees Wrapper Feature-Selection Pipeline for NILM

## 1. Overview

This folder implements an ExtraTrees-based wrapper feature-selection pipeline for
multi-appliance non-intrusive load monitoring (NILM). The purpose of this
pipeline is to identify which aggregate low-frequency and high-frequency
features are useful for appliance-level state detection and power
disaggregation.

The main research question is:

```text
Can selected aggregate low/high-frequency features improve multi-appliance NILM,
especially final appliance-power regression?
```

The pipeline treats appliance ON/OFF classification as an auxiliary task. The
final target remains appliance power regression, evaluated using NILM regression
metrics.

## 2. Experimental Design

The full dataset is split chronologically:

```text
train       60%
validation 20%
test       20%
```

The split is time-based rather than random because NILM data is time-series
data. Hyperparameter tuning and wrapper feature selection use only the training
and validation portions. The held-out test portion is reserved for the final
report only.

The pipeline contains four modelling scripts and one orchestration script:

```text
1. extratrees_hyperparameter_tuning.py
2. extratrees_nilm_forward_selection_classification.py
3. extratrees_regressor_hyperparameter_tuning.py
4. extratrees_nilm_forward_selection_classification_regression.py
5. run_extratrees_wrapper_pipeline.py
```

Together, these scripts implement steps 1-6 of the feature-selection experiment.
Deep-learning validation is a later step and is not included here.

## 3. Stage 1: Classifier Hyperparameter Tuning

Script:

```bash
python feature_selection/extratrees_nilm/extratrees_hyperparameter_tuning.py
```

Purpose:

```text
Find a suitable ExtraTreesClassifier configuration for appliance ON/OFF
prediction.
```

Input:

```text
aggregate low/high-frequency features
```

Targets:

```text
kettle_on
fridge_on
microwave_on
dishwasher_on
washingmachine_on
```

Selection metric:

```text
validation Macro F1
```

Method:

```text
Optuna proposes ExtraTreesClassifier hyperparameters.
Each trial trains on the training split.
Each trial is scored on the validation split.
The best validation Macro F1 configuration is saved.
```

Main outputs:

```text
feature_selection/results/extratrees_hyperparameter_tuning_onoff_<dataset>/
    best_hyperparameters.json
    hyperparameter_trials.csv
    optimization_history_macro_f1.png
    per_appliance_f1_by_trial.png
    runtime_vs_macro_f1.png
    hyperparameter_importance.png
    hyperparameter_slice_plots.png
```

Role in the pipeline:

```text
The selected classifier hyperparameters are loaded by the classifier feature
selection script and the classification-assisted regression script.
```

## 4. Stage 2: Classifier Forward Feature Selection

Script:

```bash
python feature_selection/extratrees_nilm/extratrees_nilm_forward_selection_classification.py
```

Purpose:

```text
Select features that are useful for predicting appliance ON/OFF states.
```

Input:

```text
aggregate low/high-frequency features
```

Targets:

```text
multi-appliance ON/OFF labels
```

Selection metric:

```text
validation Macro F1
```

Method:

```text
Start with no selected features.
At each round, try adding every remaining feature.
Train an ExtraTreesClassifier for each candidate subset.
Score each candidate subset on the validation split.
Keep the feature that gives the best validation Macro F1.
Repeat until all features are ranked.
```

Main outputs:

```text
feature_selection/results/extratrees_forward_selection_onoff_<dataset>/
    forward_selection_log.csv
    selected_features.txt
    forward_selection_macro_micro_f1.png
    forward_selection_per_appliance_f1.png
```

Important test discipline:

```text
During the full pipeline, classifier-only held-out test reporting is disabled.
This avoids exposing test results before the final regression stage.
```

For standalone classifier analysis only, enable test reporting with:

```powershell
$env:EXTRATREES_ENABLE_CLASSIFIER_TEST_REPORT="1"
python feature_selection\extratrees_nilm\extratrees_nilm_forward_selection_classification.py
```

Role in the pipeline:

```text
The validation-selected classifier feature subset is loaded by the final
classification-assisted regression script. It is used to generate predicted
ON/OFF features for the assisted regressor.
```

## 5. Stage 3: Regressor Hyperparameter Tuning

Script:

```bash
python feature_selection/extratrees_nilm/extratrees_regressor_hyperparameter_tuning.py
```

Purpose:

```text
Find a suitable ExtraTreesRegressor configuration for appliance-power
disaggregation.
```

Input:

```text
aggregate low/high-frequency features
```

Targets:

```text
kettle_power
fridge_power
microwave_power
dishwasher_power
washingmachine_power
```

Selection objective:

```text
weighted validation composite of MAE, SAE, and EA
```

The current weights are:

```text
MAE 40%
SAE 30%
EA  30%
```

Method:

```text
Optuna proposes ExtraTreesRegressor hyperparameters.
Each trial trains on the training split.
Each trial is scored on the validation split.
The best configuration is saved for the final regression feature-selection
stage.
```

Main outputs:

```text
feature_selection/results/extratrees_hyperparameter_tuning_regression_<dataset>/
    best_regressor_hyperparameters.json
    regressor_hyperparameter_trials.csv
    optimization_history_composite_score.png
    regression_metrics_by_trial.png
    runtime_vs_composite_score.png
    hyperparameter_importance.png
    hyperparameter_slice_plots.png
```

Role in the pipeline:

```text
The selected regressor hyperparameters are loaded by the final regression
feature-selection script.
```

## 6. Stage 4: Classification-Assisted Regression Feature Selection

Script:

```bash
python feature_selection/extratrees_nilm/extratrees_nilm_forward_selection_classification_regression.py
```

Purpose:

```text
Select regression features and compare direct regression against
classifier-assisted regression.
```

This script automatically loads:

```text
classifier tuned hyperparameters
classifier validation-selected feature subset
regressor tuned hyperparameters
```

### 6.1 Predicted ON/OFF Generation

The classifier branch produces predicted ON/OFF states that are used as
additional inputs for the assisted regressor.

For the training split, predicted ON/OFF states are generated using out-of-fold
prediction:

```text
Split training data into folds.
For each fold, train classifier on the other folds.
Predict ON/OFF states for the held-out fold.
Combine all held-out predictions.
```

This prevents same-row stacking leakage. The regressor does not train on ON/OFF
predictions made by a classifier that already trained on the same rows.

### 6.2 Assisted Regression Feature Selection

Assisted regression uses:

```text
selected aggregate features + predicted ON/OFF states -> appliance power
```

Feature selection objective:

```text
weighted validation rank composite of MAE, SAE, and EA
```

The assisted feature-selection log is saved as:

```text
classification_regression_forward_selection_log.csv
```

### 6.3 Direct Regression Feature Selection

Direct regression uses:

```text
selected aggregate features -> appliance power
```

A separate direct-only wrapper selection is performed so that the direct
baseline is not unfairly evaluated using features selected for the assisted
model.

The direct feature-selection log is saved as:

```text
direct_regression_forward_selection_log.csv
```

### 6.4 Final Held-Out Test

The final held-out test compares:

```text
direct_selected:
    direct regression using direct-selected features

direct_on_assisted_selected:
    direct regression using the assisted-selected feature subset

classifier_assisted_selected:
    assisted regression using assisted-selected features and predicted ON/OFF
```

The second model is included as a diagnostic same-feature comparison. The first
and third models are the fair direct-selected versus assisted-selected
comparison.

Main outputs:

```text
feature_selection/results/extratrees_forward_selection_classification_regression_<dataset>/
    classification_regression_forward_selection_log.csv
    direct_regression_forward_selection_log.csv
    selected_regression_features.txt
    final_regression_test_metrics_per_appliance.csv
    final_regression_test_predictions.csv
    regression_forward_selection_mae_sae_ea.png
    regression_forward_selection_composite_score.png
    regression_per_appliance_ea.png
    metric_curves/
    prediction_waveforms/
```

## 7. Stage 5: End-to-End Pipeline Runner

Script:

```bash
python feature_selection/extratrees_nilm/run_extratrees_wrapper_pipeline.py
```

Purpose:

```text
Run all ExtraTrees wrapper stages automatically and preserve the experiment
record.
```

The runner executes:

```text
1. classifier hyperparameter tuning
2. classifier forward feature selection
3. regressor hyperparameter tuning
4. classification-assisted and direct regression feature selection
5. final held-out test
6. output collection and snapshotting
```

Useful commands:

```bash
python feature_selection/extratrees_nilm/run_extratrees_wrapper_pipeline.py
python feature_selection/extratrees_nilm/run_extratrees_wrapper_pipeline.py --dry-run
python feature_selection/extratrees_nilm/run_extratrees_wrapper_pipeline.py --skip-existing
```

For final paper runs, avoid `--skip-existing` unless you are sure all previous
outputs are generated by the current code and dataset.

The pipeline writes a timestamped folder:

```text
feature_selection/results/extratrees_wrapper_pipeline_<dataset>_<timestamp>/
```

This folder contains:

```text
logs/
artifacts/
pipeline_manifest.json
pipeline_summary.md
```

The `artifacts/` folder snapshots important outputs from each stage so later
runs do not overwrite the experiment record.

## 8. Interpretation of the Pipeline

The experimental logic is:

```text
If high-frequency or engineered aggregate features improve ON/OFF
classification, and the predicted ON/OFF states improve final appliance-power
regression, then the selected features are meaningful for NILM.
```

The classification task is therefore not the final objective. It is an
interpretable auxiliary task used to support the regression task.

The final evidence should be based primarily on:

```text
final held-out test MAE
final held-out test SAE
final held-out test EA
per-appliance performance
direct vs classifier-assisted comparison
selected feature stability across runs/datasets
```

## 9. Remaining Step: Deep-Learning Validation

This ExtraTrees pipeline produces wrapper-selected features. The next research
stage is to test whether these selected features transfer to deep-learning NILM
models.

Suggested deep-learning comparisons:

```text
low-frequency only
all features
ExtraTrees-selected features
selected low-frequency + selected high-frequency features
```

The deep-learning validation is not implemented in this folder yet.

## 10. Methodological Notes

Current safeguards:

```text
time-based train/validation/test split
validation-only hyperparameter tuning
validation-only feature selection
held-out test used only for final reporting
out-of-fold classifier predictions for assisted-regression training
separate direct and assisted feature-selection paths
timestamped pipeline folder with output snapshots
```

Remaining limitations:

```text
single validation split may still overfit validation
blocked KFold out-of-fold prediction is offline, not strictly rolling-time
hyperparameters are tuned before feature selection
deep-learning validation is still required for the final research claim
```

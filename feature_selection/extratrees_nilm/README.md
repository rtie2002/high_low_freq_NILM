# ExtraTrees NILM Feature Selection

This folder contains the ExtraTrees-based NILM feature-selection workflow.

## 1. Classifier Hyperparameter Tuning

Run this when tuning the ON/OFF classifier:

```bash
python feature_selection/extratrees_nilm/extratrees_hyperparameter_tuning.py
```

Selection metric:

```text
Macro F1
```

## 2. Regressor Hyperparameter Tuning

Run this when tuning the appliance power regressor:

```bash
python feature_selection/extratrees_nilm/extratrees_regressor_hyperparameter_tuning.py
```

Selection objective:

```text
weighted composite of avg_nmae, avg_nrmse, avg_relative_energy_error, avg_sae, and avg_r2
```

Output used by the classification-regression forward-selection script:

```text
feature_selection/results/extratrees_hyperparameter_tuning_regression_<dataset_name>/best_regressor_hyperparameters.json
```

If this file is missing, the forward-selection script falls back to its default regressor parameters.

## 3. Classification-Regression Forward Feature Selection

Run this after tuning:

```bash
python feature_selection/extratrees_nilm/extratrees_nilm_forward_selection_classification_regression.py
```

The ON/OFF branch uses fixed classifier settings and selected classifier features. The regression branch uses tuned regressor parameters when the regressor tuning output exists.

## Workflow

```text
classifier tuning -> ON/OFF classifier settings
regressor tuning  -> power regressor settings
forward selection -> selected regression feature subset
```

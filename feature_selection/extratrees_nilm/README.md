# ExtraTrees NILM Feature Selection

This folder contains the ExtraTrees-based mini NILM workflow.

## 1. Hyperparameter Tuning

Run this first when you want to tune the ExtraTrees model automatically:

```bash
python feature_selection/extratrees_nilm/extratrees_hyperparameter_tuning.py
```

This script uses Optuna surrogate optimization to tune ExtraTrees hyperparameters on the validation set.

Outputs:

```text
feature_selection/results/extratrees_best_hyperparameters.json
feature_selection/results/extratrees_hyperparameter_trials.csv
```

If Optuna is not installed:

```bash
pip install optuna
```

## 2. Forward Feature Selection

Run this after hyperparameter tuning:

```bash
python feature_selection/extratrees_nilm/extratrees_nilm_forward_selection.py
```

This script reads the tuned hyperparameters if this file exists:

```text
feature_selection/results/extratrees_best_hyperparameters.json
```

If the JSON file does not exist, it uses default ExtraTrees parameters.

Outputs:

```text
feature_selection/results/extratrees_forward_selection_log.csv
feature_selection/results/extratrees_forward_selection_curve.png
feature_selection/results/extratrees_forward_selection_per_appliance.png
feature_selection/results/extratrees_selected_features.txt
```

## Workflow

```text
hyperparameter tuning
        ↓
best ExtraTrees parameters
        ↓
forward feature selection
        ↓
selected feature subset and F1 curves
```

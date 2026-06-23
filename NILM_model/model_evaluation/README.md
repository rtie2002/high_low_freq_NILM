# Model Evaluation

Reusable evaluation and plotting utilities for MATNILM, SGN, and future NILM models.

## What This Folder Provides

| File | Purpose |
|---|---|
| `metrics.py` | Common NILM metrics: MAE, SAE, and F1 |
| `plots.py` | Plot training curves and prediction waveforms |
| `runner.py` | Reusable NILM train/evaluate/inference loop |
| `plot_training.py` | CLI for training loss/metric plots |
| `plot_predictions.py` | CLI for aggregate/true/predicted waveform plots |
| `compute_metrics.py` | CLI for shared MAE/SAE/F1 CSV output |

## Expected Training History Format

Any model can write a CSV like:

```text
epoch,train_loss,val_loss,val_mae,val_sae,val_f1
0,0.52,0.61,34.2,0.12,0.71
1,0.48,0.57,30.5,0.10,0.76
```

Plot it:

```powershell
python -m model_evaluation.plot_training ^
  --history baseline/SGN/runs/sgn_redd/history_fridge.csv ^
  --output baseline/SGN/runs/sgn_redd/history_fridge.png ^
  --title "SGN Fridge Training"
```

## Expected Prediction CSV Format

Any model can write a prediction CSV like:

```text
readable_time,aggregate,fridge_power,pred_fridge_power,microwave_power,pred_microwave_power
2013-06-10 00:00:00,185.7,10.0,8.5,0.0,0.2
```

Plot waveform:

```powershell
python -m model_evaluation.plot_predictions ^
  --predictions predictions/test_predictions.csv ^
  --output predictions/test_waveforms.png ^
  --pair fridge:fridge_power:pred_fridge_power ^
  --pair microwave:microwave_power:pred_microwave_power ^
  --start 0 ^
  --samples 2000
```

Compute the shared metric table:

```powershell
python -m model_evaluation.compute_metrics ^
  --predictions predictions/test_predictions.csv ^
  --output predictions/test_metrics.csv ^
  --pair fridge:fridge_power:pred_fridge_power ^
  --pair microwave:microwave_power:pred_microwave_power
```

The plot shows:

1. aggregate household power
2. true appliance power
3. predicted appliance power

## Python API

```python
import pandas as pd
from model_evaluation.metrics import compute_metrics_table
from model_evaluation.plots import plot_training_history, plot_prediction_waveforms
from model_evaluation.runner import train_nilm_model, run_nilm_inference

history = pd.read_csv("history.csv")
plot_training_history(history, "history.png")

pred = pd.read_csv("predictions.csv")
plot_prediction_waveforms(
    pred,
    "waveforms.png",
    time_col="readable_time",
    aggregate_col="aggregate",
    true_pred_pairs={
        "fridge": ("fridge_power", "pred_fridge_power"),
        "microwave": ("microwave_power", "pred_microwave_power"),
    },
)

metrics = compute_metrics_table(
    pred,
    {
        "fridge": ("fridge_power", "pred_fridge_power"),
        "microwave": ("microwave_power", "pred_microwave_power"),
    },
)
print(metrics)
```

Metric CSV output is always:

```text
appliance,mae,sae,f1
```

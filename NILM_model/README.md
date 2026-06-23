# NILM Model Pipeline

This folder is the common experiment area for NILM models.

Use `main.py` as the shared entry point:

```powershell
cd "D:\Raymond\high_low_freq_NILM\NILM_model"
python main.py --model sgn --mode train
python main.py --model sgn --mode inference --checkpoint runs/sgn_redd/best_all.pt
python main.py --model sgn --mode train_inference
```

For SGN, the default target is multi-appliance output:

```text
baseline/SGN/configs/sgn_paper.json -> "default_appliance": "all"
```

So this trains one multi-output SGN checkpoint:

```powershell
python main.py --model sgn --mode train_inference --data_source csv --csv_config baseline/SGN/configs/training_data_house2.json --model_config baseline/SGN/configs/sgn_paper.json
```

Single-appliance runs are only for debugging or ablation:

```powershell
python main.py --model sgn --mode train --appliance fridge
```

The universal training and inference engine lives in:

```text
model_evaluation/runner.py
```

The model adapter that connects SGN to the universal engine lives in:

```text
models/sgn_pipeline.py
```

Each model folder should provide only model-specific pieces:

```text
model architecture
loss function
dataset adapter
hyperparameter config
```

So the clean idea is:

```text
main.py
  selects model and mode

models/
  contains model adapters, for example sgn_pipeline.py

model_evaluation/
  contains reusable training, inference, metrics, and plots

baseline/SGN/
  contains only SGN-specific model, loss, dataset, and config
```

Current supported model:

```text
sgn
```

# SGN Baseline for NILM

This folder implements **Subtask Gated Networks (SGN)** from:

> C. Shin, S. Joo, J. Yim, H. Lee, T. Moon, and W. Rhee, "Subtask Gated Networks for Non-Intrusive Load Monitoring," AAAI 2019.

The implementation is adapted to the processed REDD pickle files already shipped with `baseline/MATNILM/data/redd`.

## Folder Layout

```text
baseline/SGN/
  train.py              # compatibility wrapper for models/sgn_pipeline.py
  inference.py          # compatibility wrapper for models/sgn_pipeline.py
  requirements.txt
  ../model_evaluation/  # shared MAE/SAE/F1 metrics and plotting utilities
    runner.py           # universal NILM train/evaluate/inference loop
  configs/
    sgn_paper.json      # SGN hyperparameters
    training_data_house2.json
  sgn/
    config.py           # hyperparameters and appliance definitions
    data.py             # REDD pickle dataset/windowing
    model.py            # SGN model architecture
    losses.py           # SGN loss function: MSE(gated output) + BCE(on/off)
```

## Model

This implementation uses SGN as a multi-appliance, multi-label NILM model. One forward pass predicts power and ON/OFF labels for all selected appliances:

```text
input  -> aggregate/features window
output -> appliance power sequences [appliance, time]
output -> appliance ON/OFF sequences [appliance, time]
```

The model keeps the SGN paper idea: two CNN Seq2Seq subnetworks:

```text
Regression subnetwork      -> appliance power estimates
Classification subnetwork  -> appliance ON/OFF probabilities
Final output               -> regression * ON/OFF probability
```

## Loss Function

The loss is implemented in `sgn/losses.py`:

```text
L = MSE(gated_power, true_power) + BCE(on_probability, true_on_off)
```

The on/off label uses the SGN paper threshold:

```text
on = 1 if appliance_power > 15 W else 0
```

## Environment

For an RTX 4090 machine, install a modern CUDA PyTorch build first, then the remaining packages.

```powershell
conda create -n sgn python=3.10 -y
conda activate sgn

python -m pip install --upgrade pip
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Smoke Test

Recommended universal entry point:

```powershell
cd "D:\Raymond\high_low_freq_NILM\NILM_model"
python main.py --model sgn --mode train --debug --appliance dishwasher
```

You can also run directly from this folder:

```powershell
cd "D:\Raymond\high_low_freq_NILM\NILM_model\baseline\SGN"
python train.py --debug --appliance dishwasher --epochs 2
```

## Original SGN-Paper-Like Baseline

This follows the SGN paper settings as closely as possible. For processed REDD pickle files, use the REDD paper window:

```text
input length  = 864
output length = 64
batch size    = 16
learning rate = 0.0001
normalization = divide by std of aggregate training power
on threshold  = 15 W
optimizer     = Adam
SAE period    = 1200 REDD samples
```

For the merged CSV data in this project, `sampling_seconds` is 6, so the pipeline defaults to the UK-DALE paper window:

```text
input length  = 432
output length = 32
eval stride   = 32
```

```powershell
cd "D:\Raymond\high_low_freq_NILM\NILM_model"
python main.py --model sgn --mode train --model_config baseline/SGN/configs/sgn_paper.json
```

By default, `configs/sgn_paper.json` sets `"default_appliance": "all"`. In SGN paper mode, this trains separate appliance-specific checkpoints:

```text
runs/sgn_redd/best_dishwasher.pt
runs/sgn_redd/best_fridge.pt
runs/sgn_redd/best_kettle.pt
runs/sgn_redd/best_microwave.pt
runs/sgn_redd/best_washingmachine.pt
```

The train/evaluate/inference loop is shared in:

```text
../model_evaluation/runner.py
```

SGN-specific code only builds the SGN config, dataset, model, loss, and optimizer. To change SGN hyperparameters, edit:

```text
configs/sgn_paper.json
```

Command-line values still override the JSON when provided:

```powershell
python main.py --model sgn --mode train --model_config baseline/SGN/configs/sgn_paper.json --epochs 50 --batch_size 32
```

## SGN Variants From the Paper

Plain soft SGN:

```powershell
python train.py --preset sgn_paper --gate_mode soft
```

SGN-sp, with learnable standby power:

```powershell
python train.py --preset sgn_paper --gate_mode soft --standby_power
```

Hard SGN:

```powershell
python train.py --preset sgn_paper --gate_mode hard
```

Hard SGN-sp:

```powershell
python train.py --preset sgn_paper --gate_mode hard --standby_power
```

## Optional MATNILM-Compatible Comparison

Only use this if you specifically want to compare under the released MATNILM code defaults instead of the original SGN paper defaults:

```powershell
python train.py --preset matnilm --appliance all --epochs 200
```

Outputs are written to:

```text
runs/<run_name>/
```

The paper-style all-appliance run writes one checkpoint and history per appliance:

```text
best_<appliance>.pt
metrics_<appliance>.json
history_<appliance>.csv
history_<appliance>.png
loss_detail_<appliance>.csv
loss_detail_<appliance>.png
```

Metrics and plots are produced by the shared package:

```text
../model_evaluation/
```

This keeps SGN, MATNILM, and future NILM models on the same MAE, SAE, F1, training-loss, and waveform plotting code.

## Train On Project `training_data` CSV

The SGN trainer can also use the merged CSV in:

```text
../../training_data/multi_appliance_house2_wk24_to_wk31_merged.csv
```

The CSV setup is controlled by:

```text
configs/training_data_house2.json
```

By default it uses only the aggregate column as the SGN input feature:

```json
"feature_columns": [
  "aggregate"
]
```

To use selected engineered features, edit `feature_columns`, for example:

```json
"feature_columns": [
  "aggregate",
  "P_active",
  "I_rms",
  "PF",
  "THDI"
]
```

Available CSV appliances:

```text
kettle, fridge, microwave, dishwasher, washingmachine
```

Quick debug run:

```powershell
cd "D:\Raymond\high_low_freq_NILM\NILM_model\baseline\SGN"
python train.py --data_source csv --csv_config configs/training_data_house2.json --debug --appliance fridge
```

Full CSV run for one appliance:

```powershell
python train.py --data_source csv --csv_config configs/training_data_house2.json --appliance fridge --epochs 200
```

Full CSV paper-style run for all CSV appliances. This trains one model per appliance:

```powershell
cd "D:\Raymond\high_low_freq_NILM\NILM_model"
python main.py --model sgn --mode train --data_source csv --csv_config baseline/SGN/configs/training_data_house2.json --model_config baseline/SGN/configs/sgn_paper.json
```

The CSV file has a 6-second sampling interval, so the pipeline uses the same real-time window length as REDD's `864/64` at 3 seconds by default:

```powershell
python train.py --data_source csv --csv_config configs/training_data_house2.json --appliance fridge --epochs 200
```

## Inference

```powershell
cd "D:\Raymond\high_low_freq_NILM\NILM_model"
python main.py --model sgn --mode inference --checkpoint runs/sgn_redd/best_fridge.pt
```

For CSV-trained checkpoints:

```powershell
python main.py --model sgn --mode inference --data_source csv --csv_config baseline/SGN/configs/training_data_house2.json --checkpoint runs/sgn_redd/best_fridge.pt
```

Train and then immediately run inference:

```powershell
python main.py --model sgn --mode train_inference --data_source csv --csv_config baseline/SGN/configs/training_data_house2.json --model_config baseline/SGN/configs/sgn_paper.json
```

Inference writes these files to the selected output directory:

```text
<split>_<appliance>_predictions.npz
<split>_<appliance>_predictions.csv
<split>_<appliance>_metrics.json
<split>_<appliance>_waveforms.png
```

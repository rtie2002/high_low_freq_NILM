# SGN Baseline for NILM

This folder implements **Subtask Gated Networks (SGN)** from:

> C. Shin, S. Joo, J. Yim, H. Lee, T. Moon, and W. Rhee, "Subtask Gated Networks for Non-Intrusive Load Monitoring," AAAI 2019.

The implementation is adapted to the processed REDD pickle files already shipped with `baseline/MATNILM/data/redd`.

## Folder Layout

```text
baseline/SGN/
  train.py              # train one appliance or all appliances
  inference.py          # load a checkpoint and write predictions/metrics
  requirements.txt
  sgn/
    config.py           # hyperparameters and appliance definitions
    data.py             # REDD pickle dataset/windowing
    model.py            # SGN model architecture
    losses.py           # SGN loss function: MSE(gated output) + BCE(on/off)
    metrics.py          # MAE, SAE, F1
```

## Model

SGN is a single-appliance model. Train one model for each target appliance:

```text
dishwasher, fridge, microwave, washer_dryer
```

The model has two CNN Seq2Seq subnetworks:

```text
Regression subnetwork      -> appliance power estimate
Classification subnetwork  -> appliance on/off probability
Final output               -> regression * on_probability
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

Run from this folder:

```powershell
cd "D:\Raymond\high_low_freq_NILM\baseline\SGN"
python train.py --debug --appliance dishwasher --epochs 2
```

## Original SGN-Paper-Like Baseline

This is the default. It follows the SGN paper settings as closely as possible using the processed REDD pickle files available in this repository:

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

```powershell
python train.py --appliance all --epochs 200
```

This is equivalent to:

```powershell
python train.py --preset sgn_paper --appliance all --epochs 200
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

Each appliance gets:

```text
best_<appliance>.pt
metrics_<appliance>.json
history_<appliance>.csv
```

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
cd "D:\Raymond\high_low_freq_NILM\baseline\SGN"
python train.py --data_source csv --csv_config configs/training_data_house2.json --debug --appliance fridge
```

Full CSV run for one appliance:

```powershell
python train.py --data_source csv --csv_config configs/training_data_house2.json --appliance fridge --epochs 200
```

Full CSV run for all CSV appliances:

```powershell
python train.py --data_source csv --csv_config configs/training_data_house2.json --appliance all --epochs 200
```

The CSV file has a 6-second sampling interval. If you want the same real-time window length as REDD's `864/64` at 3 seconds, use:

```powershell
python train.py --data_source csv --csv_config configs/training_data_house2.json --appliance fridge --input_length 432 --output_length 32 --epochs 200
```

## Inference

```powershell
python inference.py --checkpoint runs/sgn_redd/best_dishwasher.pt --appliance dishwasher
```

This writes predictions and metrics to the selected output directory.

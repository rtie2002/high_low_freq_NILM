# Multi-Appliance NILM

Unified experiment framework for comparing NILM models on the **same splits** and **same metrics**, while each model keeps its own architecture, windows, loss, and dataset logic.

## Folder layout

```text
multi_appliances_NILM/
├── main.py                     # CLI entry
├── runner.py                   # train + evaluate loops
├── config/
│   ├── experiment.yaml         # data paths, CSV columns, evaluation
│   └── models/                 # per-model: architecture, windows, training
│       └── unet_nilm.yaml
│       └── mat_nilm.yaml
│   experiment_redd.yaml        # optional 4-appliance REDD experiment
├── adapters/                   # model glue + shared dataloader
│   ├── config.py               # load YAML configs
│   ├── common.py               # shared helpers (scaling, PredictionBundle, DataLoader)
│   ├── dataloader.py           # CSV load, split, windowing
│   ├── types.py                # PredictionBundle (standard test output)
│   ├── unet_nilm.py            # UNet plug-in (model + loss + data)
│   └── mat_nilm.py             # MATNILM plug-in
├── model/                      # neural network only
│   ├── UNETNILM.py
│   ├── UNETNILM_loss.py
│   ├── MATNILM.py
│   └── MATNILM_loss.py
├── datasets/                   # YOUR CSV files (gitignored)
│   └── ukdale/
│       ├── training/
│       ├── validating/
│       └── testing/
├── evaluation/                 # metrics & cross-model comparison
│   ├── metrics.py
│   └── compare.py
│
# Not in repo (created when you run train / add data):
#   datasets/ukdale/training/data.csv
#   datasets/ukdale/validating/data.csv
#   datasets/ukdale/testing/data.csv
#   runs/<experiment>/<model>/     ← checkpoints & metrics (gitignored)
```

## Design rules

| Shared across models | Model-specific |
|---------------------|----------------|
| Experiment config (`config/experiment.yaml`) | Architecture (`model/`) |
| Metrics (`evaluation/`) | Loss (`model/*_loss.py`) |
| Train/eval loop (`runner.py`) | Windowing (`config/models/`) |
| Data load (`adapters/dataloader.py`) | Adapter (`adapters/`) |
| Prediction export format (`PredictionBundle`) | Training hyperparams (`config/models/`) |
| Run output layout (`runs/`) | |

**Fair comparison:** models may differ internally, but every adapter exports the same `PredictionBundle` on the same test timesteps. Metrics are computed from that bundle only.

## Data (CSV)

Put your preprocessed CSV in each folder under `datasets/ukdale/`:

```text
datasets/ukdale/
  training/data.csv
  validating/data.csv
  testing/data.csv
```

Each file: one row per timestep (6 s). Values should already be **preprocessed** (normalized mains/power, binary states).

Required columns (names in `config/experiment.yaml` under `csv.appliances`):

```text
aggregate,                    # csv.mains_column (model yaml can override)
kettle_power, kettle_state,
fridge_power, fridge_state,
dishwasher_power, dishwasher_state,
washingmachine_power, washingmachine_state,
microwave_power, microwave_state
```

You split the data yourself — the framework just loads `training/`, `validating/`, and `testing/` CSVs.

**Training outputs** — created automatically on first `train` / `evaluate`:

```text
runs/ukdale_h1_temporal/unet_nilm/
  best.pt
  history.csv / history.json
  loss_detail.csv
  live_training_loss.png
  live_loss_components.png
  waveforms/
    validation/live/epoch_001/kettle/on_01_t1234.png   ← 5 random ON periods each
    validation/live/epoch_001/fridge/...
    validation/best/epoch_003/kettle/best_01_t5678.png
    test/live/...
  test_predictions.npz
  test_metrics.csv
  waveforms/test/kettle/on_01_t....png   ← after evaluate
```

All models share `adapters/dataloader.py`. Each model only sets `windowing` in its yaml
(input/output length, alignment, stride).

## Quick start

```powershell
cd multi_appliances_NILM

# Train UNet-NILM (UK-DALE, 5 appliances)
python main.py --model unet_nilm --mode train

# Train MATNILM (REDD, 4 appliances — use experiment_redd.yaml)
python main.py --model mat_nilm --mode train --experiment config/experiment_redd.yaml

# Or override data folder
python main.py --model unet_nilm --mode train --data-path path/to/csv/folder

# Evaluate checkpoint on test split
python main.py --model unet_nilm --mode evaluate ^
  --experiment config/experiment.yaml ^
  --model-config config/models/unet_nilm.yaml ^
  --checkpoint runs/ukdale_h1_temporal/unet_nilm/best.pt

# Compare all models under one experiment
python main.py --mode compare --experiment ukdale_h1_temporal
```

## Adding a new model

1. Add `model/YourModel.py` and `model/YourModel_loss.py`
2. Add `config/models/your_model.yaml`
3. Add `adapters/your_model.py` — reuse `adapters/dataloader.py` and `adapters/common.py`
4. Register in `main.py` → `MODELS`

Shared pieces (do not duplicate per model):

| Module | Reuse for |
|--------|-----------|
| `adapters/dataloader.py` | CSV load, windowing, optional `training_targets: full_input` |
| `adapters/common.py` | `AdapterDataMixin`, power scaling, `build_prediction_bundle` |
| `adapters/config.py` | YAML merge; optional `data.appliances` override in model yaml |
| `runner.py` | Train/eval loop |
| `evaluation/metrics.py` | MAE, SAE, F1 from `PredictionBundle` |

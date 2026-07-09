# Multi-Appliance NILM

One training pipeline. Switch **dataset** and **model** via YAML — no code changes.

## How it works

```text
--experiment config/experiment_*.yaml   → which CSVs, columns, normalization
--model + config/models/*.yaml          → window size, stride, architecture, training
         ↓
    main.py → adapter (model glue) → runner.py (train/eval loop)
         ↓
    adapters/dataloader.py (shared CSV → windows → normalize)
         ↓
    evaluation/ (metrics + plots from PredictionBundle)
```

| Layer | Controls |
|-------|----------|
| `config/experiment*.yaml` | Dataset: paths, appliances, z-score stats, eval thresholds |
| `config/models/*.yaml` | Model: `windowing`, `architecture`, `loss`, `training` |
| `adapters/*.py` | Per-model forward + loss + prediction format |
| `runner.py` | Shared epochs, checkpointing, live plots |

## Project layout

```text
multi_appliances_NILM/
├── main.py                 # CLI
├── runner.py               # train + evaluate loops
├── config/
│   ├── experiment.yaml         # UK-DALE (5 apps)
│   ├── experiment_redd.yaml    # REDD (4 apps)
│   ├── experiment_refit.yaml   # REFIT (5 apps)
│   └── models/
│       ├── multinilm.yaml      # any N appliances
│       └── mat_nilm.yaml       # fixed 4-appliance MATNILM
├── adapters/
│   ├── config.py           # load + merge YAML
│   ├── dataloader.py       # CSV, normalization, windows
│   ├── common.py           # BaseNILMAdapter, PredictionBundle
│   ├── multinilm.py
│   └── mat_nilm.py
├── model/                  # nn.Module + loss only
├── evaluation/             # metrics, plots, compare
└── datasets/               # your CSVs (gitignored)
```

## Switch dataset

```powershell
# UK-DALE (default)
python main.py --model multinilm --mode train

# REDD
python main.py --model mat_nilm --mode train --experiment config/experiment_redd.yaml

# REFIT
python main.py --model multinilm --mode train --experiment config/experiment_refit.yaml

# Custom data folder
python main.py --model multinilm --mode train --data-path D:\my\csv\folder
```

## Switch model / hyperparameters

Each model yaml owns its windowing and training settings:

```yaml
windowing:
  input_window_length: 864
  output_window_length: 64
  input_stride: 32        # train stride
  eval_stride: 64         # validation/test stride
  training_targets: output_window   # or full_input (MATNILM train mode)
```

Override with `--model-config path/to/custom.yaml`.

Optional appliance subset in model yaml:

```yaml
data:
  appliances: [fridge, microwave]   # must match experiment csv.appliances keys
```

## CSV format

Place pre-split files under `datasets/<name>/{training,validating,testing}/`.
Filenames are set in `experiment_*.yaml` → `csv.train_file`, etc.

Required columns (per experiment `csv.appliances`):

```text
aggregate                    # mains (csv.mains_column)
kettle_power, kettle_on      # example appliance pair
fridge_power, fridge_on
...
```

Values: raw watts + binary ON/OFF state. Normalization is applied in `dataloader.py` using `normalization:` stats in the experiment yaml.

## Evaluate and compare

```powershell
python main.py --model multinilm --mode evaluate ^
  --experiment config/experiment.yaml ^
  --checkpoint runs/ukdale_cross_house_5w/multinilm/best.pt

python main.py --mode compare --experiment config/experiment.yaml
```

## Add a new model

1. `model/YourModel.py` + `model/YourModel_loss.py`
2. `config/models/your_model.yaml` with `model_name: your_model`
3. `adapters/your_model.py` extending `BaseNILMAdapter`
4. Register in `main.py` → `MODELS`

Reuse as-is: `dataloader.py`, `common.py`, `runner.py`, `evaluation/`.

## Model notes

- **MultiNILM**: supports any appliance count from experiment yaml.
- **MATNILM**: architecture is fixed at **4 appliances**; use `data.appliances` in `mat_nilm.yaml` or a 4-app experiment (e.g. REDD).

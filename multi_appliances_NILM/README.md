# Multi-appliance NILM

```text
CSV  →  data/  →  model/  →  loss  →  runner.py  →  evaluation/
         ↑
      config/*.yaml
```

Live model: **MultiNILM fractional + relation attention**.

```powershell
python main.py --mode train_evaluate --model multinilm_fractional --experiment config/experiment_ukdale.yaml --model-config config/models/multinilm_fractional_relational.yaml
```

## Folders

| Folder | Open this when you want… |
|--------|--------------------------|
| `config/` | YAML: houses/CSVs (`experiment_*.yaml`) and model knobs (`models/*.yaml`). Loader is `config/__init__.py`. |
| `data/` | CSV → windows → z-score → `PredictionBundle` |
| `model/` | Networks **and** their train/eval glue. Start at `MultiNILM.py` `forward`. |
| `evaluation/` | Metrics, plots, calibration |
| `scripts/` | Dataset prep only |
| `docs/` | Equation notes |

There is no `adapters/` folder. `main.py` picks a class from `model/`. `runner.py` is the shared epoch loop.

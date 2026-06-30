# UNet-NILM Paper Reproduction

Reproduce [Faustine et al. 2020](UNet-NILM_A Deep Neural Network for Multi-tasks Appliances State Detection and Power Estimation in NILM.pdf) on bundled UK-DALE data.

## What is included

| Item | Location |
|------|----------|
| Preprocessed data | `data.zip` → extract to `data/ukdale/` |
| Author metrics | `results.zip` → `results/ukdale_UNETNiLM_quantilesresults.npy` |
| Training code | `src/experiment.py`, `src/net/` |

**Paper setup:** UK-DALE House 1 (Jan–Mar 2015), 5 appliances, 6 s sampling, multi-appliance joint model.

## Quick start

```powershell
cd NILM_model/baseline/UNETNILM
pip install -r requirements.txt

# 1) Extract preprocessed data (~950 MB)
python run_reproduce.py --extract-data

# 2) Verify author F1 matches paper Table 1 (no training)
python run_reproduce.py --verify-results

# 3) Smoke test (1 epoch, subset)
python run_reproduce.py --train --epochs 1 --sample 5000

# 4) Full reproduction (~50 epochs, GPU recommended)
python run_reproduce.py --train --epochs 50
```

## Expected results (Table 1, UNet-NILM)

| Metric | Paper (approx.) |
|--------|-----------------|
| F1-macro | 0.941 |
| F1 per app | KT 0.956, FRZ 0.962, DW 0.909, WM 0.963, MW 0.916 |
| MAE avg | ~11 W |

Author saved results in `results.zip` already show **F1-macro = 0.9412**.

## Notes

- Code loads `data/ukdale/training/*.npy` and splits 60% / 25% / 15% train/val/test internally.
- Default input: `noise_inputs.npy` (processed mains). Use `--denoise` for `denoise_inputs.npy`.
- No pretrained `.ckpt` is bundled; full retrain may differ slightly from saved `results.zip`.
- Run from `UNETNILM/` using `run_reproduce.py` (adds `src/` to path automatically).

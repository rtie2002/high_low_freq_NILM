# MATUDA publishable experiment design (owned by agent)

## Protocol (locked)
- Dataset: UK-DALE low-freq, appliances = kettle, fridge, dishwasher, WM, microwave
- Houses: **H1+H5 labeled → H2 unlabeled aggregates → H2 test** (never H3/H4)
- Task: joint multi-label ON/OFF + multi-appliance power (seq2point T=599)
- Selection: source validation `mae - f1` (no H2 labels in training objective)
- Success (minimum to claim transfer works):
  1. H2 macro-F1(EGC) ≥ H2 macro-F1(Source-Only) + **0.05**
  2. H2 SAE(EGC) ≤ 2× SAE(Source-Only) **or** explicit energy trade-off analysis
  3. Per-app P/R/F1/MAE/SAE reported; dishwasher failure analyzed if F1=0
  4. Then: 3 seeds mean±std, then MultiNILM-DA / transfer baselines, then cross-dataset

## Critical eval bug (fixed)
Pipeline `eval_reconstruction: flat` cast state probabilities with `astype(int32)` → all zeros → reported H2 F1=0 while training `val_f1` looked OK. Fixed in `adapters/common.py` (threshold before cast) and MATUDA `predict_dataloader` (binary ON).

## Run order
1. Re-eval v1 `best.pt` with fixed metrics
2. **v3** (`matuda_v3.yaml`): Hur 2-stage (30-ep source-only warmup) + confident target PL + EGC-DA
3. Source-Only / Global FC-UDA baselines
4. Scoreboard → keep or redesign

## Literature levers in v3
- Hur et al. Sensors 2022: confident multi-label pseudo-labels + domain stabilization after source training
- AHDA / Lin: conditional MMD+CORAL on FC layers (EGC)
- D’Incecco: learn a usable classifier before transfer (warmup)

## Current status
- Fix + v3 launching on training PC; judge only **fixed** H2 F1/SAE

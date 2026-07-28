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

## Run order (auto loop)
1. Current v1 EGC (already running) — finish for reference
2. **v2 EGC** (`matuda_v2.yaml`): OFF-norm gate + ON-masked MSE + stronger state weight
3. Source-Only (same backbone)
4. Global FC-UDA (same backbone)
5. Scoreboard → keep or redesign

## Model improvements already applied for v2
- Power gate blends to z-score OFF-norm (fixes constant mean-W when OFF)
- ON-masked power MSE (align regression with events)
- `state_weight: 1.5` for better detection under imbalance

## Current status (live)
- **v1 EGC** running in pipeline (~epoch 43/120); auto-loop waits then runs v2+baselines
- Source val F1 ~0.34 mid-run is a watch item (standalone MATUDA_NILM reached ~0.8); final H2 metrics decide keep vs redesign
- Scoreboard: `runs/_auto_loop/SCOREBOARD.md` after jobs finish

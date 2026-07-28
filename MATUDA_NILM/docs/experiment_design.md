# MATUDA experiment design (locked protocol + claims)

Inspired by Lin (TSG 2022), Liu (TII 2022), D’Incecco (TSG 2020), and multi-appliance
transfer papers (Li AE 2022; Sun TIM 2025). **Not** MultiNILM.

## Research claim (what the paper argues)

> A **single** shared network can jointly perform **multi-label appliance-state
> detection** and **multi-appliance power estimation**, and can be transferred to a
> new building (and later a new dataset) using **unsupervised** domain adaptation
> that aligns **fully connected** embeddings (MMD+CORAL), without target submeter labels.

Staged evidence (publishable narrative):

| Stage | Setting | Claim supported |
|-------|---------|-----------------|
| **S1** Cross-building UDA | UK-DALE H1+H5 → H2 | Multi-appliance multi-label UDA is feasible / fails honestly |
| **S2** Ablation | λ, μ, FC layers, gate | Domain loss must sit on FC tower (Lin/DAN/Deep CORAL) |
| **S3** Cross-dataset | UK-DALE ↔ REDD / REFIT | Harder shift; same multi-head + FC-UDA recipe |

## Metrics (paper-standard)

Per appliance + macro/micro:

| Task | Metric |
|------|--------|
| State (multi-label) | Precision, Recall, **F1** |
| Power (multi-target) | **MAE** (W), **SAE** (energy), optional RMSE |

Report **Δ vs source-only** so negative transfer is visible (Muaz et al.).

## Protocol S1 (now)

**Source (labeled):** UK-DALE Houses 1+5 — `training/` + `validating/` CSVs  
**Target UDA (aggregates only):** UK-DALE House 2 — `testing/` CSV (labels ignored in loss)  
**Eval:** House 2 with labels (metrics only)  
**Forbidden:** UK-DALE H3/H4 as fridge/dishwasher/clean WM sources  

Appliances (K=5): kettle, fridge, dishwasher, washing machine, microwave.

| ID | Name | λ | Notes |
|----|------|---|-------|
| B0 | Source-only | 0 | Lower reference for target |
| B1 | FC MMD+CORAL | 1.0 (warmup 5) | Additive \(L_{sup}+\lambda L_{domain}\); L2-norm FC feats |
| M0 | Selective λ_k | sched. | Stage-2 after B0/B1 |
| U1 | Few-label FT | — | Upper bound (optional) |

Model selection: **source validation MAE** (never tune on H2).

## Protocol S3 (next, after S1)

Cross-dataset intersection appliances:

| Transfer | Source | Target | Apps |
|----------|--------|--------|------|
| UK→REDD | UK-DALE H1+H5 | REDD houses | fridge, dishwasher, microwave, WM |
| UK→REFIT | UK-DALE H1+H5 | REFIT houses | 5-app set where available |
| REDD→UK | REDD | UK H2 | 4-app intersection |

Same unsupervised rule: target aggregates only during adaptation.

## Implementation

```text
C:\Users\PC\anaconda3\envs\nilm\python.exe scripts\train_matuda.py --config configs/...
```

Logs: `results/<experiment_id>/history.json`, `best.pt`, `summary.json`.

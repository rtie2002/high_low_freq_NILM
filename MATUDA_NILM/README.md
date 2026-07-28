# MATUDA_NILM

**M**ulti-**A**ppliance multi-**T**ask **U**nsupervised **D**omain **A**daptation for NILM.

## Paper experiments (only these)

| ID | Config | Result dir |
|----|--------|------------|
| B0 | `configs/matuda_s2_b0.yaml` | `results/matuda_s2_b0_source_only/` |
| B1 | `configs/matuda_s2_b1.yaml` | `results/matuda_s2_b1_fc_uda/` |
| M0 | `configs/matuda_s2_m0.yaml` | `results/matuda_s2_m0_egc/` |

Reproduce on RTX 4090:

```powershell
Set-Location D:\Raymond\high_low_freq_NILM\MATUDA_NILM
C:\Users\PC\anaconda3\envs\nilm\python.exe scripts\run_s2_ukdale.py
```

Details: `docs/PAPER_EXPERIMENTS.md`, numbers: `results/PAPER_RESULTS.md`, LaTeX: `paper/main.tex`.

## Headline (UK-DALE H1+H5 → H2)

| Method | H2 F1 | H2 MAE |
|--------|-------|--------|
| B0 | 0.087 | 29.3 |
| B1 | 0.143 | 49.6 |
| **M0 EGC-DA** | **0.469** | 32.2 |

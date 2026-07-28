# MATUDA architecture (Stage 2)

**Not MultiNILM.** Seq2point + FC-layer UDA + EGC-DA (CDAN+E-inspired).

## Windowing (paper Table)

| Item | Value |
|------|-------|
| Mode | **seq2point** (center of window) |
| Input $T$ | **599** (odd; ≈60 min at 6 s) |
| Train stride | 30 |
| Eval stride | 60 |
| Not | seq2seq / output sub-window (MATNILM-style) |

## Network

```text
x (B,1,599)
  → multi-scale stem k={3,5,9} → proj C=96
  → TCN 8 blocks, dilation 2^{0..7}
  → GAP → (B,96)
  → FC1 512, FC2 256, FC3 128     ← MMD+CORAL / EGC-DA here
  → state head (K) + power head (K), optional gate
```

## Losses

- $L_{sup}$ = $2\cdot$MSE(power) + BCE(logits, pos_weight from ON rates)
- $L_{domain}$ on **FC embeddings** (L2-normalized): $\mu$MMD+(1-$\mu$)CORAL, $\mu=0.4$
- Mix: convex $(1-\lambda)L_{sup}+\lambda\tilde L_{domain}$, `domain_scale=equal`, $\lambda=0.6$, warmup 10
- **EGC-DA (M0):** entropy gate on samples + appliance-conditional CORAL on last FC

## Configs

| ID | File | DA |
|----|------|-----|
| B0 | `configs/matuda_s2_b0.yaml` | none |
| B1 | `configs/matuda_s2_b1.yaml` | global FC |
| M0 | `configs/matuda_s2_m0.yaml` | egc |

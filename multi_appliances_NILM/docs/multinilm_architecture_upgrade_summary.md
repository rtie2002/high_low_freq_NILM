# MultiNILM architecture upgrades — summary

UK-DALE multi-appliance, **no domain adaptation** in these runs  
(`experiment_id: ukdale(no domain adaptation)`).  
Checkpoint monitor: `val_mae_minus_f1`. Seed: 2026.

---

## Current stack (kept)

```text
aggregate
  → multi-scale stem (k=3/5/9) + residual
  → staged CNN [32 → 64 → 128]
  → TCN (8 residual blocks, dilations 1…128)
  → per-appliance local decoder (2× Conv k=3 + residual)
  → 1×1 power / state heads
  → hard state gate (STE in train)
```

| Setting | Value |
|---------|--------|
| `channel_schedule` | `[32, 64, 128]` |
| `hidden_channels` | `128` |
| `num_blocks` | `8` |
| `dropout` | `0.15` |
| `gate_mode` | `hard` |
| `head_local_layers` | `2` (`k=3`, residual) |
| `use_multiscale_stem` | `true` (`detail_kernels: [3,5,9]`) |
| Params | **~1.21M** |
| Domain adaptation | **off** |
| Shape / slope loss | **off** |

Config: `config/models/multinilm.yaml`

---

## Upgrade log

| # | Change | What | Outcome |
|---|--------|------|---------|
| — | Stem + shape loss (reverted) | Multi-scale stem **and** slope MSE together | Training worse → **removed** |
| 1 | Hard gate | Binary ON/OFF gate on power (STE) | Sharper waveform edges (visual) |
| 2 | Local appliance head | Per-app `2×Conv1d(k=3)` + residual before 1×1 heads | ~159K → ~263K; solid val ~0.75 F1 |
| 3 | Multi-scale stem only | Parallel `k=3/5/9`, fuse + residual; **no** shape loss | +~1K params; small MAE/SAE gain |
| 4 | Widen backbone | `[32,64,128]`, 8 blocks, dropout 0.15 | **~1.21M**; large val F1/MAE gain |
| — | DA / shape loss | Lin-style MMD/CORAL; slope loss | **Not used** in these val-focused runs |

---

## Three measured performance steps (validation)

Same protocol; numbers from `best.pt` evaluation tables.

| Stage | Model | Val MAE (W) | Val SAE | Val F1 | Val micro-F1 | Params |
|-------|--------|-------------|---------|--------|--------------|--------|
| **①** | Hard gate + local head | 16.69 | 11.87 | 0.751 | 0.805 | ~263K |
| **②** | + multi-scale stem | **15.90** | **11.22** | 0.750 | 0.804 | ~264K |
| **③** | + widen `[32,64,128]` ×8 | **12.97** | **9.80** | **0.851** | **0.912** | **1.21M** |

### ① → ② (multi-scale stem)

- Power error slightly better (MAE ↓ ~0.8 W, SAE ↓ ~0.7).
- F1 essentially flat.
- Test F1 rose a little (0.12 → 0.17) but still poor.

### ② → ③ (widen) — largest gain

- Val F1: **0.75 → 0.85** (+0.10).
- Val MAE: **15.9 → 13.0**; SAE: **11.2 → 9.8**.
- Highlights: dishwasher F1 **0.45 → 0.75**; fridge MAE **~20 → ~10**.

### Test (no DA) — for contrast

| Stage | Test F1 | Test micro-F1 | Test MAE (W) |
|-------|---------|---------------|--------------|
| ① | 0.123 | 0.216 | 25.3 |
| ② | 0.166 | 0.274 | 25.4 |
| ③ | 0.132 | 0.431 | 23.9 |

Widening **improves source-domain (val)** strongly; **overall test F1 does not follow** — house / domain gap remains (larger F1 gap val→test after widen). Micro-F1 on test rose mainly via fridge.

---

## Stage ③ per-appliance (widened best)

**Validation**

| Appliance | MAE | SAE | F1 |
|-----------|-----|-----|-----|
| kettle | 2.71 | 2.10 | 0.936 |
| fridge | 10.25 | 4.81 | 0.927 |
| dishwasher | 11.01 | 9.43 | 0.749 |
| washingmachine | 12.16 | 5.68 | 0.875 |
| microwave | 28.73 | 26.98 | 0.769 |
| **OVERALL** | **12.97** | **9.80** | **0.851** |

**Test** (still no DA)

| Appliance | MAE | SAE | F1 |
|-----------|-----|-----|-----|
| kettle | 19.63 | 19.66 | 0.039 |
| fridge | 43.41 | 38.44 | 0.459 |
| dishwasher | 36.98 | 37.07 | 0.000 |
| washingmachine | 8.25 | 8.24 | 0.035 |
| microwave | 11.27 | 9.80 | 0.128 |
| **OVERALL** | **23.91** | **22.64** | **0.132** |

---

## Takeaways

1. **Hard gate** helps sharp edges; **local heads** help redraw short shapes; **widen** helps capacity for val metrics most.
2. **Multi-scale stem alone** is a small MAE tweak, not a F1 breakthrough.
3. **Do not** re-bundle stem + strong shape loss without isolating experiments.
4. Next for **test / new house**: enable **domain adaptation** (`domain_adaptation.enabled: true`, `lambda_domain > 0`) on this widened base — not further blind width increases.
5. Optional next for **val waveform detail**: keep current width; try early TCN blocks with `dilation=1` only (cheap).

---

## One-liner

Hard gate → local heads → multi-scale stem → widen: three val steps end at **F1 ≈ 0.85 / MAE ≈ 13 W**; test still needs **DA**.

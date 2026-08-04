# MATUDA vs MultiNILM — Architecture and Parameter Comparison

Counts below are measured from the **current** configs:

| Model | Config |
|-------|--------|
| **MATUDA** | `config/models/matuda.yaml` |
| **MultiNILM** | `config/models/multinilm.yaml` |

Both use a shared TCN-style temporal encoder and **five per-appliance heads** (kettle, fridge, dishwasher, washing machine, microwave). The supervised loss is the same MultiNILM multitask objective (MSE + BCE, `task_balance: equal`). Domain adaptation, when enabled, differs in **where** features are taken from.

---

## 1. High-level block diagram

```text
MultiNILM                                         MATUDA (now)
─────────                                         ────────────
aggregate (B, T, 1)                               aggregate (B, 1, T)
        │                                                 │
multi-scale stem (k=3,5,9 → 16 ch each)           multi-scale stem (k=3,5,9 → 16 ch each)
        │                                                 │
staged widen: 32 → 64 → 128                       staged widen: 32 → 64 → 128
        │                                                 │
8× TCN residual blocks (C=128, k=5)               8× TCN residual blocks (C=128, k=5)
        │                                                 │
        │                                         compact 1×1 FC: 128→256→192→128
        │                                         (Lin-style DA hooks after time-pool)
        │                                                 │
5× appliance heads (local residual, C=128)        5× appliance heads (local residual, C=128)
        │                                                 │
power + state (hard gate)                         power + state (hard gate)
```

**Both use the same MultiNILM stem / TCN / heads.** MATUDA keeps an explicit **compact FC tower** for Lin-style multi-layer MMD+CORAL. MultiNILM has no FC tower; optional DA hooks attach to late TCN / aligned maps (`temporal_4`, `temporal_6`, `aligned`).

---

## 2. Parameter summary by module

| Module | MATUDA params | MATUDA % | MultiNILM params | MultiNILM % |
|--------|---------------|----------|------------------|-------------|
| Stem / front-end | 53,952 | 4.1% | 53,952 | 4.5% |
| TCN / temporal encoder | **658,432** | **50.0%** | **658,432** | **54.5%** |
| FC tower (DA) | **107,072** | **8.1%** | **0** | — |
| Appliance heads (×5) | **496,650** | **37.7%** | **496,650** | **41.1%** |
| **Total** | **1,316,106** | 100% | **1,209,034** | 100% |

Takeaway: MATUDA ≈ MultiNILM **+ ~107K FC**. Stem, TCN, and heads now match MultiNILM exactly (same building blocks / widths).

---

## 3. Channel / layer dimensions

### 3.1 Stem

| | MATUDA | MultiNILM |
|--|--------|-----------|
| Multi-scale kernels | 3, 5, 9 | 3, 5, 9 |
| Branch channels | 16 each → fuse to 32 | 16 each → fuse to 32 |
| After stem | staged `32→64→128` | staged `32→64→128` |
| TCN input width | **128** | **128** |

### 3.2 TCN (both: 8 blocks)

| | MATUDA | MultiNILM |
|--|--------|-----------|
| Channels | 128 | 128 |
| Kernel | 5 | 5 |
| Structure | `ResidualTemporalBlock` (shared code) | same |
| Dilations | \(2^i\) capped by `max_dilation: 64` | same |
| Params | 658,432 | 658,432 |

### 3.3 FC tower (MATUDA only — compact Lin analogues)

| Layer | Weight shape | Bias | Params | Role |
|-------|--------------|------|--------|------|
| FC1 | `(256, 128, 1)` | 256 | 33,024 | 128 → 256 |
| FC2 | `(192, 256, 1)` | 192 | 49,344 | 256 → 192 |
| FC3 | `(128, 192, 1)` | 128 | 24,704 | 192 → 128 |
| **Sum** | | | **107,072** | Time-pooled \(Z^{(\ell)}\) for DA |

Old MATUDA used `96→512→256→128` (~214K). New tower is **~half the size** and starts from MultiNILM’s `C=128`.

MultiNILM: **no FC tower**. Domain features (if DA on) are pooled from selected TCN feature maps.

### 3.4 Appliance heads — dimensions (per appliance)

Both models use **5 identical `ApplianceHead` modules** (same class).

| Layer | Weight shape | Meaning |
|-------|--------------|---------|
| Local decoder 1 | `(128, 128, 3)` + BN | temporal refine |
| Local decoder 2 | `(128, 128, 3)` + BN | temporal refine (+ residual) |
| State head | `(1, 128, 1)` | 128 → 1 logit |
| Power head | `(1, 128, 1)` | 128 → 1 power |
| Gate | hard (STE) | ON/OFF blend with OFF-norm |

**Per-head params:** 99,330  
**Five heads:** 5 × 99,330 = **496,650**

```text
shared (B, 128, T)   # MATUDA: last FC map; MultiNILM: TCN / aligned
  → Conv1d 128→128, k=3 + BN + GELU
  → Conv1d 128→128, k=3 + BN + GELU  (+ residual)
  → state: Conv1d 128→1
  → power: Conv1d 128→1  (hard-gated)
```

---

## 4. Side-by-side head comparison

| Item | MATUDA | MultiNILM |
|------|--------|-----------|
| Head class | `ApplianceHead` (shared) | `ApplianceHead` |
| Input channels to head | 128 (last FC) | 128 (TCN out) |
| Hidden width | **128** | **128** |
| Local temporal layers | 2× Conv k=3 **+ residual + BN** | same |
| State / power out | 1 channel each | 1 channel each |
| Params per head | ~99K | ~99K |
| Params all 5 heads | ~497K | ~497K |

---

## 5. Domain-adaptation feature dims (when DA is on)

| | MATUDA | MultiNILM (yaml default) |
|--|--------|---------------------------|
| Feature sources | FC1, FC2, FC3 after **time mean-pool** | `temporal_4`, `temporal_6`, `aligned` |
| Feature shapes | \(Z\in\mathbb{R}^{B\times 256}\), \(B\times 192\), \(B\times 128\) | pooled maps with channel width **128** |
| Distance | \(\mu\) MMD + \((1-\mu)\) CORAL, \(\mu=0.4\) | same hybrid (`domain_method: both`) |
| Default in yaml | DA **on**, \(\lambda=0.6\) | DA **off**, \(\lambda=0\) |

---

## 6. Short conclusion

1. **Both use the same MultiNILM TCN backbone** (stem + staged widen + 8× C=128 k=5 + residual heads).
2. MATUDA’s only structural extra is a **compact FC tower** `128→256→192→128` (~8% of params) for Lin-style FC-layer DA.
3. Total size: MultiNILM ~1.21M; MATUDA ~1.32M (= MultiNILM + FC).
4. Head I/O is identical (1 power + 1 state logit per appliance per timestep).

---

## 7. Change log — align MATUDA dims with MultiNILM

**Goal:** keep Lin-style FC DA, but make every other block match MultiNILM so capacity / inductive bias are comparable.

### 7.1 What changed (old → new)

| Block | Old MATUDA | New MATUDA (= MultiNILM + compact FC) |
|-------|------------|----------------------------------------|
| Stem branches | 32 ch × (k=3,5,9) | **16 ch** × (k=3,5,9) |
| After stem | 1×1 proj → **96** | staged **32→64→128** |
| TCN | 8× C=**96**, k=**3**, 2-conv Lin block | 8× C=**128**, k=**5**, `ResidualTemporalBlock` |
| FC tower | **96→512→256→128** (~214K) | **128→256→192→128** (~107K) |
| Heads | refine C=**64**, no residual | MultiNILM `ApplianceHead` C=**128**, residual |
| Total params | ~0.85M | **~1.32M** |

### 7.2 Why shrink the FC tower

- MultiNILM trunk already ends at **C=128**; a fat `→512→256` tower was oversized relative to that width and cost ~25% of old MATUDA.
- Compact `128→256→192→128` still gives **three** Lin-style layers for multi-layer MMD+CORAL, at ~half the old FC cost (~8% of new total).
- Heads still read **128-d** maps, same as MultiNILM.

### 7.3 Implementation notes

- MATUDA now **imports** `MultiScaleWaveformStem`, `StagedFeatureExtractor`, `ResidualTemporalBlock`, and `ApplianceHead` from `model/MultiNILM.py` (same code path as the baseline).
- Config keys in `matuda.yaml` mirror MultiNILM (`channel_schedule`, `hidden_channels`, `detail_branch_channels`, `tcn_kernel_size`, `head_local_layers`, …) plus `fc_dims`.
- Layout remains `(B, 1, T)` inside the net; the adapter still permutes dataloader `(B, T, 1)`.
- **Not yet done:** FC-as-DA-only side branch (TCN→heads; pool→FC→DA). Current path is still `TCN → FC → heads`. Side-branch is the next optional step.

### 7.4 Expected training implication

- Supervised capacity should behave closer to MultiNILM (same stem/TCN/heads + MultiNILM lr/clip settings).
- DA still acts on pooled FC maps; if val state BCE rises again, prefer lowering \(\lambda\) / freezing DA rather than widening FC.

---

*Generated from live `count_parameters` / `named_parameters` on the configs above. Re-run if yaml channel widths change.*

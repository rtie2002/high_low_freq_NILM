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
MultiNILM                                         MATUDA
─────────                                         ──────
aggregate (B, T, 1)                               aggregate (B, 1, T)
        │                                                 │
multi-scale stem (k=3,5,9 → 16 ch each)           multi-scale stem (k=3,5,9 → 32 ch each)
        │                                                 │
staged widen: 32 → 64 → 128                       1×1 proj → 96 ch
        │                                                 │
8× TCN residual blocks (C=128, k=5)               8× TCN residual blocks (C=96, k=3)
        │                                                 │
        │                                         1×1 FC tower: 96→512→256→128
        │                                         (Lin-style DA hooks after time-pool)
        │                                                 │
5× appliance heads (local residual, C=128)        5× appliance heads (refine C=64)
        │                                                 │
power + state (hard gate)                         power + state (hard gate)
```

**Both use a TCN.** MultiNILM has **no separate FC tower**; optional DA hooks attach to late TCN / aligned maps (`temporal_4`, `temporal_6`, `aligned`). MATUDA adds an explicit **FC tower** for Lin-style multi-layer MMD+CORAL.

---

## 2. Parameter summary by module

| Module | MATUDA params | MATUDA % | MultiNILM params | MultiNILM % |
|--------|---------------|----------|------------------|-------------|
| Stem / front-end | 9,952 | 1.2% | 53,952 | 4.5% |
| TCN / temporal encoder | **443,904** | **52.0%** | **658,432** | **54.5%** |
| FC tower (DA) | **213,888** | **25.1%** | **0** | — |
| Appliance heads (×5) | 185,610 | 21.8% | **496,650** | **41.1%** |
| **Total** | **853,354** | 100% | **1,209,034** | 100% |

Takeaway: the heaviest block in **both** models is the **TCN**. MATUDA’s extra cost is mainly the **FC tower (~0.21M)**. MultiNILM is larger overall because of a **wider TCN (128 vs 96)** and **heavier heads (128-d residual local decoder)**.

---

## 3. Channel / layer dimensions

### 3.1 Stem

| | MATUDA | MultiNILM |
|--|--------|-----------|
| Multi-scale kernels | 3, 5, 9 | 3, 5, 9 |
| Branch channels | 32 each → concat 96 | 16 each → concat 48, fuse to 32 |
| After stem | `Conv1d` 96→96 | staged `32→64→128` (`channel_schedule`) |
| TCN input width | **96** | **128** |

### 3.2 TCN (both: 8 blocks)

| | MATUDA | MultiNILM |
|--|--------|-----------|
| Channels | 96 | 128 |
| Kernel | 3 | 5 |
| Structure | 2× Conv1d per residual block | 1× dilated Conv + GroupNorm residual |
| Dilations | \(2^0 \ldots 2^7\) | capped by `max_dilation: 64` |
| Params | 443,904 | 658,432 |

Per MATUDA block: two layers of shape `(96, 96, 3)` ≈ 2×(96×96×3 + 96) = 55,488; ×8 = 443,904.

Per MultiNILM block: one `(128, 128, 5)` conv + norm ≈ 82,304; ×8 = 658,432.

### 3.3 FC tower (MATUDA only — Lin fc6–fc8 analogues)

| Layer | Weight shape | Bias | Params | Role |
|-------|--------------|------|--------|------|
| FC1 | `(512, 96, 1)` | 512 | 49,664 | 96 → 512 |
| FC2 | `(256, 512, 1)` | 256 | 131,328 | 512 → 256 |
| FC3 | `(128, 256, 1)` | 128 | 32,896 | 256 → 128 |
| **Sum** | | | **213,888** | Time-pooled \(Z^{(\ell)}\in\mathbb{R}^{B\times D}\) for DA |

MultiNILM: **no FC tower**. Domain features (if DA on) are pooled from selected TCN feature maps.

### 3.4 Appliance heads — dimensions (per appliance)

Both models use **5 identical heads** (one per appliance).

#### MATUDA head (`head_hidden: 64`, `head_kernel_size: 3`)

| Layer | Weight shape | Meaning |
|-------|--------------|---------|
| Refine conv 1 | `(64, 128, 3)` | shared embed 128 → 64, k=3 |
| Refine conv 2 | `(64, 64, 3)` | 64 → 64, k=3 |
| State head | `(1, 64, 1)` | 64 → 1 logit |
| Power head | `(1, 64, 1)` | 64 → 1 power |
| Gate | hard (STE) | ON/OFF blend with OFF-norm |

**Per-head params:** 37,122  
**Five heads:** 5 × 37,122 = **185,610**

Tensor flow (one head):

```text
shared (B, 128, T)
  → Conv1d 128→64, k=3
  → Conv1d 64→64, k=3
  → state: Conv1d 64→1
  → power: Conv1d 64→1  (hard-gated)
```

#### MultiNILM head (`hidden_channels: 128`, `head_local_layers: 2`, residual)

| Layer | Weight shape | Meaning |
|-------|--------------|---------|
| Local decoder 1 | `(128, 128, 3)` + GroupNorm | temporal refine |
| Local decoder 2 | `(128, 128, 3)` + GroupNorm | temporal refine (+ residual) |
| State head | `(1, 128, 1)` | 128 → 1 logit |
| Power head | `(1, 128, 1)` | 128 → 1 power |
| Gate | hard (STE) | ON/OFF blend with OFF-norm |

**Per-head params:** 99,330  
**Five heads:** 5 × 99,330 = **496,650**

Tensor flow (one head):

```text
shared (B, 128, T)
  → Conv1d 128→128, k=3 + norm
  → Conv1d 128→128, k=3 + norm  (+ residual)
  → state: Conv1d 128→1
  → power: Conv1d 128→1  (hard-gated)
```

---

## 4. Side-by-side head comparison

| Item | MATUDA | MultiNILM |
|------|--------|-----------|
| Input channels to head | 128 (last FC) | 128 (TCN out) |
| Hidden width | **64** | **128** |
| Local temporal layers | 2× Conv k=3 (no residual) | 2× Conv k=3 **+ residual + GN** |
| State / power out | 1 channel each | 1 channel each |
| Params per head | ~37K | ~99K |
| Params all 5 heads | ~186K | ~497K |

---

## 5. Domain-adaptation feature dims (when DA is on)

| | MATUDA | MultiNILM (yaml default) |
|--|--------|---------------------------|
| Feature sources | FC1, FC2, FC3 after **time mean-pool** | `temporal_4`, `temporal_6`, `aligned` |
| Feature shapes | \(Z\in\mathbb{R}^{B\times 512}\), \(B\times 256\), \(B\times 128\) | pooled maps with channel width **128** |
| Distance | \(\mu\) MMD + \((1-\mu)\) CORAL, \(\mu=0.4\) | same hybrid (`domain_method: both`) |
| Default in yaml | DA **on**, \(\lambda=0.6\) | DA **off**, \(\lambda=0\) |

---

## 6. Short conclusion

1. **Yes — both use a TCN**; it is the largest parameter block in each model (~52–55%).
2. MATUDA adds a **512→256→128 FC tower** (~25% of its params) for Lin-style FC-layer DA.
3. MultiNILM puts more capacity into a **wider TCN** and **wider residual heads** (64 vs 128 hidden), so total size is larger (~1.21M vs ~0.85M) even without an FC tower.
4. Head output dimensions are the same conceptually (1 power + 1 state logit per appliance per timestep); the difference is the **hidden width and local decoder design**.

---

*Generated from live `count_parameters` / `named_parameters` on the configs above. Re-run if yaml channel widths change.*

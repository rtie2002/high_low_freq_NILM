# Why Li et al. (Applied Energy 2022) Case iii is Non-Reproducible on UK-DALE H3/H4

**Paper:** Dandan Li et al., *Transfer learning for multi-objective non-intrusive load monitoring in smart building*, Applied Energy, 329:120223, 2023 (online 2022).  
**DOI:** [10.1016/j.apenergy.2022.120223](https://doi.org/10.1016/j.apenergy.2022.120223)

**Finding:** Their **Case iii** results (REFIT → UK-DALE, Table 6) **cannot be faithfully reproduced** if one follows the paper’s stated test houses (**UK-DALE House 3 and House 4**). This is **not** because our local UK-DALE copy is incomplete: official online metadata matches what we have.

---

## 1. What the paper claims

| Item | Claim in the paper |
|------|--------------------|
| Case iii | Pre-train on **REFIT** House 11; fine-tune & test on **UK-DALE Houses 3 & 4** |
| Table 1 | UK-DALE houses **3, 4** with **53, 19** “number of appliances” |
| Text (§5.1) | “Houses 3 and 4 have relatively complete data”; period **03/May/2013 – 13/May/2013** |
| Table 6 metrics | Reported for **Fridge, Dishwasher, Washing machine, Kettle** (regression SAE / EpD / NDE) |
| Transfer type | **Not unsupervised**: dense layers (and proposed CNN fine-tune) use **labeled UK-DALE target data** |

---

## 2. What official UK-DALE actually has (H3 / H4)

Checked against:

- Author metadata: [JackKelly/UK-DALE_metadata](https://github.com/JackKelly/UK-DALE_metadata) (`building3.yaml`, `building4.yaml`)
- Dataset paper: Kelly & Knottenbelt, *Sci. Data* 2015, Table 1 ([DOI](https://doi.org/10.1038/sdata.2015.7))
- UK-DALE **2017** release (longer House 1 only; H3/H4 channel set unchanged)
- Local copy: `dataset_preprocess/UK_DALE/UKDALE2017/metadata/`

### House 3 — **5 meters total** (1 mains + 4 plugs)

| Channel | Appliance |
|---------|-----------|
| 1 | mains |
| 2 | kettle |
| 3 | electric space heater |
| 4 | laptop |
| 5 | projector |

Recording ends **~2013-04-08**.

### House 4 — **6 meters total** (1 mains + 5 plugs)

| Channel | Content |
|---------|---------|
| 1 | mains |
| 2 | TV + DVD + digibox + lamp (**combined**) |
| 3 | kettle + radio (**combined**) |
| 4 | gas boiler |
| 5 | freezer |
| 6 | washing machine + microwave + breadmaker (**combined**) |

### Sci. Data Table 1 meter counts

| House | Total meters |
|-------|--------------|
| 1 | **54** |
| 2 | **20** |
| 3 | **5** |
| 4 | **6** |
| 5 | **26** |

**UK-DALE 2017 does not add fridge / dishwasher / clean WM channels to H3 or H4.**

---

## 3. Why Case iii is totally non-reproducible (as written)

### 3.1 Missing ground-truth appliances for Table 6

Table 6 requires per-appliance labels for:

| Appliance | On real H3? | On real H4? |
|-----------|-------------|-------------|
| Fridge | **No** | **No** (only freezer) |
| Dishwasher | **No** | **No** |
| Washing machine | **No** | Only **mixed** with microwave + breadmaker |
| Kettle | Yes | Yes, but mixed with radio |

Without separate fridge / dishwasher / WM submeters, you **cannot** compute the reported SAE / EpD / NDE in a standard supervised NILM sense on H3/H4.

### 3.2 Table 1 channel counts contradict H3/H4

Paper Table 1:

```text
UK-DALE   House index: 3, 4    Number of appliances: 53, 19
```

Official reality:

- H3, H4 → **5, 6** meters  
- **53 ≈ House 1 (~54)**, **19 ≈ House 2 (~19–20)**

This is strong evidence of a **house-index error** (likely meant Houses **1 and 2**, but wrote **3 and 4**), not “extra hidden channels” in some UK-DALE mirror.

### 3.3 Stated date range excludes House 3

Paper: UK-DALE samples from **03 May 2013 – 13 May 2013**.  
Official House 3 timeframe ends **~08 April 2013**.

→ **House 3 has no data in the May 2013 window** they specify.

### 3.4 “Relatively complete data” is false for H3/H4

Houses with the common five baseline appliances (kettle, microwave, fridge, dishwasher, washing machine) are typically **H1 / H2 / H5**, not H3/H4.  
H3/H4 are the **least** instrumented UK-DALE homes.

### 3.5 Not an issue with “our incomplete download”

Cross-check of online official metadata = local metadata.  
**No alternate public UK-DALE source** lists 53 / 19 meters on Houses 3 / 4.

---

## 4. What *is* reproducible (with a corrected reading)

If one **ignores** the literal “House 3, 4” labels and instead uses houses whose meter counts match Table 1 (**House 1 & 2**), then:

- The five baseline appliances exist with usable labels  
- The **method** (REFIT pretrain → target **labeled** fine-tune of dense / CNN layers) can be re-implemented  
- Exact Table 6 numbers may still differ due to preprocessing, windows, and undocumented splits  

That is a **protocol fix by reinterpretation**, not a faithful reproduction of the written H3/H4 Case iii.

---

## 5. Their training process (phase by phase)

This is **supervised / semi-supervised transfer learning with target labels**, **not** unsupervised domain adaptation (no MMD/CORAL).  
Architecture: **one-to-many seq2point-style CNN** — 4 CNN layers (shared features) + 2 dense layers (multi-appliance outputs). Input window length **599**.

### Overview diagram

```text
Phase 0   Build windows (length 599, slide by 1)
    │
Phase 1   PRE-TRAIN on REFIT (source) — full network
    │         4×CNN + 2×Dense, all trainable
    │
Phase 2   BUILD target model — copy 4×CNN weights; new Dense (N outputs)
    │
Phase 3   RETRAIN Dense only on TARGET (labeled) — CNN frozen
    │
Phase 4   FINE-TUNE 4×CNN on TARGET (labeled) — very small LR  (proposed method)
    │         Benchmark [Zhong]: often stops after Phase 3
    │
Phase 5   TEST on target hold-out
```

### Phase 0 — Data windowing (all cases)

| Setting | Value |
|---------|--------|
| Input window | **599** samples (seq2point-style; predict mid-window appliance powers) |
| Slide | **1** sample per step |
| Resample | Other datasets resampled to match REFIT **8 s** spacing |
| Split | Random **60% / 20% / 20%** train / val / test from samples in Table 2 |

### Phase 1 — Pre-train on source (REFIT)

| | |
|--|--|
| Data | REFIT **House 11** (~199.6K samples in Table 2) |
| Trainable | **Entire** network: 4 CNN + 2 Dense |
| Epochs (max) | **300** |
| Batch size | **1024** |
| Learning rate | **1e-2** |
| Goal | Learn generic temporal features + multi-appliance mapping on source |

Output of this phase: a **pre-trained one-to-many model**.

### Phase 2 — Create the transfer (base) model

Paper’s five transfer steps start here:

1. Obtain the pre-trained model (Phase 1).  
2. Create a base model with the **same first four CNN layers** (weights copied).  
3. Set Dense output size \(N\) = **number of appliances in the target house**.  
4. (Next phases train on target.)

CNN = “generic / transferable” features; Dense = “appliance-specific” head for the new domain.

### Phase 3 — Retrain Dense layers on target (labeled)

| | |
|--|--|
| Data | **Labeled** target houses (Case iii: paper says UK-DALE 3,4 — **problematic**, see §2–3) |
| Trainable | **Last 2 Dense layers only** |
| CNN | **Frozen** (weights from REFIT) |
| Epochs (max) | **50** |
| Batch size | **1024** |
| Learning rate | **1e-2** |
| Goal | Adapt multi-appliance outputs to target domain using target labels |

This is the **benchmark** transfer style (Zhong-style: freeze CNN, train FC/Dense on target).

### Phase 4 — Fine-tune CNN on target (proposed extra step)

| | |
|--|--|
| Data | Same **labeled** target data |
| Trainable | **First 4 CNN layers** (fine-tune) |
| Epochs (max) | **20** |
| Batch size | **1024** |
| Learning rate | **1e-4** (much smaller than Phase 3) |
| Goal | Adapt low-level features to domain shift (REFIT → UK-DALE / REDD / SC-EDNRR) |

Paper argues this is why their model beats the benchmark that only does Phase 3.

### Phase 5 — Testing

| Case | Pre-train | Fine-tune / retrain target | Test |
|------|-----------|----------------------------|------|
| i | REFIT 11 | REFIT 20 | REFIT 20 |
| ii | REFIT 11 | REDD 1,2,5 | REDD 1,2 |
| iii | REFIT 11 | UK-DALE **3,4** (as written) | UK-DALE **3,4** |
| iv | REFIT 11 | SC-EDNRR | SC-EDNRR |

Metrics: SAE, EpD (Wh), NDE, ODPE (%).

### Hyper-parameter summary (Table 3)

| Stage | Max epochs | Batch | LR |
|-------|------------|-------|-----|
| Pre-train (full net) | 300 | 1024 | 1e-2 |
| Retrain Dense | 50 | 1024 | 1e-2 |
| Fine-tune CNN | 20 | 1024 | 1e-4 |

### Contrast with our unsupervised DA

| | Li et al. (this paper) | Our MultiNILM DA (Lin-style) |
|--|------------------------|------------------------------|
| Target labels | **Required** (Phases 3–4) | **Not used** |
| Domain loss | None (MMD/CORAL not used) | MMD + CORAL on features |
| Mechanism | Freeze / retrain / fine-tune weights | Align \(Z_S\) and \(Z_T\) while supervised on source |
| “Transfer” meaning | **Weight transfer + target supervised adapt** | **Feature distribution alignment** |

So even if Case iii house IDs were fixed to H1/H2, their pipeline is still a **different problem** from unsupervised cross-house DA.

---

## 6. Bottom-line statement (for notes / related work)

> Li et al. (Applied Energy 2022) Case iii claims REFIT→UK-DALE transfer on **Houses 3 and 4** with metrics for fridge, dishwasher, washing machine, and kettle. Official UK-DALE metadata (Jack Kelly GitHub, Sci. Data 2015, UK-DALE 2017) shows H3/H4 have only **5–6 meters**, lack fridge/dishwasher (and clean WM), and H3 has **no data** in the paper’s May 2013 window, while Table 1’s **53 / 19** appliance counts match **Houses 1 / 2**, not 3 / 4. Therefore Case iii, **as written for H3/H4, is non-reproducible**. This is a documentation inconsistency in the paper, not missing data in our UK-DALE copy. Their training process is phase-wise **pre-train → freeze CNN → retrain Dense on labeled target → fine-tune CNN on labeled target** (see §5), not unsupervised DA.

---

## 7. Implications for our MultiNILM work

1. Do **not** use this table as a fair H3/H4 multi-appliance transfer benchmark.  
2. Prefer **H1 / H2 / H5** for UK-DALE multi-appliance experiments (as we already do).  
3. Their “success” is **label-efficient fine-tune transfer**, not unsupervised DA on H3/H4.  
4. When citing this paper, note the house-index / meter-count inconsistency.  
5. If we want a fair comparison to *their method*, implement Phases 1–5 on **H1/H2** (or our H1+H5→H2 with a labeled H2 fine-tune split), not unsupervised MMD/CORAL alone.

---

*Documented after reading the PDF and verifying Jack Kelly official metadata + Sci. Data 2015 + UK-DALE 2017.*

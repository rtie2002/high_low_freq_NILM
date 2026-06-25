# SGN ON/OFF Classification Fix (UK-DALE Cross-House)

This document explains why the dishwasher classifier collapsed to “always OFF”, why the final regression stayed small even when timing looked right, and what we changed in the training pipeline to fix it.

---

## 1. How SGN works (paper recap)

SGN (Shin et al., AAAI 2019) has two CNN heads:

| Head | Output | Role |
|------|--------|------|
| Regression | `power` | Predicts appliance power (normalized) |
| Classification | `on_prob` | Predicts ON probability (sigmoid) |

Final prediction:

```
gated_power = power × on_prob        (soft gate)
```

Training loss (paper Eq. 8):

```
L = L_output + L_on
L_output = MSE(gated_power, true_power)
L_on     = BCE(on_prob, true_on_label)
```

Labels come from your CSV columns (e.g. `dishwasher_on`, `dishwasher_power`). The model does **not** recompute ON/OFF at train time.

---

## 2. Symptoms we saw

On validation (House 2, cross-house):

- **Blue** (`dishwasher true`): clear ~2000 W rectangular pulses
- **Red** (`dishwasher pred`): small humps (~100–400 W) or flat near 0
- **Green** (`predicted ON`): missing — classifier rarely reached `on_prob ≥ 0.5`

So the model sometimes learned *when* something happened (red bumps aligned with blue), but not *full ON/OFF* or *full magnitude*.

---

## 3. Root causes

### 3.1 BCE class imbalance (main ON/OFF failure)

Dishwasher is OFF most of the time. In each batch, OFF timesteps dominate BCE:

```
Example batch: 20 ON timesteps + 236 OFF timesteps (equal BCE weight)

Gradient pressure toward OFF  ≈ 236
Gradient pressure toward ON   ≈ 20

→ “Always predict OFF” is the easy minimum for BCE
```

You **do** have ON/OFF labels in the data. The problem is not missing labels — it is that plain BCE is biased toward the majority class unless you rebalance loss or sampling.

### 3.2 Soft-gate attenuates regression (low red line even when timing is OK)

Paper gradient to regression (through `L_output`):

```
∂L_output/∂regression ∝ on_prob × (true_power − gated_power)
```

If `on_prob ≈ 0.1–0.3`:

- Gated output stays small: `800 W × 0.2 ≈ 160 W`
- Regression gets weak gradient on ON windows
- 99% OFF windows pull regression toward 0

Green bands need `on_prob ≥ 0.5`. Below that, red can move a little without ever looking like a real ON cycle.

### 3.3 Cross-house makes classification harder

Paper UK-DALE setup: train houses 1, 3, 4, 5 → test house 2 (same dataset, similar usage).

Our setup: train H1+H5, val/test H2 (different house, different aggregate patterns). The classifier is less confident on H2, so `on_prob` stays below 0.5 more often.

### 3.4 What is **not** the paper’s fix

The paper uses only `L_output + L_on` with soft gate. It does **not** use:

- Weighted BCE / `pos_weight`
- Weighted random sampling
- Extra regression loss on ON windows
- Hard gate (hard SGN is a variant and often worse than soft)

Our changes are **training extensions** for sparse, cross-house dishwasher — not a reproduction of the original paper loss alone.

---

## 4. Solution (three parts)

Config: `configs/sgn_ukdale_cross_house.json`

```json
"bce_pos_weight": 30.0,
"oversample_on": true,
"reg_on_weight": 1.0
```

### 4.1 Weighted BCE — `bce_pos_weight` (primary fix for ON/OFF)

**File:** `sgn/losses.py`

ON timesteps get higher weight in BCE:

```
Same batch with bce_pos_weight = 30:

Gradient toward ON  ≈ 20 × 30 = 600
Gradient toward OFF ≈ 236 × 1  = 236

→ Classifier is pushed to predict ON when label is ON
```

This directly addresses “model not willing to predict ON” while still using **your** `dishwasher_on` labels.

### 4.2 Weighted random sampling — `oversample_on` (extra balance)

**File:** `model_evaluation/runner.py` → `make_dataloader(..., oversample_on=True)`

Uses `WeightedRandomSampler` on the **training** loader only:

- Windows that contain any ON timestep in the output region are sampled more often
- Val/test loaders stay uniform (realistic evaluation)

At startup you should see something like:

```
WeightedRandomSampler: 500 ON / 24000 OFF windows, ON weight = 48.0x
```

This is backup: even if some batches would be OFF-heavy, training still sees enough ON windows. **`bce_pos_weight` is the main fix; sampling is insurance.**

### 4.3 ON-only direct regression — `reg_on_weight` (fix magnitude)

**File:** `sgn/losses.py`

Extra term (not in paper):

```
L_reg_on = MSE(raw_power, true_power)   only where true_on = 1
L_total  = L_output + L_on + reg_on_weight × L_reg_on
```

Why: regression no longer depends only on a weak gated gradient when `on_prob` is low. On labeled ON timesteps, the regression head gets full-strength supervision to reach ~2000 W (normalized).

Set `reg_on_weight: 0` to match paper loss exactly.

### 4.4 ON-only gated output — `gated_on_weight` (fix half-magnitude red line)

**File:** `sgn/losses.py`

When green bands appear but red only reaches ~50% of blue (~1000 W vs ~2000 W), the usual cause is:

```
gated_power = regression × on_prob
2000 W × on_prob ≈ 0.5  →  plot shows ~1000 W
```

`reg_on_weight` trains **raw** regression; plots show **gated** output. Add:

```
L_gated_on = MSE(gated_power, true_power)   only where true_on = 1
```

Gradients update **both** regression and `on_prob` so the product reaches full true power on ON windows.

Config example: `gated_on_weight: 3.0`, `reg_on_weight: 2.0`.

---

## 5. Data and labels (unchanged)

Still using your preprocessing labels:

| Column | Source |
|--------|--------|
| `dishwasher_power` | Meter power, capped by aggregate |
| `dishwasher_on` | Algorithm 1 labeling (`ukdale.yaml`: threshold 50 W, min on/off duration 300 samples @ 6 s) |

CSV split config: `configs/training_data_ukdale_cross_house.json`  
(Filenames may be `multi_appliance_*_cross_house.csv` on D: — same data, different names.)

---

## 6. What to expect after the fix

| Epoch range | Expected |
|-------------|----------|
| Early | Green bands appear during true dishwasher cycles |
| Mid | Red line rises inside green bands toward true ~2000 W |
| If too many false green bands | Lower `bce_pos_weight` (e.g. 15) |
| If green OK but red still low | Raise `reg_on_weight` (e.g. 2–3) |

Live plots: green = `pred_dishwasher_on_prob ≥ 0.5` in `model_evaluation/plots.py`.

---

## 7. How to run

```powershell
cd D:\Raymond\high_low_freq_NILM\NILM_model
python main.py --model sgn --mode train_inference --data_source csv `
  --csv_config baseline/SGN/configs/training_data_ukdale_cross_house.json `
  --model_config baseline/SGN/configs/sgn_ukdale_cross_house.json `
  --run_dir runs/sgn_ukdale_cross_house
```

Paper-faithful hyperparameters (no oversample / pos_weight / reg_on): use `sgn_ukdale_paper.json` instead.

---

## 8. Code map

| Piece | Location |
|-------|----------|
| Loss (`L_output`, `L_on`, `L_reg_on`, `bce_pos_weight`) | `baseline/SGN/sgn/losses.py` |
| Config fields | `baseline/SGN/sgn/config.py`, `configs/sgn_ukdale_cross_house.json` |
| Weighted sampler | `model_evaluation/runner.py` → `make_dataloader` |
| Pipeline wiring | `models/sgn_pipeline.py` |
| ON/OFF labels in CSV | `dataset_preprocess/ukdale_processing_multi_appliance.py` + `config/preprocess/ukdale.yaml` |
| Waveform plots | `model_evaluation/plots.py`, `model_evaluation/runner.py` |

---

## 9. Summary

| Problem | Cause | Fix |
|---------|-------|-----|
| No green (never ON) | BCE dominated by OFF class | `bce_pos_weight: 30` |
| Unstable ON exposure in training | Rare ON windows in some batches | `oversample_on: true` |
| Red too small despite timing | Soft gate × low `on_prob` | `reg_on_weight: 1.0` on true-ON timesteps |

The architecture and labels were fine; training needed **class-balanced classification loss** and **direct regression supervision on ON windows** for cross-house dishwasher.

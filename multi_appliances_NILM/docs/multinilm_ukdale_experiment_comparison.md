# MultiNILM UK-DALE Experiment Comparison

This document records the configuration changes applied to MultiNILM on the shared UK-DALE cross-house setup, and compares results from the **first run** (864→256 center window, heavy overlap) against the **second run** (480→480 full-sequence, val-first setup).

Both runs used:

- **Experiment:** `config/experiment_ukdale.yaml` (`ukdale_first_model`)
- **Model:** `multinilm` (~159K parameters)
- **Seed:** 2026
- **Split:** Train H1+H5 (3 weeks); Val last week H1+H5; **Test H2 (cross-house)**
- **Appliances:** kettle, fridge, dishwasher, washingmachine, microwave

---

## 1. Configuration changes (Run 1 → Run 2)

| Setting | Run 1 (baseline) | Run 2 (current `multinilm.yaml`) | Rationale |
|---------|------------------|----------------------------------|-----------|
| `input_window_length` | 864 | **480** | Align with Transfer baseline window |
| `output_window_length` | 256 | **480** | Full-sequence supervision |
| `output_alignment` | center | **end** | Match dataloader label alignment for 480/480 |
| `input_stride` | 16 | **240** | Less redundant windows; reduce overfitting |
| `eval_stride` | 256 | **240** | Consistent eval stride with training |
| `eval_reconstruction` | flat | **overlap_mean** | Required when stride < output length; fixes waveform x-axis |
| `training_targets` | output_window | **full_input** | Supervise all 480 output steps |
| `checkpoint_monitor` | `mae` | **`val_mae_minus_f1`** | Val-first; balance normalized MAE and F1 (see **§1.3**) |
| `checkpoint_mae_space` | (default) | **normalized** | Match Transfer checkpoint rule |
| `evaluation.pred_on_source` | (default / combined) | **`state_head`** | F1 from state head only (≥ 0.5) |
| `early_stop_patience` | (default) | **60** | Allow longer convergence |
| `epochs` | 100 | **200** | More room before early stop |
| `weight_decay` | — | **0** | Match Transfer recipe |
| `scheduler` | — | **none** (with `lr_scheduler` preset kept) | Match Transfer; preset ready if enabled |
| `lambda_state` | — | **1** | State loss weight (sweep 2–3 suggested next) |

### 1.1 Impact summary

Not every setting changed the **reported F1/MAE numbers** equally. Some changes improve learning; others fix evaluation/plotting; others only affect which epoch is saved.

| Change | Estimated impact on Val F1 / MAE | Why |
|--------|----------------------------------|-----|
| Window **480/480 + stride 240** | **Large** | Main driver of Run 2 gains |
| `training_targets: full_input` | **Medium** (pairs with 480/480) | More supervised timesteps per window |
| `checkpoint: val_mae_minus_f1` | **Medium** | Saves a different (F1-friendlier) epoch |
| `pred_on_source: state_head` | **Medium** | Changes how F1 is computed at eval |
| `eval_reconstruction: overlap_mean` | **Low for F1** | Fixes timeline/plots; not a training change |
| `early_stop: 60`, `epochs: 200` | **Small** | More time to converge |

**Takeaway:** Run 2 improved mainly because of **windowing + supervision length**, supported by **checkpoint and F1 evaluation choices**. It was not a single-knob change.

### 1.2 Detailed explanation of each change

#### Window 480/480 + stride 240 — **large impact**

**What changed**

- Run 1: 864-sample input, predict center **256** steps, train stride **16** (heavy overlap).
- Run 2: **480** in, **480** out, stride **240** (50% overlap, same as Transfer baseline).

**Why it matters**

1. **Stride 16** creates many nearly identical windows → easy to memorize local patterns → high train score, weak val F1 (Run 1 kettle / washingmachine F1 ≈ 0.03–0.10).
2. **480/480** matches the Transfer recipe and gives the model a consistent temporal scale for multi-appliance events.
3. **Stride 240** reduces redundancy → better generalization and **~25× faster epochs** (37 s → 1.4 s on RTX 4090).

**Evidence in our runs:** Largest F1 jumps are on kettle (0.03 → 0.89 val) and washingmachine (0.11 → 0.83 val). Val MAE got worse (41 → 61 W), so this trade-off is real—not all metrics improved.

---

#### `training_targets: full_input` — **medium impact (tied to 480/480)**

**What it does**

- Run 1 (`output_window`): loss and labels only on the **center 256** timesteps inside the 864 input.
- Run 2 (`full_input`): loss on **all 480** output timesteps the model predicts.

**Why it matters**

- The state (ON/OFF) head sees supervision at every step in the window, not just the middle slice.
- Especially helps appliances with short ON bursts (kettle, microwave) and long cycles (washingmachine) where the center crop can miss event boundaries.

**Note:** This setting is meaningful together with `output_window_length: 480`. Changing to `full_input` alone without matching output length would not help.

---

#### `checkpoint: val_mae_minus_f1` — **medium impact**

**What it does**

- Run 1: save `best.pt` when **validation MAE** is lowest.
- Run 2: save when **normalized MAE − F1** is lowest (same idea as Transfer baseline checkpoint).

**Why it matters**

- The checkpoint rule decides **which epoch** you evaluate. Run 1 best epoch 42 optimized MAE; Run 2 best epoch 187 optimized a **balance** of MAE and F1.
- A model with good MAE can still have poor ON/OFF F1 (Run 1 fridge F1 OK but kettle terrible). The composite metric pushes the saved model toward better state detection.

**Caveat:** This does not change how the model learns each step—it changes **model selection**. Re-evaluating Run 1’s weights with `val_mae_minus_f1` selection could recover some F1, but would not fix stride-16 overfitting by itself.

See **§1.3** for the full mathematical definition.

---

#### `pred_on_source: state_head` — **medium impact (on reported F1)**

**What it does**

- Defines binary ON/OFF for **F1, plots, and saved predictions**:
  - `state_head`: ON if `sigmoid(state_logit) ≥ 0.5`
  - `power_threshold`: ON if denormalized power > appliance threshold
  - `combined`: ON if either condition is true

**Why it matters**

- `combined` can **inflate F1** by marking ON from power spikes even when the state head says OFF.
- `state_head` matches Transfer evaluation and aligns F1 with what the classification head actually learned.

**Caveat:** This affects **metrics and plots**, not the training loss (unless you also change labels). Comparing Run 1 and Run 2 F1 is fairest when both use `state_head`; if Run 1 used `combined`, part of the F1 gap may be evaluation definition, not only windowing.

---

#### `eval_reconstruction: overlap_mean` — **low direct impact on F1**

**What it does**

- After sliding-window inference, map window outputs back onto the CSV timeline.
- **`flat`**: concatenate windows in batch order → when stride < output length, CSV row indices go **backwards** → zig-zag waveform plots.
- **`overlap_mean`**: average predictions that cover the same CSV row → monotonic time axis.

**Why it matters**

- **Does not change training** or the per-batch validation loop during training.
- Fixes **final evaluation bundle**, waveform PNGs, and test metrics that use reconstructed timelines.
- Run 1 used `eval_stride: 256` with `output_window_length: 256` (no overlap) so `flat` was valid. Run 2 **requires** `overlap_mean` (or equivalent) for correct plots and consistent eval.

---

#### `early_stop_patience: 60`, `epochs: 200` — **small impact**

**What they do**

- Stop training if the checkpoint metric does not improve for 60 epochs.
- Allow up to 200 epochs before hard stop.

**Why it matters**

- Run 1 stopped at epoch 57; Run 2 ran 200 epochs with best at 187.
- Gives the smaller MultiNILM model more time to fit val houses, but most of the F1 gain came from window/checkpoint changes, not from “training longer” alone.

**Evidence:** Run 2 best epoch is very late (187), so patience 60 helped. But Run 1 with the old window setup would still overfit early even with 200 epochs.

---

#### Other settings (supporting, not primary)

| Setting | Role |
|---------|------|
| `checkpoint_mae_space: normalized` | Puts MAE on same scale as F1 in the composite checkpoint score |
| `weight_decay: 0`, `scheduler: none` | Match Transfer training recipe; minor for this comparison |
| `lambda_state: 1` | Weight of BCE state loss vs power loss; suggested sweep 2–3 next |

### 1.3 Checkpoint monitor — mathematical definition

During training, each validation epoch computes metrics on sliding-window batches, then **averages per batch** (Transfer baseline style). The checkpoint rule picks which epoch’s weights are saved as `best.pt`.

**Notation**

| Symbol | Meaning |
|--------|---------|
| \(A\) | Number of appliances (5 on UK-DALE) |
| \(B\) | Number of validation batches in one epoch |
| \(N_b\) | Flattened timesteps in batch \(b\) (all window outputs in the batch) |
| \(y_{b,t,a}, \hat{y}_{b,t,a}\) | Normalized true / predicted power for appliance \(a\) |
| \(z_{b,t,a}, \hat{z}_{b,t,a}\) | True / predicted ON/OFF (binary) for F1 |
| \(\sigma_a\) | Denormalization scale (std) for appliance \(a\) from training stats |

Implementation: `runner.py` → `_validation_batch_metrics`, `_state_f1_logs`, `_epoch_score`.

---

#### Per-batch power MAE

**Normalized MAE** (macro over appliances):

\[
\mathrm{MAE}^{\mathrm{norm}}_b = \frac{1}{A} \sum_{a=1}^{A} \frac{1}{N_b} \sum_{t=1}^{N_b} \left| \hat{y}_{b,t,a} - y_{b,t,a} \right|
\]

**MAE in watts** (denormalize first, then same macro mean):

\[
\mathrm{MAE}^{\mathrm{W}}_b = \frac{1}{A} \sum_{a=1}^{A} \frac{1}{N_b} \sum_{t=1}^{N_b} \left| \sigma_a \hat{y}_{b,t,a} - \sigma_a y_{b,t,a} \right|
\]

Epoch-level values are the batch mean:

\[
\mathrm{MAE}^{\mathrm{norm}} = \frac{1}{B} \sum_{b=1}^{B} \mathrm{MAE}^{\mathrm{norm}}_b,
\qquad
\mathrm{MAE}^{\mathrm{W}} = \frac{1}{B} \sum_{b=1}^{B} \mathrm{MAE}^{\mathrm{W}}_b
\]

Run 1 also logs `mae` from the training step (`MultiNILM_loss.py`): macro mean of **denormalized** absolute error in watts, then averaged over batches. That is the same *unit* as MAE-W but computed inside the loss forward pass rather than in `_validation_batch_metrics`.

---

#### Per-batch macro F1 (ON/OFF)

For each appliance \(a\), over all timesteps in batch \(b\):

\[
\mathrm{TP}_a = \sum_t \mathbf{1}\{z=1,\, \hat{z}=1\},
\quad
\mathrm{FP}_a = \sum_t \mathbf{1}\{z=0,\, \hat{z}=1\},
\quad
\mathrm{FN}_a = \sum_t \mathbf{1}\{z=1,\, \hat{z}=0\}
\]

\[
F1_a = \frac{2\,\mathrm{TP}_a}{2\,\mathrm{TP}_a + \mathrm{FP}_a + \mathrm{FN}_a}
\]

**Macro F1** (what `val_f1` uses for checkpoint):

\[
F1 = \frac{1}{A} \sum_{a=1}^{A} F1_a
\]

Predicted ON/OFF \(\hat{z}\) comes from `pred_on_source` (Run 2: `state_head`, i.e. ON when sigmoid(logit) ≥ 0.5). True \(z\) follows `state_label_source` (threshold on watts or CSV labels).

Epoch-level:

\[
F1_{\text{epoch}} = \frac{1}{B} \sum_{b=1}^{B} F1_b
\]

(Micro F1 is also logged as `val_mif1` but is **not** used for checkpoint selection.)

---

#### Run 1: `checkpoint_monitor: mae` (or `val_mae`)

**Selection rule:** save epoch \(e\) that minimizes batch-averaged validation MAE in **watts**:

\[
e^{*} = \min_{e} \;\mathrm{MAE}^{\mathrm{W}}(e)
\]

**Direction:** lower is better (`mode = min`).

**Example (Run 1):** best epoch 42, checkpoint score **43.58** ≈ MAE-W at that epoch (watts). F1 is **not** part of this score — a checkpoint can win on power error while ON/OFF F1 is poor (Run 1 kettle val F1 ≈ 0.03).

---

#### Run 2: `checkpoint_monitor: val_mae_minus_f1`

**Composite score** (Transfer baseline: `normalized_MAE − F1`):

\[
S(e) = \mathrm{MAE}^{\mathrm{ckpt}}(e) - F1_{\mathrm{epoch}}(e)
\]

where MAE-ckpt is chosen by `checkpoint_mae_space`:

| `checkpoint_mae_space` | MAE term used in S |
|------------------------|--------------------|
| `normalized` (Run 2 default) | MAE-norm — typical range ~0.05–1.0 |
| `watts` | MAE-W — typical range tens of watts |

**Selection rule:**

\[
e^{*} = \min_{e} \; S(e)
\]

**Direction:** lower is better.

**Why subtract F1?** Both terms are treated as “costs” in a **minimize** framework:

- Lower MAE → good (reduces \(S\))
- Higher F1 → good (subtracting a larger F1 also reduces \(S\))

So the rule prefers epochs that **both** regress power accurately **and** classify ON/OFF well. It is a simple scalar trade-off, not a weighted sum with a tunable \(\lambda\):

\[
S = \mathrm{MAE}^{\mathrm{ckpt}} - F1
\]

**Scale matching:** With `checkpoint_mae_space: normalized`, MAE and F1 are both on a similar scale (roughly 0.1–1), so neither term dominates purely because of units. With `watts`, MAE would be ~40–60 W while F1 is in [0, 1], and MAE would dominate — normalized space is preferred for balancing.

**Example (Run 2, best epoch 187):**

\[
S = -0.262
\]

So MAE-norm − F1 = −0.262. If F1 ≈ 0.70 at that epoch, then MAE-norm ≈ 0.44. The score is **negative** when F1 is larger than normalized MAE — that is expected and desirable.

---

#### Side-by-side summary

| | Run 1 (`mae`) | Run 2 (`val_mae_minus_f1`) |
|---|---------------|----------------------------|
| **Formula** | min MAE-W | min (MAE-norm − F1) |
| **Uses F1?** | No | Yes |
| **MAE space** | Watts (from step logs) | Normalized (config default) |
| **Best epoch** | 42 (early, MAE-optimal) | 187 (late, MAE–F1 balance) |
| **Typical best score** | ~43 W | ~−0.26 (unitless composite) |

**Important:** Final reported tables (validation/test CSV after `best.pt`) recompute MAE in **watts** with overlap reconstruction and post-processing. The checkpoint formulas above apply only to **which epoch is saved during training**, not necessarily to the watt-scale MAE numbers in the evaluation report.

### Related framework fixes (Run 2 period)

These code changes support the new windowing and plotting setup:

| Fix | File(s) | Purpose |
|-----|---------|---------|
| Auto `overlap_mean` when `eval_stride < output_window_length` | `adapters/config.py`, `adapters/common.py` | Prevent non-monotonic CSV x-axis even if yaml says `flat` |
| Sort waveform points by CSV row before plotting | `evaluation/plots.py` | Safety net against zig-zag lines |
| Full-split inference for live waveforms | `evaluation/live_monitor.py` | Do not limit waveform plots to `plot_max_batches` |
| Pass state **probabilities** (not binarized) into overlap reconstruction | `adapters/multinilm.py` | Correct overlap averaging for ON/OFF |
| Secondary `lr_scheduler:` preset block | `adapters/config.py`, yaml | Keep scheduler params while `scheduler: none` |

---

## 2. Training summary

| Item | Run 1 | Run 2 |
|------|-------|-------|
| Epochs completed | 57 (early stop) | 200 |
| Best epoch | 42 | 187 |
| Best checkpoint score | 43.58 (`mae`, lower better) | −0.262 (`val_mae_minus_f1`, lower better) |
| Total training time | 34m 50s | 4m 35s |
| Avg time / epoch | ~37s | ~1.4s |
| GPU | RTX 4090 | RTX 4090 |

Run 2 is much faster per epoch because `input_stride: 240` produces far fewer windows than `stride: 16`.

**Loss curves (qualitative):**

- **Run 1:** Training loss decreases smoothly; validation loss stays volatile after ~epoch 10 (typical overfit on heavily overlapping windows).
- **Run 2:** Train/val track together until ~epoch 50; validation loss then shows spikes (composite checkpoint metric + per-batch val aggregation). Best checkpoint still selected at epoch 187.

---

## 3. Validation metrics (primary focus)

Run 2 validation numbers below are from the **updated model** (480/480 config + OFF-norm gate blend `off_norm = -mean/std`).

| Appliance | Run 1 MAE (W) | Run 2 MAE (W) | Run 1 F1 | Run 2 F1 |
|-----------|---------------|---------------|----------|----------|
| kettle | 40.0 | **3.2** | 0.027 | **0.899** |
| fridge | **17.2** | 19.8 | **0.858** | 0.852 |
| dishwasher | 59.9 | **9.5** | 0.053 | **0.439** |
| washingmachine | 55.0 | **15.6** | 0.105 | **0.843** |
| microwave | 31.5 | **29.4** | **0.668** | 0.662 |
| **overall** | 40.7 | **15.5** | 0.342 | **0.739** |

| Aggregate | Run 1 | Run 2 |
|-----------|-------|-------|
| Val SAE | 34.8 | **10.6** |

Per-appliance SAE (Run 2): kettle 2.08, fridge 7.11, dishwasher 7.69, washingmachine 9.13, microwave 26.86.

**Validation conclusion:** Run 2 wins on **both overall MAE** (40.7 → **15.5 W**) and **overall F1** (0.34 → **0.74**). The OFF-norm gate fix removes the mean-watt spike when appliances are OFF and greatly improves power MAE (especially kettle, dishwasher, washingmachine) while keeping strong ON/OFF F1.

---

## 4. Test metrics (cross-house H2, secondary)

| Appliance | Run 1 MAE (W) | Run 2 MAE (W) | Run 1 F1 | Run 2 F1 |
|-----------|---------------|---------------|----------|----------|
| kettle | 57.3 | 72.0 | 0.018 | **0.099** |
| fridge | 50.2 | 46.9 | 0.228 | **0.350** |
| dishwasher | 145.4 | 108.7 | **0.032** | 0.016 |
| washingmachine | 89.5 | 106.8 | 0.005 | **0.359** |
| microwave | 45.9 | 45.9 | 0.161 | 0.159 |
| **overall** | 77.7 | **76.1** | 0.089 | **0.197** |

| Aggregate | Run 1 | Run 2 |
|-----------|-------|-------|
| Test micro F1 | 0.066 | **0.335** |
| Test SAE | 67.3 | 64.0 |

**Test conclusion:** Run 2 still wins on overall F1 (~2×) and slightly on MAE. Both runs show a large validation→test F1 drop (cross-house domain gap). Run 2’s absolute test F1 is higher, but the val→test gap is larger because validation F1 improved more.

---

## 5. Validation vs test gap (Run 2)

*Test metrics below are from the earlier Run 2 eval before the OFF-norm gate fix; re-run test after retraining to refresh.*

| Appliance | Val F1 | Test F1 | F1 gap |
|-----------|--------|---------|--------|
| kettle | 0.899 | 0.099 | −0.800 |
| fridge | 0.852 | 0.350 | −0.502 |
| dishwasher | 0.439 | 0.016 | −0.423 |
| washingmachine | 0.843 | 0.359 | −0.484 |
| microwave | 0.662 | 0.159 | −0.503 |
| **overall** | **0.739** | **0.197** | **−0.542** |

| Metric | Val | Test | Gap |
|--------|-----|------|-----|
| MAE (W) | 15.5 | 76.1 | +60.6 |

This pattern matches the intentional **cross-house** test split (H2 vs train/val H1+H5). It is not necessarily a regression from the config change.

---

## 6. Overall verdict

| Criterion | Better run |
|-----------|------------|
| Validation F1 (primary goal) | **Run 2** |
| Validation MAE | **Run 2** |
| Test F1 | **Run 2** (pending re-eval with OFF-norm fix) |
| Test MAE | **Run 2** (slightly; pending re-eval) |
| Training efficiency | **Run 2** |
| Generalization val→test (F1 gap) | Run 1 (smaller gap, but lower absolute test F1) |

**Recommendation:** Keep **Run 2** (`480/480`, stride 240, `val_mae_minus_f1`, `state_head`, `overlap_mean`, OFF-norm gate blend) as the MultiNILM UK-DALE baseline. Updated Run 2 beats Run 1 on validation MAE and F1.

---

## 7. Suggested next steps

1. **`lambda_state` sweep** (2–3) to improve power regression without sacrificing F1.
2. **Dishwasher** remains weak on test in both runs — consider per-appliance loss weight or threshold review.
3. **Cross-house gap:** acceptable if val-first; for better test F1, add houses or domain-adaptation later.
4. **Compare against `transfer_multi_appliance`** on the same split using `docs/model_comparison_ukdale.md`.
5. **Waveform plots:** use `overlap_mean` and synced code; remaining pred jitter is model quality, not x-axis reconstruction.

---

## 8. Run metadata (reference)

### Run 1

```json
{
  "experiment_id": "ukdale_first_model",
  "checkpoint_monitor": "mae",
  "windowing": {
    "input_window_length": 864,
    "output_window_length": 256,
    "output_alignment": "center",
    "input_stride": 16,
    "eval_stride": 256,
    "eval_reconstruction": "flat",
    "training_targets": "output_window"
  },
  "epochs_completed": 57,
  "best_epoch": 42
}
```

### Run 2

```json
{
  "experiment_id": "ukdale_first_model",
  "checkpoint_monitor": "val_mae_minus_f1",
  "checkpoint_mae_space": "normalized",
  "windowing": {
    "input_window_length": 480,
    "output_window_length": 480,
    "output_alignment": "end",
    "input_stride": 240,
    "eval_stride": 240,
    "eval_reconstruction": "overlap_mean",
    "training_targets": "full_input"
  },
  "epochs_completed": 200,
  "best_epoch": 187
}
```

---

*Last updated: Run 2 validation metrics refreshed after OFF-norm gate fix (overall MAE 15.5 W, F1 0.739).*

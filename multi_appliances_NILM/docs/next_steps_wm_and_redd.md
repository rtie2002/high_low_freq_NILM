# Next steps: washing-machine ON failure and REDD limits

Status note after UK-DALE `multinilm_fractional` (H1+H5 → H2) and the restored REDD 6 s export (2026-08-12).

## Decision: architecture vs new domain data vs hyperparameter tuning

**Choose new domain data (and DA). Do not spend the next cycle on architecture or more HP search.**

| Option | Do it now? | Why |
|--------|------------|-----|
| **1. New domain data** | **Yes — first** | WM ON fails because H2’s aggregate signature is different and weak. More houses (REDD now, REFIT next) + unsupervised DA is the only way to attack that. It is also what the paper claim needs. |
| **2. Hyperparameter tuning** | **No — pause** | Current yaml already has a working fridge/kettle setup. More LR / window / `k` / dropout sweeps will not create H2 washer activations or fix a state head that never fires. Tune later, after a second dataset + DA baseline exist, and only a few knobs (`pos_weight`, gate threshold, λ). |
| **3. Architecture upgrade** | **No — last** | Dual-stream / extra fractional channels already failed to move WM. The backbone *can* learn ON (kettle, fridge). A wider TCN will not invent a REDD/UK-DALE house-2 washer that is not in the data. |

Concrete next work under option 1:

1. Train source-only on REDD (H1+H3 → H2; **no H2 WM column**).
2. Same `multinilm_fractional`, turn DA on, compare B0 / B1 / M0.
3. Then REFIT (many homes, real WM), not another stem.

One exception: a **single** loss/gating change (stronger WM `pos_weight` or ON-masked power) is not “architecture” and is worth one UK-DALE rerun if DA still misses WM ON.

## Current problems

### 1. Washing-machine head does not learn ON (UK-DALE test)

On House 2 test plots the WM **state head** stays off (or only fragments) through the heater phase (~2 kW). With

- `evaluation.pred_on_source: state_head`
- `architecture.gate_mode: soft_train_hard_eval`
- `evaluation.regate_power_with_pred_on: true`

a missed ON also zeros watts, so the red power line looks dead.

This is **not** a missing CSV label. Train WM ON is long, clean Algorithm-1 cycles (H1: 89 events, ~90 min median). The classifier fails the decision boundary.

Why WM ON is harder than kettle / microwave:

| | Kettle / microwave | Washing machine |
|--|--|--|
| ON length | seconds–minutes | ~40–90 min continuous label |
| When ON, share of aggregate | often dominates | often buried (train: ~70% of ON time WM &lt; 25% of aggregate) |
| Power while “ON” | always high | ~53% of train ON samples &lt; 200 W (motor / idle inside the cycle) |
| Test cue (H2) | aggregate ON median ~3 kW | aggregate ON median ~428 W vs OFF ~210 W |

Fridge (~46% ON, periodic ~90 W) and kettle (huge spike) give the shared backbone an easy gradient. WM looks like a long mid-power blob; MSE is happy predicting near-zero; BCE at `gate_threshold: 0.5` never fires.

Cross-house shift hits **ON first**: train H1/H5 aggregate when WM ON is still elevated (~1350 W median); H2 that cue collapses.

Dual-stream / extra fractional channels did **not** fix this. Architecture is not the bottleneck.

### 2. REDD cannot copy the UK-DALE H1+H5 → H2 WM protocol

Official REDD **lists** a washer dryer on House 2 (NILMTK: `washer_dryer` → meter **7**). The channel exists; **usage does not**.

Raw meter 7: mean ~2 W, max ~55 W, essentially standby. Algorithm-1 → **0 ON**. Known REDD quirk (NILMTK activation counts also report 0 washer events in building 2).

| House | Calendar span | Actual 6 s coverage (after gaps) | WM usable? |
|-------|----------------|----------------------------------|------------|
| 1 | 18 Apr–24 May 2011 (~5.2 wk) | ~2.6 wk | yes (~2.3% ON, ch 10+20) |
| 2 | ~5.0 wk | ~2.0 wk | **no** |
| 3 | 16 Apr–30 May (~6.4 wk) | ~2.4 wk | yes (~2.9% ON, ch 13) |

REDD looks like 5–6 weeks on the calendar; long dropouts leave only **~2–2.6 weeks** of real samples per house.

Fridge / dishwasher / microwave on H1–H3 are usable. H4 washer looks alive but H4–H6 do not share the same 4-appliance set, so they stay out of the default protocol.

Exported files:

- `dataset_preprocess/created_data/REDD/redd_house{1,2,3}_lf_6s.csv`
- `multi_appliances_NILM/datasets/redd/{training,validating,testing}/`

Current split `prepare_redd_crosshouse_split.py` is H1+H3 → H2. **Do not report H2 WM F1/MAE** from that split.

---

## What we can do next (priority order)

Do **not** start another stem / TCN / fusion upgrade. Order: honest protocol → more domains / DA → WM-aware learning signal → architecture only if ON is still dead.

### A. Fix the evaluation protocol (this week, no new model)

1. **UK-DALE** — keep H1+H5 labeled → H2 unlabeled/eval for kettle, fridge, dishwasher, microwave. Keep reporting WM on H2, but treat it as the hard transfer case (not a reason to keep changing the backbone).
2. **REDD 4-appliance** — two tables, not one:
   - **Cross-house H1+H3 → H2** for fridge, dishwasher, microwave only (drop or N/A the WM column).
   - **WM-only transfer** H1 ↔ H3 (time hold-out on the source house for val). That is the only honest REDD WM number.
3. Write this split rule into `experiment_redd.yaml` comments and the paper protocol section when results exist.
4. Optional: add a one-line skip / `nan` in metrics when an appliance has 0 test ON so H2 WM cannot silently look like “perfect OFF”.

### B. Add domain data (highest leverage for the actual failure)

WM ON fails because the **aggregate signature is house-specific and low-SNR**, not because the CNN is too shallow.

1. **Train and score REDD** with the protocol in A (source H1+H3, target H2 without WM). Confirms the pipeline on a second dataset.
2. **Turn on unsupervised DA** (`domain_adaptation.enabled: true`, existing MMD+CORAL / EGC) with a clean comparison:
   - B0: source-only (no DA)
   - B1: naive global UDA
   - M0: proposed DA
   Same windows, loss, seed. House-2 UK-DALE and House-2 REDD (3 appliances) as unlabeled targets.
3. **REFIT next** (not more REDD houses). REFIT has many homes and real WM usage; it is the only way to get more than two REDD washer domains. Build a UK-DALE-style 6 s (or native 8 s) multi-appliance CSV with the same `{app}_power` / `{app}_on` schema.
4. Do not mix UK-DALE and REDD in one training run until each dataset has its own honest split and a source-only baseline.

### C. Give the WM state head a clearer learning signal (if A+B still miss ON)

Do **one** of these, then retrain; do not stack them.

1. **ON-masked / ON-emphasized power loss** so the ~2 kW heater samples are not drowned by 95% OFF MSE.
2. **Stronger WM `pos_weight`** (or a cap-aware per-appliance weight). Auto `(1−p)/p` is only ~17 for WM vs ~150–180 for kettle/microwave — WM is rare *and* long, so BCE still under-fires.
3. **Lower WM gate threshold** or delay hard regate until the state head is calibrated (train soft, eval with a WM-specific threshold from val PR curve).
4. **Do not relabel the whole cycle as OFF** just to make F1 easier. Algorithm-1 long ON is the paper label; if you try a “heater-only ON” variant, keep it as an ablation, not a silent CSV change.

### D. New features only if they match WM character

Useful if they target **long multimodal cycles** and **ON while buried in the aggregate**:

- event / cycle memory (ON duration so far, recent energy)
- heater vs drum timescale (short high-power vs long mid-power)
- ON-masked fractional or stats, not another generic `k` on the full trace

Skip another dual-stream stem or extra α channels unless C shows the state head is now firing and power is still wrong.

### E. Architecture last

Shared TCN + per-appliance heads already learn fridge and kettle ON. Wider stages / more blocks will not invent H2’s WM signature from H1. Revisit architecture only after A–C (and DA) if WM recall is still near zero.

---

## Suggested run order

1. Document REDD protocol (this file + `experiment_redd.yaml` comments).
2. Source-only REDD train: H1+H3 → H2, report 3 appliances; separate H1↔H3 WM number.
3. Same model, UK-DALE source-only vs DA (B0 / B1 / M0) — existing `multinilm_fractional` yaml, DA switch only.
4. One WM loss/gating change (C) on UK-DALE; keep if H2 WM ON recall rises without killing fridge.
5. REFIT preprocess + third-domain split when 2–4 are stable.

## What not to do

- Report REDD H2 washing-machine F1 as a transfer result.
- Spend the next iteration on fusion / dual-stream / more fractional `k`.
- Put failed pilots in the paper tables; keep them as comparison only if they are honest baselines (e.g. global UDA hurting MAE).

## Commands already used

```text
python dataset_preprocess/redd_processing_multi_appliance.py --split_houses 1,2,3 --full_range
python multi_appliances_NILM/scripts/prepare_redd_crosshouse_split.py --force-copy
```

Train REDD (when ready) with `config/experiment_redd.yaml` + `config/models/multinilm_fractional.yaml`. Change `experiment_id` on the model yaml when switching dataset or DA mode.

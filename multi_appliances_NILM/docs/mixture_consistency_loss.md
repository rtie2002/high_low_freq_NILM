# Mixture consistency loss (from Conv-NILM-Net)

Status: design note — not wired into training yet.  
Related model: `multinilm_fractional` (shared TCN + per-appliance heads).

---

## 1. Where this idea comes from

**Paper:** Alami et al., *Conv-NILM-Net, a causal and multi-appliance model for energy source separation*, arXiv:2208.02173 (v2).

**Core framing (paper §1, Eq. 1):** NILM is single-channel source separation

\[
\bar y(t) = \sum_{i=1}^{C} y^{(i)}(t) + e(t)
\]

- \(\bar y\): aggregate (mixture)
- \(y^{(i)}\): appliance \(i\) (one “channel”)
- \(e(t)\): noise / unmodelled load

**Architecture idea (paper §3):** Conv-TasNet-style pipeline

```text
aggregate → Encoder → Z
         → Separator → masks m_1 … m_C
         → s_i = Z ⊙ m_i
         → Decoder → ŷ_1 … ŷ_C
```

One forward pass outputs **C appliance power channels** at once (true multi-appliance source separation), not one network per appliance.

**Important paper detail we reuse carefully:**  
In speech Conv-TasNet, masks often satisfy \(\sum_i m_i = 1\) (perfect reconstruction).  
Conv-NILM-Net **relaxes** that, because in NILM \(e(t) \neq 0\) (other loads, noise). See paper §3.1 around Eq. (3).

We do **not** plan to replace MultiNILM with their encoder–separator–decoder.  
We only borrow the **mixture / additivity** inductive bias as a loss term.

---

## 2. What we take (and what we skip)

| From the paper | Take? | Notes |
|----------------|-------|--------|
| Multi-appliance = multi-channel separation | Yes (conceptually) | Matches our multi-head outputs |
| Soft mixture / residual (do not force \(\sum m_i=1\)) | Yes | Core of our loss design |
| Sum MSE over appliances (not mean) | Already close | Our `loss_power` is sum over appliances |
| Full Conv-TasNet mask network | No | Keep MultiNILM + fractional frontend |
| Causal online TCN only | No | We use offline windows |
| GLU in separator | Optional later | Not required for mixture loss |

---

## 3. How it maps to *our* case

### Batch tensors (already in the codebase)

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `x` | `(B, T)` | Normalized aggregate |
| `y` | `(B, T, A)` | Normalized appliance powers (A=5) |
| `z` | `(B, T, A)` | ON/OFF labels |
| `power_pred` | `(B, T, A)` | Model output (normalized) |

UK-DALE appliances (see `config/experiment_ukdale.yaml`):

`kettle, fridge, dishwasher, washingmachine, microwave`.

### Physical identity (watt space)

\[
P_{\mathrm{agg}}(t)
  = \sum_{k=1}^{5} P_k(t) + P_{\mathrm{other}}(t)
\]

We only supervise **five** appliances. \(P_{\mathrm{other}}\) (lights, TV, …) is never predicted.

**Therefore we must not enforce**

\[
\sum_{k=1}^{5} \hat P_k = P_{\mathrm{agg}}
\]

That would force unexplained load into the five heads and hurt accuracy.

### Constraint that *does* fit our case

Predicted sum of the five appliances should **not exceed** the aggregate (physically impossible):

\[
L_{\mathrm{mix}}
  = \mathrm{mean}\Big(
      \mathrm{ReLU}\big(
        \sum_k \hat P_k^{(\mathrm{W})} - P_{\mathrm{agg}}^{(\mathrm{W})}
      \big)^2
    \Big)
\]

- If \(\sum \hat P_k \le P_{\mathrm{agg}}\): ReLU is 0 → no penalty (slack = other load).
- If \(\sum \hat P_k > P_{\mathrm{agg}}\): penalty → stops channels from “stealing” power.

This is the Conv-NILM-Net spirit (mixture + residual \(e(t)\)) adapted to a **partial** appliance set.

### Critical: compute in **watts**, not normalized space

`experiment_ukdale.yaml` uses **different** mean/std per appliance and for aggregate (e.g. fridge `std=50`, dishwasher `std=1000`).  
Summing normalized channels is meaningless.

Implementation must:

1. Denormalize `power_pred` per appliance → watts  
2. Denormalize `x` with aggregate stats → watts  
3. Apply \(L_{\mathrm{mix}}\) in watts  

Gradients still flow through denormalization (affine), so training stays end-to-end.

---

## 3b. Detailed walkthrough: how \(L_{\mathrm{mix}}\) works in *our* pipeline

### Step 0 — What one training step already does

```text
batch = (x, y, z)
  x: (B, T)      normalized aggregate
  y: (B, T, 5)   normalized appliance powers
  z: (B, T, 5)   ON/OFF

power_pred, state_logits = model(x)     # MultiNILMFractional
L_NILM = power MSE + balanced state BCE   # existing MultiNILMLoss
```

\(L_{\mathrm{mix}}\) is an **extra scalar** computed from `power_pred` and `x` only (no need for `y`/`z`).

### Step 1 — Undo normalization (same stats as the dataloader)

From `config/experiment_ukdale.yaml`:

| Signal | mean | std |
|--------|------|-----|
| aggregate | 400 | 500 |
| kettle | 100 | 500 |
| fridge | 50 | 50 |
| dishwasher | 700 | 1000 |
| washingmachine | 400 | 700 |
| microwave | 60 | 300 |

Training / dataloader uses:

\[
x_{\mathrm{norm}} = \frac{P_{\mathrm{agg}} - 400}{500},
\qquad
y_{k,\mathrm{norm}} = \frac{P_k - \mu_k}{\sigma_k}
\]

So for the loss we invert:

\[
\hat P_k^{(\mathrm{W})}
  = \hat y_{k,\mathrm{norm}} \cdot \sigma_k + \mu_k
\]

\[
P_{\mathrm{agg}}^{(\mathrm{W})}
  = x_{\mathrm{norm}} \cdot 500 + 400
\]

In code (broadcast over batch and time):

```python
# power_pred: (B, T, 5), x: (B, T)
# app_std, app_mean: shape (5,) registered from experiment yaml
power_w = power_pred * app_std + app_mean          # (B, T, 5) watts
agg_w   = x * agg_std + agg_mean                   # (B, T) watts
```

**Why not sum in normalized space?**  
One unit of fridge-normalized power ≈ 50 W; one unit of dishwasher-normalized ≈ 1000 W.  
`power_pred.sum(-1)` would treat them as equal “counts,” which is wrong physically.

### Step 2 — Sum the five predicted channels

```python
sum_pred_w = power_w.sum(dim=-1)   # (B, T)  watts
```

At each time \(t\) this is \(\sum_{k=1}^{5} \hat P_k(t)\): the model’s claim for “how much of the house is these five appliances.”

### Step 3 — Overshoot vs aggregate (ReLU)

```python
diff = sum_pred_w - agg_w          # (B, T)
overshoot = diff.clamp_min(0.0)    # ReLU(diff)
```

| Case at time \(t\) | `diff` | `overshoot` | Meaning |
|--------------------|--------|-------------|---------|
| sum = 600 W, agg = 800 W | −200 | **0** | OK: 200 W can be “other” |
| sum = 800 W, agg = 800 W | 0 | **0** | OK: no other load this instant |
| sum = 1200 W, agg = 800 W | +400 | **400** | Impossible → penalize |

ReLU is the “one-sided” gate: we only care when predictions **overshoot**; undershoot is allowed because of \(P_{\mathrm{other}}\).

### Step 4 — Turn overshoot into a scalar loss

```python
L_mix = overshoot.pow(2).mean()   # mean over B and T
```

Squaring: large violations (e.g. 2000 W overshoot) hurt much more than tiny noise.  
`mean`: average over the whole batch/window so the scale is stable vs batch size.

Full formula again:

\[
L_{\mathrm{mix}}
  = \frac{1}{BT}
    \sum_{b=1}^{B}\sum_{t=1}^{T}
    \Big[
      \mathrm{ReLU}\Big(
        \sum_{k=1}^{5}\hat P_{b,t,k}^{(\mathrm{W})}
        - P_{\mathrm{agg},b,t}^{(\mathrm{W})}
      \Big)
    \Big]^2
\]

### Step 5 — Add to the existing training loss

```python
L = L_NILM + lambda_mix * L_mix
# (+ domain term if DA enabled)
```

- `lambda_mix = 0` → behavior unchanged (feature off).  
- Start small (e.g. `0.05`): mix is an **auxiliary** constraint, not the main NILM objective.  
- Too large → model may under-predict everything (always stay under aggregate) → weak F1/MAE.

`backward()` then updates MultiNILM (and fractional frontend if trainable) so that heads stop assigning more watts than the meter shows.

### Concrete numeric toy (one timestep)

Suppose at \(t\):

- True aggregate = **900 W** → \(x = (900-400)/500 = 1.0\)
- Model predicts (already denormed for clarity):

| Appliance | \(\hat P\) (W) |
|-----------|----------------|
| kettle | 0 |
| fridge | 100 |
| dishwasher | 0 |
| washingmachine | 500 |
| microwave | 0 |
| **sum** | **600** |

`diff = 600 - 900 = -300` → ReLU = 0 → \(L_{\mathrm{mix}}\) contribution **0**. Fine.

Now a bad prediction:

| Appliance | \(\hat P\) (W) |
|-----------|----------------|
| kettle | 2000 |
| fridge | 100 |
| dishwasher | 0 |
| washingmachine | 0 |
| microwave | 800 |
| **sum** | **2900** |

`diff = 2900 - 900 = 2000` → ReLU = 2000 → term \(2000^2 = 4\times10^6\).  
Gradient pushes kettle/microwave (and shared encoder) to lower those powers.

### Gradient intuition (what “stops stealing” means)

- Only timesteps with overshoot get gradient from \(L_{\mathrm{mix}}\).  
- Gradient flows into **every** appliance that contributed to the sum (and into shared TCN features).  
- Combined with per-appliance MSE on labeled source data: MSE says “match ground truth”; mix says “don’t invent watts that aren’t in the meter.”  
- On **unlabeled H2**, if we only have `x`, we can still compute \(L_{\mathrm{mix}}\) (no `y` needed) as a weak physics regularizer.

### What this does *not* do

- Does **not** force \(\sum \hat P_k = P_{\mathrm{agg}}\) (other load remains free).  
- Does **not** fix brand/house signature shift by itself (still need AdaBN / better features for WM).  
- Does **not** replace SGN: SGN zeros power when OFF; mix stops **joint** overshoot when several heads are ON.

### Pseudocode in one place (copy-paste mental model)

```python
# inside MultiNILMLoss.forward / adapter.step
power_w = power_pred * app_std + app_mean       # (B,T,A) watts
agg_w   = x * agg_std + agg_mean                # (B,T) watts
L_mix   = (power_w.sum(-1) - agg_w).clamp_min(0).pow(2).mean()

L = L_nilm + lambda_mix * L_mix
```

Wire `app_mean/std` and `agg_mean/std` once from `adapters/dataloader.py` (`NormalizationStats` / `NILMDataLoader.norm`), same source as `denorm_to_watts`.

---

## 4. How we implement it (planned wiring)

No change to `model/MultiNILM_fractional.py` (wrapper stays frontend → backbone).

### 4.1 YAML (`config/models/multinilm_fractional.yaml`)

```yaml
loss:
  # ... existing keys ...
  lambda_mix: 0.05          # start small; 0 = off
  mix_mode: overshoot       # ReLU(sum_pred - agg)^2 in watts
```

Optional later modes (not required for v1):

- `overshoot` — recommended default  
- `l2_with_residual` — learn / ignore residual head (heavier)

### 4.2 Loss (`model/MultiNILM_loss.py`)

Add something equivalent to:

```python
def mixture_overshoot_loss(
    power_pred_norm,   # (B, T, A)
    aggregate_norm,    # (B, T)
    appliance_mean,    # (A,)
    appliance_std,     # (A,)
    agg_mean: float,
    agg_std: float,
) -> torch.Tensor:
    # watts
    power_w = power_pred_norm * appliance_std + appliance_mean
    agg_w = aggregate_norm * agg_std + agg_mean
    overshoot = (power_w.sum(dim=-1) - agg_w).clamp_min(0.0)
    return overshoot.pow(2).mean()
```

Total objective:

\[
L = L_{\mathrm{NILM}} + \lambda_{\mathrm{mix}} \, L_{\mathrm{mix}}
  \quad (+ \text{optional domain term if DA on})
\]

Log `loss_mix` next to `loss_power` / `loss_state`.

### 4.3 Adapter (`adapters/multinilm.py` → `step`)

After `power_pred, state_logits = model(x)`:

1. Pass `x` (aggregate) into the loss (or compute \(L_{\mathrm{mix}}\) in `step` and add to `out.loss`).  
2. Read denorm stats from `self._data_loader()` (same source as `denorm_to_watts` / `loss_scale`).  
3. If `lambda_mix == 0`, skip (zero cost).

`MultiNILMFractionalAdapter` inherits `step` → no separate fractional adapter change if logic lives in base `MultiNILMAdapter` + shared loss.

### 4.4 Unlabeled target (H2) — optional

On a target batch that only has aggregate `x_T`:

- Supervised \(L_{\mathrm{NILM}}\) is unavailable / unused.  
- \(L_{\mathrm{mix}}\) still needs only `power_pred(x_T)` and `x_T`.

Can be used as a light unsupervised regularizer on H2 (same spirit as “use unlabeled aggregate”), without UM-Adapt complexity.

---

## 5. How we put this restriction (soft loss, not a hard clamp)

**Idea in one line:** make sure the five appliance outputs (in watts) do not overshoot the aggregate — nothing else.

We do **not** hard-clip the network output (`power_pred = min(power_pred, …)`).  
We add a **soft training penalty**: if overshoot happens, loss goes up and gradients push the model to stop.

```text
restriction type: soft (loss)
where:            training objective only
when active:      lambda_mix > 0
formula:          L += λ_mix * mean( ReLU(Σ P̂_k^(W) − P_agg^(W) )² )
```

### Why soft, not hard?

| Soft \(L_{\mathrm{mix}}\) | Hard clamp on outputs |
|---------------------------|------------------------|
| Model *learns* not to overshoot | Masks a bad prediction at the last second |
| Gradients reach shared TCN / heads | Often no learning signal |
| Can turn off with `lambda_mix: 0` | Always on, harder to ablate |

### Where it sits in the forward/train path

```text
CSV watts
  → normalize (global μ,σ)
  → MultiNILMFractional → power_pred, state   [existing architecture restrictions apply here]
  → L_NILM (MSE + BCE)                         [existing loss restrictions]
  → (+ NEW) denorm to watts → L_mix overshoot   [this restriction]
  → L_total.backward()
```

Eval-time watt clamps (`min_power_watts`, `max_on_power`) stay as they are; they are **post-process**, not this training restriction.

---

## 5b. How many restrictions we already have (inventory)

Before adding \(L_{\mathrm{mix}}\), the pipeline already has several constraints. They act at different layers.

### A. Inside the **model** (architecture — always on while training/inferring)

| # | Restriction | What it does | Where |
|---|-------------|--------------|--------|
| 1 | **SGN state gate** | If state looks OFF, power is pulled toward `off_norm` instead of raw power head | `MultiNILM.py` → `power = gate * power_raw + (1-gate)*off_norm` |
| 2 | **Per-appliance `off_norm`** | OFF baseline in *normalized* space (from experiment config) | `appliance_off_norm_normalized` → each head |
| 3 | **Gate mode** | `soft_train_hard_eval`: soft gate in train, hard threshold at eval | `gate_mode` in yaml |

These are **per-appliance** (each head alone). They do **not** look at “sum of five vs aggregate.”

### B. Inside the **training loss** (already in `MultiNILMLoss` / optim)

| # | Restriction | What it does | Where |
|---|-------------|--------------|--------|
| 4 | **Power MSE** | Each appliance power must match label \(y\) | `L_power` |
| 5 | **State BCE** (+ optional `pos_weight: auto`) | ON/OFF must match label \(z\); rare ON up-weighted | `L_state` |
| 6 | **Task balance** | `task_balance: equal` rescales state vs power magnitudes | `_balanced_state_term` |
| 7 | **Weight decay** | L2 on weights (`weight_decay: 0.0001`) | optimizer |
| 8 | **Gradient clip** | `gradient_clip: 1.0` caps update size | training loop |
| 9 | **Domain loss (optional)** | CORAL/MMD if `domain_adaptation.enabled` and `lambda_domain>0` | currently **off** in fractional yaml (`enabled: false`) |

### C. **Eval / post-process only** (not training gradients)

| # | Restriction | What it does | Where |
|---|-------------|--------------|--------|
| 10 | **`min_power_watts`** | Tiny predicted power → treat as 0 for metrics/plots | `evaluation.power_postprocess` |
| 11 | **`max_on_power_watts`** | Cap absurd ON power when postprocess enabled | `experiment_ukdale.yaml` |
| 12 | **Denorm `max(·,0)`** | Numpy denorm floors negative watts at 0 for reporting | `NormalizationStats.denorm` |

### D. **Not** a power restriction

| Piece | Note |
|-------|------|
| Fractional frontend | Feature expand (raw + α); not a power sum constraint |
| PAD-lite cross-appliance | Soft feature mix between heads; no watt budget |
| Window length / stride | Data sampling, not output physics |

### Count (current fractional setup)

- **Hard-ish architecture:** ~3 (gate + off_norm + gate mode)  
- **Soft train losses / optim:** ~5 active (MSE, BCE, task balance, weight decay, grad clip); DA optional/off  
- **Eval-only:** ~3 (min power, max ON power, denorm floor)  
- **New \(L_{\mathrm{mix}}\):** +1 soft train restriction — the **first** that constrains **joint** \(\sum_k \hat P_k\) vs \(P_{\mathrm{agg}}\)

### Gap that \(L_{\mathrm{mix}}\) fills

```text
Already have:  per-appliance “if OFF → small power” (SGN)
Missing:       all ON heads together must not exceed the meter
Add:           L_mix overshoot (soft)
```

---

## 5c. Relation to what we already have (short)

| Component | Role vs \(L_{\mathrm{mix}}\) |
|-----------|------------------------------|
| Per-appliance heads | The five “channels” we sum |
| SGN gate | Per-appliance OFF → low power; mix constrains **cross-appliance** overshoot |
| PAD-lite `CrossApplianceDistill` | Feature mix between heads; mix loss is **output-space** physics |
| Fractional frontend | Unrelated; leave as-is |
| Global CORAL/MMD | Domain alignment; orthogonal to mixture additivity |

---

## 6. What we are *not* implementing (from that paper)

1. Replacing MultiNILM with Conv-NILM-Net masks.  
2. Hard \(\sum_i \hat P_i = P_{\mathrm{agg}}\) or \(\sum_i m_i = 1\).  
3. Causal-only convolutions for online EMS.  
4. Training one model without state heads (we keep multi-label ON/OFF + power).

---

## 7. Suggested validation

1. Set `lambda_mix: 0.05`, train vs identical run with `0`.  
2. Check logs: `loss_mix` should fall; val F1/MAE should not collapse.  
3. On waveforms: fewer windows where \(\sum \hat P_k \gg P_{\mathrm{agg}}\).  
4. SAE often improves when overshoot is reduced; F1 may move little (state-driven).

If `lambda_mix` is too large, model may under-predict all appliances (safe but weak). Keep λ small.

---

## 8. References

- Alami et al., Conv-NILM-Net, arXiv:2208.02173 — https://arxiv.org/abs/2208.02173  
- Luo & Mesgarani, Conv-TasNet (speech parent), arXiv:1809.07454 — https://arxiv.org/abs/1809.07454  
- Local model config: `config/models/multinilm_fractional.yaml`  
- Local experiment norms: `config/experiment_ukdale.yaml`

# MultiNILM task loss balance (power ↔ state)

This note explains how MultiNILM combines **power MSE** and **state BCE** so their weights are controllable without guessing a fragile `lambda_state` scale.

Code: `model/MultiNILM_loss.py`  
Config: `config/models/multinilm.yaml` → `loss.task_balance`, `loss.lambda_state`

---

## Problem

The supervised NILM objective is:

\[
L_{\text{power}} = \sum_{i=1}^{A} \mathrm{MSE}_i,\qquad
L_{\text{state}} = \sum_{i=1}^{A} \mathrm{BCE}_i
\]

These two scalars live on **different scales**:

| Term | Typical scale drivers |
|------|------------------------|
| \(L_{\text{power}}\) | Z-score MSE on continuous power |
| \(L_{\text{state}}\) | BCE, often inflated by large `pos_weight` for rare ON events |

So the old fixed mix

\[
L_{\text{NILM}} = L_{\text{power}} + \lambda_{\text{state}} \cdot L_{\text{state}}
\]

with `lambda_state: 1` does **not** mean “equal importance”.  
One term can dominate gradients even when \(\lambda_{\text{state}}=1\).

`pos_weight: auto` only balances **ON vs OFF inside BCE**. It does **not** balance power vs state.

---

## Solution: `task_balance: equal`

We keep the additive form, but **rescale the state term each batch** so its magnitude matches power. Then `lambda_state` becomes a **preference**, not a scale hack.

### Formula

Raw sums (unchanged):

\[
L_{\text{power}} = \sum_i \mathrm{MSE}_i,\qquad
L_{\text{state}} = \sum_i \mathrm{BCE}_i
\]

Balanced state contribution:

\[
\text{state\_term}
=
\lambda_{\text{state}}
\cdot
L_{\text{state}}
\cdot
\left(
\frac{L_{\text{power}}}{L_{\text{state}}}
\right)_{\text{stop-grad}}
\]

**Numeric example** (`λ=1`): \(L_{\text{power}}=2\), \(L_{\text{state}}=8\)

\[
\text{ratio}=2/8=0.25,\quad
\text{state\_term}=8\times 0.25=2,\quad
L_{\text{NILM}}=2+2=4
\]

→ equal contribution (not \(2+8=10\)).

Total supervised loss:

\[
L_{\text{NILM}} = L_{\text{power}} + \text{state\_term}
\]

- Gradients still flow through \(L_{\text{state}}\) (and \(L_{\text{power}}\)).
- Only the **ratio** is detached (`stop-grad`), so it acts as a magnitude normalizer.
- With \(\lambda_{\text{state}} = 1\): \(\text{state\_term} = L_{\text{power}}\) → equal weights.

### Config

```yaml
loss:
  task_balance: equal   # none | equal
  lambda_state: 1       # 1 = equal after balance; 2 = state twice as strong
  pos_weight: auto
```

| `task_balance` | Meaning |
|----------------|---------|
| `equal` | Auto-match magnitudes; `lambda_state=1` ⇒ equal power ↔ state |
| `none` | Old behavior: \(L = L_{\text{power}} + \lambda_{\text{state}} L_{\text{state}}\) (must tune λ for scale) |

---

## How to read the training log

Epoch summaries print:

```text
  nilm    power=1.23  |  state_raw=4.56  |  state_term=1.23
```

| Field | Meaning |
|-------|---------|
| `power` | \(L_{\text{power}}\) (raw MSE sum) |
| `state_raw` | \(L_{\text{state}}\) (raw BCE sum, before balance) |
| `state_term` | What actually enters \(L_{\text{NILM}}\) |

**Check:** with `task_balance: equal` and `lambda_state: 1`, you should see

\[
\texttt{power} \approx \texttt{state\_term}
\]

even if `state_raw` is much larger or smaller.

---

## Relation to domain adaptation

Domain loss is **separate** and still additive:

\[
L = L_{\text{NILM}} + \lambda_{\text{domain}} \cdot L_{\text{domain}}
\]

Task balance only fixes **power ↔ state** inside \(L_{\text{NILM}}\).  
It does **not** auto-balance domain vs NILM (that is a later / separate knob).

---

## What this does *not* fix

| Issue | Handled by |
|-------|------------|
| ON/OFF class imbalance | `pos_weight: auto` |
| Power vs state scale | `task_balance: equal` |
| One appliance dominating the sum of 5 | Not yet (equal sum over appliances) |
| Domain vs NILM scale | Tune `lambda_domain` or add adaptive DA balance later |

---

## Quick reference

```text
L_power  = sum_i MSE_i
L_state  = sum_i BCE_i

# task_balance: equal, lambda_state = 1
L_NILM   = L_power + L_state * (L_power / L_state).detach()
         = L_power + L_power     # equal weights

# then optionally
L        = L_NILM + lambda_domain * L_domain
```

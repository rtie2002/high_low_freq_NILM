# SAMNet architecture notes (Section B)

Paper: *SAMNet: Toward Latency-Free Non-Intrusive Load Monitoring via Multi-Task Deep Learning* (IEEE TSG, 2022).

This note explains **§B Network Architecture** (Fig. 3): the four blocks and how they connect for joint **state detection** + **energy disaggregation**.

---

## Goal of the architecture

Build a **latency-free** NILM model that does two related tasks at once:

1. **State detection** — is the appliance ON/OFF at each time?
2. **Energy disaggregation** — what is its power \(y(t)\)?

Instead of two independent nets, SAMNet uses **multi-task sharing** (inspired by **MMoE**: Multi-gate Mixture-of-Experts) so the two tasks share low-level features but keep task-specific heads.

---

## Four parts (Fig. 3)

| Symbol | Name | Role |
|--------|------|------|
| \(F(x,\theta_f)\) | **Experts learner** | Shared bottom: extract \(E\) “expert” feature streams from aggregate \(x\) |
| \(G(x,\theta_g)\) | **Gate** | Per-task soft selection / weighting of those experts |
| \(H(\cdot,\theta_h)\) | **Tower** | Task-specific head → \(\hat s\) (state) or \(\hat y_r\) (raw power) |
| \(C(x,w)\) / \(C(y)\) | **Automatic ON/OFF marking** | Turn continuous appliance power \(y\) into soft state labels for training (no hand threshold \(\tau\)) |

High-level data flow:

```text
aggregate window x
        │
        ▼
┌───────────────────┐
│ Experts F (dilated│──► f ∈ R^{E×T}   (E experts × time)
│ TCN shared bottom)│
└─────────┬─────────┘
          │
   ┌──────┴──────┐
   ▼             ▼
Gate_A        Gate_B     (task-specific gates + self-attention fusion)
   │             │
   ▼             ▼
 z_A           z_B
   │             │
Tower_A       Tower_B
   │             │
   ▼             ▼
 ŝ = σ(·)     ŷ_r
   │             │
   └──────┬──────┘
          ▼
     ŷ_out = ŷ_r ⊙ ŝ     (SGN-style gate; Eq. 15)
```

---

## 1) Experts learner \(F\)

- **Why:** State and power tasks are strongly correlated → share early features.
- **What:** A **dilated TCN** produces shared expert features  
  \[
  f = F(x,\theta_f),\quad f \in \mathbb{R}^{E \times T}
  \]
  \(E\) = number of experts, \(T\) = window length.
- **Vs MMoE:** Classic MMoE often uses **FC** experts. SAMNet uses **dilated TCN** so experts can “see” a **long temporal context** in parallel (important for their latency-free / large-context claim), not only pointwise FC mixes.

Think of \(f\) as \(E\) parallel temporal embeddings of the same aggregate window.

---

## 2) Gate \(G\)

- **Why:** Different tasks should use experts **differently** (MMoE idea).
- **What:** Each task \(j\) has its own gate  
  \[
  g_j = G(x,\theta_g^j),\quad g_j \in \mathbb{R}^{E \times T}
  \]
  typically **FC + softmax** over experts so \(\sum_i g_j(i,t)=1\).
- Tasks: \(j=A\) state detection, \(j=B\) energy disaggregation.

**Fusion is not a plain weighted sum.** They use **scaled dot-product self-attention**:

- Query ← gate \(g_j\)
- Key / Value ← experts \(f\)

\[
z_j = \mathrm{softmax}\!\left(\frac{(g_j W_Q)(f W_K)^\top}{\sqrt{d_k}}\right) f
\]

So the gate can attend over the **whole sequence** when deciding how much each expert matters for that task. Output \(z_j\) is the **task-specific assembled** representation.

---

## 3) Tower \(H\)

- **Why:** After sharing + gating, each task still needs its own predictor.
- **What:** Small **two-layer FC towers**:

\[
\hat s = \sigma\!\big(H_A(z_A,\theta_h^A)\big)
\qquad
\hat y_r = H_B(z_B,\theta_h^B)
\]

- \(\hat s(t) \in (0,1)\): predicted ON probability  
- \(\hat y_r(t)\): unconstrained (or regression) power before state gating  

---

## 4) Automatic ON/OFF marking \(C\)

**Problem they criticize:** Many multi-task NILM papers build state labels by **hand threshold** on appliance power \(y\):

\[
s(t)=\mathbf{1}\{y(t)>\tau\}
\]

Different \(\tau\) → different labels → brittle.

**Their \(C\):** map power to a **soft** ON probability with sigmoid (and a small offset \(\epsilon\) to ignore tiny noise):

\[
P(s(t)=\mathrm{on}\mid y(t)) = \sigma(y(t)-\epsilon)
\]

That soft label is used as supervision for the state tower (instead of a hard arbitrary watt threshold).  
(\(C(x,w)\) in the intro naming is this automatic marking path from appliance power; in the equations it is written as \(\sigma(y-\epsilon)\).)

---

## How the two tasks combine at the output

Same idea as **SGN** (subtask gated network):

\[
\hat y_{\mathrm{out}} = \hat y_r \odot \hat s
\]

- If \(\hat s\approx 0\) (OFF), final power is forced near 0.  
- If \(\hat s\approx 1\) (ON), final power ≈ regression head.

**Loss** (Section C, for context):

\[
L = \lambda L_c + L_r
\]

- \(L_c\): BCE between automatic/soft state labels and \(\hat s\)  
- \(L_r\): MSE between true \(y\) and **gated** \(\hat y_{\mathrm{out}}=\hat y_r\odot\hat s\)

So gradients couple classification and regression through the product gate.

---

## One-sentence summary

**SAMNet = dilated-TCN mixture-of-experts (shared) + per-task attention gates + two towers (state & power) + soft auto state labels from \(y\), with final power = regression ⊙ state — multi-task sharing for latency-free NILM, not a separate UDA module.**

---

## Relation to our MultiNILM (quick)

| SAMNet | MultiNILM (typical) |
|--------|---------------------|
| Shared dilated experts \(F\) | Shared stem + TCN backbone |
| Per-task gate + attention | Soft/hard state gate on power head (SGN-like) |
| Towers \(H_A,H_B\) | Per-appliance `state_head` + `power_head` |
| Soft \(\sigma(y-\epsilon)\) labels | Usually **hard threshold** from experiment yaml |
| Multi-task BCE+MSE on gated power | Same family of losses |

Useful ideas to steal: **attention-gated experts**, **soft auto state labels** (less brittle than fixed \(\tau\)), keep **\(\hat y\odot\hat s\)** coupling. Cross-dataset “transfer” in the paper is still **train then test**, not unsupervised DA.

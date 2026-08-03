# Device & Time Invariant Features for Transferable NILM

**Paper:** Schirmer & Mporas, *Device and Time Invariant Features for Transferable Non-Intrusive Load Monitoring*, IEEE Open Access Journal of Power and Energy, 2022.  
**Local PDF:** `Device_and_Time_Invariant_Features_for_Transferable_Non-Intrusive_Load_Monitoring.pdf` (repo root)

---

## Why feature invariance matters in NILM

Same-house supervised NILM often reaches **80–95%** accuracy. Pre-trained models transferred to a **new household** usually drop sharply, because the same appliance *type* does not look the same across homes: different brands, usage habits, and temporal alignment.

Ground-truth appliance meters are expensive, so **transferable** (cross-house) models need features that stay stable across domains.

---

## The feature-invariance problem (three mismatches)

Compare the same appliance type from two manufacturers (e.g. two fridges, two washing machines). Three systematic differences appear:

| # | Mismatch | What changes | Physical intuition |
|---|----------|--------------|--------------------|
| 1 | **Power scaling (y-axis)** | Steady-state power differs (e.g. fridge A ≈ 75 W, fridge B ≈ 100 W) | Same electrical topology, different component size → amplitude scales |
| 2 | **Time shifts (x-axis)** | On/off edges and cycle timing are not aligned across homes | Different usage schedules and event timing |
| 3 | **State probabilities** | On/off duty cycles differ a lot (user-driven) | How often the user runs the device ≠ the device’s internal active states |

A transferable NILM pipeline must therefore produce **device-characteristic** features that are largely **invariant** to (1)–(3).

---

## Proposed solution (pipeline)

```
aggregated power
    → framing
    → fractional calculus          (time-shift robustness)
    → normalized KLE spectrogram   (scale / brand robustness)
    → 2D CNN (one model per device)
    → active-state post-processing (user-duty-cycle robustness)
    → estimated appliance power
```

---

## How to implement each invariance (math + examples)

Paper pipeline (Fig. 2 / Sec. III): framing → fractional calculus → normalized KLE → CNN → active-state post-processing.

Notation:
- \(p_{\mathrm{agg}}(t)\): aggregated active power (W), \(t = 1,\ldots,T\)
- Frame length \(L\), KLE order \(\tilde{N} < L\)
- \(K\) fractional orders \(\alpha_1,\ldots,\alpha_K\) (paper best: \(K=8\))
- Transfer setup preferred \(\tilde{N}=256\); in-house \(\tilde{N}=64\)

---

### 1. Scaling invariance — normalized KLE (Sec. II-A, III-B)

#### Why

Same fridge type, two brands: steady-state \(\approx 75\,\mathrm{W}\) vs \(\approx 100\,\mathrm{W}\). Circuits are similar; frequency content mostly **scales** with the fundamental. Align spectra by transforming to frequency space and normalizing.

#### Math (per fractional signal \(P_\alpha\))

1. Take one frame \(\tau\) of length \(L\) from a fractional signal \(D^{\alpha_k} p_{\mathrm{agg}}\). Call it \(P_\alpha \in \mathbb{R}^L\).

2. Build the **autocorrelation matrix (ACM)** of order \(\tilde{N}\):

\[
R_{PP}(n) = \mathrm{ACF}\{P_\alpha\}(n),\quad
\Theta_{PP} =
\begin{bmatrix}
R_{PP}(0) & \cdots & R_{PP}(\tilde{N}-1) \\
\vdots & \ddots & \vdots \\
R_{PP}(\tilde{N}-1) & \cdots & R_{PP}(0)
\end{bmatrix}
\tag{6}
\]

3. Eigendecompose \(\Theta_{PP} = Q \Lambda Q^\top\) with orthonormal columns \(Q = [q_0,\ldots,q_{\tilde{N}-1}]\).

4. **KLE transform** (Eq. 7–8):

\[
\tilde{P}_\alpha = Q^\top P_\alpha,\qquad
P_\alpha = Q\,\tilde{P}_\alpha = \sum_{i=0}^{\tilde{N}-1} (q_i^\top P_\alpha)\, q_i
\]

Subspace components (SCs) \(p_i = (q_i^\top P_\alpha)\, q_i\) are treated as near-sinusoidal → extract **magnitude** \(A_\alpha \in \mathbb{R}^{\tilde{N}}\) and **phase** \(\Phi_\alpha \in \mathbb{R}^{\tilde{N}}\).

5. Repeat for every \(\alpha_k\), \(k=1,\ldots,K\), stack into spectrograms:

\[
A,\ \Phi \in \mathbb{R}^{\tilde{N} \times K}
\]

6. **Normalize** magnitude/phase (batch-norm over the KLE spectrum, plus mean–std as in Sec. VI-B):

\[
x \leftarrow \frac{|x - \bar{x}|}{\sigma}
\tag{14}
\]

then batch-normalize CNN inputs/layers. Idea: divide out scale so fridge-75W and fridge-100W share similar harmonic **shape**.

#### Toy example (scaling)

Suppose two fridges in the frequency domain (fundamental + 2 harmonics), before norm:

| Brand | \(A_0\) (fund.) | \(A_1\) | \(A_2\) |
|-------|------------------|---------|---------|
| Bosch | 75 | 15 | 7.5 |
| Samsung | 100 | 20 | 10 |

Normalize by fundamental \(A_0\):

| Brand | \(A_0/A_0\) | \(A_1/A_0\) | \(A_2/A_0\) |
|-------|-------------|-------------|-------------|
| Bosch | 1.0 | 0.20 | 0.10 |
| Samsung | 1.0 | 0.20 | 0.10 |

→ identical **device signature** after scale removal. Paper does this via KLE + batch-norm rather than a hand-coded \(A_i/A_0\), but the physical goal is the same.

#### Pseudocode

```python
# P_alpha: frame of length L from fractional signal
# N_tilde: KLE order (e.g. 256 for transfer)
R = acf(P_alpha, max_lag=N_tilde)          # R_PP(0)..R_PP(N-1)
Theta = toeplitz(R)                        # Eq. (6)
eigvals, Q = eigh(Theta)                   # Q unitary
P_tilde = Q.T @ P_alpha[:N_tilde]          # Eq. (7); trim/pad as needed
A_alpha, Phi_alpha = mag_phase_from_SCs(P_tilde, Q)
# stack over alpha_k → A, Phi; then BatchNorm / mean-std
```

---

### 2. Time-shift invariance — fractional calculus (Sec. II-B, III-A)

#### Why

On/off edges of the same appliance are not time-aligned across houses. Raw windows look different under a shift. Fractional derivatives add multi-scale temporal structure that is more shift-tolerant and works well with CNNs.

#### Math (Grünwald–Letnikov, Eq. 4–5)

Fractional derivative of order \(\alpha \in \mathbb{R}\) on interval \([t_0, t]\):

\[
{}_{t_0}D_t^\alpha \, p(t)
= \lim_{h \to 0}
\frac{1}{h^\alpha}
\sum_{j=0}^{\lfloor k \rfloor}
(-1)^j \binom{\alpha}{j}\, p(t - jh)
\tag{4}
\]

with \(k = (t-t_0)/h\) and binomial weights via Gamma:

\[
\binom{\alpha}{j}
= \frac{\Gamma(\alpha+1)}{\Gamma(j+1)\,\Gamma(\alpha-j+1)}
\tag{5}
\]

In discrete NILM practice (\(h=1\) sample):

\[
(D^\alpha p)[t]
\approx \sum_{j=0}^{J}
(-1)^j \binom{\alpha}{j}\, p[t-j]
\]

with memory length \(J\) (e.g. frame length or a fixed truncation).

Paper uses **\(K\)** orders \(\alpha_1,\ldots,\alpha_K\) (optimized \(K=8\)), producing \(K\) fractional signals \(D^{\alpha_k} p_{\mathrm{agg}}\). Each is then KLE-transformed → 2D map \(A,\Phi \in \mathbb{R}^{\tilde{N}\times K}\).

#### Toy example (time shift + fractional)

Integer derivative \(\alpha=1\) is a differencer: \(p[t]-p[t-1]\) (edge detector).  
Half-order \(\alpha=0.5\) mixes present and past with decaying weights from \(\binom{0.5}{j}\).

Raw step (fridge on at \(t=10\)):

```
p:   ... 0 0 0 75 75 75 ...
```

Shifted (on at \(t=12\)):

```
p':  ... 0 0 0 0 0 75 75 ...
```

CNNs on raw \(p\) vs \(p'\) see misaligned edges. On \(D^\alpha p\) for several \(\alpha\), the **shape of the transient** (rise signature) is similar; only its location moves. Stacking many \(\alpha\) gives the CNN a richer “edge / memory” view than one raw frame.

#### Pseudocode

```python
def gl_fractional(p, alpha, J):
    # p: 1D array length T
    out = np.zeros_like(p, dtype=float)
    # precompute w_j = (-1)^j * C(alpha, j)
    w = [(-1)**j * binom_gamma(alpha, j) for j in range(J+1)]
    for t in range(T):
        s = 0.0
        for j in range(min(J, t) + 1):
            s += w[j] * p[t - j]
        out[t] = s  # h=1
    return out

frac_signals = [gl_fractional(p_agg, alpha, J) for alpha in alphas]  # K of them
# then KLE each → columns of A, Phi
```

---

### 3. State-probability invariance — active-state post-processing (Sec. II-C, III-C)

#### Why

House A runs the washing machine 1×/week; house B runs it 5×/week. Off-duration statistics differ by **user**, not by machine. Once ON, internal states (rinse / wash / spin) are **device**-dependent. So only correct **active** estimates.

#### Math (Eq. 9–10)

1. CNN outputs continuous estimate \(\hat{p}'\) for appliance \(m\).

2. Device is ON if \(\hat{p}' > \theta\) (power threshold).

3. Fit **fuzzy c-means** on **active** training powers of appliance \(m\) → cluster centers \(s_m^1,\ldots,s_m^N\) (active states only; ignore off).

4. Nearest active center (Eq. 9):

\[
n_{\min} = \arg\min_{1 \le n \le N} \big\| \hat{p}' - s_m^n \big\|
\]

5. Update with appliance-specific margin \(\varepsilon\) (Eq. 10) — **only active states** are rewritten; near-zero / off is left alone:

\[
\hat{p}_m =
\begin{cases}
\hat{p}', & \text{if } \hat{p}' \le \varepsilon \\[4pt]
s_m^{n_{\min}}, & \text{if } \hat{p}' > \varepsilon
\end{cases}
\]

So when the CNN says “on” (\(\hat{p}' > \varepsilon\)), the estimate is **quantized** to the nearest fuzzy c-means active-state center. Off / near-off (\(\hat{p}' \le \varepsilon\)) is not forced to match across houses.

#### Toy example (washing machine)

Fuzzy c-means on ON samples finds centers:

| State | Center \(s\) |
|-------|--------------|
| rinse | 200 W |
| wash  | 500 W |
| spin  | 800 W |

Threshold \(\theta = 50\,\mathrm{W}\), margin \(\varepsilon = 50\,\mathrm{W}\) (illustrative).

| CNN \(\hat{p}'\) | Action (Eq. 10) |
|------------------|-----------------|
| 8 W (\(\le \varepsilon\)) | keep 8 — off / user-dependent, **not** snapped |
| 480 W (\(> \varepsilon\)) | snap to nearest center → **500 W** (wash) |
| 650 W (\(> \varepsilon\)) | snap to nearest → **500** or **800** (whichever closer) |
| 790 W (\(> \varepsilon\)) | snap → **800 W** (spin) |

House with long OFF periods never pollutes the centers, because OFF samples are excluded from c-means.

#### Pseudocode

```python
# Fit once on source (or labeled) active powers for appliance m
active = y_m[y_m > theta]
centers = fuzzy_cmeans(active, n_clusters=N)  # s_m^1 .. s_m^N

def postprocess(p_hat, centers, eps):
    # Eq. (10): leave near-zero; if active, snap to nearest center
    if p_hat <= eps:
        return p_hat
    n_min = np.abs(centers - p_hat).argmin()
    return centers[n_min]
```

---

## Putting it together (one forward pass)

1. Frame \(p_{\mathrm{agg}}\) (length \(L\)).
2. Compute \(K\) fractional signals \(D^{\alpha_k} p_{\mathrm{agg}}\) (Eq. 4–5).
3. For each \(\alpha_k\): ACM → eigendecomp → KLE → \(A_{\alpha_k}, \Phi_{\alpha_k}\) (Eq. 6–8).
4. Stack + mean–std / batch-norm → input tensor for 2D CNN.
5. CNN regresses \(\hat{p}'\) (one model per appliance, as in paper).
6. Active-state fuzzy c-means snap (Eq. 9–10) → final \(\hat{p}_m\).

Best paper protocol (#6): magnitudes + phases + raw fractional samples + post-processing.

---

## Experimental takeaways (paper)

- Datasets: **REDD** (classic NILM) and **REFIT** (many houses, multi-brand appliances).
- Targets: kettle, microwave, dishwasher, fridge, washing machine.
- Best reported transfer gain vs prior transfer NILM: about **13.1% MAE** improvement (REDD vs best literature baseline in their comparison).
- Ablations: magnitude KLE helps; **phase alone** is weak; combining fractional KLE + raw fractional samples + post-processing is strongest.
- Normalization is critical for transfer: no norm ≪ min–max ≪ mean–std ≪ mean–std **+ batch norm** on CNN layers.
- Transfer setups prefer **longer frames** (\(\tilde{N}=256\) vs \(64\) in-house), matching the idea that more context is needed to absorb local brand/usage differences.

---

## Take-home for multi-domain / DA NILM

Feature invariance here is **hand-designed**:

1. **Amplitude / brand** → spectral shape, normalized  
2. **Timing** → fractional / temporal descriptors  
3. **Usage frequency** → do not align off-duty; only constrain **active** states  

This is complementary to **learned** domain adaptation (MMD, CORAL, adversarial DA): Schirmer et al. make the *input representation* more invariant; DA methods typically make *latent embeddings* more domain-invariant after a shared encoder.

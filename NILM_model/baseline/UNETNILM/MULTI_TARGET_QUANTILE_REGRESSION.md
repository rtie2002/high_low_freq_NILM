# UNet-NILM: Multi-Label State Estimation and Multi-Target Quantile Regression

This note provides a formal exposition of the two core learning objectives in **UNet-NILM** (Faustine et al., BuildSys/NILM Workshop 2020):

1. **Appliance state profile generation (Seq2Quantile)** — how raw power traces are converted into clean ON/OFF labels (Paper §2.4).
2. **Multi-label state estimation** — joint detection of ON/OFF states for all appliances via per-appliance softmax classification and cross-entropy loss (Equation 5).
3. **Multi-target quantile regression** — probabilistic power estimation via pinball loss over multiple quantile levels (Equations 2–4).

Both tasks are trained jointly from shared deep representations, yielding a multi-task NILM system that simultaneously classifies appliance activity and estimates power with distributional uncertainty.

---

## 1. Problem Setting

At each time step \(t\), given aggregate power input features \(\mathbf{x}(t)\), the model must predict:

| Symbol | Meaning |
|--------|---------|
| \(s_m(t) \in \{0, 1\}\) | Ground-truth ON/OFF state of appliance \(m\) |
| \(y_m(t)\) | Ground-truth normalised power of appliance \(m\) |
| \(\hat{s}_m(t)\) | Predicted state logits / probabilities for appliance \(m\) |
| \(\hat{y}_m^{(\tau_n)}(t)\) | Predicted \(\tau_n\)-quantile of power for appliance \(m\) |
| \(M\) | Number of target appliances |
| \(T\) | Number of time steps in a training sequence |
| \(N_\tau\) | Number of quantile levels \(\{\tau_1, \ldots, \tau_{N_\tau}\}\) |
| \(\tau_n \in (0, 1)\) | The \(n\)-th quantile level (e.g. 0.1, 0.5, 0.9) |

UNet-NILM treats these as **coupled multi-task outputs** from a single network: a classification head produces state logits for all \(M\) appliances, and a regression head produces \(N_\tau\) quantile surfaces per appliance.

---

## 2. Appliance State Profile — Seq2Quantile (Paper §2.4)

Before the neural network can learn ON/OFF detection, we need **ground-truth state labels** \(s_m(t)\) for each appliance. UNet-NILM builds these labels from the appliance power trace using a method called **Seq2Quantile** (sequence-to-quantile), instead of the **slope algorithm** used in earlier NILM papers [17, 18].

### 2.1 The basic idea (plain language)

Each appliance is treated as a **two-state switch**:

| State | Meaning |
|-------|---------|
| **OFF** (\(s_m = 0\)) | Appliance is not consuming meaningful power |
| **ON** (\(s_m = 1\)) | Appliance is actively consuming power |

Raw sub-meter readings are **noisy** — small spikes, meter jitter, or brief glitches can make a truly OFF appliance look briefly ON. Seq2Quantile **smooths** the power trace first, then applies a power threshold to decide ON vs OFF.

Think of it as: *“Look at a short window of recent power readings, take a robust middle value (the median), and compare that to a known ON threshold.”*

### 2.2 Step-by-step procedure

For each appliance \(m\) at each time step \(t\):

**Step 1 — Collect a sliding window of raw power**

Take the next \(w_m\) samples from the appliance power profile:

\[
y_m(t),\; y_m(t+1),\; \ldots,\; y_m(t + w_m - 1)
\]

We write this compactly as \(y_m(t : t + w_m)\).

**Step 2 — Compute the quantile (median filter)**

\[
y_m^{(\tau)} = Q_\tau\!\bigl(y_m(t : t + w_m)\bigr)
\]

where \(Q_\tau(\cdot)\) is the \(\tau\)-quantile of the window. The paper uses **\(\tau = 0.5\)**, i.e. the **median**:

\[
y_m^{(0.5)} = \text{median of the window}
\]

Half the values in the window are above this level and half are below. This removes short noise spikes that would confuse a simple threshold on a single raw sample.

**Step 3 — Compare to ON-power threshold**

Each appliance has a fixed threshold \(p_m^{\text{on}}\) (in Watts):

\[
s_m(t) =
\begin{cases}
1 \quad \text{(ON)}  & \text{if } y_m^{(\tau)} \ge p_m^{\text{on}} \\
0 \quad \text{(OFF)} & \text{otherwise}
\end{cases}
\]

### 2.3 Window size \(w_m\)

The window length is **different per appliance** and scales with how long the appliance typically stays ON:

\[
w_m \propto \frac{T_{\text{on}}}{T_s}
\]

| Symbol | Meaning |
|--------|---------|
| \(T_{\text{on}}\) | Mean duration of an ON episode for appliance \(m\) |
| \(T_s\) | Sampling interval (6 s in UNet-NILM) |
| \(w_m\) | Number of samples in the sliding window |

**Intuition:**

- **Short-burst appliances** (kettle, microwave) → small window → reacts quickly to brief ON events.
- **Long-cycle appliances** (fridge, dishwasher, washing machine) → large window → smooths over compressor cycles or multi-phase wash programs.

In the reference implementation (`load_data.py`), fixed window sizes are used:

| Appliance | Window \(w_m\) | At 6 s sampling |
|-----------|----------------|-----------------|
| Kettle | 10 | ~60 s |
| Microwave | 10 | ~60 s |
| Fridge | 50 | ~300 s (~5 min) |
| Dishwasher | 50 | ~300 s |
| Washing machine | 50 | ~300 s |

### 2.4 Visual intuition

```
Raw power (noisy):     ___/|\__/|\_________/|\_/|____
                              ↑ glitches

Seq2Quantile (median): _______/‾‾‾‾\_________/‾‾\____
                              ↑ smooth ON plateaus

Threshold p_on:        --------|--------|--------|----
                              ↑
                         ON if smoothed ≥ threshold
```

The median acts as a **low-pass filter**: brief spikes disappear, sustained ON periods remain visible.

### 2.5 Why Seq2Quantile instead of the slope algorithm?

| Approach | How it works | Drawback |
|----------|--------------|----------|
| **Slope algorithm** [17, 18] | Detect ON/OFF from sudden **changes** in power (edges) | Sensitive to noise; false edges from meter jitter |
| **Seq2Quantile** (UNet-NILM) | Smooth with median, then threshold **level** | More robust; better for noisy 6 s data |

Seq2Quantile asks *“Is the typical power level in this window high enough?”* rather than *“Did power just jump?”*

### 2.6 Worked example

Suppose for the **kettle** at time \(t\), the window has 10 samples (6 s each) with power readings (W):

```
[0, 0, 5, 10, 2100, 2200, 2150, 2180, 0, 0]
```

- Sorted: `[0, 0, 0, 0, 5, 10, 2100, 2150, 2180, 2200]`
- Median \(Q_{0.5}\) = average of 5th and 6th values = \((5 + 10) / 2 =\) **7.5 W**
- Threshold \(p_{\text{on}}^{\text{kettle}} = 2000\) W
- 7.5 < 2000 → **OFF** at this window position

If the window is mostly during boiling:

```
[2100, 2200, 2150, 2180, 2200, 2100, 2150, 0, 0, 0]
```

- Median ≈ **2150 W**
- 2150 ≥ 2000 → **ON**

### 2.7 Implementation in the codebase

In `src/data/load_data.py`, Seq2Quantile is implemented as `quantile_filter` followed by `binarization`:

```python
meter = quantile_filter(ukdale_appliance_data[app]['window'], power, p=50)  # Q_0.5
state = binarization(meter, ukdale_appliance_data[app]['on_power_threshold'])  # s_m(t)
meter = (meter - mean) / std   # normalise smoothed power for regression target
```

The **same smoothed power** (`meter`) is saved as the regression target \(y_m(t)\); the binarised version becomes the classification label \(s_m(t)\) stored in `states.npy`.

### 2.8 Relation to the rest of the pipeline

```
Raw appliance power
        │
        ▼
  Seq2Quantile (§2.4)  ──►  smoothed power y_m(t)  ──►  regression target (Eq. 4)
        │
        ▼
  Threshold p_m^on     ──►  ON/OFF label s_m(t)    ──►  classification target (Eq. 5)
```

The neural network **does not** run Seq2Quantile at inference time. Seq2Quantile is a **preprocessing step** used only to create training labels (and smoothed power targets) from the dataset.

---

## 3. Multi-Label Learning for State Estimation

### 3.1 Background and Motivation

**Multi-label learning** is a supervised paradigm in which each input instance is associated with **one or more labels** drawn from a label set, rather than a single mutually exclusive class [6, 31]. In NILM, the natural label set at time \(t\) is the vector of appliance states:

\[
\mathbf{s}(t) = \bigl(s_1(t), s_2(t), \ldots, s_M(t)\bigr), \quad s_m(t) \in \{0, 1\},
\]

where \(s_m(t) = 1\) indicates appliance \(m\) is ON and \(s_m(t) = 0\) indicates it is OFF. Multiple appliances may be ON simultaneously, making this a genuine multi-label problem rather than standard single-label multi-class classification.

Prior NILM work has demonstrated that multi-label formulations are a viable alternative to conventional per-appliance or single-target approaches [2, 3, 6, 17, 18, 20, 27]:

- **Temporal multi-label classification** with engineered meta-features [2]
- **Multi-label and meta-classification frameworks** surveyed systematically [27]
- **Deep multi-label appliance recognition** using sigmoid activations [19, 29]

However, many deep NILM models adopt a **single-task learning strategy** — training one network per appliance or optimising one objective at a time — which does not exploit joint structure across co-active loads.

### 3.2 Sigmoid vs. Softmax Formulations

Two principal DNN extensions exist for multi-label state detection:

#### Approach A: Sigmoid + threshold (common in prior work)

A single network outputs \(M\) independent probabilities via the sigmoid function:

\[
P(s_m = 1 \mid \mathbf{x}) = \sigma(\hat{s}_m) = \frac{1}{1 + e^{-\hat{s}_m}}, \quad m = 1, \ldots, M.
\]

Each appliance is treated as an **independent binary classification** problem. Because sigmoid outputs are continuous in \((0, 1)\), an **additional threshold mechanism** is required to convert probabilities into hard ON/OFF decisions:

\[
\hat{s}_m^{\text{hard}} = \mathbb{1}\bigl[\sigma(\hat{s}_m) \ge \theta_m\bigr],
\]

where \(\theta_m\) is typically tuned per appliance on a validation set. Threshold selection introduces an extra hyperparameter and can be sensitive to class imbalance and domain shift.

#### Approach B: Softmax per appliance (UNet-NILM)

UNet-NILM instead applies a **softmax over two mutually exclusive states** (OFF, ON) **independently for each appliance** \(m\):

\[
P\bigl(s_m = k \mid \mathbf{x}\bigr) = \frac{\exp\bigl(\hat{s}_m^{(k)}(t)\bigr)}{\displaystyle\sum_{j=1}^{2} \exp\bigl(\hat{s}_m^{(j)}(t)\bigr)}, \quad k \in \{0, 1\},
\]

where \(\hat{s}_m^{(k)}(t)\) denotes the logit for state \(k\) of appliance \(m\) at time \(t\).

**Key advantages:**

| Property | Sigmoid | Softmax (UNet-NILM) |
|----------|---------|---------------------|
| Normalisation | Per-label, independent | Per-appliance, sums to 1 over OFF/ON |
| Hard decision | Requires threshold \(\theta_m\) | Argmax over 2 states — **no threshold tuning** |
| State coupling | None within an appliance | OFF and ON probabilities are complementary |
| Loss | Binary cross-entropy (per label) | Categorical cross-entropy (per appliance) |

The softmax formulation implicitly captures the **competitive relationship between OFF and ON** for each appliance: increasing the probability of one state necessarily decreases the other. As demonstrated in [6], this removes the need for post-hoc threshold calibration while maintaining the multi-label structure across appliances.

### 3.3 Cross-Entropy Loss (Equation 5)

Model parameters are learned by minimising the **multi-label cross-entropy** between the predicted softmax distributions and the ground-truth states:

\[
\mathcal{L}_{CE}\bigl(\hat{\mathbf{s}}, \mathbf{s}\bigr) =
-\frac{1}{TM}
\sum_{t=1}^{T}
\sum_{m=1}^{M}
\sum_{k=1}^{2}
s_m^{(k)}(t) \cdot
\log
\left(
\frac{\exp\bigl(\hat{s}_m^{(k)}(t)\bigr)}
{\displaystyle\sum_{j=1}^{2} \exp\bigl(\hat{s}_m^{(j)}(t)\bigr)}
\right).
\tag{5}
\]

#### Notation

| Term | Meaning |
|------|---------|
| \(s_m^{(k)}(t)\) | One-hot encoding of the true state: \(s_m^{(k)} = 1\) if appliance \(m\) is in state \(k\), else 0 |
| \(\hat{s}_m^{(k)}(t)\) | Raw logit output by the network for appliance \(m\), state \(k\) |
| Denominator \(\sum_{j=1}^{2} \exp(\hat{s}_m^{(j)})\) | Softmax normaliser over OFF (\(j=0\)) and ON (\(j=1\)) |

When the true state is encoded as an integer label \(z_m(t) \in \{0, 1\}\), Equation (5) simplifies to the **negative log-likelihood of the correct class**:

\[
\mathcal{L}_{CE} =
-\frac{1}{TM}
\sum_{t=1}^{T}
\sum_{m=1}^{M}
\log
\left(
\frac{\exp\bigl(\hat{s}_m^{(z_m(t))}(t)\bigr)}
{\displaystyle\sum_{j=1}^{2} \exp\bigl(\hat{s}_m^{(j)}(t)\bigr)}
\right).
\]

This is standard **categorical cross-entropy** applied independently to each of the \(M\) appliances at each time step, then averaged over \(T \times M\) terms.

#### Interpretation

- **\(t\) index:** Each time step in the input window contributes equally to the loss.
- **\(m\) index:** All \(M\) appliances are supervised jointly — the model learns shared features useful for detecting any appliance.
- **\(k\) index:** Only the one-hot component \(s_m^{(k)} = 1\) contributes (the true state); the other state receives zero weight in the sum.
- **Normalisation \(1/(TM)\):** Yields a per-time-step, per-appliance average, stabilising gradients as sequence length or appliance count changes.

#### Inference

At test time, the predicted hard label for appliance \(m\) at time \(t\) is:

\[
\hat{z}_m(t) = \arg\max_{k \in \{0,1\}} P\bigl(s_m = k \mid \mathbf{x}(t)\bigr)
= \arg\max_{k} \hat{s}_m^{(k)}(t).
\]

No threshold is required.

### 3.4 Implementation Correspondence

In `src/net/modules.py`, the state head produces logits of shape \((B, 2, M)\):

```python
self.fc_out_state = nn.Linear(1024, output_size * 2)
states_logits = self.fc_out_state(mlp_out).reshape(B, 2, -1)
```

In `src/net/model_pl.py`, Equation (5) is implemented as:

```python
loss_nll = F.nll_loss(F.log_softmax(logits, dim=1), z)
prob, pred = torch.max(F.softmax(logits, dim=1), dim=1)
```

Here `logits` has shape \((B, 2, M)\), `z` holds integer labels \(z_m(t) \in \{0, 1\}\), and `pred` is the argmax ON/OFF decision. The variable is named `loss_nll` because `F.nll_loss` applied to `log_softmax` outputs is equivalent to cross-entropy.

---

## 4. Non-Parametric Density Estimate (Equation 2)

### 4.1 Statement

A non-parametric probabilistic density estimate \(\hat{f}(y)\) is obtained by collecting \(N_\tau\) quantile estimates:

\[
\hat{f}(y) = \left\{ \hat{y}^{(\tau_n)},\; n = 1, \ldots, N_\tau \;\middle|\; \tau_n \in (0, 1) \right\}
\tag{2}
\]

### 4.2 Interpretation

Equation (2) does **not** specify a closed-form parametric density (such as a Gaussian \(\mathcal{N}(\mu, \sigma^2)\)). Instead, it defines the predictive distribution **implicitly** through its quantile function.

Recall that for a random variable \(Y\) with cumulative distribution function (CDF) \(F_Y(y) = P(Y \le y)\), the \(\tau\)-quantile \(q_\tau\) satisfies:

\[
F_Y(q_\tau) = \tau, \quad \tau \in (0, 1).
\]

Equivalently, \(q_\tau\) is the value below which a fraction \(\tau\) of the probability mass lies. By estimating \(q_{\tau_1}, q_{\tau_2}, \ldots, q_{\tau_{N_\tau}}\) at multiple levels, one obtains a **discrete sampling of the inverse CDF** (quantile function):

\[
\hat{F}^{-1}(\tau_n) = \hat{y}^{(\tau_n)}.
\]

From this collection, several downstream quantities can be derived without assuming a parametric family:

- **Central tendency:** median \(\hat{y}^{(0.5)}\)
- **Dispersion:** inter-quantile range \(\hat{y}^{(0.9)} - \hat{y}^{(0.1)}\)
- **Tail behaviour:** lower/upper quantiles (e.g. 2.5th and 97.5th percentiles)
- **Approximate density:** via numerical differentiation of the estimated quantile function or kernel smoothing between quantile points

This is why the approach is described as *non-parametric*: the shape of \(\hat{f}(y)\) is determined entirely by the learned quantile values, not by a fixed functional form with a small number of parameters.

### 4.3 Role in NILM

Appliance power at a given time step is often **heterogeneous and multi-modal** (e.g. standby vs. active states, variable cycle phases). A single mean squared error (MSE) target encourages the model to predict conditional expectations that may lie between physically plausible operating levels. Quantile regression instead asks: *"What power level is exceeded with probability \(1 - \tau\)?"* — a question that remains meaningful under skewed or multimodal conditional distributions.

In the reference implementation (`src/experiment.py`), the default quantile grid is:

\[
\tau \in \{0.0025,\; 0.1,\; 0.5,\; 0.9,\; 0.975\}
\]

The median prediction \(\hat{y}^{(0.5)}\) is typically used as the point estimate at inference, while the full set supports uncertainty characterisation.

---

## 5. Pinball Loss (Equation 3)

### 5.1 Statement

Learning is accomplished by minimising the **pinball loss** (quantile loss). Let the residual be:

\[
r = y - \hat{y},
\]

where \(y\) is the observed power and \(\hat{y}\) is the predicted value at a specific quantile level \(\tau\). The pinball loss is defined as:

\[
\rho_\tau(r) =
\begin{cases}
(\tau - 1) \cdot r & \text{if } r \ge 0, \\[6pt]
\tau \cdot r & \text{if } r < 0.
\end{cases}
\tag{3}
\]

This can be written compactly as:

\[
\rho_\tau(r) = \max\bigl\{ \tau \cdot r,\; (\tau - 1) \cdot r \bigr\}.
\]

### 5.2 Derivation and Intuition

The pinball loss is the **consistent loss function for quantile estimation**. A predicted value \(\hat{y}\) minimises the expected loss \(\mathbb{E}[\rho_\tau(Y - \hat{y})]\) if and only if \(\hat{y}\) equals the \(\tau\)-quantile of the conditional distribution of \(Y\).

**Asymmetric penalty.** Unlike squared error, which penalises over- and under-estimation symmetrically, \(\rho_\tau\) assigns different costs depending on the sign of the residual:

| Condition | Residual | Loss | Interpretation |
|-----------|----------|------|----------------|
| \(r \ge 0\) (under-prediction) | \(y > \hat{y}\) | \((\tau - 1)\, r\) | Weight \((\tau - 1)\), negative since \(r > 0\) |
| \(r < 0\) (over-prediction) | \(y < \hat{y}\) | \(\tau\, r\) | Weight \(\tau\), negative since \(r < 0\) |

Because \(\tau \in (0,1)\), we have \(\tau - 1 < 0\). The magnitudes of the two branches differ:

- For **small \(\tau\)** (e.g. 0.1): under-predictions are penalised lightly \((\tau - 1 = -0.9)\) relative to over-predictions \((\tau = 0.1)\). The model is pushed toward the **lower tail** of the distribution.
- For **large \(\tau\)** (e.g. 0.9): over-predictions are penalised lightly. The model is pushed toward the **upper tail**.
- For **\(\tau = 0.5\)**: both branches have equal magnitude \(|r|/2\), recovering a symmetric absolute-error criterion (up to scaling).

**Visual analogy.** The piecewise-linear shape of \(\rho_\tau(r)\) resembles a "pinball" resting in a wedge — hence the name.

### 5.3 Worked Example

Suppose \(\tau = 0.9\), \(y = 500\) W, and the model predicts \(\hat{y} = 450\) W.

\[
r = y - \hat{y} = 50 \ge 0 \quad \Rightarrow \quad \rho_{0.9}(50) = (0.9 - 1) \times 50 = -5.
\]

If instead \(\hat{y} = 550\) W:

\[
r = -50 \quad \Rightarrow \quad \rho_{0.9}(-50) = 0.9 \times (-50) = -45.
\]

Over-prediction incurs a **nine times larger** penalty than the same-magnitude under-prediction at \(\tau = 0.9\), consistent with targeting the upper decile of the distribution.

### 5.4 Implementation Correspondence

In `src/net/utils.py`, the `QuantileLoss` module implements Equation (3) exactly:

```python
error = targets - inputs          # r = y - ŷ
loss = torch.max(quantiles * error, (quantiles - 1) * error)
```

Here `inputs` holds the predicted quantiles \(\hat{y}^{(\tau_n)}\), `targets` holds the observed power \(y\), and `quantiles` holds the vector \([\tau_1, \ldots, \tau_{N_\tau}]\). The loss is averaged over batch, time, appliance, and quantile dimensions.

---

## 6. Multi-Target Quantile Objective (Equation 4)

### 6.1 Statement

The deep neural network for multi-target quantile regression is trained by minimising:

\[
\mathcal{L}(\rho_\tau)\bigl(\hat{\mathbf{y}}_\tau, \mathbf{y}\bigr) =
\frac{1}{T M}
\sum_{t=1}^{T}
\sum_{n=1}^{N_\tau}
\sum_{m=1}^{M}
\rho_{\tau_n}\!\left( \hat{y}_m^{(\tau_n)}(t) - y_m(t) \right).
\tag{4}
\]

Note: the paper writes the argument of \(\rho_{\tau_n}\) as \(\hat{y}_m(t)^{(\tau_n)} - y_m(t)\), which is algebraically identical to \(y_m(t) - \hat{y}_m^{(\tau_n)}(t)\) up to sign convention in the piecewise definition. The implementation uses \(r = y - \hat{y}\) as in Section 5.

### 6.2 Structure of the Objective

Equation (4) aggregates pinball loss over **three nested indices**:

```
For each time step t = 1 … T
  For each quantile level n = 1 … N_τ
    For each appliance m = 1 … M
      Compute ρ_{τ_n}( y_m(t) − ŷ_m^{(τ_n)}(t) )
Average over T × M  (and implicitly over N_τ in the summation)
```

| Index | Role |
|-------|------|
| \(t\) | Temporal dimension — each sample in the input sequence contributes equally |
| \(n\) | Quantile dimension — each quantile head is supervised at its own level |
| \(m\) | Appliance dimension — all \(M\) appliances are regressed jointly |

The normalisation factor \(1/(TM)\) yields a **per-time-step, per-appliance average loss**, scaled by the number of quantile levels through the inner sum.

---

## 7. Joint Multi-Task Training Objective

UNet-NILM optimises **both** the state classification loss (Equation 5) and the quantile regression loss (Equation 4) simultaneously from shared network features:

\[
\mathcal{L}_{\text{total}} =
\mathcal{L}_{CE}\bigl(\hat{\mathbf{s}}, \mathbf{s}\bigr)
+
\mathcal{L}(\rho_\tau)\bigl(\hat{\mathbf{y}}_\tau, \mathbf{y}\bigr).
\]

| Component | Equation | Task | Output head |
|-----------|----------|------|-------------|
| \(\mathcal{L}_{CE}\) | (5) | Multi-label ON/OFF detection | `fc_out_state` → \((B, 2, M)\) logits |
| \(\mathcal{L}(\rho_\tau)\) | (4) | Multi-target power quantile regression | `fc_out_power` → \((B, N_\tau, M)\) quantiles |

This is implemented in `src/net/model_pl.py` (`NILMnet._step`):

```python
loss = loss_nll + loss_mse   # L_CE + L(ρ_τ)
```

where `loss_nll` corresponds to Equation (5) and `loss_mse` (the quantile loss, despite the variable name) corresponds to Equation (4).

### Architectural Flow

```
Aggregate power sequence x(t)
        │
        ▼
  Shared encoder (UNet / CNN)
        │
        ├──────────────────────────────┐
        ▼                              ▼
  fc_out_state (2 × M)          fc_out_power (N_τ × M)
        │                              │
        ▼                              ▼
  Softmax → ON/OFF (Eq. 5)      Quantile estimates (Eq. 4)
        │                              │
        └────────── L_total ───────────┘
```

The multi-task design ensures that state detection informs power estimation and vice versa: the classification head learns to identify active appliances while the regression head learns the conditional power distribution given aggregate context.

---

## 8. Predictive Uncertainty

### 8.1 Why Quantiles Provide Uncertainty "Almost for Free"

Standard point-regression (MSE) requires a separate mechanism — such as Monte Carlo dropout, deep ensembles, or explicit variance heads — to estimate predictive uncertainty. Quantile regression embeds uncertainty in the **primary prediction target**: if the model is uncertain about appliance power at time \(t\), the quantile estimates \(\hat{y}^{(\tau_1)}, \ldots, \hat{y}^{(\tau_{N_\tau})}\) will be **spread apart**; if the model is confident, they will **collapse toward a common value**.

No additional probabilistic head or sampling procedure is required at inference time.

### 8.2 Practical Uncertainty Measures

From the estimated quantile set (Equation 2), the following diagnostics are readily available:

| Measure | Formula | Interpretation |
|---------|---------|----------------|
| Point estimate | \(\hat{y}^{(0.5)}\) | Median predicted power |
| Prediction interval | \(\bigl[\hat{y}^{(0.025)},\; \hat{y}^{(0.975)}\bigr]\) | Approximate 95% interval |
| Spread | \(\hat{y}^{(0.9)} - \hat{y}^{(0.1)}\) | Inter-decile range |
| Asymmetry | \(\hat{y}^{(0.9)} - \hat{y}^{(0.5)}\) vs. \(\hat{y}^{(0.5)} - \hat{y}^{(0.1)}\) | Skew of predictive distribution |

### 8.3 Trust and Interpretability in NILM

Uncertainty quantification addresses a central limitation of black-box NILM models: a low MAE does not reveal whether the model is **extrapolating** beyond its training distribution (e.g. cross-house transfer from UK-DALE House 1 to House 2). Wide prediction intervals at test time signal reduced reliability, enabling:

- **Selective prediction** — suppress or flag disaggregated outputs when uncertainty exceeds a threshold
- **Human-in-the-loop monitoring** — prioritise manual review of high-uncertainty intervals
- **Model comparison** — two models with similar MAE may differ substantially in calibration of their uncertainty bands

---

## 9. Summary

| Section | Name | Purpose |
|---------|------|---------|
| §2.4 | Seq2Quantile | Smooth raw power with median filter; threshold to create ON/OFF labels |
| (5) | Cross-entropy \(\mathcal{L}_{CE}\) | Multi-label ON/OFF state detection via per-appliance softmax |
| (2) | Quantile collection | Non-parametric predictive distribution via \(N_\tau\) inverse-CDF samples |
| (3) | Pinball loss \(\rho_\tau(r)\) | Asymmetric loss targeting the \(\tau\)-quantile consistently |
| (4) | Multi-target objective \(\mathcal{L}(\rho_\tau)\) | Aggregates pinball loss over time, appliances, and quantile levels |
| — | \(\mathcal{L}_{\text{total}} = \mathcal{L}_{CE} + \mathcal{L}(\rho_\tau)\) | Joint multi-task training of state detection and power estimation |

Together, these equations define the complete UNet-NILM learning framework: **softmax-based multi-label classification** for reliable appliance state detection, combined with **multi-target quantile regression** for calibrated distributional power estimates — supporting accurate, reliable, and interpretable disaggregation in practical energy-monitoring systems.

---

## References

- Faustine, A., Pereira, L., Bousbiat, H., & Kulkarni, S. (2020). **UNet-NILM: A Deep Neural Network for Multi-tasks Appliances State Detection and Power Estimation in NILM.** *Proceedings of the 5th International Workshop on Non-Intrusive Load Monitoring (NILM)*, co-located with ACM BuildSys 2020.
- Koenker, R., & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33–50.
- Zhang, M.-L., & Zhou, Z.-H. (2014). A Review on Multi-Label Learning Algorithms. *IEEE Transactions on Knowledge and Data Engineering*, 26(8), 1819–1837.
- Implementation: `src/data/load_data.py` (`quantile_filter`, `binarization`), `src/net/modules.py`, `src/net/model_pl.py` (`NILMnet`), `src/net/utils.py` (`QuantileLoss`).

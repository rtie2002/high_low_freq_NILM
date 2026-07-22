# Paper Analysis: Deep Domain Adaptation for NILM (Lin et al., IEEE TSG 2022)

**Full title:** Deep Domain Adaptation for Non-Intrusive Load Monitoring Based on a Knowledge Transfer Learning Network  
**Authors:** Jun Lin, Jin Ma, Jianguo Zhu, Huishi Liang  
**Venue:** *IEEE Transactions on Smart Grid*, Vol. 13, No. 1, pp. 280–292, Jan. 2022  
**DOI:** [10.1109/TSG.2021.3115910](https://doi.org/10.1109/TSG.2021.3115910)  
**Local PDF:** `Deep_Domain_Adaptation_for_Non-Intrusive_Load_Monitoring_Based_on_a_Knowledge_Transfer_Learning_Network.pdf`

---

## 1. One-sentence summary

Train a **TCN disaggregator on labeled source houses**, and simultaneously **align feature distributions with unlabeled target-house aggregate**, so the model transfers to an unseen house **without any target appliance labels**.

---

## 2. Problem they solve

Standard supervised NILM (CNN / LSTM / seq2point) works well when train and test come from the **same distribution**. In practice:

| Issue | Why it hurts NILM |
|-------|-------------------|
| Cross-house / cross-dataset shift | Different appliances, habits, wiring → `P_source(X) ≠ P_target(X)` |
| Label scarcity | Target house usually has only mains (no submeter labels) |
| Temporal dynamics | Load patterns change over time; shallow features transfer poorly |
| Naive fine-tuning | Fine-tuning FC layers on tiny labeled target data is unstable (representation bias) |

**Goal:** Learn a representation that is:

1. **Discriminative** — good for appliance power regression (source labels)  
2. **Domain-invariant** — similar feature stats for source and target aggregates  

so the regressor trained on source can be applied directly to target.

---

## 3. Method overview (TLN)

They call the model a **Transfer Learning Network (TLN)**.

```text
Source (labeled):  aggregate X_S + appliance y_S
Target (unlabeled): aggregate X_T only
                    │
                    ▼
         Shared TCN feature extractor
         (5 residual dilated-causal blocks)
                    │
                    ▼
         FC layers (fc6–fc8)  ← domain adaptation applied here
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   Regression loss L_R    Domain loss L_domain
   (MSE on source only)   (MMD + CORAL on source vs target features)
                    │
                    ▼
         L = (1 − λ) L_R + λ L_domain
```

**Key design choices:**

- **Shared weights** between source and target streams (same network sees both).
- **No target appliance labels** during training (unlike D'Incecco seq2point fine-tuning).
- **Multi-layer adaptation** on fc6–fc8 (not a single adaptation layer).
- **Sequence-to-sequence** output (window → same-length appliance sequence), causal — suitable for real-time use.

In math notation:

$$
\mathcal{L} = (1-\lambda)\,\mathcal{L}_{R} + \lambda\,\mathcal{L}_{\mathrm{domain}}.
$$

### 3.1 How data is input (with dimensions)

#### 正确理解 / Correct idea（中文优先）

**一句话：** 同一套网络权重 $\Theta$，源域和目标域数据**分开各跑一遍**；两个损失合在一起反向传播，更新这一套 $\Theta$。

```text
真正是这样（对的）：

源域数据 X_S ──► 网络(同一套权重 Θ) ──► 预测 Ŷ_S + 特征 Z_S
目标域数据 X_T ──► 网络(同一套权重 Θ) ──►           特征 Z_T

然后：
  回归损失   = Ŷ_S 和真实标签 Y_S 的误差     （只有源域有标签）
  域对齐损失 = 让 Z_S 和 Z_T 的分布尽量像
  总损失     = 两者加权
  反向传播   → 更新这一套 Θ
```

```text
This is the correct picture:

Source data   X_S ──► network (same weights Θ) ──► prediction Ŷ_S + features Z_S
Target data   X_T ──► network (same weights Θ) ──►                 features Z_T

Then:
  Regression loss   = error between Ŷ_S and true labels Y_S   (source has labels only)
  Domain-alignment loss = make distributions of Z_S and Z_T as similar as possible
  Total loss        = weighted sum of the two losses
  Backpropagation   → update this single shared Θ
```

| 步骤 | 做什么 | 说明 |
|------|--------|------|
| 1 | 把 $X\_S$ 送进网络 | 得到预测 $\hat{Y}\_S$ 和特征 $Z\_S$ |
| 2 | 把 $X\_T$ 再送进**同一个**网络 | 只得到特征 $Z\_T$（目标域没有电器标签） |
| 3 | 算回归损失 $\mathcal{L}\_R$ | 只比较 $\hat{Y}\_S$ 与 $Y\_S$ |
| 4 | 算域对齐损失 $\mathcal{L}\_{\mathrm{domain}}$ | 比较 $Z\_S$ 与 $Z\_T$（MMD + CORAL） |
| 5 | $\mathcal{L}=(1-\lambda)\mathcal{L}\_R+\lambda\mathcal{L}\_{\mathrm{domain}}$ | 合成总损失 |
| 6 | `backward` + `optimizer.step` | **一次**更新共享权重 $\Theta$ |

**不是**两个独立的孪生网络，也**不是**一个 block 同时接两个输入口。  
就是：**同一套 block，源域、目标域数据分开输入（调用两次）**。

English equivalent (proper notation):

> **Why equations looked broken:** Cursor Markdown preview treats `_text_` as *italic*, so `_` inside LaTeX (like `\mathcal{L}_{R}`) gets eaten. Below, every subscript uses `\_` so it displays correctly. For Overleaf / thesis, you can change `\_` back to `_`.

**Simple meaning (no math):**

```text
1) Pass source windows through the network  →  prediction Yhat_S + features Z_S
2) Pass target windows through SAME network →  features Z_T only
3) L_R      = how wrong Yhat_S is vs true Y_S          (needs labels)
4) L_domain = how different Z_S and Z_T look           (push them to be similar)
5) L        = (1-λ)*L_R + λ*L_domain                   (mix the two goals)
6) Backprop updates the single shared weight set Θ
```

**Forward (shared parameters $\Theta$):**  
*Meaning: same network $f_\Theta$, called on source and on target.*

$$
\hat{\mathbf{Y}}\_S,\; \mathbf{Z}\_S = f\_{\Theta}(\mathbf{X}\_S),\qquad \mathbf{Z}\_T = f\_{\Theta}(\mathbf{X}\_T).
$$

**Regression loss $\mathcal{L}\_{R}$:**  
*Meaning: only source has labels; measure power prediction error (MSE).*

$$
\mathcal{L}\_{R} = \mathrm{MSE}\big(\hat{\mathbf{Y}}\_S,\; \mathbf{Y}\_S\big).
$$

**Domain loss $\mathcal{L}\_{\mathrm{domain}}$:**  
*Meaning: on FC layers 6–8, make source features and target features look alike (MMD = mean distance, CORAL = covariance distance).*

$$
\mathcal{L}\_{\mathrm{domain}} = \sum\_{l=6}^{8}\Big[ \mu\,\mathrm{MMD}^{2}\big(\mathbf{Z}\_S^{(l)},\mathbf{Z}\_T^{(l)}\big) + (1-\mu)\,\mathcal{L}\_{\mathrm{CORAL}}\big(\mathbf{Z}\_S^{(l)},\mathbf{Z}\_T^{(l)}\big) \Big].
$$

**Total loss $\mathcal{L}$:**  
*Meaning: $\lambda$ balances “predict well on source” vs “look the same across houses” (paper uses $\lambda=0.6$).*

$$
\mathcal{L} = (1-\lambda)\,\mathcal{L}\_{R} + \lambda\,\mathcal{L}\_{\mathrm{domain}}.
$$

**Parameter update:**  
*Meaning: one gradient step on the shared weights.*

$$
\Theta \leftarrow \Theta - \eta\,\nabla\_{\Theta}\mathcal{L}.
$$

**Compact summary:**

$$
\mathbf{X}\_S \xrightarrow{f\_{\Theta}} (\hat{\mathbf{Y}}\_S,\mathbf{Z}\_S),\quad \mathbf{X}\_T \xrightarrow{f\_{\Theta}} \mathbf{Z}\_T,\quad \min\_{\Theta}\;\mathcal{L}.
$$

**Copy-paste for Overleaf / thesis (normal `_`, not escaped):**

```latex
\hat{\mathbf{Y}}_S, \mathbf{Z}_S = f_{\Theta}(\mathbf{X}_S),
\qquad
\mathbf{Z}_T = f_{\Theta}(\mathbf{X}_T).

\mathcal{L}_{R}
= \mathrm{MSE}(\hat{\mathbf{Y}}_S, \mathbf{Y}_S).

\mathcal{L}_{\mathrm{domain}}
= \sum_{l=6}^{8}
\Bigl[
\mu\,\mathrm{MMD}^{2}(\mathbf{Z}_S^{(l)},\mathbf{Z}_T^{(l)})
+ (1-\mu)\,
\mathcal{L}_{\mathrm{CORAL}}(\mathbf{Z}_S^{(l)},\mathbf{Z}_T^{(l)})
\Bigr].

\mathcal{L}
= (1-\lambda)\,\mathcal{L}_{R}
+ \lambda\,\mathcal{L}_{\mathrm{domain}}.

\Theta \leftarrow \Theta - \eta\,\nabla_{\Theta}\mathcal{L}.
```

#### Important: the block does **not** take two inputs at once

##### Plain English

Imagine one coffee machine (the network).

- You put in house-A coffee beans → get a drink (source prediction).
- You put in house-B coffee beans → get another drink (target features).

It is the **same machine** (same settings / weights). You do **not** pour both bags into the machine at the same time as two separate spouts.

In NILM:

| What | Role |
|------|------|
| Coffee machine | `tcn1`…`tcn5` + `fc6`…`fc8` (weights $\Theta$) |
| House-A beans | source windows $\mathbf{X}_S$ (has labels $\mathbf{Y}_S$) |
| House-B beans | target windows $\mathbf{X}_T$ (no labels) |

##### Tiny numbers (batch size $B=2$, window $T=4$)

**Source batch** (2 windows from labeled houses):

$$
\mathbf{X}_S =
\begin{bmatrix}
320 & 450 & 1800 & 400 \\
310 & 305 & 312 & 308
\end{bmatrix}
\quad\text{shape }(2,1,4)
$$

**Target batch** (2 windows from unlabeled house):

$$
\mathbf{X}_T =
\begin{bmatrix}
500 & 520 & 510 & 900 \\
480 & 2000 & 1900 & 470
\end{bmatrix}
\quad\text{shape }(2,1,4)
$$

**Way 1 — two calls (easiest to understand)**

```text
Step 1:  feed X_S into the network  →  get Yhat_S and features Z_S
Step 2:  feed X_T into the SAME network  →  get features Z_T
Step 3:  L_R = error(Yhat_S, Y_S)          # only source has labels
         L_domain = distance(Z_S, Z_T)     # make Z_S and Z_T look similar
Step 4:  update the network weights once
```

The network never sees “two inputs” in one layer. It sees **one batch at a time**.

**Way 2 — stack then split (same math, one call)**

Some code stacks the 2 source rows and 2 target rows into **one bigger batch of 4 rows**:

$$
\mathbf{X}_{\mathrm{cat}} =
\begin{bmatrix}
320 & 450 & 1800 & 400 \\
310 & 305 & 312 & 308 \\
500 & 520 & 510 & 900 \\
480 & 2000 & 1900 & 470
\end{bmatrix}
\quad\text{shape }(4,1,4) = (2B,1,T)
$$

```text
row 0: source window 0
row 1: source window 1
row 2: target window 0
row 3: target window 1
```

Run **one** forward: all 4 rows go through `tcn1`…`tcn5` independently (like 4 separate samples in a normal DataLoader batch).

Then **split** the output features:

```text
Z_cat has 4 feature vectors (one per row)
Z_S = first 2 rows
Z_T = last 2 rows
```

That is all “concatenate along the batch axis” means: **put source and target samples in the same list of examples**, not fuse two channels or two time series into one sample.

```text
WRONG:  one sample = [source_signal AND target_signal] as 2 channels
RIGHT:  batch = [source0, source1, target0, target1]   # 4 separate samples
```

##### Diagram

```text
CORRECT mental model (shared weights, two calls):

        X_S ──► [ tcn1 ... tcn5 ; fc6..fc8 ] ──► Yhat_S , Z_S
                      ▲
                      │ same weights Θ
                      │
        X_T ──► [ tcn1 ... tcn5 ; fc6..fc8 ] ──►            Z_T


EQUIVALENT (stack batch, one call, then split):

        X_cat = [X_S rows ; X_T rows]   shape (4,1,4) when B=2
                │
                ▼
        [ tcn1 ... tcn5 ; fc6..fc8 ]     one forward
                │
                ▼
        Z_cat (4 feature rows)
           ├── first B rows  → Z_S
           └── last  B rows  → Z_T
```

##### Pseudocode

```python
# Way 1: two calls
Yhat_S, Z_S = model(X_S)      # X_S shape (2,1,4)
_,      Z_T = model(X_T)      # X_T shape (2,1,4)

# Way 2: one call (identical result if no BatchNorm train quirks)
X_cat = torch.cat([X_S, X_T], dim=0)   # shape (4,1,4)
Yhat_cat, Z_cat = model(X_cat)
Yhat_S, Z_S = Yhat_cat[:2], Z_cat[:2]
Z_T = Z_cat[2:]
```

**Remember:** each row is still one house’s aggregate window of shape $(1,T)$. Source and target are different **rows**, not two inputs into one block.

#### Notation and paper settings

| Symbol | Meaning | Typical value (paper) |
|--------|---------|------------------------|
| $T$ | Sliding window length | $600$ samples ($\approx 1$ h at $6$ s) |
| $l_s$ | Window stride | $8$ (UK-DALE / REFIT), $2$ (REDD) |
| $B$ | Mini-batch size (same for both domains) | e.g. $B=32$ (illustrative; paper uses equal source/target batch size) |
| $C_{\mathrm{in}}$ | Input channels | $1$ (aggregate active power only) |
| Appliance | One target appliance per model | e.g. kettle, fridge, WM, … |

#### Step A — cut long traces into windows

Raw house aggregate is a 1-D time series. A sliding window of length $T$ produces one sample:

$$
\mathbf{x} = (x_1,x_2,\ldots,x_T) \in \mathbb{R}^{T}.
$$

For the **source** house(s), each window also has a matching appliance label sequence of the same length (seq2seq):

$$
\mathbf{y} = (y_1,y_2,\ldots,y_T) \in \mathbb{R}^{T}.
$$

For the **target** house, only $\mathbf{x}_T$ is available (no $\mathbf{y}_T$ during training).

**Concrete UK-DALE-style example** (task $U^{A}\to U2$, appliance = dishwasher):

| Stream | What you load                                  | After windowing                            |
| --------| ------------------------------------------------| --------------------------------------------|
| Source | Aggregate + dishwasher from UK-DALE houses ≠ 2 | many windows $(\mathbf{x}_S,\mathbf{y}_S)$ |
| Target | Aggregate only from UK-DALE house 2            | many windows $\mathbf{x}_T$ (no labels)    |

One source window:

```text
x_S[t] = whole-house power at times t … t+599     shape (600,)
y_S[t] = dishwasher power at times t … t+599      shape (600,)
```

One target window:

```text
x_T[t] = whole-house power at times t … t+599     shape (600,)
y_T     = NOT USED in training
```

#### Step B — form a training mini-batch (two tensors)

At each optimisation step the loader draws $B$ source windows and $B$ target windows:

$$
\mathbf{X}_S \in \mathbb{R}^{B \times 1 \times T},\quad
\mathbf{Y}_S \in \mathbb{R}^{B \times 1 \times T},\quad
\mathbf{X}_T \in \mathbb{R}^{B \times 1 \times T}.
$$

With $B=32$, $T=600$:

| Tensor | Role | Shape |
|--------|------|-------|
| $\mathbf{X}_S$ | Source aggregates | $(32,\ 1,\ 600)$ |
| $\mathbf{Y}_S$ | Source appliance powers (labels) | $(32,\ 1,\ 600)$ |
| $\mathbf{X}_T$ | Target aggregates (unlabeled) | $(32,\ 1,\ 600)$ |

They are **not** concatenated along the batch axis for the regression head. Both go through the shared backbone; only $\mathbf{X}_S$ is paired with $\mathbf{Y}_S$ for $\mathcal{L}_{R}$.

#### Step C — forward through shared `tcn1`–`tcn5` and `fc6`–`fc8`

Each block still has **one** input tensor of shape $(B,1,T)$. Source and target are two calls (or two halves of a stacked batch), not two inputs into one layer:

```text
Call 1 (source):
  X_S (32,1,600) ──► tcn1→…→tcn5 → fc6→fc7→fc8 ──► Yhat_S (32,1,600), Z_S

Call 2 (target):
  X_T (32,1,600) ──► tcn1→…→tcn5 → fc6→fc7→fc8 ──►                 Z_T
                     ▲
                     └── identical weights Θ (not a second copy of the network)
```

**Approx. shapes through the network** (channel widths are paper-style schematic; exact FC widths are implementation details):

| Stage | Source stream | Target stream | Notes |
|-------|---------------|---------------|-------|
| Input | $(B,1,T)=(32,1,600)$ | $(32,1,600)$ | 1-D power window |
| After `tcn1`–`tcn5` | $(B,C,T)$ e.g. $(32,C,600)$ | $(32,C,600)$ | causal dilated TCN keeps length $T$ |
| After flatten / pool to FC | $(B,D)$ | $(B,D)$ | vector per window (or per timestep if seq FC) |
| `fc6` features $\mathbf{Z}^{(6)}$ | $(B,D_6)$ | $(B,D_6)$ | used in $\mathcal{L}_{\mathrm{domain}}$ |
| `fc7` features $\mathbf{Z}^{(7)}$ | $(B,D_7)$ | $(B,D_7)$ | used in $\mathcal{L}_{\mathrm{domain}}$ |
| `fc8` / output | $\hat{\mathbf{Y}}_S\in\mathbb{R}^{B\times 1\times T}$ | *(no label loss)* | seq2seq appliance power |

Paper design intent: TCN is **sequence-to-sequence**, so the appliance prediction for source matches input length:

$$
\hat{\mathbf{Y}}_S \in \mathbb{R}^{B \times 1 \times T}
\quad\text{(e.g. }(32,1,600)\text{)}.
$$

Domain adaptation compares **feature matrices** at layers $l\in\{6,7,8\}$:

$$
\mathbf{Z}_S^{(l)},\ \mathbf{Z}_T^{(l)} \in \mathbb{R}^{B \times D_l}.
$$

#### Step D — losses on this batch

$$
\mathcal{L}_{R}
= \mathrm{MSE}\big(\hat{\mathbf{Y}}_S,\ \mathbf{Y}_S\big)
\quad\text{shapes: both }(B,1,T),
$$

$$
\mathcal{L}_{\mathrm{domain}}
= \sum_{l=6}^{8}
\left[
\mu\,\mathrm{MMD}^{2}\left(\mathbf{Z}_S^{(l)},\mathbf{Z}_T^{(l)}\right)
+(1-\mu)\,
\mathcal{L}_{\mathrm{CORAL}}\left(\mathbf{Z}_S^{(l)},\mathbf{Z}_T^{(l)}\right)
\right],
$$

$$
\mathcal{L}
= (1-\lambda)\,\mathcal{L}_{R} + \lambda\,\mathcal{L}_{\mathrm{domain}}.
$$

Backprop updates **one** set of weights $\Theta$ (shared TCN + FC).

#### Tiny numeric toy example (easier to visualise)

Suppose $T=4$, $B=2$ (toy only; paper uses $T=600$):

```text
X_S =
[[ 320, 450, 1800, 400 ],   # source window 0 (aggregate W)
 [ 310, 305,  312, 308 ]]   # source window 1
shape (2, 1, 4)

Y_S =
[[   0,   0, 1500,   0 ],   # dishwasher power for window 0
 [   0,   0,    0,   0 ]]   # OFF for window 1
shape (2, 1, 4)

X_T =
[[ 500, 520,  510, 900 ],   # target house aggregate only
 [ 480, 2000, 1900, 470 ]]
shape (2, 1, 4)
# no Y_T
```

Both $X_S$ and $X_T$ enter the shared `tcn1`–`tcn5`. Only $(X_S,Y_S)$ build $\mathcal{L}_{R}$. Feature stats of both streams build $\mathcal{L}_{\mathrm{domain}}$.

#### What is *not* done

| Incorrect mental model | Actual paper design |
|------------------------|---------------------|
| Concatenate $X_S$ and $X_T$ into one $(2B,1,T)$ and treat as one labeled batch | Two streams; only source has labels |
| Feed $[x_S; x_T]$ as 2-channel input | Single channel; two forward passes / parallel batch with shared $\Theta$ |
| Need target appliance labels | Target uses **aggregate only** |

#### Inference (after training)

Only the target aggregate is needed:

$$
\mathbf{X}_T \in \mathbb{R}^{B\times 1\times T}
\ \xrightarrow{\ \Theta\ }\
\hat{\mathbf{Y}}_T \in \mathbb{R}^{B\times 1\times T}.
$$

No domain loss, no source batch.

---

## 4. Architecture details

### 4.1 Temporal Convolutional Network (TCN)

Why TCN instead of LSTM/standard CNN:

| Property | Benefit for NILM |
|----------|------------------|
| Dilated **causal** conv | Large receptive field; no future leakage |
| Residual blocks | Train deeper nets without vanishing gradients |
| Seq2seq same length | Real-time disaggregation of full window |

Dilated conv (filter size $k$, dilation $d$):

$$
F(s) = (x *_d f)(s) = \sum_{i=0}^{k-1} f(i)\, x_{s - d\cdot i}.
$$

Typical stacking: $d = 1, 2, 4, \ldots$ → exponential history coverage with few layers.

Each residual block: 2× dilated causal conv + ReLU + weight norm + dropout + 1×1 identity skip.

**Final architecture (AlexNet-inspired):**

- `tcn1`–`tcn5`: shared general features (transferable)
- `fc6`–`fc8`: task-specific layers where domain discrepancy is measured and reduced

Hyperparameters they settled on (tuned on REDD leave-one-house):

| Param | Value |
|-------|-------|
| Shared TCN layers `n_tcn` | 5 |
| FC layers `n_fc` | 3 |
| Window `T` | 600 samples (~1 hour at 6 s) |
| Stride | 8 (UK-DALE/REFIT), 2 (REDD, data-limited) |
| Domain mix `μ` | 0.4 |
| Loss mix `λ` | 0.6 |

### 4.2 Training objective — loss explained simply (procedure)

> **Preview note:** Cursor Markdown eats `_` inside math (treats it as italic). Display equations below use `\_`. For Overleaf / thesis, use the copy-paste block at the end (normal `_`).

#### What the loss is trying to do (one paragraph)

The model must do two jobs at the same time:

1. **Learn to disaggregate** — on labeled source houses, predicted appliance power should match the true power.
2. **Learn house-invariant features** — mid/late features from source aggregates and from unlabeled target aggregates should look statistically similar, so the disaggregator still works on a new house.

Job 1 alone → overfits the source house.  
Job 2 alone → features become similar but useless.  
The paper **adds** both losses with weight $\lambda$.

#### Training procedure (one mini-batch)

```text
Step 1  Load source batch:   X_S (aggregate), Y_S (appliance power labels)
        Load target batch:   X_T (aggregate only; NO appliance labels)

Step 2  Forward source:  (Yhat_S, Z_S) = f_Θ(X_S)
        Forward target:  Z_T           = f_Θ(X_T)     # same network Θ

Step 3  Regression loss L_R:
          compare Yhat_S vs Y_S   (MSE)
          # "How wrong is the power prediction on the labeled house?"

Step 4  Domain loss L_domain:
          compare features Z_S vs Z_T on FC layers 6,7,8
          using MMD (mean gap) + CORAL (covariance gap)
          # "Do source and target features look like the same distribution?"

Step 5  Total loss:
          L = (1-λ) * L_R + λ * L_domain
          paper: λ = 0.6

Step 6  Backprop → update the single shared Θ
```

| Symbol | Plain meaning |
|--------|----------------|
| $\mathbf{X}\_S$, $\mathbf{Y}\_S$ | Source aggregate windows + appliance labels |
| $\mathbf{X}\_T$ | Target aggregate windows (no labels) |
| $\hat{\mathbf{Y}}\_S$ | Predicted appliance power on source |
| $\mathbf{Z}\_S$, $\mathbf{Z}\_T$ | Hidden features at FC layers (used for alignment) |
| $\mathcal{L}\_R$ | “Predict power correctly” (source only) |
| $\mathcal{L}\_{\mathrm{domain}}$ | “Make source/target features similar” |
| $\lambda$ | How much to care about domain alignment (0.6) |
| $\mu$ | Inside domain loss: mix MMD vs CORAL (0.4) |

---

#### Loss A — regression (source only)

**Simple idea:** for each labeled source window, predicted power should be close to true power. Use mean squared error (MSE). Target house is **not** in this term.

$$
\mathcal{L}\_{R} = \frac{1}{n\_s}\sum\_{i=1}^{n\_s}\big(\mathbf{y}\_i - \hat{\mathbf{y}}\_i\big)^{2}.
$$

- $n\_s$: number of source samples in the batch / set  
- $\mathbf{y}\_i$: true appliance power sequence  
- $\hat{\mathbf{y}}\_i$: network prediction  

---

#### Loss B — domain alignment in detail (MMD + CORAL)

**Goal in one sentence:** make the hidden features of the **source house(s)** and the **target house** look like they came from the same distribution, so a regressor trained on source labels still works on the target.

After the shared TCN, the paper measures this gap on fully connected features $\mathbf{Z}\_{\mathcal{S}}^{l}$ and $\mathbf{Z}\_{\mathcal{T}}^{l}$ at layers $l \in \{6,7,8\}$ (fc6–fc8). Two distances are used:

| Tool | What it matches | Order of statistics |
|------|-----------------|---------------------|
| **MMD** | Feature **centers** (means) after a kernel map | 1st order |
| **CORAL** | Feature **shape / correlations** (covariances) | 2nd order |

Matching only centers is not enough: two clouds can share a center but one is elongated and the other is round. That is why the paper mixes MMD and CORAL.

---

##### B1. MMD (Maximum Mean Discrepancy) — paper Eq. (6)

**Core idea**

If two distributions are the same, then after mapping samples into a high-dimensional feature space, their **means (centers)** should coincide. MMD is the distance between those two centers.

**Formula (paper Eq. 6), written for features at one layer:**

$$
\mathrm{MMD}\big(\mathbf{Z}\_{\mathcal{S}}, \mathbf{Z}\_{\mathcal{T}}\big) = \left\| \frac{1}{n\_s}\sum\_{i=1}^{n\_s}\phi\big(\mathbf{z}\_{\mathcal{S}}^{i}\big) - \frac{1}{n\_t}\sum\_{j=1}^{n\_t}\phi\big(\mathbf{z}\_{\mathcal{T}}^{j}\big) \right\|\_{\mathcal{H}}.
$$

**Term-by-term:**

| Symbol | Meaning |
|--------|---------|
| $\mathbf{z}\_{\mathcal{S}}^{i}$ | $i$-th source sample feature (one window’s FC vector) |
| $\mathbf{z}\_{\mathcal{T}}^{j}$ | $j$-th target sample feature |
| $n\_s$, $n\_t$ | Number of source / target samples in the batch |
| $\phi(\cdot)$ | Map from the original feature space into a high-dim RKHS $\mathcal{H}$ |
| $\frac{1}{n\_s}\sum \phi(\mathbf{z}\_{\mathcal{S}}^{i})$ | Center (mean) of all mapped **source** features |
| $\frac{1}{n\_t}\sum \phi(\mathbf{z}\_{\mathcal{T}}^{j})$ | Center (mean) of all mapped **target** features |
| $\|\cdot\|\_{\mathcal{H}}$ | Distance between those two centers in Hilbert space |

**Why map with $\phi$?**

In the original feature space, source and target clouds may be tangled and hard to compare with a simple Euclidean mean. Mapping into an RKHS makes distribution differences easier to measure.

**Gaussian kernel (what the paper uses)**

You do **not** explicitly build huge high-dimensional vectors. With a **Gaussian kernel**, MMD can be computed with a **kernel trick** in the original space (fast, standard practice). The paper uses a Gaussian kernel as the practical implementation of $\phi$.

**One-line summary of MMD**

> Map both sides to a high-dim space and measure how far apart their **centers** are (1st-order statistics).

---

##### B2. Deep CORAL (Correlation Alignment) — paper Eqs. (7)–(11)

**Core idea**

Matching centers alone is not enough. Example: two clouds can share the same mean, but one is a long ellipse and the other is a disk — distributions still differ. CORAL aligns **second-order** information: the **covariance matrices** (shape and feature-to-feature correlations).

**Overall CORAL loss (paper Eq. 7):**

$$
\mathcal{L}\_{\mathrm{CORAL}} = \frac{1}{4L^{2}}\left\| \mathbf{C}\_{\mathcal{S}} - \mathbf{C}\_{\mathcal{T}} \right\|\_{F}^{2}.
$$

| Symbol | Meaning |
|--------|---------|
| $\mathbf{C}\_{\mathcal{S}}$, $\mathbf{C}\_{\mathcal{T}}$ | Covariance matrices of source / target features |
| $L$ | Feature dimension of the current layer |
| $\|\cdot\|\_{F}^{2}$ | Squared Frobenius norm = sum of squared entries of the matrix difference |
| $\frac{1}{4L^{2}}$ | Normalization so the loss does not explode when $L$ is large |

**Plain meaning:** subtract the two covariance matrices; if they differ a lot, CORAL loss is large; training pushes $\mathbf{C}\_{\mathcal{S}} \approx \mathbf{C}\_{\mathcal{T}}$.

**How to compute the covariances (paper Eqs. 8–9)**

Source covariance:

$$
\mathbf{C}\_{\mathcal{S}} = \frac{1}{n\_s-1}\left( \mathbf{X}\_{\mathcal{S}}^{\top}\mathbf{X}\_{\mathcal{S}} - \frac{1}{n\_s}\big(\mathbf{1}^{\top}\mathbf{X}\_{\mathcal{S}}\big)^{\top}\big(\mathbf{1}^{\top}\mathbf{X}\_{\mathcal{S}}\big) \right).
$$

Target covariance $\mathbf{C}\_{\mathcal{T}}$ is the same formula with $\mathbf{X}\_{\mathcal{T}}$ and $n\_t$ (Eq. 9).

| Piece | Role |
|-------|------|
| $\mathbf{1}$ | Column vector of ones |
| $\mathbf{1}^{\top}\mathbf{X}\_{\mathcal{S}}$ | Sum of features over samples (per dimension) |
| $\mathbf{X}\_{\mathcal{S}}^{\top}\mathbf{X}\_{\mathcal{S}}$ | Uncentered second-moment / Gram term |
| $\frac{1}{n\_s}(\cdots)$ | Mean correction (center the features) |
| $\frac{1}{n\_s-1}$ | Unbiased sample covariance denominator |

**Gradients (paper Eqs. 10–11) — what they mean in practice**

The paper also writes $\partial \mathcal{L}\_{\mathrm{CORAL}} / \partial X\_{\mathcal{S}}^{ij}$ (and the target analogue) so Adam can update weights. In PyTorch / TensorFlow you **do not hand-code** those derivatives: if you implement Eq. (7) with tensors, **autograd** computes the gradients automatically. Eqs. (10)–(11) are the theoretical justification that CORAL is differentiable end-to-end.

**One-line summary of CORAL**

> Align the **shape and correlations** of the two feature clouds (2nd-order statistics), not only their centers.

---

##### B3. Mixed domain loss — paper Eq. (12)

**Why mix both?**

- MMD → pull **centers** together (~40% with $\mu=0.4$)  
- CORAL → pull **covariances / shape** together (~60% with $1-\mu=0.6$)  
- Either alone is weaker (paper ablation on $\mu$)

**Formula (paper Eq. 12):**

$$
\mathcal{L}\_{\mathrm{domain}}\big(\mathbf{Z}\_{\mathcal{S}}, \mathbf{Z}\_{\mathcal{T}}\big) = \sum\_{l=l\_1}^{l\_2}\Big[ \mu\,\mathrm{MMD}^{2}\big(\mathbf{Z}\_{\mathcal{S}}^{l}, \mathbf{Z}\_{\mathcal{T}}^{l}\big) + (1-\mu)\,\mathcal{L}\_{\mathrm{CORAL}}\big(\mathbf{Z}\_{\mathcal{S}}^{l}, \mathbf{Z}\_{\mathcal{T}}^{l}\big) \Big].
$$

| Setting | Paper value | Meaning |
|---------|-------------|---------|
| $l\_1$, $l\_2$ | $6$, $8$ | Apply on **fc6, fc7, fc8** (sum over three layers) |
| $\mu$ | $0.4$ | Weight on MMD vs CORAL |
| Multi-layer | yes | Stronger than adapting a single FC layer |

**Pipeline for $\mathcal{L}\_{\mathrm{domain}}$ (UK-DALE-style mental model):**

```text
1. Extract features
   Source houses (e.g. H1+H5) aggregate  →  network  →  Z_S at fc6, fc7, fc8
   Target house  (e.g. H2)     aggregate  →  same Θ   →  Z_T at fc6, fc7, fc8

2. For each layer l = 6,7,8:
      compute MMD^2(Z_S^l, Z_T^l)      # how far apart are the centers?
      compute L_CORAL(Z_S^l, Z_T^l)    # how different are the covariance shapes?

3. Mix with μ:
      layer_loss = μ * MMD^2 + (1-μ) * L_CORAL

4. Sum over layers 6..8  →  L_domain

5. Backprop L_domain (together with L_R)
   → push TCN + FC parameters so source and target features look alike
```

**Intuition for your cross-house setup:**  
House 1/5 features and House 2 features are forced toward the same “mean + shape” in FC space, without needing House 2 appliance labels.

---

#### Loss C — total objective (what is minimized)

**Simple idea:** mix “predict correctly” and “align domains”.

$$
\min\_{\Theta}\;\mathcal{L} = (1-\lambda)\,\mathcal{L}\_{R} + \lambda\,\mathcal{L}\_{\mathrm{domain}}.
$$

| $\lambda$ | What happens |
|-----------|----------------|
| $\lambda \to 0$ | Only MSE → overfit source, bad transfer |
| $\lambda \to 1$ | Only alignment → features similar but may not disaggregate well |
| $\lambda \approx 0.6$ | Paper’s balance (best in their REDD tuning) |

After training, at test time you only run $f\_\Theta(\mathbf{X}\_T)$ — no domain loss, no source batch.

---

#### Picture of the two losses

```text
                    same network f_Θ
         ┌──────────────────────────────────┐
X_S ───► │ TCN + FC                         │───► Yhat_S ──► L_R = MSE(Yhat_S, Y_S)
         │                                  │───► Z_S ──┐
X_T ───► │ (same weights)                   │───► Z_T ──┴─► L_domain = MMD+CORAL(Z_S,Z_T)
         └──────────────────────────────────┘
                              │
                              ▼
              L = (1-λ) L_R + λ L_domain  →  update Θ
```

---

#### Copy-paste LaTeX for Overleaf / thesis (normal `_`)

```latex
% --- MMD (Eq. 6) ---
\mathrm{MMD}(X_{\mathcal{S}}, X_{\mathcal{T}})
= \left\|
\frac{1}{n_s}\sum_{i=1}^{n_s}\phi(x_{\mathcal{S}}^{i})
-
\frac{1}{n_t}\sum_{j=1}^{n_t}\phi(x_{\mathcal{T}}^{j})
\right\|_{\mathcal{H}}.

% --- CORAL (Eq. 7) ---
\mathcal{L}_{\mathrm{CORAL}}
= \frac{1}{4L^{2}}\|C_{\mathcal{S}} - C_{\mathcal{T}}\|_{F}^{2}.

% --- Covariances (Eqs. 8--9) ---
C_{\mathcal{S}}
= \frac{1}{n_s-1}\left(
X_{\mathcal{S}}^{\top}X_{\mathcal{S}}
- \frac{1}{n_s}(\mathbf{1}^{\top}X_{\mathcal{S}})^{\top}(\mathbf{1}^{\top}X_{\mathcal{S}})
\right).

C_{\mathcal{T}}
= \frac{1}{n_t-1}\left(
X_{\mathcal{T}}^{\top}X_{\mathcal{T}}
- \frac{1}{n_t}(\mathbf{1}^{\top}X_{\mathcal{T}})^{\top}(\mathbf{1}^{\top}X_{\mathcal{T}})
\right).

% --- Domain loss (Eq. 12), layers fc6--fc8 ---
\mathcal{L}_{\mathrm{domain}}(Z_{\mathcal{S}}, Z_{\mathcal{T}})
= \sum_{l=l_1}^{l_2}
\Bigl[
\mu\,\mathrm{MMD}^{2}(Z_{\mathcal{S}}^{l}, Z_{\mathcal{T}}^{l})
+ (1-\mu)\,
\mathcal{L}_{\mathrm{CORAL}}(Z_{\mathcal{S}}^{l}, Z_{\mathcal{T}}^{l})
\Bigr],
\quad l_1=6,\; l_2=8,\; \mu=0.4.

% --- Joint objective ---
\mathcal{L}
= (1-\lambda)\,\mathcal{L}_{R}
+ \lambda\,\mathcal{L}_{\mathrm{domain}},
\quad \lambda=0.6.
```

---

## 5. Experimental setup

### Datasets (all resampled to **6 s**)

| Dataset | Houses | Role |
|---------|--------|------|
| REDD | 6 (USA) | Within + cross dataset |
| UK-DALE | 5 (UK) | Within + cross dataset |
| REFIT | 20 (UK) | Within + cross dataset |

### Appliances

Washing machine, dishwasher, microwave, fridge, kettle (kettle absent on REDD).

### Transfer protocol

Notation: `RA → R3` means train on all REDD houses except 3 (labeled source), adapt using unlabeled aggregate of house 3 (target), test on house 3.

Two regimes:

1. **Within-dataset:** other houses in same dataset → leave-one-house  
2. **Across-dataset:** e.g. all REFIT houses → UK-DALE house 2  

**Critical:** target appliance power is **never** used in training (only for evaluation).

### Metrics

- **MAE** — pointwise absolute error  
- **SAE** — aggregate energy error over 6-hour blocks (`T_L = 3600`)

### Baselines

| Method | Role |
|--------|------|
| AlexNet-style (no DA) | Ablation: remove domain loss |
| T-CNN, T-GRU ([Murray et al.]) | Transferable NN baselines |
| TS2P (D'Incecco et al.) | Cross-domain seq2point (often needs target fine-tune) |

---

## 6. Main results (what they claim)

### Within same dataset (Table II)

- TLN beats baselines on most appliances (MAE / SAE).  
- Unlike TS2P, **no target labels / fine-tuning** needed.  
- Example waveforms (`UA → U2`): better WM / DW / fridge / kettle tracking; less OFF over-disaggregation.

### Across datasets (Table IV)

Harder shift (country / metering). Still best among five methods; reported relative MAE gains vs best baseline up to ~24% (WM), ~22% (DW).

### Ablations / sensitivity

| Finding | Takeaway |
|---------|----------|
| `μ = 0.4` best; edges `μ=0` or `1` worse | Combine MMD + CORAL |
| `λ = 0.6` best | Joint optimization needed |
| More source houses help, but not always monotonically | Regression vs domain loss trade-off |
| Noise (SNR 10–25 dB) | TLN more robust than baselines |
| Appliance OFF in target (synthetic zero) | Low false ON / over-disaggregation |

### Compute

Training is offline and heavier than plain CNN; inference is seconds. T-GRU is smaller but slower to train.

---

## 7. Assumptions and limitations (authors admit)

Effective transfer assumes:

1. Source and target tasks are the **same** (disaggregate appliance power from mains).  
2. Distributions are **not too different** (otherwise **negative transfer**).  
3. One architecture fits both domains.

They note there is **no clear threshold** for “too different,” and flag negative transfer as future work. Also: this is **single-appliance** (one model per appliance), not multi-appliance joint heads.

---

## 8. How this relates to your MultiNILM / UK-DALE work

| Aspect | Lin et al. (this paper) | Your current pipeline |
|--------|-------------------------|------------------------|
| Output | **One appliance** at a time | **5 appliances** jointly (power + state) |
| Target labels | **None** (unsupervised DA) | Not used in Optuna; test = H2 |
| Domain alignment | Explicit MMD + CORAL on FC features | **None** — pure supervised on H1+H5 |
| Backbone | Dilated causal TCN | Staged CNN + residual temporal blocks |
| Window | 600 @ 6 s | Tunable (e.g. 128–864) |
| Metrics | MAE, SAE | `val_mae_minus_f1`, MAE, F1, etc. |
| Cross-house | Leave-one-house + cross-dataset | Train H1+H5, test H2 |

### What you can borrow for cross-house MultiNILM

1. **Unlabeled H2 aggregate during training**  
   During MultiNILM training on H1+H5, also feed H2 mains windows into a shared encoder and add `L_domain` on mid/late features. No H2 appliance labels required.

2. **MMD / CORAL (or both) on appliance-shared features**  
   Align features before the multi-appliance heads so the shared encoder is house-invariant; keep per-appliance heads discriminative.

3. **λ schedule**  
   Start with small λ (learn to disaggregate), increase toward 0.5–0.6 so domain alignment does not destroy early learning.

4. **Eval protocol alignment**  
   Report leave-one-house (and optionally REFIT→UK-DALE) the way this paper does — reviewers expect it for “generalization” claims.

5. **What not to copy blindly**  
   - Causal-only TCN may hurt offline batch accuracy vs bidirectional temporal blocks.  
   - Single-appliance setup understates multi-appliance interference (your harder setting).  
   - Seq2seq causal windows of 600 may not match your `full_input` / stride design.

### Suggested thesis contribution gap

Lin et al. = **single-appliance + unsupervised domain adaptation**.  
Your stack = **multi-appliance multi-task**.  

A strong next step: **multi-appliance MultiNILM + domain adaptation on unlabeled target house** (combine MATNilm-style multi-head with Lin-style MMD/CORAL). Few journal papers do both rigorously on UK-DALE leave-one-house.

---

## 9. Key equations (cheat sheet)

Aggregate model:

$$
x_t = \sum_{i=1}^{N} y_t^{i} + \varepsilon_t.
$$

| Symbol | Meaning |
|--------|---------|
| $\mathcal{L}_{R}$ | Source MSE regression loss |
| $\mathcal{L}_{\mathrm{domain}}$ | $\mu\cdot\mathrm{MMD}^{2}+(1-\mu)\cdot\mathcal{L}_{\mathrm{CORAL}}$ on FC layers |
| $\mathcal{L}$ | $(1-\lambda)\mathcal{L}_{R}+\lambda\mathcal{L}_{\mathrm{domain}}$ |
| $\mu$ | Balance MMD vs CORAL (0.4) |
| $\lambda$ | Balance regression vs domain (0.6) |

---

## 10. Citation

```bibtex
@article{lin2022deep,
  title   = {Deep Domain Adaptation for Non-Intrusive Load Monitoring
             Based on a Knowledge Transfer Learning Network},
  author  = {Lin, Jun and Ma, Jin and Zhu, Jianguo and Liang, Huishi},
  journal = {IEEE Transactions on Smart Grid},
  volume  = {13},
  number  = {1},
  pages   = {280--292},
  year    = {2022},
  doi     = {10.1109/TSG.2021.3115910}
}
```

---

## 11. Bottom line

This paper is a **top-tier (IEEE TSG) blueprint for unsupervised cross-house NILM**: shared TCN + joint MSE and MMD/CORAL. It shows you can transfer to an unseen house using **only target mains**. For your PhD, treat it as the reference for **domain adaptation**; extend it from single-appliance to your **multi-appliance MultiNILM** setting if you want a clear journal-level contribution.

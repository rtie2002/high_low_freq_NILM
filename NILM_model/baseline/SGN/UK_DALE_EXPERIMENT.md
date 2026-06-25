# UK-DALE SGN Experiment — Formulation Notes

Training objective, regularization, checkpoint selection, and data split for the UK-DALE SGN baseline in this repo.

**Reference:** Shin et al., *Subtask Gated Networks for Non-Intrusive Load Monitoring*, AAAI 2019.

**Math in this file:** use `$...$` (inline) and `$$...$$` (display). Preview in VS Code/Cursor with a Markdown math extension, or on GitHub.

---

## 1. Model output (SGN gating)

Input window $x$: aggregate power, length $L_{\mathrm{in}} = 432$ samples at 6 s sampling (~43 min).

**Regression branch** (normalized appliance power):

$$
\hat{p} = f_{\mathrm{reg}}(x) \in \mathbb{R}^{L_{\mathrm{out}}}, \qquad L_{\mathrm{out}} = 32
$$

**Classification branch** (ON probability):

$$
\hat{z} = \sigma\!\left(f_{\mathrm{cls}}(x)\right) \in (0,1)^{L_{\mathrm{out}}}
$$

**Soft gate** (default, `gate_mode: soft`):

$$
\hat{y}_{\mathrm{gated}} = \hat{p} \odot \hat{z}
$$

**Hard gate** (`gate_mode: hard`):

$$
\hat{y}_{\mathrm{gated}} = \hat{p} \odot \mathbf{1}[\hat{z} \ge 0.5]
$$

**Normalization** (scale $s$ = std of aggregate power on the training split):

$$
x' = \frac{x}{s}, \qquad p' = \frac{p}{s}
$$

The network input is $x'$. Targets are $p'$. The model output $\hat{p}$ is already in normalized units, so gated prediction is $\hat{y}_{\mathrm{gated}} = \hat{p} \odot \hat{z}$.

---

## 2. Training loss (SGN paper form)

Per window: true normalized power $p'$, true ON/OFF labels $o_t \in \{0,1\}$.

### 2.1 Regression — MSE on gated output

$$
\mathcal{L}_{\mathrm{out}} = \frac{1}{L_{\mathrm{out}}} \sum_{t=1}^{L_{\mathrm{out}}} \left( \hat{y}_{\mathrm{gated},t} - p'_t \right)^2
$$

### 2.2 Classification — BCE on ON probability

**Paper:** hard labels $o_t \in \{0,1\}$.

**This repo (optional):** label smoothing with $\varepsilon = 0.05$:

$$
\tilde{o}_t = o_t (1 - \varepsilon) + 0.5\,\varepsilon
$$

So OFF $\to 0.025$, ON $\to 0.975$ when $\varepsilon = 0.05$.

$$
\mathcal{L}_{\mathrm{on}} = -\frac{1}{L_{\mathrm{out}}} \sum_{t=1}^{L_{\mathrm{out}}} \left[ \tilde{o}_t \log \hat{z}_t + (1-\tilde{o}_t) \log (1-\hat{z}_t) \right]
$$

(Set $\varepsilon = 0$ to match the paper exactly.)

### 2.3 Total loss (optimized every training step)

$$
\mathcal{L} = \mathcal{L}_{\mathrm{out}} + \mathcal{L}_{\mathrm{on}}
$$

Paper form:

$$
\mathcal{L}_{\mathrm{SGN}} = \mathrm{MSE}(\hat{y}_{\mathrm{gated}}, p') + \mathrm{BCE}(\hat{z}, o)
$$

---

## 3. Weight decay (L2 regularization)

**Paper:** Adam, learning rate $\eta = 10^{-4}$, no weight decay.

**This repo:** $\lambda = 10^{-4}$:

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L} - \eta \lambda \theta
$$

Same idea as adding $\lambda \|\theta\|_2^2$ to the loss (L2 penalty on weights $\theta$).

---

## 4. Best checkpoint vs training loss

Training always minimizes $\mathcal{L} = \mathcal{L}_{\mathrm{out}} + \mathcal{L}_{\mathrm{on}}$.

**Checkpoint selection** uses a separate validation score (early stopping).

### Original (combined validation loss)

$$
\mathcal{L}_{\mathrm{total}}^{\mathrm{val}} = \mathcal{L}_{\mathrm{out}}^{\mathrm{val}} + \mathcal{L}_{\mathrm{on}}^{\mathrm{val}}
$$

$$
e^* = \arg\min_e \; \mathcal{L}_{\mathrm{total}}^{\mathrm{val}}(e)
$$

Config: `"early_stop_metric": "f1"` in `sgn_ukdale_cross_house.json` (cross-house val on house~2).

Paper-faithful alternative: `"early_stop_metric": "total_loss"` in `sgn_paper.json`.

**Early stopping:** no improvement for `patience` epochs $\Rightarrow$ stop.

Other metrics: `mae`, `output_loss`, `total_loss` (code minimizes loss; maximizes F1).

---

## 5. UK-DALE data split

Built with `dataset_preprocess/build_sgn_ukdale_splits.py` (default `--val_source test_house`):

| Split | File | Content |
|-------|------|---------|
| Train | `multi_appliance_training_cross_house.csv` | Houses 1 \& 5, last 28 days each |
| Val | `multi_appliance_validating_cross_house.csv` | House 2, last 4 days |
| Test | `multi_appliance_testing_cross_house.csv` | House 2, first 24 days (disjoint from val) |

Config: `configs/training_data_ukdale_cross_house.json`

---

## 6. Sliding windows

| Symbol | Value | Real time @ 6 s |
|--------|-------|-----------------|
| $L_{\mathrm{in}}$ | 432 | ~43 min input |
| $L_{\mathrm{out}}$ | 32 | ~3 min output |
| $S_{\mathrm{train}}$ | 32 (UK-DALE) | stride between windows |
| $S_{\mathrm{eval}}$ | 32 | stride at val/test |

Window starts: $0,\; S,\; 2S,\; \ldots$

Paper REDD often uses $S_{\mathrm{train}} = 1$. UK-DALE uses $S_{\mathrm{train}} = 32$ to reduce overlap memorization.

---

## 7. ON/OFF labels (preprocessing)

From CSV columns (`fridge_on`, etc.), built offline in `ukdale_processing_multi_appliance.py`:

$$
o_t = \begin{cases} 1 & \text{if } P_t \ge \tau_{\mathrm{house}} \\ 0 & \text{otherwise} \end{cases}
$$

Thresholds in `config/preprocess/ukdale.yaml`:

| Appliance | Default $\tau$ | House override |
|-----------|----------------|----------------|
| fridge | 50 W | — |
| washingmachine | 20 W | house 5: 25 W |
| kettle | 200 W | — |
| microwave | 200 W | — |
| dishwasher | 50 W | — |

CSV training uses these precomputed labels. REDD pickle path in this codebase can use a 15 W rule inside the loader instead.

---

## 8. Paper vs this UK-DALE run

| Component | SGN paper | `sgn_paper.json` | `sgn_ukdale_cross_house.json` |
|-----------|-----------|------------------|-------------------------------|
| $\mathcal{L}_{\mathrm{out}}$ | MSE on gated power | Same | Same |
| $\mathcal{L}_{\mathrm{on}}$ | BCE, hard labels | Same | Same |
| Optimizer | Adam, $\eta=10^{-4}$ | Same | Adam, $\eta=2\times10^{-4}$ |
| Batch size | 16 | 16 | 256 |
| $L_{\mathrm{in}} / L_{\mathrm{out}}$ | 432 / 32 @ 6 s | Same | Same |
| $S_{\mathrm{train}}$ | 1 | 1 | 32 |
| Best checkpoint | total val loss | total val loss | val F1 |
| Val split | — | — | house 2 (cross-house) |
| Train houses | 1, 3, 4, 5 (1 week) | 1, 5 (28 d) | 1, 5 (28 d) |
| Test house | 2 | 2 | 2 |

**Use `sgn_paper.json`** for paper-faithful reproduction.  
**Use `sgn_ukdale_cross_house.json`** for cross-house transfer experiments (recommended).

---

## 9. Config files and command

| File | Role |
|------|------|
| `configs/training_data_ukdale_cross_house.json` | Train/val/test CSV paths (cross-house split) |
| `configs/sgn_ukdale_cross_house.json` | Fast UK-DALE training (stride 32, F1 early stop) |
| `configs/sgn_paper.json` | Paper-faithful hyperparameters |
| `sgn/losses.py` | $\mathcal{L}_{\mathrm{out}} + \mathcal{L}_{\mathrm{on}}$ |
| `model_evaluation/runner.py` | Early stopping, waveform saves |

**Train (UK-DALE):**

```powershell
cd NILM_model

python main.py --model sgn --mode train_inference --data_source csv `
  --csv_config baseline/SGN/configs/training_data_ukdale_cross_house.json `
  --model_config baseline/SGN/configs/sgn_ukdale_cross_house.json `
  --run_dir runs/sgn_ukdale_cross_house
```

**Outputs per appliance** (under `run_dir`):

- `best_{appliance}.pt` — best checkpoint (by `early_stop_metric`, after `min_epochs`)
- `live_waveform_{appliance}.png` — latest epoch, **validation** (house 5 last week)
- `live_waveform_{appliance}_test.png` — latest epoch, **test** (house 2)
- `best_waveform_{appliance}.png` — best checkpoint on validation
- `best_waveform_{appliance}_test.png` — best checkpoint on test

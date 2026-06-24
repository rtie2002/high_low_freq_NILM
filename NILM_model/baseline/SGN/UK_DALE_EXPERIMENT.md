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

Config: `"early_stop_metric": "total_loss"` (default in `sgn_paper.json`).

### Optional regularized setting (power only)

Used only in `sgn_ukdale_reg.json`:

$$
e^* = \arg\min_e \; \mathcal{L}_{\mathrm{out}}^{\mathrm{val}}(e)
$$

Config: `"early_stop_metric": "output_loss"`.

Used because validation ON/OFF loss was overfitting while validation power loss stayed stable.

**Early stopping:** no improvement for `patience = 30` epochs $\Rightarrow$ stop.

Other metrics: `mae`, `f1` (code minimizes $-\mathrm{F1}$).

---

## 5. UK-DALE data split

| Split | File | Content |
|-------|------|---------|
| Train + val source | `NILM_model/data/multi_appliance_house1_5_lf_2weeks.csv` | Houses 1 & 5, 14 days each |
| Test | `NILM_model/data/multi_appliance_house2_lf.csv` | House 2, 7 days |

Validation split (`val_mode: by_house_tail`, `val_last_days: 7`):

$$
\mathcal{D}_{\mathrm{train}} = \mathcal{D}_{\mathrm{house\,1}} \cup \left\{ x \in \mathcal{D}_{\mathrm{house\,5}} : t(x) < t_{\max}^{(5)} - 7\,\mathrm{days} \right\}
$$

$$
\mathcal{D}_{\mathrm{val}} = \left\{ x \in \mathcal{D}_{\mathrm{house\,5}} : t(x) \ge t_{\max}^{(5)} - 7\,\mathrm{days} \right\}
$$

$$
\mathcal{D}_{\mathrm{test}} = \mathcal{D}_{\mathrm{house\,2}} \quad \text{(entire test CSV)}
$$

**Paper target:** train houses $\{1,3,4,5\}$, test house $2$, last week only. Here we use houses $\{1,5\}$ for 2 weeks (houses 3 & 4 not usable for all five appliances).

Approx. row counts @ 6 s: train ~302k, val ~101k, test ~101k.

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

| Component | SGN paper | `sgn_paper.json` (default) | `sgn_ukdale_reg.json` (optional) |
|-----------|-----------|----------------------------|----------------------------------|
| $\mathcal{L}_{\mathrm{out}}$ | MSE on gated power | Same | Same |
| $\mathcal{L}_{\mathrm{on}}$ | BCE, hard labels | Same | BCE + $\varepsilon=0.05$ smoothing |
| Optimizer | Adam, $\eta=10^{-4}$ | Same | Adam + $\lambda=10^{-4}$ weight decay |
| Batch size | 16 | 16 | 16 |
| $L_{\mathrm{in}} / L_{\mathrm{out}}$ | 432 / 32 @ 6 s | Same | Same |
| $S_{\mathrm{train}}$ | 1 | 1 | 32 |
| Best checkpoint | total val loss | total val loss | $\mathcal{L}_{\mathrm{out}}^{\mathrm{val}}$ only |
| Train houses | 1, 3, 4, 5 (1 week) | 1, 5 (2 weeks) | 1, 5 (2 weeks) |
| Test house | 2 | 2 | 2 |

**Use `sgn_paper.json` for the first UK-DALE run** (closest to the paper).  
**Use `sgn_ukdale_reg.json` only if** train ON/OFF loss collapses to ~0 while val ON/OFF explodes.

### Why the regularized run can look worse

The experimental config (`sgn_ukdale_reg.json`) was added to fight classification overfitting, but it can hurt overall results:

1. **`train_stride: 32`** — ~32× fewer training windows; rare ON events (microwave, kettle) are seen less often.
2. **`early_stop_metric: output_loss`** — can pick a very early checkpoint (e.g. epoch 14) when power loss plateaus, even though F1 keeps improving later (epoch 40+).
3. **Label smoothing + weight decay** — softer ON/OFF targets and smaller weights can reduce confident ON predictions → lower F1 on sparse appliances.

Regression is not always broken: e.g. washingmachine test MAE ~11 W can be reasonable while F1 stays low because the ON/OFF branch under-trained or never fires on house 2.

---

## 9. Config files and command

| File | Role |
|------|------|
| `configs/sgn_paper.json` | Paper-faithful hyperparameters (default for UK-DALE) |
| `configs/sgn_ukdale_reg.json` | Optional regularized variant (stride 32, WD, label smoothing) |
| `configs/training_data_ukdale_paper.json` | Train/test CSV paths, val split |
| `sgn/losses.py` | $\mathcal{L}_{\mathrm{out}} + \mathcal{L}_{\mathrm{on}}$ |
| `model_evaluation/runner.py` | Early stopping, waveform saves |

**Train (UK-DALE):**

```powershell
cd NILM_model

python main.py --model sgn --mode train --data_source csv `
  --csv_config baseline/SGN/configs/training_data_ukdale_paper.json `
  --model_config baseline/SGN/configs/sgn_paper.json `
  --run_dir runs/sgn_ukdale
```

**Outputs per appliance** (under `run_dir`):

- `best_{appliance}.pt` — best checkpoint
- `live_waveform_{appliance}.png` — latest epoch (validation)
- `best_waveform_{appliance}.png` — best val epoch
- `best_waveform_{appliance}_test.png` — best model on house 2 test

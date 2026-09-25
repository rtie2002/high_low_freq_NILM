# Concerns with the MultiNILM training loss

The implementation in `model/MultiNILM_loss.py` matches the formulas in
`docs/multinilm_loss_equations.md`. The concern is the design: eight weighted
terms, tied together by one batch-wise scale, are more than this experiment can
attribute to a result. This note states the three concrete problems, derives
them from the loss, and records the simpler objective to try next.

The numbers below come from two finished runs, both seed 2026, 200 epochs,
`task_balance: equal`, `lambda_state: 0.8`, and smoothing off:

- `new one` — no random mix
- `Random Mix 0.5 (8w)` — random mix with probability 0.5

ON rates used in Section 3 are the ones already tabulated in
`multinilm_loss_equations.md` Section 5.1, measured on the 3-week training CSV.
The 8-week CSV has the same houses and a longer window, so the rates are the
right order; they are not remeasured here.

---

## 0. What is optimized now

For appliance \(i\), the head outputs a raw power \(R\) and a state logit \(s\).
With `gate_mode: soft`,

\[
p=\sigma(s),\qquad
\hat y=p\,R+(1-p)\,y_{\mathrm{off}},
\]

where \(y_{\mathrm{off}}\) is 0 W after z-scoring. Every power term therefore
also trains \(p\).

Per appliance, before the task balance,

\[
\begin{aligned}
L_{P,i}
&=L_{\mathrm{MSE},i}
+1.0\,L_{\mathrm{on},i}
+0.5\,L_{\mathrm{off},i}
+0.15\,L_{\Delta,i}
+0.25\,L_{\mathrm{relE},i},\\[4pt]
L_{S,i}
&=L_{\mathrm{BCE},i}
+1.0\,L_{\mathrm{FP},i}
+w_{\mathrm{smooth}}\,L_{\mathrm{smooth},i}.
\end{aligned}
\]

The finished runs had \(w_{\mathrm{smooth}}=0\). The yaml now sets it to 0.15.
Sum over the five appliances, \(L_P=\sum_i L_{P,i}\) and \(L_S=\sum_i L_{S,i}\).
With `task_balance: equal` the scalar that `backward()` sees is

\[
\boxed{
L
=L_P
+\lambda\,L_S\,
\operatorname{stopgrad}\!\left(\frac{L_P}{L_S}\right),
\qquad \lambda=0.8.
}
\]

In the forward pass the second term equals \(\lambda L_P\), so the logged
`L_NILM` is always \(1.8\,L_P\). In the backward pass the ratio is a constant,
and the gradient is

\[
\nabla L
=\nabla L_P
+\underbrace{\lambda\,\frac{L_P}{L_S}}_{\text{effective state weight}}
\nabla L_S.
\]

`lambda_state: 0.8` is the forward-value preference. The number that multiplies
the state gradient is \(\lambda L_P/L_S\), and that number moves every batch.

---

## 1. The state gradient weight moves during training

Read \(\lambda L_P/L_S\) from `loss_detail.csv`
(`0.8 * train_loss_power / train_loss_state`):

| Epoch | No mix | Random mix |
|---:|---:|---:|
| 1 | 26.0 | 24.4 |
| 10 | 27.2 | 25.3 |
| 50 | 17.9 | 19.9 |
| 100 | 13.3 | 17.4 |
| 200 | 9.7 | 14.4 |

Within one run the multiplier falls by a factor of about 2.8 (no mix) or 1.8
(random mix). Three consequences follow.

**A configured \(\lambda\) is a moving target.** Setting `lambda_state`
from 0.8 to 1.6 would double the multiplier, but the training itself already
changes the multiplier by more than that. A sweep of \(\lambda\) under
`task_balance: equal` does not compare fixed preferences.

**Random mix is mixed with a weight change.** At epoch 200 the no-mix run
trains the state head with multiplier 9.7 and the random-mix run with 14.4.
Part of the test-set gain can be "the state head was trained harder late in
training", which is a property of the balance rule under a harder power loss.

**Adding smoothing is mixed with the same change.** Smoothing enters \(L_S\):

\[
L_S' = L_S + w_{\mathrm{smooth}}\sum_i L_{\mathrm{smooth},i},
\qquad
\text{new multiplier}
=\lambda\,\frac{L_P}{L_S'}
<\lambda\,\frac{L_P}{L_S}.
\]

BCE and the false-ON term then receive a smaller gradient, even though their
yaml weights did not change. A smoothing run is therefore two changes: the new
term, and a quieter BCE. `train_loss_state_smooth` in `loss_detail.csv` is
enough to compute how large the second change was.

---

## 2. One global ratio ties unrelated appliances together

The ratio uses the sums, so one appliance's power error rescales every
appliance's state gradient:

\[
\text{multiplier}
=\lambda\,
\frac{\sum_{j=1}^{5} L_{P,j}}{\sum_{j=1}^{5} L_{S,j}}.
\]

Shares at epoch 200:

| | Kettle | Fridge | Dishwasher | Washing machine | Microwave |
|---|---:|---:|---:|---:|---:|
| Share of \(L_P\), no mix | 0.26 | 0.18 | 0.07 | 0.12 | **0.37** |
| Share of \(L_S\), no mix | 0.05 | **0.75** | 0.13 | 0.05 | 0.02 |
| Share of \(L_P\), random mix | 0.21 | 0.10 | 0.06 | 0.13 | **0.49** |
| Share of \(L_S\), random mix | 0.04 | **0.54** | 0.22 | 0.16 | 0.04 |

Microwave dominates \(L_P\) because its normalization standard deviation is
111 W: a 1 kW error is about \((1000/111)^2 \approx 81\) in z-score MSE, while
a 100 W fridge error (std 60 W) is about 3. Fridge dominates \(L_S\) because it
is on about 40% of the time, so its BCE stays large after the other appliances'
ON/OFF predictions have sharpened.

The multiplier is therefore close to "microwave power error, divided by fridge
classification loss". When the microwave power error grows, the kettle,
dishwasher, washing machine, and fridge state heads are all trained harder.
Nothing in the measurement says those four decisions should track the microwave
regression.

---

## 3. Several terms push the same error, so a yaml weight is a ratio

### 3.1 One ON sample versus one OFF sample in the power loss

Write \(N=BT\) for the number of samples in the batch, \(r_i\) for the ON rate
of appliance \(i\), \(N_{\mathrm{on},i}=r_i N\), and
\(N_{\mathrm{off},i}=(1-r_i)N\). For a single sample whose squared z-score
error is \(e^2\):

\[
\begin{aligned}
\text{cost if the sample is ON}
&= \frac{e^2}{N} + 1.0\,\frac{e^2}{N_{\mathrm{on},i}},\\[4pt]
\text{cost if the sample is OFF}
&= \frac{e^2}{N} + 0.5\,\frac{e^2}{N_{\mathrm{off},i}}.
\end{aligned}
\]

The first piece is the all-sample MSE. The second piece is the conditional
ON-MSE or OFF-MSE, which divides by the count of that class, so a rare class
is up-weighted. Cancel \(e^2/N\):

\[
\frac{\text{cost of one ON sample}}{\text{cost of one OFF sample}}
=
\frac{1+1/r_i}{1+0.5/(1-r_i)}
=
\frac{(1+r_i)/r_i}{(1.5-r_i)/(1-r_i)}.
\]

With the tabulated ON rates:

| Appliance | ON rate \(r_i\) | ON-sample cost / OFF-sample cost |
|---|---:|---:|
| Microwave | 0.509% | 131 |
| Kettle | 1.110% | 60 |
| Washing machine | 4.447% | 15 |
| Dishwasher | 5.799% | 12 |
| Fridge | 40.34% | 1.9 |

`power_on_weight: 1.0` and `power_off_weight: 0.5` therefore mean, for the
microwave, that missing an ON sample of a given size costs about 131 times a
false ON of the same size. Predicting ON on a doubtful pulse is cheap. That
matches the REFIT House 20 microwave result under random mix: recall 0.67,
precision 0.40, and 0.89 of the true microwave energy coming from false-ON
samples (see the prediction-file diagnosis).

The weights would mean "ON twice as important as OFF" only if both terms were
means over the same set of samples. They are means over different sets.

### 3.2 `pos_weight` mostly moves the probability scale

For BCE with ON weight \(w\) and OFF weight 1, the loss at one sample with
true ON probability \(q\) is

\[
\ell(p)=-\,w\,q\log p-(1-q)\log(1-p).
\]

Set the derivative to zero:

\[
\frac{\partial \ell}{\partial p}
=-\frac{wq}{p}+\frac{1-q}{1-p}=0
\quad\Rightarrow\quad
p^{\star}=\frac{wq}{wq+(1-q)}.
\]

The network is not asked to output the true probability \(q\). It is asked to
output \(p^{\star}\). With the cap \(w=12\) and \(q=0.4\),

\[
p^{\star}=\frac{12\times 0.4}{12\times 0.4+0.6}=\frac{4.8}{5.4}=0.89.
\]

With the fridge weight \(w=1.48\) and the same \(q=0.4\),

\[
p^{\star}=\frac{1.48\times 0.4}{1.48\times 0.4+0.6}=0.50.
\]

A larger `pos_weight` lifts the whole probability curve. Validation calibration
then picks a new threshold to maximize F1, which moves up with the curve.
The calibrated thresholds of the random-mix run sit where this predicts:
kettle 0.89, microwave 0.90, fridge 0.38. Ranking metrics such as average
precision are unchanged by a monotone remapping, so `pos_weight_cap` is a weak
lever for AP and a strong lever for the raw probability values.

Three terms then pull \(p\) in overlapping directions on true-OFF samples:
BCE with \(w=12\) upward, \(L_{\mathrm{FP}}=p^2\) downward, and OFF-MSE
downward through the gate \(\partial \hat y/\partial s=(R-y_{\mathrm{off}})p(1-p)\).
The threshold sweep discards the resulting operating point and chooses another
one. The yaml numbers do not describe the decision the metrics use.

### 3.3 Delta, relative energy, and smoothing have no separate evidence yet

\(L_{\Delta}\) penalizes a wrong step in z-scored power, and only on pairs
where at least one sample is truly ON. \(L_{\mathrm{relE}}\) penalizes a wrong
window total in watts. \(L_{\mathrm{smooth}}\) penalizes a change in
\(\log p\) between adjacent samples. No run has removed one of these while
keeping the rest fixed, so their contribution to the test numbers is unknown.

\(L_{\Delta}\) rewards a sharp power edge. \(L_{\mathrm{smooth}}\) penalizes a
sharp probability edge until the log-probability jump exceeds \(\tau=4\).
At a true switch both gradients are active and point at different shapes of
the same edge.

---

## 4. What the source papers optimize

### 4.1 Subtask Gated Networks

Shin, Joo, and Moon, "Subtask Gated Networks for Non-Intrusive Load
Monitoring", AAAI 2019 ([ar5iv](https://ar5iv.labs.arxiv.org/html/1811.06692)).
The gated output and its two losses are

\[
\hat y_t=\hat p_t\,\hat o_t,
\]

\[
\mathcal{L}_{\mathrm{output}}
=\frac{1}{T}\sum_t (y_t-\hat p_t\hat o_t)^2,
\qquad
\mathcal{L}_{\mathrm{on}}
=-\sum_t \big(o_t\log\hat o_t+(1-o_t)\log(1-\hat o_t)\big),
\]

\[
\mathcal{L}=\mathcal{L}_{\mathrm{output}}+\mathcal{L}_{\mathrm{on}}.
\]

Their reported 15–30% error reduction over the REDD and UK-DALE baselines was
obtained without tuning the network, the hyperparameters, or the weight
between the two terms (Section 4 of that paper). They also report that
training on \(\mathcal{L}_{\mathrm{output}}\) alone is worse, so the plain BCE
term earns its place.

Two scale differences matter if we copy the formula. Their BCE is a **sum**
over timesteps; ours is a **mean** over the batch and time, so `lambda_state: 1`
is the same expression and a different numeric balance. They train one
appliance per network; we sum five appliances first.

### 4.2 MS-TCN smoothing

Abu Farha and Gall, "MS-TCN: Multi-Stage Temporal Convolutional Network for
Action Segmentation", CVPR 2019
([arXiv:1903.01945](https://arxiv.org/abs/1903.01945)). Per stage,

\[
\mathcal{L}_s=\mathcal{L}_{\mathrm{cls}}+\lambda\,\mathcal{L}_{T\text{-MSE}},
\qquad \lambda=0.15,\ \tau=4.
\]

On 50Salads, \(\lambda\in\{0.05,0.15,0.25\}\) moves F1@10 only from 74.1 to
76.3 to 74.7. Raising \(\tau\) from 4 to 5 drops F1@10 from 76.3 to 66.6,
because confident true boundaries get penalized. If smoothing stays, \(\tau\)
should stay at 4 and \(\lambda\) is the knob, and it should be added only after
the state weight is a fixed number (Section 1).

---

## 5. Simpler objective to test

Match SGN's two-term form. This is a yaml change; every weight already exists
in `MultiNILM_loss.py`.

```yaml
loss:
  task_balance: none          # state_term = lambda_state * L_S, fixed
  lambda_state: 1.0
  pos_weight: auto
  pos_weight_cap: 12
  state_fp_weight: 0.0
  state_smooth_weight: 0.0
  power_on_weight: 0.0
  power_off_weight: 0.0
  power_delta_weight: 0.0
  power_energy_relative_weight: 0.0
```

The optimized scalar becomes

\[
\boxed{
L
=\sum_{i=1}^{5} L_{\mathrm{MSE},i}
+\lambda\sum_{i=1}^{5} L_{\mathrm{BCE},i},
\qquad \lambda=1.
}
\]

`L_NILM` in the log is then the real sum, and it can fall because the state
term improved. The two knobs left are `lambda_state` and `pos_weight_cap`.

Compare this run with the current full loss, both with random mix on and
smoothing off, two seeds each. Read validation average precision first: it
does not depend on the calibrated threshold. Then read calibrated F1, MAE, and
the per-house energy ratio. Keep the simple loss if it is within the
seed-to-seed gap of the full loss.

If it loses, put back one term at a time, two seeds each, in this order:
ON-MSE at a small weight such as 0.1 (this is the 131× lever in Section 3.1),
then the false-ON term, then relative energy if the energy ratio gets worse.
Keep a term only when the validation AP or the targeted test symptom moves by
more than the seed gap.

After the weights are fixed, the smoothing ablation is one change. Try
`state_smooth_weight` in \(\{0.05, 0.15\}\) and leave `state_smooth_tau: 4`.

Choose every value on the validation split. Report UK-DALE House 2 and
REFIT House 20 only after the choice is made.

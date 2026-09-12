# MultiNILM Loss: Complete Top-Down Derivation

This document derives the exact scalar loss optimized by
`config/models/multinilm_fractional_relational.yaml`. It follows the
implementation in `model/MultiNILM_loss.py` and the parameter wiring in
`adapters/multinilm.py`.

The derivation is specific to the current configuration snapshot:

```yaml
architecture:
  gate_mode: soft

loss:
  task_balance: equal
  lambda_state: 0.8
  pos_weight: auto
  pos_weight_cap: 12
  state_fp_weight: 1.0
  power_on_weight: 1.0
  power_off_weight: 0.5
  power_delta_weight: 0.15
  power_delta_on_only: true
  power_energy_weight: 0.0
  power_energy_relative_weight: 0.25
  energy_floor_watts: 10
  aggregate_consistency_weight: 1.0
  aggregate_tolerance_watts: 30
  aggregate_loss_scale_watts: 1000
  domain_method: both
  domain_mu: 0.4
  domain_mix: convex
  domain_scale: equal
  lambda_domain: 0.0

domain_adaptation:
  enabled: false
```

## 1. Complete Objective at a Glance

The general implementation supports supervised NILM and optional domain
adaptation:

$$
L =
\begin{cases}
(1-\lambda_D)L_{\mathrm{NILM}}+\lambda_D L_{D,\mathrm{term}},
& \text{DA enabled with convex mixing},\\
L_{\mathrm{NILM}}+\lambda_D L_{D,\mathrm{term}},
& \text{DA enabled with additive mixing},\\
L_{\mathrm{NILM}}, & \text{DA disabled}.
\end{cases}
$$

The current experiment has domain adaptation disabled and
$\lambda_D=0$. Therefore, the scalar passed to `backward()` is exactly

$$
\boxed{
L=L_{\mathrm{NILM}}
=L_P+L_{S,\mathrm{term}}+L_A
}
$$

where

$$
L_P=\sum_{i=1}^{A}L_{P,i},
\qquad
L_S=\sum_{i=1}^{A}L_{S,i},
$$

$$
L_{S,\mathrm{term}}
=0.8L_S\operatorname{stopgrad}
\left(\frac{L_P}{\max(L_S,10^{-8})}\right),
$$

and $L_A$ is the one-sided aggregate consistency loss. The current model has
$A=5$ appliances.

### 1.1 Single derivation chain from the total loss to all components

To read the objective from the top down, define

$$
\mathcal{P}_i
=L_{\mathrm{base},i}
+L_{\mathrm{on},i}
+0.5L_{\mathrm{off},i}
+0.15L_{\Delta,i}
+0.25L_{\mathrm{relE},i},
$$

$$
\mathcal{S}_i
=L_{\mathrm{BCE},i}
+L_{\mathrm{FP},i}.
$$

The complete active loss can then be expanded in one chain:

$$
\begin{aligned}
L
&=L_P+L_{S,\mathrm{term}}+L_A\\
&=L_P
+0.8L_S\operatorname{stopgrad}
\left(\frac{L_P}{\max(L_S,10^{-8})}\right)
+L_A\\
&=\sum_{i=1}^{5}L_{P,i}
+0.8\left(\sum_{i=1}^{5}L_{S,i}\right)
\operatorname{stopgrad}
\left(
\frac{\sum_{i=1}^{5}L_{P,i}}
{\max(\sum_{i=1}^{5}L_{S,i},10^{-8})}
\right)
+L_A\\
&=\sum_{i=1}^{5}\mathcal{P}_i
+0.8\left(\sum_{i=1}^{5}\mathcal{S}_i\right)
\operatorname{stopgrad}
\left(
\frac{\sum_{i=1}^{5}\mathcal{P}_i}
{\max(\sum_{i=1}^{5}\mathcal{S}_i,10^{-8})}
\right)
+L_A\\
&=\sum_{i=1}^{5}
\left[
L_{\mathrm{base},i}
+L_{\mathrm{on},i}
+0.5L_{\mathrm{off},i}
+0.15L_{\Delta,i}
+0.25L_{\mathrm{relE},i}
\right]\\
&\quad+0.8\sum_{i=1}^{5}
\left[
L_{\mathrm{BCE},i}
+L_{\mathrm{FP},i}
\right]
\operatorname{stopgrad}
\left(
\frac{
\sum_{i=1}^{5}
\left[
L_{\mathrm{base},i}
+L_{\mathrm{on},i}
+0.5L_{\mathrm{off},i}
+0.15L_{\Delta,i}
+0.25L_{\mathrm{relE},i}
\right]
}{
\max\left(
\sum_{i=1}^{5}
\left[
L_{\mathrm{BCE},i}
+L_{\mathrm{FP},i}
\right],
10^{-8}
\right)
}
\right)
+L_A.
\end{aligned}
$$

The terminal terms in this chain are calculated directly from samples as

$$
\begin{aligned}
L_{\mathrm{base},i}
&=\frac{1}{BT}\sum_{b,t}(\hat y_{bti}-y_{bti})^2,\\
L_{\mathrm{on},i}
&=\frac{\sum_{b,t}z_{bti}(\hat y_{bti}-y_{bti})^2}
{\max(\sum_{b,t}z_{bti},1)},\\
L_{\mathrm{off},i}
&=\frac{\sum_{b,t}(1-z_{bti})(\hat y_{bti}-y_{bti})^2}
{\max(\sum_{b,t}(1-z_{bti}),1)},\\
L_{\Delta,i}
&=\frac{\sum_{b,t\ge2}m^{\Delta}_{bti}
(\Delta\hat y_{bti}-\Delta y_{bti})^2}
{\max(\sum_{b,t\ge2}m^{\Delta}_{bti},1)},\\
L_{\mathrm{relE},i}
&=\frac{1}{B}\sum_b
\frac{|\sum_t\hat P_{bti}-\sum_tP_{bti}|}
{\sum_tP_{bti}+10T},\\
L_{\mathrm{BCE},i}
&=-\frac{1}{BT}\sum_{b,t}
\left[w_i^+z_{bti}\log p_{bti}
+(1-z_{bti})\log(1-p_{bti})\right],\\
L_{\mathrm{FP},i}
&=\frac{\sum_{b,t}(1-z_{bti})p_{bti}^2}
{\max(\sum_{b,t}(1-z_{bti}),1)},\\
L_A
&=\frac{1}{BT}\sum_{b,t}
\left[
\frac{\max(\sum_i\hat P_{bti}-X^W_{bt}-30,0)}{1000}
\right]^2.
\end{aligned}
$$

Finally, all terminal predictions and masks reduce to model outputs and labels:

In the equations below, $\sigma(\cdot)$ is the sigmoid function, whereas
$\sigma_i$ is the target-power normalization standard deviation for appliance
$i$.

$$
\begin{aligned}
p_{bti}&=\sigma(s_{bti}),\\
\hat y_{bti}&=p_{bti}R_{bti}+(1-p_{bti})y_{\mathrm{off},i},\\
\hat P_{bti}&=\max(\sigma_i\hat y_{bti}+\mu_i,0),\\
P_{bti}&=\max(\sigma_i y_{bti}+\mu_i,0),\\
m^{\Delta}_{bti}&=\max(z_{bti},z_{b,t-1,i}),\\
w_i^+&=\min\left(
\frac{1-\operatorname{clip}(r_i,10^{-4},1-10^{-4})}
{\operatorname{clip}(r_i,10^{-4},1-10^{-4})},
12
\right).
\end{aligned}
$$

This is the shortest complete path from the final scalar $L$ to the raw model
outputs $R,s$ and training labels $y,z$.

Numerically, when $L_S>10^{-8}$,

$$
L_{S,\mathrm{term}}=0.8L_P,
\qquad
L\approx1.8L_P+L_A.
$$

This numerical identity does **not** mean that the state loss disappears. The
ratio is detached, so its gradient is

$$
\boxed{
\nabla_\theta L
=\nabla_\theta L_P
+0.8\left(\frac{L_P}{L_S}\right)_{\mathrm{stopgrad}}
\nabla_\theta L_S
+\nabla_\theta L_A.
}
$$

The following sections expand every term in this expression.

## 2. Notation and Tensor Shapes

| Symbol | Meaning | Shape |
|---|---|---|
| $B$ | batch size | scalar; currently 64 |
| $T$ | output timesteps | scalar; currently 1024 |
| $A$ | appliances | scalar; currently 5 |
| $i$ | appliance index | $1,\ldots,A$ |
| $y_{bti}$ | normalized true appliance power | $(B,T,A)$ |
| $\hat y_{bti}$ | normalized gated power prediction | $(B,T,A)$ |
| $R_{bti}$ | raw normalized power-head output | $(B,T,A)$ |
| $z_{bti}$ | true binary ON/OFF state | $(B,T,A)$ |
| $s_{bti}$ | predicted state logit | $(B,T,A)$ |
| $p_{bti}=\sigma(s_{bti})$ | predicted ON probability | $(B,T,A)$ |
| $X_{bt}$ | normalized aggregate input | $(B,T)$ |
| $\mu_i,\sigma_i$ | appliance normalization statistics | one pair per appliance |

The target normalization and inverse transform are

$$
y_{bti}=\frac{P_{bti}-\mu_i}{\sigma_i},
\qquad
P_{bti}=\max(\sigma_i y_{bti}+\mu_i,0).
$$

The clamp to zero is used only by the physical-watt energy and aggregate
terms. The normalized pointwise losses still receive gradients for negative
predictions.

## 3. Soft State-Gated Power Output

Before any loss is calculated, each appliance head combines its raw power and
state outputs:

$$
p_{bti}=\sigma(s_{bti}),
$$

$$
y_{\mathrm{off},i}=\frac{0-\mu_i}{\sigma_i},
$$

$$
\boxed{
\hat y_{bti}
=p_{bti}R_{bti}+(1-p_{bti})y_{\mathrm{off},i}.
}
$$

After inverse normalization this is equivalent to a soft watt-space gate:

$$
\hat P_{bti}=p_{bti}R^{W}_{bti}+(1-p_{bti})0
=p_{bti}R^{W}_{bti}.
$$

The normalized OFF values for the current training statistics are:

| Appliance | $y_{\mathrm{off},i}$ |
|---|---:|
| kettle | -0.100285 |
| fridge | -0.713795 |
| dishwasher | -0.152018 |
| washing machine | -0.122535 |
| microwave | -0.136206 |

The gate couples the power and state tasks. Its local derivatives are

$$
\frac{\partial\hat y}{\partial R}=p,
\qquad
\frac{\partial\hat y}{\partial s}
=(R-y_{\mathrm{off}})p(1-p).
$$

Consequently, every power-side loss updates both the power head and the state
head. A low $p$ attenuates the power-head gradient, while a nonzero
$p(1-p)$ lets the power error supervise the state logit indirectly.

## 4. Per-Appliance Power Loss

Define the normalized pointwise error

$$
e_{bti}=\hat y_{bti}-y_{bti}.
$$

### 4.1 Base pointwise MSE

$$
L_{\mathrm{base},i}
=\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}e_{bti}^{2}.
$$

This term includes both ON and OFF samples in their natural frequency.

### 4.2 Conditional ON MSE

Let

$$
N_{\mathrm{on},i}=\sum_{b,t}z_{bti}.
$$

Then

$$
L_{\mathrm{on},i}
=\frac{\sum_{b,t}z_{bti}e_{bti}^{2}}
{\max(N_{\mathrm{on},i},1)}.
$$

This is a conditional mean, not a sum over positive samples. One batch with
few ON samples does not automatically receive a smaller ON loss, provided at
least one ON sample exists.

### 4.3 Conditional OFF MSE

Let

$$
N_{\mathrm{off},i}=\sum_{b,t}(1-z_{bti}).
$$

Then

$$
L_{\mathrm{off},i}
=\frac{\sum_{b,t}(1-z_{bti})e_{bti}^{2}}
{\max(N_{\mathrm{off},i},1)}.
$$

### 4.4 ON-adjacent power-difference loss

For $t=2,\ldots,T$, define

$$
\Delta\hat y_{bti}=\hat y_{bti}-\hat y_{b,t-1,i},
\qquad
\Delta y_{bti}=y_{bti}-y_{b,t-1,i},
$$

and

$$
m^{\Delta}_{bti}=\max(z_{bti},z_{b,t-1,i}).
$$

Because `power_delta_on_only: true`, the active delta loss is

$$
L_{\Delta,i}
=\frac{
\sum_{b,t=2}^{T}m^{\Delta}_{bti}
(\Delta\hat y_{bti}-\Delta y_{bti})^2
}{
\max(\sum_{b,t=2}^{T}m^{\Delta}_{bti},1)
}.
$$

The mask includes any adjacent pair for which either endpoint is ON. It
therefore supervises ON/OFF edges and power variations inside ON periods; it
is not restricted only to true transition locations.

### 4.5 Absolute energy loss: implemented but inactive

The code can calculate the normalized-window energy error

$$
L_{\mathrm{absE},i}
=\frac{1}{BT}\sum_b
\left|\sum_t\hat y_{bti}-\sum_t y_{bti}\right|.
$$

Its configured coefficient is `power_energy_weight: 0.0`, so this term is not
part of the current optimized loss.

### 4.6 Relative energy loss in physical watts

First convert the normalized predictions and targets to nonnegative watts:

$$
\hat P_{bti}=\max(\sigma_i\hat y_{bti}+\mu_i,0),
\qquad
P_{bti}=\max(\sigma_i y_{bti}+\mu_i,0).
$$

Define watt-sample energy within each training window:

$$
\hat E_{bi}=\sum_{t=1}^{T}\hat P_{bti},
\qquad
E_{bi}=\sum_{t=1}^{T}P_{bti}.
$$

The current relative energy loss is

$$
L_{\mathrm{relE},i}
=\frac{1}{B}\sum_b
\frac{|\hat E_{bi}-E_{bi}|}
{E_{bi}+10T}.
$$

The $10T$ floor prevents the ratio from exploding for windows with little or
no true appliance energy. The omitted sampling-period multiplier would appear
in both numerator and denominator and therefore cancels in this relative
ratio.

### 4.7 Complete active power loss

For one appliance,

$$
\boxed{
L_{P,i}
=L_{\mathrm{base},i}
+1.0L_{\mathrm{on},i}
+0.5L_{\mathrm{off},i}
+0.15L_{\Delta,i}
+0.25L_{\mathrm{relE},i}.
}
$$

Across all appliances,

$$
\boxed{L_P=\sum_{i=1}^{5}L_{P,i}.}
$$

If appliance $i$ has ON fraction $r_i^{(batch)}$ in the current batch, the
pointwise part can also be rewritten as

$$
L_{\mathrm{base},i}+L_{\mathrm{on},i}+0.5L_{\mathrm{off},i}
=(1+r_i^{(batch)})L_{\mathrm{on},i}
+(1.5-r_i^{(batch)})L_{\mathrm{off},i}.
$$

Under ordinary random window sampling, the dataset ON rate gives the expected
coefficient. For microwave, $r_{MW}=0.005090$, giving approximately

$$
1.005L_{\mathrm{on},MW}+1.495L_{\mathrm{off},MW}.
$$

Thus the current pointwise power objective does not place a larger explicit
coefficient on microwave ON error than on conditional OFF error.

## 5. Per-Appliance State Loss

### 5.1 Automatic positive-class weight

For appliance $i$, the training ON rate and its numerically clipped value are

$$
r_i=\frac{N_{\mathrm{on},i}}
{N_{\mathrm{on},i}+N_{\mathrm{off},i}}.
$$

$$
\tilde r_i=\operatorname{clip}(r_i,10^{-4},1-10^{-4}).
$$

The adapter calculates

$$
w_i^{+}=\min\left(\frac{1-\tilde r_i}{\tilde r_i},12\right).
$$

Using `mixed_ukdale_refit_3w/training/multi_appliance_training.csv`, the
current values are:

| Appliance | ON rate | Raw $N_{off}/N_{on}$ | Effective $w_i^+$ |
|---|---:|---:|---:|
| kettle | 1.1104% | 89.059 | 12.000 |
| fridge | 40.3365% | 1.479 | 1.479 |
| dishwasher | 5.7991% | 16.244 | 12.000 |
| washing machine | 4.4466% | 21.489 | 12.000 |
| microwave | 0.5090% | 195.456 | 12.000 |

### 5.2 Weighted binary cross-entropy

`binary_cross_entropy_with_logits` applies the sigmoid internally. Expanded,

$$
L_{\mathrm{BCE},i}
=-\frac{1}{BT}\sum_{b,t}
\left[
w_i^{+}z_{bti}\log p_{bti}
+(1-z_{bti})\log(1-p_{bti})
\right].
$$

Only the positive term is multiplied by $w_i^{+}$; the negative weight is 1.

### 5.3 Explicit false-positive probability loss

The code adds a second OFF-state penalty:

$$
L_{\mathrm{FP},i}
=\frac{
\sum_{b,t}(1-z_{bti})p_{bti}^{2}
}{
\max(\sum_{b,t}(1-z_{bti}),1)
}.
$$

This overlaps with the negative part of BCE. BCE penalizes
$-\log(1-p)$ at true OFF samples, while this term separately penalizes $p^2$.

### 5.4 Complete active state loss

For one appliance,

$$
\boxed{
L_{S,i}
=L_{\mathrm{BCE},i}
+1.0L_{\mathrm{FP},i}.
}
$$

Across all appliances,

$$
\boxed{L_S=\sum_{i=1}^{5}L_{S,i}.}
$$

## 6. Dynamic Power-State Magnitude Balance

The current `task_balance: equal` mode calculates

$$
c_S=\operatorname{stopgrad}
\left(\frac{L_P}{\max(L_S,10^{-8})}\right),
$$

$$
\boxed{L_{S,\mathrm{term}}=0.8c_SL_S.}
$$

The stop-gradient operation is essential:

* In the forward pass, $L_{S,\mathrm{term}}$ has magnitude $0.8L_P$.
* In the backward pass, $c_S$ is treated as a constant ruler.
* This balances scalar loss magnitudes, not gradient norms.
* One global $c_S$ is shared by all appliances; it is not calculated per
  appliance.

Increasing one appliance's BCE weight can increase $L_S$ and reduce $c_S$.
Therefore, changing `pos_weight_cap` does not multiply the total state gradient
by the same factor. It mainly redistributes state gradients among appliances
and between positive and negative samples.

## 7. Aggregate Consistency Loss

The normalized aggregate input is converted to watts:

$$
X^{W}_{bt}=\max(\sigma_X X_{bt}+\mu_X,0).
$$

The total predicted power of the five modeled appliances is

$$
\hat P^{\mathrm{sum}}_{bt}=\sum_{i=1}^{5}\hat P_{bti}.
$$

Only physically impossible over-allocation beyond the 30 W tolerance is
penalized:

$$
u_{bt}=\max(
\hat P^{\mathrm{sum}}_{bt}-X^{W}_{bt}-30,
0),
$$

$$
\boxed{
L_A=1.0\frac{1}{BT}\sum_{b,t}
\left(\frac{u_{bt}}{1000}\right)^2.
}
$$

This loss is intentionally one-sided. It does not require the five predicted
appliances to sum to the aggregate because unmodeled appliances and background
load are allowed to remain.

## 8. Fully Expanded Active Training Objective

Substituting all active branches gives

$$
\boxed{
\begin{aligned}
L
={}&\sum_{i=1}^{5}
\left[
L_{\mathrm{base},i}
+L_{\mathrm{on},i}
+0.5L_{\mathrm{off},i}
+0.15L_{\Delta,i}
+0.25L_{\mathrm{relE},i}
\right]\\
&+0.8
\left[
\sum_{i=1}^{5}
\left(
L_{\mathrm{BCE},i}
+L_{\mathrm{FP},i}
\right)
\right]\\
&\quad\times
\operatorname{stopgrad}
\left(
\frac{
\sum_{i=1}^{5}
\left[
L_{\mathrm{base},i}
+L_{\mathrm{on},i}
+0.5L_{\mathrm{off},i}
+0.15L_{\Delta,i}
+0.25L_{\mathrm{relE},i}
\right]
}{
\max\left(
\sum_{i=1}^{5}
\left[
L_{\mathrm{BCE},i}
+L_{\mathrm{FP},i}
\right],
10^{-8}
\right)
}
\right)\\
&+L_A.
\end{aligned}
}
$$

All power terms above operate on the soft state-gated prediction
$\hat y=pR+(1-p)y_{\mathrm{off}}$. Therefore, the apparently separate power
and state branches are coupled before this total objective is evaluated.

## 9. Optional Domain Adaptation Terms: Currently Inactive

The code supports domain adaptation, but the current experiment does not use
it. If enabled, each selected feature map $(B,C,T)$ is first mean-pooled over
time to obtain $(B,C)$.

For layer $l$, Deep CORAL is

$$
L_{\mathrm{CORAL},l}
=\frac{1}{4D_l^2}
\|C_{S,l}-C_{T,l}\|_F^2,
$$

and RBF-MMD is

$$
L_{\mathrm{MMD},l}
=\mathbb{E}[k(Z_S,Z'_S)]
+\mathbb{E}[k(Z_T,Z'_T)]
-2\mathbb{E}[k(Z_S,Z_T)],
$$

$$
k(u,v)=\exp\left(-\frac{\|u-v\|^2}{2\sigma^2}\right).
$$

With `domain_method: both` and `domain_mu: 0.4`, the configured discrepancy
would be

$$
L_D=\sum_l
\left[0.4L_{\mathrm{MMD},l}+0.6L_{\mathrm{CORAL},l}\right].
$$

With `domain_scale: equal`,

$$
L_{D,\mathrm{term}}
=L_D\operatorname{stopgrad}
\left(\frac{L_{\mathrm{NILM}}}{\max(L_D,10^{-8})}\right).
$$

With convex mixing, the final objective would be

$$
L=(1-\lambda_D)L_{\mathrm{NILM}}
+\lambda_D L_{D,\mathrm{term}}.
$$

These equations are documented for completeness only. With
`domain_adaptation.enabled: false` and `lambda_domain: 0.0`, the implementation
returns $L_D=0$ and optimizes only the supervised objective in Section 8.

## 10. Quantities That Are Logged but Not Optimized Separately

The reported training MAE is

$$
\mathrm{MAE}_{\mathrm{log}}
=\frac{1}{A}\sum_i
\sigma_i\operatorname{mean}_{b,t}
|\hat y_{bti}-y_{bti}|.
$$

It is for logging only and is not added to $L$.

`loss_power_per_appliance` contains the complete $L_{P,i}$, including delta
and relative-energy terms; it is not pure MSE. `loss_state_per_appliance`
contains BCE and false-positive terms. These values are detached
for logging after the differentiable total loss has already been assembled.

Validation threshold calibration, minimum-ON cleanup, gap merging, final hard
power gating, F1, SAE, and evaluation MAE are also outside the training loss.
They cannot send gradients into the model.

## 11. Gradient Paths Through the Complete Model

The power-head parameters receive gradients from

* base MSE;
* conditional ON and OFF MSE;
* ON-adjacent delta loss;
* relative energy loss; and
* aggregate consistency.

Because $\partial\hat y/\partial R=p$, all of these gradients are attenuated
when the state probability is low.

The state-head parameters receive gradients from

* weighted BCE;
* explicit false-positive loss;
* every power term through the soft gate; and
* aggregate consistency through the soft gate.

The shared frontend, TCN, task-attention, and cross-appliance relation modules
receive gradients from both task families and from all five appliance heads.

## 12. Engineering Assessment

The loss is scientifically expressive but contains overlapping constraints:

1. Base MSE already contains ON and OFF errors; conditional ON/OFF MSE adds
   both errors again with different normalization.
2. BCE already penalizes false positives; $L_{\mathrm{FP}}$ adds a second OFF
   probability penalty.
3. Pointwise, delta, relative-energy, and aggregate terms can prefer different
   waveform compromises.
4. Global task balancing couples the state scale of all appliances instead of
   balancing each appliance independently.
5. Soft gating makes the regression and classification objectives more tightly
   coupled than the high-level formula $L_P+L_S$ suggests.

The previous `state_fp_weight: 0.0` ablation reduced test performance, so the
explicit false-positive term remains active. The current experiment instead
simplifies `head_local_layers` from 2 to 1 while keeping this loss unchanged.

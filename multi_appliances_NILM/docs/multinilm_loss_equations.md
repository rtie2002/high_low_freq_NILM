# MultiNILM Loss Equations

This is the active loss in `multinilm_fractional_relational.yaml`. It is calculated separately for each appliance and then combined across all five appliances.

Notation: $i$ is the appliance, $z_i\in\{0,1\}$ is the true state, $s_i$ is the state logit, $p_i=\sigma(s_i)$ is the ON probability, $y_i$ is normalized true power, and $P_i$ is power in watts.

## 1. State-Gated Regression

$$
p_{b,t,i}=\sigma(s_{b,t,i}),
$$

$$
\hat y_{b,t,i}
=p_{b,t,i}\hat R_{b,t,i}
+(1-p_{b,t,i})y_{off,i}.
$$

**Meaning:** the state prediction gates the regression output. A low ON probability moves predicted power toward the normalized 0 W value.

## 2. Regression Loss for Appliance $i$

Define

$$
e_{b,t,i}=\hat y_{b,t,i}-y_{b,t,i},
\qquad
m^{\Delta}_{b,t,i}=\max(z_{b,t,i},z_{b,t-1,i}).
$$

### Loss terms

$$
L_{MSE,i}=\operatorname{mean}_{b,t}(e_{b,t,i}^{2}),
$$

$$
L_{on,i}=
\frac{\sum_{b,t}z_{b,t,i}e_{b,t,i}^{2}}
{\max(\sum_{b,t}z_{b,t,i},1)},
$$

$$
L_{off,i}=
\frac{\sum_{b,t}(1-z_{b,t,i})e_{b,t,i}^{2}}
{\max(\sum_{b,t}(1-z_{b,t,i}),1)},
$$

$$
L_{\Delta,i}=
\frac{\sum_{b,t}m^{\Delta}_{b,t,i}
(\Delta\hat y_{b,t,i}-\Delta y_{b,t,i})^{2}}
{\max(\sum_{b,t}m^{\Delta}_{b,t,i},1)},
$$

$$
L_{E,i}=\frac{1}{B}\sum_b
\frac{\left|\sum_t\hat P_{b,t,i}-\sum_tP_{b,t,i}\right|}
{\sum_tP_{b,t,i}+10T}.
$$

| Term | Meaning |
|---|---|
| $L_{MSE,i}$ | Pointwise waveform error over all samples |
| $L_{on,i}$ | Extra power accuracy while the appliance is ON |
| $L_{off,i}$ | Suppresses false power while the appliance is OFF |
| $L_{\Delta,i}$ | Matches rises, falls, and local waveform shape near ON periods |
| $L_{E,i}$ | Matches total appliance consumption inside each window |

### Complete regression loss

$$
\boxed{
L_{power,i}=L_{MSE,i}
+1.0L_{on,i}
+0.5L_{off,i}
+0.15L_{\Delta,i}
+0.25L_{E,i}
}
$$

This is the complete **regression-side loss for one appliance**. Its components work together as follows:

| Contribution | Weight | Effect on training |
|---|---:|---|
| $L_{MSE,i}$ | 1.0 implicit | Fits the complete predicted power sequence at every timestep |
| $L_{on,i}$ | 1.0 | Adds a second, ON-only power objective so rare ON waveforms are not overwhelmed by OFF samples |
| $L_{off,i}$ | 0.5 | Adds extra pressure toward 0 W during true OFF periods, reducing false power |
| $L_{\Delta,i}$ | 0.15 | Gives a smaller auxiliary penalty to wrong slopes and edges, helping waveform shape without dominating amplitude fitting |
| $L_{E,i}$ | 0.25 | Corrects window-level underprediction or overprediction of total appliance consumption |

`L_MSE`, `L_on`, and `L_off` overlap intentionally. A true ON sample contributes to the base MSE and the separately averaged ON loss. A true OFF sample contributes to the base MSE and the separately averaged OFF loss. This compensates for the large class imbalance in NILM, where OFF samples are much more common.

The weights are relative coefficients, not percentages. For example, `0.25` does not mean that energy contributes exactly 25% of the gradient because the five loss terms have different numerical scales.

Because $\hat y_i$ is already state-gated, gradients from $L_{power,i}$ update the regression head and can also pass through the soft state probability into the state head. The separate $L_{state,i}$ in the next section is still required to train explicit ON/OFF classification.

## 3. State Loss for Appliance $i$

The automatic positive-class weight is

$$
w_i^{+}=\min\left(\frac{1-r_i}{r_i},12\right),
$$

where $r_i$ is the training ON rate.

### Loss terms

$$
L_{BCE,i}=\operatorname{BCEWithLogits}(s_i,z_i;w_i^{+}),
$$

$$
L_{FP,i}=
\frac{\sum_{b,t}(1-z_{b,t,i})p_{b,t,i}^{2}}
{\max(\sum_{b,t}(1-z_{b,t,i}),1)}.
$$

For event boundaries:

$$
q_{b,t,i}=p_{b,t-1,i}(1-p_{b,t,i})
+(1-p_{b,t-1,i})p_{b,t,i},
$$

$$
q^{*}_{b,t,i}=|z_{b,t,i}-z_{b,t-1,i}|,
$$

$$
L_{transition,i}
=\frac{1}{2}\operatorname{mean}_{q^{*}=1}[-\log q]
+\frac{1}{2}\operatorname{mean}_{q^{*}=0}[-\log(1-q)].
$$

| Term | Meaning |
|---|---|
| $L_{BCE,i}$ | Predicts ON/OFF state and gives rare ON samples more weight |
| $L_{FP,i}$ | Suppresses high ON probability at true OFF positions |
| $L_{transition,i}$ | Trains correct event start, stop, width, and continuity |

### Complete state loss

$$
\boxed{
L_{state,i}=L_{BCE,i}+1.0L_{FP,i}+0.20L_{transition,i}
}
$$

## 4. Combine All Appliances

$$
L_{power}=\sum_{i=1}^{5}L_{power,i},
\qquad
L_{state}=\sum_{i=1}^{5}L_{state,i}.
$$

**Meaning:** each appliance has separate regression and state losses. The five losses are summed before power/state balancing.

## 5. Dynamic Power/State Balance

$$
s_{balance}=\operatorname{stopgrad}\left(
\frac{L_{power}}{\max(L_{state},10^{-8})}
\right),
$$

$$
L_{state\_term}=0.8L_{state}s_{balance}.
$$

**Meaning:** places state loss on a numerical scale comparable to power loss while still backpropagating through $L_{state}$.

## 6. Aggregate Constraint

$$
L_{agg}=\operatorname{mean}_{b,t}\left[
\frac{\operatorname{ReLU}
(\sum_i\hat P_{b,t,i}-X_{b,t}-30)}{1000}
\right]^2.
$$

**Meaning:** penalizes the sum of appliance predictions only when it exceeds aggregate power by more than 30 W. Unknown household load is allowed.

## 7. Final Loss

$$
\boxed{
L_{NILM}=L_{power}+L_{state\_term}+L_{agg}
}
$$

Equivalently,

$$
\boxed{
L_{NILM}
=\sum_iL_{power,i}
+0.8\left(\sum_iL_{state,i}\right)
\operatorname{stopgrad}\left(
\frac{\sum_iL_{power,i}}
{\max(\sum_iL_{state,i},10^{-8})}
\right)
+L_{agg}.
}
$$

The legacy energy term (`power_energy_weight: 0.0`) and domain loss (`lambda_domain: 0.0`) are disabled and are not included.

# MultiNILM loss simplification

The simplified experiment uses
`config/models/multinilm_fractional_relational_sgn.yaml`. Only its loss
configuration and `experiment_id` differ from the full-loss configuration.
The model, random mix, data splits, and evaluation are unchanged.

## Before: full loss

For appliance $i$, let $L_{P,i}$ be its power loss and $L_{S,i}$ its state
loss:

$$
L_{P,i}=L_{MSE,i}+L_{ON,i}+0.5L_{OFF,i}+0.15L_{\Delta,i}+0.25L_{relE,i},
$$

$$
L_{S,i}=L_{BCE,i}+L_{FP,i}.
$$

After summing the five appliances, the old dynamic balance was

$$
L_{old}=L_P+0.8L_S\operatorname{stopgrad}\left(\frac{L_P}{L_S}\right).
$$

Therefore, the effective state-loss weight changed on every batch.

## After: simplified SGN-style loss

The model still uses the state probability to gate its power output:

$$
p_i=\sigma(s_i),\qquad
\hat y_i=p_iR_i+(1-p_i)y_{off,i}.
$$

The new training loss is

$$
\boxed{
L_{new}
=\sum_{i=1}^{5}L_{MSE}(\hat y_i,y_i)
+\lambda_{state}\sum_{i=1}^{5}L_{BCE}(s_i,z_i),
\qquad \lambda_{state}=1.
}
$$

The BCE keeps the existing automatic positive-class weight:

$$
w_i^+=\min\left(\frac{N_{OFF,i}}{N_{ON,i}},12\right).
$$

This is SGN-style rather than an exact reproduction: this repository averages
BCE over batch and time, sums five appliances, and uses `pos_weight`.

## Configuration change

```yaml
experiment_id: Random Mix 0.5 SGN Simple Loss (8w)

loss:
  task_balance: none
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

Now `L_NILM` is the real fixed sum $L_P+L_S$. The two main values to tune later
are `lambda_state` and `pos_weight_cap`, using validation AP first.

`loss_energy_relative` may still appear as a diagnostic value in the log. Its
weight is zero, so it contributes no gradient and is not part of `L_NILM`.

From `multi_appliances_NILM/`, run:

```powershell
python main.py --mode train_evaluate --model multinilm_fractional `
  --experiment config/experiment_mixed_ukdale_refit_8w.yaml `
  --model-config config/models/multinilm_fractional_relational_sgn.yaml
```

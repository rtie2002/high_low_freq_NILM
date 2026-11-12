# Five-Appliance Power Disaggregation

The system takes one whole-house meter trace and estimates five appliance power traces. It first trains a convolutional neural network (CNN) to reconstruct individually metered appliances. At inference, its decoder generates candidate appliance traces. Adding those traces and an estimate of the remaining household power gives a candidate meter reading that can be checked against the observation.

| Phase | Available data | What changes | Result |
| --- | --- | --- | --- |
| Stage 1 — Training | Five labelled appliance power traces | Encoder and decoder weights | A trained waveform decoder |
| Stage 2 — Inference | One observed whole-house power trace | Candidate waveform codes and the estimated background | Five estimated appliance traces |

The current streamlined implementation uses [`../train.py`](../train.py) for Stage 1, [`../cnn.py`](../cnn.py) for the convolutional neural network (CNN), and [`../inverse.py`](../inverse.py) for Stage 2. Stage 2 uses a nonnegative constant background per window. Section 4 describes a proposed learned-background extension. The numerical examples illustrate the calculations.

Run both stages from the inverse optimisation project directory. Use fresh run directories for a new experiment.

```powershell
python train.py --run-dir runs/forward_cnn
python inverse.py --checkpoint runs/forward_cnn/best.pt --split validation --run-dir runs/inverse_validation
```

The training checkpoint is selected by validation reconstruction mean squared error. Inverse evaluation writes appliance metrics, background diagnostics and representative waveforms to its own run directory.

## 1. Define the signals and read one training window

Let \(t=1,\ldots,T\) index time within a window, and let \(i=1,\ldots,5\) index the kettle, refrigerator, dishwasher, washing machine, and microwave, in that order.

| Symbol | Meaning |
| --- | --- |
| \(p_i(t)\) | Power measured by appliance \(i\)'s submeter, in watts |
| \(y(t)\) | Power measured by the whole-house meter, in watts |
| \(r(t)\) | Effective residual, covering unmonitored loads and meter or alignment discrepancies |
| \(\mathbf p_i,\mathbf y,\mathbf r\) | The corresponding traces across the entire window |
| \(\mathbf z_i\) | The code from which the decoder generates appliance \(i\)'s waveform |
| \(\hat p_i(t)\) | Estimated or reconstructed appliance power at time \(t\) |

The observation model is additive.

$$
y(t)=\sum_{i=1}^{5}p_i(t)+r(t).
$$

Consider four aligned samples from a training household. All readings are in watts.

| Time | Kettle | Refrigerator | Dishwasher | Washing machine | Microwave | Whole-house meter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 100 | 0 | 0 | 0 | 400 |
| 2 | 0 | 100 | 0 | 0 | 1500 | 1900 |
| 3 | 0 | 100 | 0 | 0 | 1500 | 1900 |
| 4 | 0 | 100 | 0 | 0 | 0 | 400 |

Both the whole-house meter and the five submeters are available for this training example. We can therefore calculate the effective training residual.

$$
\mathbf r_{\mathrm{train}}
=\mathbf y_{\mathrm{train}}-\sum_{i=1}^{5}\mathbf p_{i,\mathrm{train}}
=[400,1900,1900,400]-[100,1600,1600,100]
=[300,300,300,300]\ \mathrm W.
$$

At time 2, the refrigerator and microwave explain \(100+1500=1600\ \mathrm W\). The other \(300\ \mathrm W\) belongs to the effective residual. It could include lighting, other appliances, and measurement discrepancies. The calculation gives one combined trace.

## 2. Stage 1 — Train the five-appliance autoencoder

### 2.1. Prepare the input and target

The training script cuts contiguous windows from the preprocessed comma-separated values (CSV) files. The current configuration uses 1024 rows per window and a stride of 512 rows within each sequence. It divides each power value by the same scale \(S=4000\ \mathrm W\).

$$
\mathbf x_i=\frac{\mathbf p_i}{S}.
$$

The microwave input and its training target are both \([0,0.375,0.375,0]\). The refrigerator input and target are both \([0.025,0.025,0.025,0.025]\). The five traces form an input of shape \([B,5,T]\), where \(B\) is the batch size. The network gets each appliance's identity from its fixed channel index. Its training data are power traces.

### 2.2. Encode and reconstruct each appliance

$$
\mathbf z_i=E_\psi(\mathbf x_i),
\qquad
\tilde{\mathbf x}_i=G_\theta(\mathbf z_i,i).
$$

\(E_\psi\) is the encoder with weights \(\psi\). It compresses one waveform into a smaller array of numbers. \(G_\theta\) is the decoder with weights \(\theta\). It takes those numbers and the appliance identity and reconstructs one waveform. The five appliances share the encoder and decoder weights. Each has its own code. With the configured 1024-sample window, each code contains two channels of 128 values.

Suppose the microwave reconstruction, converted back to watts, is \([5,1430,1520,20]\). Its errors relative to the submeter are \([5,-70,20,20]\ \mathrm W\). Its mean squared error (MSE) in watts is

$$
\frac{5^2+(-70)^2+20^2+20^2}{4}=1431.25\ \mathrm W^2.
$$

The implementation computes MSE in scaled power. This microwave example contributes a per-trace MSE of \(1431.25/4000^2\) in that scale.

### 2.3. Update the network and save the checkpoint

The batch loss compares each reconstruction with the corresponding submeter trace.

$$
\mathcal L_{\mathrm{train}}(\psi,\theta)=
\frac{1}{5BT}
\sum_{n=1}^{B}\sum_{i=1}^{5}\sum_{t=1}^{T}
\left[\tilde x_{n,i}(t)-x_{n,i}(t)\right]^2.
$$

Backpropagation computes gradients for the encoder and decoder weights. Adam updates those weights. A microwave error contributes to the loss even if another channel offsets it in the sum. The script saves the weights with the lowest validation reconstruction MSE as **best.pt**.

The current objective uses five submeter targets. The microwave's target already isolates its measured power, so the effective residual has no term in this objective. The code also computes the sum of the five reconstructions using fixed addition. This sum represents the monitored appliances and does not enter the training loss.

Stage 2 takes the trained decoder \(G_{\theta^\star}\) from the checkpoint. Good validation reconstruction shows that the encoder can find codes for measured appliance waveforms. During inference, the codes have to be found using the whole-house meter.

## 3. Stage 2 — Infer five appliance traces from a new meter window

### 3.1. Start with the meter reading and initialise the codes

For a separate inference example, suppose an unseen household has the same four meter readings.

$$
\mathbf y_{\mathrm{obs}}=[400,1900,1900,400]\ \mathrm W.
$$

The inference routine receives only this trace. The new household's submeter readings and effective residual are unknown to it. Freeze the decoder weights at \(\theta^\star\), then initialise five codes \(\mathbf z_1^{(0)},\ldots,\mathbf z_5^{(0)}\), for example with a seeded random initialisation. Superscript \(k\) counts optimisation steps. Index \(t\) still refers to the measurement time.

At each step, decode the current codes and add the five candidates point by point.

$$
\hat{\mathbf p}_i^{(k)}=S G_{\theta^\star}(\mathbf z_i^{(k)},i),
\qquad
\mathbf s^{(k)}=\sum_{i=1}^{5}\hat{\mathbf p}_i^{(k)}.
$$

Here \(G_{\theta^\star}\) produces scaled power and multiplication by \(S\) gives watts. The worked calculations below use watts throughout. An implementation can use scaled power throughout by dividing the meter and background by \(S\) as well.

### 3.2. Estimate the residual under a simple background assumption

The whole-house meter includes the five targets and other loads. For the first baseline, approximate the background over the window by one unknown nonnegative value \(b\) in watts. For a fixed candidate sum \(\mathbf s\), define the pointwise gap \(d(t)=y_{\mathrm{obs}}(t)-s(t)\). The value of \(b\) that minimises the squared gap after background correction is

$$
\hat b(\mathbf z)=
\underset{b\geq 0}{\operatorname{argmin}}\,
\frac{1}{T}\sum_{t=1}^{T}[d(t)-b]^2
=\max\left(0,\frac{1}{T}\sum_{t=1}^{T}d(t)\right).
$$

The unconstrained minimum is the mean because differentiating the loss with respect to \(b\) gives \(2(b-\operatorname{mean}_t d(t))\). The maximum with zero enforces the assumed nonnegative background. We recalculate this number for each candidate. The entire gap \(d(t)\) also contains appliance estimation errors. If a separate background value could freely equal \(d(t)\) at every point, every appliance candidate would fit the meter exactly.

### 3.3. Compute the first loss and update the codes

Suppose the initial candidates give a refrigerator trace of \([100,100,100,100]\ \mathrm W\) and near-zero power for the four other appliances, rounded to zero here. The gap is \([300,1800,1800,300]\ \mathrm W\). Its mean is \(1050\ \mathrm W\), so the candidate whole-house reading is \([1150,1150,1150,1150]\ \mathrm W\).

$$
\mathcal L_{\mathrm{meter}}(\mathbf z)
=\frac{1}{T}\sum_{t=1}^{T}
\left[s(t)+\hat b(\mathbf z)-y_{\mathrm{obs}}(t)\right]^2.
$$

The signed error, candidate minus observed, is \([750,-750,-750,750]\ \mathrm W\). The loss is \(562{,}500\ \mathrm W^2\). Raising the background alone would raise all four readings. Improving the match requires a candidate sum with more power in the middle relative to the ends.

Backpropagation calculates how a small change to each code changes the loss through the decoder. For plain gradient descent, one update is

$$
\mathbf z_i^{(k+1)}
=\mathbf z_i^{(k)}-\alpha_k\nabla_{\mathbf z_i}\mathcal L_{\mathrm{meter}}(\mathbf z^{(k)}),
$$

where \(\alpha_k\) is the step size. Adam uses the gradients and their running statistics to update the codes. The decoder weights stay fixed. The next iteration decodes the updated codes and recalculates the candidate sum, background, and loss.

### 3.4. Work through an update with numbers

For a hand calculation, hold the other four appliance candidates fixed and use a simplified microwave decoder. Let one adjustable number \(a\) control the pulse height in watts so the candidate microwave trace is \([0,a,a,0]\). The trained CNN uses a multidimensional code, so this one-number decoder illustrates the update rule.

For \(0\leq a\leq1500\), the mean gap is \(1050-a/2\) watts, and the meter loss simplifies to

$$
\mathcal L_{\mathrm{meter}}(a)=\frac{(1500-a)^2}{4},
\qquad
\frac{d\mathcal L_{\mathrm{meter}}}{da}=\frac{a-1500}{2}.
$$

At \(a=0\), the derivative is \(-750\). A gradient descent step with \(\alpha=1.6\) in this example gives \(a_{\mathrm{new}}=0-1.6(-750)=1200\ \mathrm W\). The decoder then produces \([0,1200,1200,0]\ \mathrm W\). Recalculate the gap, background, and error.

| Quantity | Initial candidate | After one illustrative update | Exact-fit candidate in the example |
| --- | --- | --- | --- |
| Microwave power (W) | [0, 0, 0, 0] | [0, 1200, 1200, 0] | [0, 1500, 1500, 0] |
| Five-appliance sum (W) | [100, 100, 100, 100] | [100, 1300, 1300, 100] | [100, 1600, 1600, 100] |
| Meter minus five-appliance sum (W) | [300, 1800, 1800, 300] | [300, 600, 600, 300] | [300, 300, 300, 300] |
| Estimated background (W) | 1050 | 450 | 300 |
| Candidate whole-house meter (W) | [1150, 1150, 1150, 1150] | [550, 1750, 1750, 550] | [400, 1900, 1900, 400] |
| Meter MSE (W²) | 562,500 | 22,500 | 0 |

In this simplified example, each update with that step size leaves one fifth of the remaining pulse-height error. Further steps give \(1440\) W and \(1488\) W, approaching \(1500\) W. The exact-fit column shows the limiting candidate.

### 3.5. Stop and return the five estimates

The solver stops after the configured iteration budget and retains the five decoded waveforms from the lowest-loss candidate for each window. For the exact-fit candidate above, the output is

| Appliance | Estimated power (W) |
| --- | --- |
| Kettle | [0, 0, 0, 0] |
| Refrigerator | [100, 100, 100, 100] |
| Dishwasher | [0, 0, 0, 0] |
| Washing machine | [0, 0, 0, 0] |
| Microwave | [0, 1500, 1500, 0] |

The estimated background of \(300\ \mathrm W\) and the candidate whole-house reading can be saved for diagnostics. The five appliance traces are the disaggregation output.

## 4. Optional extension — Learn a background waveform

A constant background gives the solver one value to adjust per window. That assumption can miss sudden changes from unmonitored loads. To represent those changes, calculate \(\mathbf r_{\mathrm{train}}\) from aligned training-house meters and fit a separate background autoencoder. Its decoder \(H_\phi\) reconstructs the effective residual. Use a signed output if modelling a signed residual that includes measurement and alignment discrepancies. Use the same power scale as for the five appliance decoders.

$$
\mathbf x_r=\mathbf r_{\mathrm{train}}/S,\qquad
\tilde{\mathbf x}_r=H_\phi(E_{\mathrm{bg}}(\mathbf x_r)),\qquad
\mathcal L_{\mathrm{bg}}=\frac{1}{T}\sum_{t=1}^{T}[\tilde x_r(t)-x_r(t)]^2.
$$

At inference, freeze both decoders and initialise a sixth code \(\mathbf z_r\) along with the five appliance codes. Generate the candidate background \(\hat{\mathbf r}=S H_{\phi^\star}(\mathbf z_r)\). The whole-house candidate is now \(\sum_i\hat{\mathbf p}_i+\hat{\mathbf r}\). Update all six codes using the meter error and any specified code penalties. For example, penalties based on training-code statistics can favour code values near those seen during training. The solver still receives only \(\mathbf y_{\mathrm{obs}}\) for each new window.

The extra flexibility creates a trade-off. In the example, a microwave pulse with a constant \(300\ \mathrm W\) background fits the meter. A microwave that stays off and a background of \([300,1800,1800,300]\ \mathrm W\) fit it equally well. A kettle producing the same pulse can also fit. The decoder constrains the shapes each code can produce. The observed sum alone cannot resolve identical explanations.

## 5. Verify the method on held-out households

| Check | Procedure | What it measures |
| --- | --- | --- |
| Appliance reconstruction | Reconstruct validation submeter windows and inspect errors during operation, short pulses, and off periods. | Whether the decoder can represent the target waveforms. |
| Appliance separation | Use the sum of five validation submeters as a controlled observation. Keep the individual traces for scoring. | Whether the solver separates the target appliances under a known zero-background condition. |
| Real-meter disaggregation | Keep the same appliance checkpoint and real validation windows. Compare no background, a constant background, and a learned background. | How background treatment affects device-level error, false alarms, and missed events. |
| Held-out test | Fix the checkpoint and inverse settings. Run inference on real test-house meter traces, then use the test submeters to score the outputs. | How well the method generalises to unseen households. |

Report appliance-level mean absolute error (MAE), MAE during operation, and on/off F1 score. Inspect false alarms, missed events, and representative waveforms. Meter fit measures consistency with the observation. The submeter comparisons measure whether the model assigned power to the correct appliances.

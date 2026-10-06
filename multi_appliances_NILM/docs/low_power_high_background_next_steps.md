# Low-Power Appliances Under Strong Aggregate Background

## 1. Current problem

Low-power appliances, especially the fridge, perform poorly when other loads are active. The main difficulty is not simply that the aggregate power is high. The important factor is whether the target appliance's change is visible relative to simultaneous changes from other appliances and unknown background loads.

For example:

- A 100 W fridge can still be detected on top of a stable 3000 W background because its switching edge remains visible.
- The same fridge can become difficult to detect on a 500 W aggregate if an unknown load changes by 200 W at the same time.
- A microwave has higher power, but its short pulse can still be confused with appliances having similar transient shapes.

Therefore, the real problem is **target-to-interference ratio**, not aggregate wattage alone.

## 2. Evidence from the literature

- [HIFDA](https://www.nature.com/articles/s41597-025-04859-3): low sampling rates make low-power appliances difficult to identify; higher-frequency measurements provide more distinctive signatures.
- [Performance Gap Between Real and Denoised Aggregates](https://arxiv.org/abs/2008.10985): NILM performance drops on real aggregates containing unknown and untracked loads compared with denoised aggregates.
- [Bits and Watts](https://doi.org/10.1145/2674061.2675039): low-power and similar appliances are fundamental NILM identification problems; additional signal information can improve observability.
- [Iterative Load Disaggregation](https://ojs.aaai.org/index.php/AAAI/article/view/8162): appliances can be separated iteratively by subtracting estimated loads from the aggregate.
- [Differentiable Mixture Consistency](https://research.google/pubs/differentiable-consistency-constraints-for-improved-deep-speech-enhancement/): source estimates can be constrained to reconstruct the input mixture.
- [PCEN](https://research.google.com/pubs/archive/45911.pdf): local gain control can expose weak signals under strong, changing background conditions.

PCEN and speech-separation methods have not been proven directly for this project. They are cross-domain ideas that require controlled NILM experiments.

## 3. First step: diagnose the failure correctly

Do not change the architecture yet. First measure performance against the local interference level.

For appliance \(i\):

\[
x(t)=y_i(t)+r_i(t),
\]

where \(x\) is aggregate power, \(y_i\) is the target appliance and \(r_i\) contains all other appliances and background power.

Use a change-domain signal-to-interference ratio for each true ON event \(e\):

\[
\Delta\mathrm{SNR}_{i,e}
=10\log_{10}
\frac{\sum_{t\in e}(\Delta y_i(t))^2}
{\sum_{t\in e}(\Delta(x(t)-y_i(t)))^2+\epsilon}.
\]

Suggested groups:

| \(\Delta\mathrm{SNR}\) | Interpretation |
|---:|---|
| Below -20 dB | Target is almost hidden by interference |
| -20 to -10 dB | Very difficult |
| -10 to 0 dB | Moderate difficulty |
| Above 0 dB | Target change is relatively clear |

For every appliance and group, report:

- event precision and recall;
- false events per hour;
- ON-MAE;
- switching-edge timing error;
- waveform correlation or DTW distance;
- number of events in the group.

This experiment will determine whether failures are caused by high absolute aggregate power, simultaneous background changes, or both.

## 4. First model experiment: one local-contrast input

After completing the current task-attention ablation, restore the selected baseline and make only one new change: add a local-contrast channel inspired by PCEN.

\[
m_t=\operatorname{EMA}(x_t),
\]

\[
u_t=x_t-m_t,
\]

\[
s_t=\operatorname{EMA}(|u_t|),
\]

\[
c_t=\operatorname{clip}\left(
\frac{u_t}{(\epsilon+s_t)^\alpha}
\right).
\]

The model should receive both signals:

```text
Raw aggregate ------------------------> absolute power information
      |
      +--> baseline removal
             |
             +--> local normalization -> weak-event contrast
                                            |
Raw path + contrast path -------------------+
                                            v
                              Multi-appliance model
```

Important controls:

- Keep the raw aggregate channel so the model retains absolute watt information.
- Calculate the contrast from power in watts, not directly from a signed z-score.
- Use an epsilon floor, robust scale estimate and clipping to avoid amplifying sensor noise.
- Do not change the loss, architecture or post-processing in the same experiment.
- Compare at least two seeds using the same validation and test protocol.

## 5. Only if local contrast is insufficient

Test a shared residual-refinement stage:

\[
r_i(t)=x(t)-\sum_{j\ne i}\operatorname{stopgrad}(\hat y_j(t)),
\]

\[
\hat y_i^{\mathrm{refined}}
=f_{\mathrm{shared}}(x,r_i,e_i),
\]

where \(e_i\) is an appliance embedding.

```text
Aggregate -> shared multi-appliance prediction
                         |
                         v
             subtract other predicted loads
                         |
                         v
              appliance residual signal
                         |
                         v
          shared lightweight refinement head
```

This retains the multi-appliance contribution because the initial estimator and refinement head remain shared. `stopgrad` is needed to reduce error propagation between appliance branches.

This is a larger architecture experiment and must not be combined with the local-contrast experiment.

## 6. When architecture changes cannot solve it

If performance consistently collapses at extremely low \(\Delta\mathrm{SNR}\), the 8-second active-power input may not contain enough information to distinguish the appliance from unknown loads. More attention blocks, class weights or post-processing cannot reconstruct information that is absent from the input.

The next physical solution would be to add information such as:

- higher-frequency current and voltage measurements;
- harmonics or transient descriptors;
- high-frequency event embeddings fused with low-frequency temporal context;
- another relevant sensor or contextual signal.

## 7. Agreed experiment order

1. Finish the current task-attention ablation.
2. Implement \(\Delta\mathrm{SNR}\)-conditioned evaluation without changing training.
3. Confirm whether fridge and microwave failures follow decreasing \(\Delta\mathrm{SNR}\).
4. Add exactly one local-contrast input channel.
5. Run the same seeds and compare per-appliance and per-SNR-bin results.
6. Try shared residual refinement only if the local-contrast experiment is insufficient.
7. If very-low-SNR cases remain unidentifiable, investigate high-frequency or additional input information.

## 8. What not to do next

- Do not add another large attention module immediately.
- Do not increase positive class weights or ON-loss weights as the first solution; this may increase false positives without revealing the hidden signal.
- Do not change input features, architecture, loss and post-processing together.
- Do not judge success only from global MAE or overall F1; these metrics can hide low-SNR failure cases.

The immediate next task is therefore **diagnostic evaluation by local \(\Delta\mathrm{SNR}\)**, followed by a single controlled local-contrast feature experiment.

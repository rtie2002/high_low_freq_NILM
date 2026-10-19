# MART-NILM Ablation Study Summary & Mathematical Reference

> **Scope**: Controlled experimental progression starting from the **Recovery Baseline (2026-10-08)** to the final paper models (2026-10-14).  
> **Code Alignment**: Fully verified against [`model/MARTNILM.py`](../model/MARTNILM.py), [`model/MARTNILM_loss.py`](../model/MARTNILM_loss.py), and [`docs/multinilm_experiment_log.md`](multinilm_experiment_log.md).

---

## 1. Core Mathematical Formulations

### 1.1 Architecture & Gating
- **Aggregate Input**: $\mathbf{x} \in \mathbb{R}^{B \times 1 \times T}$ (Raw aggregate mains power only).
- **Multi-Scale Stem + IBN**:
  $$\mathbf{H}_{\text{stem}} = \mathrm{Concat}\Big(\mathrm{Conv}_{k=3}(\mathbf{x}), \mathrm{Conv}_{k=5}(\mathbf{x}), \mathrm{Conv}_{k=9}(\mathbf{x})\Big), \quad \mathbf{H}_{\text{IBN}} = \big[\mathrm{IN}(\mathbf{H}_{1:C/2}) \,\|\, \mathrm{BN}(\mathbf{H}_{C/2+1:C})\big]$$
- **Temporal Receptive Field (5-block Dilated TCN)**:
  $$\mathbf{H}^{(\ell+1)} = \mathbf{H}^{(\ell)} + \mathrm{Dropout}\Big(\mathrm{ReLU}\big(\mathrm{BN}(\mathrm{Conv1D}_{k=5, d=2^{\ell}}(\mathbf{H}^{(\ell)}))\big)\Big), \quad \ell \in \{0, 1, 2, 3, 4\}$$
- **Cross-Appliance Relation Attention**:
  $$\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times T \times A \times d}, \quad \mathbf{A} = \mathrm{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right), \quad \mathbf{F}_i^{\text{rel}} = \mathbf{F}_i + \alpha_{\text{rel}} \cdot \mathbf{g}_i \odot \big(\mathbf{A}\mathbf{V}\big)_i$$
  where $\alpha_{\text{rel}} = 0.25$, $A=5$ appliances, and $\mathbf{g}_i = \sigma(\mathrm{Linear}([\mathbf{F}_i \,\|\, \mathbf{msg}_i]))$.
- **Physical Output State-Power Gate**:
  $$\hat{y}_{b,t,i} = g_{b,t,i} \cdot \hat{p}_{b,t,i} + (1 - g_{b,t,i}) \cdot y_i^{\text{off}}, \quad g_{b,t,i} = \sigma(\hat{z}_{b,t,i})$$
  *(At evaluation time, a calibrated hard mask $g = \mathbb{I}(\sigma(\hat{z}) \ge \tau_i)$ is applied to suppress zero-power leakage).*

---

### 1.2 Multi-Objective Loss Formulation
$$\mathcal{L}_{\text{NILM}} = \mathcal{L}_{\text{power}} + \lambda_{\text{state}} \sum_{i=1}^{M} r_i S_i$$

1. **Power Objective**:
   $$\mathcal{L}_{\text{power}} = \sum_{i=1}^M \Big(\mathrm{MSE}_i + w_{\text{on}}\mathrm{MSE}_i^{\text{on}} + w_{\Delta}\mathrm{MSE}_i^{\Delta}\Big)$$
   - **All-sample MSE**: $\mathrm{MSE}_i = \frac{1}{BT}\sum_{b,t} (\hat{y}_{b,t,i} - y_{b,t,i})^2$
   - **ON-state MSE**: $\mathrm{MSE}_i^{\text{on}} = \frac{\sum_{b,t} z_{b,t,i}(\hat{y}_{b,t,i} - y_{b,t,i})^2}{\max(\sum z, 1)}$
   - **Edge/Delta MSE**: $\mathrm{MSE}_i^{\Delta} = \frac{\sum_{b,t} \max(z_{b,t,i}, z_{b,t-1,i})(\Delta \hat{y} - \Delta y)^2}{\max(N_i^\Delta, 1)}$

2. **State Objective**:
   $$S_i = \mathrm{BCEWithLogits}(\hat{z}_{b,t,i}, z_{b,t,i}; w_i^{\text{pos}}) + w_{\text{fp}} \frac{\sum_{b,t} (1 - z_{b,t,i})\sigma(\hat{z}_{b,t,i})^2}{\max(\sum (1-z), 1)}$$

3. **Clipped Per-Appliance Dynamic Balancing Rule**:
   $$r_{\text{global}} = \frac{\mathcal{L}_{\text{power}}}{\mathcal{L}_{\text{state}}}, \quad r_i = \mathrm{clip}\left(\frac{P_i}{S_i}, \; \frac{r_{\text{global}}}{c}, \; c \cdot r_{\text{global}}\right), \quad c = 3.0$$

---

## 2. Table 1: What We Deleted / Rejected
*All candidate simplifications and modifications discarded since 2026-10-08 (Numbers placed directly beside category/component):*

| # | Category | Component / Trial | Baseline Numbers | Ablated Numbers | $\Delta$ Performance Impact | What was Deleted / Tested & Proof |
|:---:|:---|:---|:---|:---|:---|:---|
| **1** | **Input Features** | **13-channel Feature Bank** | 13-channel: **$15.586\text{ W}$**, F1 **$0.772$** | Raw-only: **$14.590\text{ W}$**, F1 **$0.771$** | **$+0.996\text{ W}$** MAE penalty for 13 ch | Deleted 12 handcrafted features (rolling mean/std, delta, GL); raw-only is cleaner and lower error. |
| **2** | **Input Features** | **EMA-Residual Feature** | Val MAE $14.78\text{ W}$, MW F1 **$0.714$** | Val MAE $14.92\text{ W}$, MW F1 **$0.642$** | **$-0.072$** microwave F1 drop | Added causal $x[t] - \mathrm{EMA}_{45}(x)[t]$; generated artificial $1\text{ kW}$ false pulse on REFIT microwave. |
| **3** | **Augmentation** | **Partial Focal Mix ($p=0.25$)** | $p=1.0$: Val AP **$0.795$**, FPR **$0.171$** | $p=0.25$: Val AP **$0.784$**, FPR **$0.205$** | **$-0.011$** AP, **$+0.034$** fridge FPR | Reduced focal mix to 25% of synthetic windows; degraded ranking and increased fridge false alarms. |
| **4** | **Augmentation** | **Target Alignment / Jitter** | Val MAE **$14.88\text{ W}$**, F1 **$0.782$** | Val MAE **$16.44\text{ W}$**, F1 **$0.760$** | **$+1.56\text{ W}$** MAE, **$-0.022$** F1 | Shifted labels $\pm 2$ samples to match aggregate edges; attaching to local edges confused overlapping loads. |
| **5** | **Training Duration**| **10-Week Training Split** | 5w REFIT MAE **$11.28\text{ W}$**, FPR **$0.609$** | 10w REFIT MAE **$11.73\text{ W}$**, FPR **$0.742$** | **$+0.133$** fridge FPR ($>800\text{ W}$) | Doubled rows ($1.86\text{M} \rightarrow 3.68\text{M}$); more data increased false fridge activations on high background. |
| **6** | **Loss Function** | **Core Loss Simplification** | Val MAE **$14.59\text{ W}$**, MW Det **$63.7\%$** | Val MAE **$15.13\text{ W}$**, MW Det **$56.8\%$** | **$+0.54\text{ W}$** MAE, **$-6.9\%$** MW Det | Removed OFF-MSE, $\Delta y$-MSE, and FP penalty; microwave detection collapsed, false alarms rose $93 \rightarrow 101$. |
| **7** | **Loss Function** | **Compact Loss Simplification**| Val MAE **$14.59\text{ W}$**, MW NRMSE **$0.579$** | Val MAE **$15.11\text{ W}$**, MW NRMSE **$0.789$** | **$+0.210$** MW event NRMSE | Removed OFF-MSE and relative energy error; microwave event waveform reconstruction severely degraded. |
| **8** | **Loss Function** | **Unrestricted Local Balance** | Global: Val MAE **$14.78\text{ W}$**, Ratio $\approx 17$ | Local: Val MAE **$15.37\text{ W}$**, Ratio $\approx 177$ | **$10\times$** state gradient jump | Balanced each appliance by $P_i/S_i$ without clipping; over-amplified microwave state gradient, causing false pulses. |
| **9** | **Architecture** | **w/o Temporal Encoder** | Test F1 **$0.796$**, Test ON-MAE **$242.9\text{ W}$** | Test F1 **$0.710$**, Test ON-MAE **$288.1\text{ W}$** | **$-0.086$** F1, **$+45.13\text{ W}$** ON-MAE | Replaced 5-block dilated TCN with identity mapping; catastrophic collapse, continuous false fridge cycles. |
| **10** | **Architecture** | **Truncated TCN (3 vs 5 blk)** | 5 blk: Val MAE **$14.59\text{ W}$**, Det **$63.7\%$** | 3 blk: Val MAE **$16.60\text{ W}$**, Det **$51.4\%$** | **$+2.01\text{ W}$** MAE, **$-12.3\%$** MW Det | Cut dilated TCN blocks from 5 to 3; insufficient receptive field, microwave energy error doubled to $73.5\%$. |
| **11** | **Architecture** | **w/o Appliance Decoders** | Test ON-MAE **$242.9\text{ W}$**, F1 **$0.796$** | Test ON-MAE **$265.8\text{ W}$**, F1 **$0.783$** | **$+22.90\text{ W}$** Test ON-MAE | Replaced 5 private decoders with $1\times 1$ readouts; cannot isolate appliance signatures, severe fragmentation. |
| **12** | **Architecture** | **Decoder Depth (1 vs 2 blk)** | 2 blk: Val MAE **$14.59\text{ W}$**, NRMSE **$0.579$**| 1 blk: Val MAE **$15.08\text{ W}$**, NRMSE **$0.812$**| **$+0.49\text{ W}$** MAE, **$+0.233$** NRMSE | Cut local residual blocks from 2 to 1 in heads; degrades amplitude reconstruction on active appliances. |
| **13** | **Architecture** | **BatchNorm instead of IBN** | IBN: Val MAE **$14.59\text{ W}$**, Fridge **$20.4\text{ W}$** | BN: Val MAE **$15.89\text{ W}$**, Fridge **$26.3\text{ W}$** | **$+5.91\text{ W}$** Fridge MAE, **$+144$** FPs | Replaced stem Instance-Batch Norm with standard BN; stripped house-invariant normalization. |
| **14** | **Architecture** | **GroupNorm for Temporal TCN**| BN: Val MAE **$14.42\text{ W}$**, MW F1 **$0.521$** | GN: Val MAE **$14.32\text{ W}$**, MW F1 **$0.434$** | **$-0.087$** REFIT microwave F1 | Replaced BatchNorm with GroupNorm in TCN; removed cross-sample amplitude context. |
| **15** | **Architecture** | **Shared BiGRU Context** | TCN: Val MAE **$14.59\text{ W}$**, FPs **$228$** | +BiGRU: Val MAE **$16.89\text{ W}$**, FPs **$2,461$**| **$+2,233$** fridge false alarms | Appended BiGRU ($128$ units) after TCN; full-window recurrent mixing propagated ambiguous noise. |
| **16** | **Architecture** | **Narrow Hidden Width ($96$ ch)** | 128 ch: Val MAE **$14.59\text{ W}$**, Det **$63.7\%$** | 96 ch: Val MAE **$15.86\text{ W}$**, Det **$48.6\%$** | **$+1.27\text{ W}$** MAE, **$-15.1\%$** MW Det | Cut channels from 128 to 96; under-capacity disproportionately harmed sparse transient loads. |
| **17** | **Gating** | **Removing State-Power Gate** | Gated: Test OFF-MAE **$5.97\text{ W}$**, MAE **$9.13\text{ W}$** | Ungated: Test OFF-MAE **$10.80\text{ W}$**, MAE **$12.82\text{ W}$** | **$+4.82\text{ W}$** OFF-MAE, **$+3.70\text{ W}$** MAE | Removed soft state multiplier ($y = \hat{p}$); un-gated network hallucinates non-zero power when OFF. |
| **18** | **Gating** | **Detaching Power Gate** | Joint: Val MAE **$14.59\text{ W}$**, MW F1 **$0.474$** | Detached: Val MAE **$15.14\text{ W}$**, MW F1 **$0.431$** | **$-0.043$** MW F1, **$+43$** false events | Stopped power loss backprop through state head; destroyed state/power synchronization. |
| **19** | **Gating** | **Removing Eval Hard Mask** | Gated: Val MAE **$14.42\text{ W}$**, REFIT **$10.25\text{ W}$** | Ungated: Val MAE **$23.71\text{ W}$**, REFIT **$17.20\text{ W}$** | **$+9.29\text{ W}$** Val, **$+6.94\text{ W}$** REFIT | Disabled inference binary gate (`apply_to_power: false`); massive low-level baseline leakage. |

---

## 3. Table 2: What We Confirmed Is Working / Retained
*The verified core mechanisms forming the final MART-NILM model (Numbers placed directly beside category/component):*

| # | Category | Component / Setting | Retained Numbers | Ablated Numbers | $\Delta$ Numerical Gain | Exact Formulation & Proven Evidence |
|:---:|:---|:---|:---|:---|:---|:---|
| **1** | **Input** | **Raw Aggregate Input** | Val MAE: **$14.590\text{ W}$**, F1: **$0.771$** | 13-channel: Val MAE **$15.586\text{ W}$** | **$-0.996\text{ W}$** lower MAE | Single raw channel $\mathbf{x} \in \mathbb{R}^{B \times 1 \times T}$; zero phase delay, lower error than 13 handcrafted channels. |
| **2** | **Augmentation** | **Focal-Event Stratified Mix** | Val AP: **$0.795$**, FPR: **$0.171$** | Uniform: Val AP **$0.787$**, FPR **$0.209$** | **$+0.008$** AP, **$-0.038$** fridge FPR | $p=1.0$; 1 focal active appliance across 5 residual bins $[0, 100, 200, 400, 800, \infty)\text{ W}$. |
| **3** | **Optimization**| **Clipped Local Balancing** | Val F1: **$0.782$**, REFIT MAE: **$9.840\text{ W}$** | Global: Val F1 **$0.770$**, REFIT **$11.067\text{ W}$** | **$+0.012$** F1, **$-1.23\text{ W}$** REFIT MAE | $r_i = \mathrm{clip}(P_i/S_i, r/3, 3r)$, $\lambda_{\text{state}} = 0.8$; prevents high-power microwave from dictating fridge state. |
| **4** | **Architecture**| **Multi-Scale Stem + IBN** | UK-DALE5 MW ON-MAE: **$589.5\text{ W}$** | Single-scale: MW ON-MAE **$696.5\text{ W}$** | **$-107.0\text{ W}$** microwave error | Conv1D kernels $[3, 5, 9]$ + IBN-1D; captures short transients and normalizes house-to-house voltage drift. |
| **5** | **Architecture**| **5-Layer Dilated TCN** | Test MAE: **$9.125\text{ W}$**, F1: **$0.796$** | w/o TCN: Test MAE **$10.565\text{ W}$**, F1 **$0.710$** | **$-1.44\text{ W}$** MAE, **$+0.086$** F1 | 5 residual blocks ($d \in \{1, 2, 4, 8, 16\}$); essential 33-min receptive field without downsampling. |
| **6** | **Architecture**| **Appliance Decoders** | Test ON-MAE: **$242.94\text{ W}$**, F1: **$0.796$** | w/o Decoders: Test ON-MAE **$265.84\text{ W}$** | **$-22.90\text{ W}$** Test ON-MAE | 5 private heads with 2 local residual blocks + channel gate; eliminates signal fragmentation. |
| **7** | **Architecture**| **Relation Attention** | REFIT20 MW F1: **$0.549$**, Test F1: **$0.796$** | w/o Attention: REFIT20 MW F1 **$0.496$** | **$+0.053$** REFIT MW F1 | Cross-appliance multi-head attention ($\alpha_{\text{rel}} = 0.25$); exchanges energy budget context between appliances. |
| **8** | **Gating** | **Joint Soft Power Gate** | Test OFF-MAE: **$5.971\text{ W}$**, MAE: **$9.125\text{ W}$**| w/o Gate: Test OFF-MAE **$10.795\text{ W}$** | **$-4.82\text{ W}$** OFF-MAE, **$-3.70\text{ W}$** MAE | $\hat{y} = \hat{p} \cdot \sigma(\hat{z})$ with joint gradient flow; strictly enforces zero-power output during OFF periods. |
| **9** | **Loss Function**| **Auxiliary Loss Terms** | Val MAE: **$14.590\text{ W}$**, MW Det: **$63.7\%$** | Core loss: Val MAE **$15.134\text{ W}$**, Det **$56.8\%$** | **$-0.54\text{ W}$** MAE, **$+6.9\%$** MW Det | $\Delta y$-MSE ($w_\Delta = 0.5$) aligns switching edges; FP penalty ($w_{\text{fp}} = 0.2$) stops background misdetections. |
| **10** | **Decoding** | **Duration Postprocessing** | REFIT20 MW F1: **$0.447$**, Precision: **$0.396$** | Raw threshold: MW F1 **$0.420$**, Prec **$0.351$** | **$+0.027$** F1, **$+0.045$** precision | Min ON duration $32\text{ s}$, max merge gap $24\text{ s}$; eliminates short spurious submeter noise fragments. |

---

## 4. Key Takeaways for Presentation

1. **The OFF-Dominated Trap**: Overall MAE in NILM is $>90\%$ dominated by OFF intervals. Removing gating or channel attention may appear neutral or slightly reduce all-sample MAE by predicting zero, but **ON-state MAE severely degrades** and physical realism is lost.
2. **Specialized Module Roles**:
   - **TCN & IBN**: Anchor low-frequency, high-duty-cycle stability (eliminates false fridge activations).
   - **Multi-Scale Stem & Relation Attention**: Resolve high-frequency, sparse, high-power switching transients (essential for kettle and microwave).

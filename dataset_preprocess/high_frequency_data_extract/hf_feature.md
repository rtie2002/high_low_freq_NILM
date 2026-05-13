# NILM Multi-Domain Feature Engine: The Unified Ph.D. Whitepaper

This document is the definitive technical reference for the `hf_feature.py` engine. It combines mathematical rigor, circuit physics, implementation logic, and a complete formula glossary for all **55 extracted features**.

---

## 1. Mathematical Foundation: Windowing & FFT Normalization

### 1.1 The Energy Conservation Problem
**Idea**: In a discrete system, applying a window (like Hann) reduces the total energy of the signal. Without normalization, all power features would be underestimated by ~50%.
**Math**:
We apply a Hann window $w[n]$ and a normalization constant $C_{win}$:
$$C_{win} = \frac{N}{\sum_{n=0}^{N-1} w[n]} \approx 2.0$$
**Code**:
```python
window = np.hanning(N)
window_norm = window / (window.sum() / N)
v_fft = np.fft.rfft(v_t * window_norm)
```

---

## 2. Level 1: Time-Domain (Morphological & Energy)

### 2.1 Basic Power Metrics
**Math**:
*   Active Power ($P$): Average of instantaneous power $p(t) = v(t)i(t)$.
    $$P = \frac{1}{N} \sum_{n=0}^{N-1} v[n] \cdot i[n]$$
*   Crest Factor ($CF$): Ratio of peak value to RMS.
    $$CF_i = \frac{\max|i[n]|}{\sqrt{\frac{1}{N} \sum i[n]^2}}$$

**Physical Meaning**:
*   **Active Power**: The real work done (Heating, Mechanical).
*   **Crest Factor**: High $CF_i$ (e.g., > 3.0) indicates **Switching Mode Power Supplies (SMPS)**. Heaters have $CF_i \approx 1.41$.
**Code**:
```python
v_rms = np.sqrt(np.mean(v_t ** 2))
p_active = np.mean(v_t * i_t)
fci = np.max(np.abs(i_t)) / (i_rms + 1e-9)
```

### 2.2 Statistical Moments (Shape Stats)
**Math**:
*   **Skewness**: $E[(i-\mu)^3] / \sigma^3$
*   **Kurtosis**: $E[(i-\mu)^4] / \sigma^4 - 3$

**Physical Meaning**:
*   **Skewness**: Asymmetric current often indicates **half-wave rectification**.
*   **Kurtosis**: High Kurtosis distinguishes **impulsive loads** (sharp spikes) from continuous ones.

---

## 3. Level 2: Frequency-Domain (Spectral Fingerprinting)

### 3.1 Robust Harmonics ($I_k$)
**Idea**: Due to grid drift, the 50Hz fundamental might actually be 49.95Hz. A single-bin FFT check would miss the energy.
**Math (Band Energy Integration)**:
Instead of a single bin, we integrate over a bandwidth $B = \pm 15$ Hz:
$$I_k = \sqrt{\sum_{f=k f_0 - B}^{k f_0 + B} |X(f)|^2}$$
**Code**:
```python
bin_bw = max(1, int(round(15.0 / (fs / N))))
ik = np.sqrt(np.sum(i_amp[idx - bin_bw : idx + bin_bw + 1]**2))
```

### 3.2 Spectral Entropy ($H$)
**Math**:
$$H = -\sum p_i \log(p_i), \quad p_i = \frac{|X(f_i)|^2}{\sum |X(f_j)|^2}$$
**Physical Meaning**: 
*   **Low Entropy**: Energy is concentrated in clear harmonics (Motors).
*   **High Entropy**: Energy is spread across the spectrum (Arcing, PWM switching).

### 3.3 Log-Frequency Spectral Envelope ($Env_j$)
**Idea**: We use a log-octave scale to group high-frequency residuals.
**Math (Normalization)**:
To isolate **Spectral Shape** from **Power Level**, we normalize the envelope vector $\mathbf{E}$:
$$\hat{E}_j = E_j / \sum_{k} E_k$$
**Physical Meaning**: This is the **"Visual Silhouette"** of the appliance.

---

## 4. Level 3: Time-Frequency (Wavelet Transients)

### 4.1 Discrete Wavelet Transform (DWT)
**Math**:
$$DWT\_E_j = \frac{1}{N_j} \sum_{n=1}^{N_j} d_j[n]^2$$
**Physical Meaning**: Captures the "switching shiver" — the sub-cycle transient that occurs at the exact moment of turn-on/off.

---

## 5. Complete Formula Glossary (All 55 Columns)

| Column Name(s) | Mathematical Definition | Domain |
| :--- | :--- | :--- |
| `V_rms` / `I_rms` | $\sqrt{\frac{1}{N} \sum x[n]^2}$ | L1 Basic |
| `P_active` | $\frac{1}{N} \sum (v[n] \cdot i[n])$ | L1 Basic |
| `S_apparent` | $V_{rms} \cdot I_{rms}$ | L1 Basic |
| `PF` | $P_{active} / S_{apparent}$ | L1 Basic |
| `Fcv` / `Fci` | $\max|x[n]| / X_{rms}$ | L1 Morphology |
| `I_skew` / `V_skew` | $E[(x-\mu)^3] / \sigma^3$ | L1 Shape |
| `I_kurt` | $E[(i-\mu)^4] / \sigma^4 - 3$ | L1 Shape |
| `I_std` / `V_std` | Standard Deviation of the block | L1 Shape |
| `I1` ... `I15` | $\sqrt{\sum_{f \in [k f_0 \pm 15]} \text{Amp}_i(f)^2}$ | L2 Harmonics |
| `V1` ... `V15` | $\sqrt{\sum_{f \in [k f_0 \pm 15]} \text{Amp}_v(f)^2}$ | L2 Harmonics |
| `IH` / `VH` | $\sqrt{\sum_{k \ge 2} X_k^2}$ | L2 Distortion |
| `THDI` / `THDV` | $X_H / X_1$ | L2 Distortion |
| `I_BP_low/mid/high`| Sum of squared Amps in specific Hz bands | L2 Band Energy |
| `V_BP_low` | Sum of squared Amps in 50-500Hz | L2 Band Energy |
| `I_spec_entropy` | $-\sum p \log p$ (Spectral complexity) | L2 Descriptor |
| `I_env_0` ... `I_env_7`| Normalized Log-Octave Band Energies | L2 Envelope |
| `DWT_E0` ... `DWT_E4`| Mean Squared Detail Coefficients (db4) | L3 Wavelet |

---

## 6. Summary for PhD Thesis Appendix

| Category | Features | Research Value |
| :--- | :--- | :--- |
| **Atomic** | 7 | Fundamental energy consumption level. |
| **Morphology** | 5 | Circuit topology identification. |
| **Harmonics** | 16 | Robust fingerprint against grid drift. |
| **Spectral** | 17 | Magnitude-invariant shape & complexity. |
| **Wavelet** | 5 | Switching transients & arcing noise. |

---
*Version: 5.0 (Super Unified PhD Whitepaper)*
*Author: Antigravity AI Orchestrator*

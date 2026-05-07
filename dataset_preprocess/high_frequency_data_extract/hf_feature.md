# High-Frequency Feature Extraction — Theory & Implementation

> [!IMPORTANT]
> **Consistency Rule**: This document is the theoretical twin of `hf_feature.py`.
> Any changes to the calculation logic in the code **must** be mirrored here.

---

## Overview: The 4-Layer Pipeline

Raw waveforms `v(t)` and `i(t)` → **3 Python functions** → ~70 features

```
compute_hf_features()              ← Orchestrator
    ├── get_rms()                  ← Layer 1: Time-Domain
    ├── get_power_metrics()        ← Layer 1: Time-Domain
    ├── get_harmonic_analysis()    ← Layer 2 + 3: FFT + Distortion
    └── get_wavelet_features()     ← Layer 4: DWT
```

---

## The Core Concept: Multi-Domain "Digital Fingerprinting"

The philosophy behind `hf_feature.py` is that no single feature can identify an appliance perfectly. Instead, we perform a **multi-dimensional autopsy** of the electricity:

### A. The Workflow Path
1.  **Normalization**: Convert raw ADC counts (integers) into physical units (Volts, Amps).
2.  **Steady-State Decomposition (FFT)**: Assuming the appliance is running stably, we use FFT as a "prism" to see the harmonic fingerprint.
3.  **Phase-Aware Cross-Analysis**: We don't just look at current; we look at how current *interacts* with voltage at every harmonic frequency. This reveals Inductive (motors) vs. Resistive (heaters) behavior.
4.  **Transient Detection (Wavelet)**: We look for high-frequency "shivers" in the waveform that occur during switching events.

### B. Why this matters for Ph.D. Research?
By combining these domains, we create a **holographic vector**. If two appliances have the same Power ($P$), they might have different 3rd Harmonics ($I_3$). If they have the same $I_3$, they might have different Reactive Power ($Q_3$). If everything else is the same, their **Wavelet Transient** ($DWT$) will likely be different.

---

## The Cycle-Splitting & Vectorization Algorithm: The "How"

Before any features are extracted, the engine performs a **Phase-Averaged Reshaping** operation. This is the mechanism that achieves "splitting with windows."

### 1. The Splitting Logic (Windowing)
We don't use sliding windows; instead, we use **Strict Cycle Windowing** to ensure each window contains exactly one fundamental period.
- **Cycle Length ($M$)**: Defined as $f_s / f_0 = 16000 / 50 = 320$ samples.
- **Window Count ($N$)**: For 6 seconds at 16kHz, we have $96,000 / 320 \approx 300$ windows.

### 2. The Reshape Algorithm (Vectorization)
Instead of a slow `for` loop, we utilize a **2D Matrix Reshape**. This allows the CPU to process all 300 cycles in parallel.

**Step A: Truncation**
We remove partial samples at the end to ensure integer cycles:
$$L_{aligned} = \lfloor \text{len}(v\_t) / 320 \rfloor \times 320$$

**Step B: Matrix Transformation (The -1 Wildcard)**
The 1D array is folded into a 2D matrix using the syntax `reshape(-1, 320)`. This fixes the width to one cycle and automatically calculates the number of rows.

```python
# [300 cycles, 320 points per cycle]
v_cycles = v_t[:L_aligned].reshape(-1, 320)
i_cycles = i_t[:L_aligned].reshape(-1, 320)
```

#### **Detailed Code Breakdown**:
- **`[:L_aligned]` (Truncation)**: This slice removes any partial samples at the very end of the 6s block. Numpy's `reshape` requires the total number of elements to be **exactly divisible** by the target width (320). Truncation prevents "Incompatible Shape" errors.
- **`320` (The Physical Anchor)**: This number is not arbitrary. It represents one 20ms period at 16kHz. By fixing the width at 320, we guarantee that every row starts at the same phase point relative to the previous row.
- **`-1` (The Automatic Window Wildcard)**: This is a dynamic instruction. It tells Numpy: *"I have fixed the cycle width to 320; you figure out how many rows (N) can fit into the total length."* 
    - For 6 seconds, $N = 96,000 / 320 = 300$. 
    - For 3 seconds, $N = 48,000 / 320 = 150$. 
    - This allows the code to work automatically even if the input block is shorter than 6 seconds (e.g., at the end of a dataset).

#### **Dynamic Robustness (Edge Cases)**:
What if the last block of data is shorter than expected?
1.  **Partial Window Handling**: The floor division `// samples_per_cycle` ensures we only process complete 20ms periods. Any remaining fractional samples (less than 320) are ignored by the slice `[:num_cycles*320]`.
2.  **Short Block Handling**: If the block is extremely short (less than 320 samples), `num_cycles` will be `0`. The code includes a guard: `if num_cycles < 1: return res`, which prevents a `ZeroDivisionError` and returns an empty feature set safely.

**Visualization of the Resulting Matrix (`v_cycles`)**:
```text
            [Column 0 ... Column 319]  <- Time within 20ms cycle
Window 1:   [ v[0]    ... v[319]    ]
Window 2:   [ v[320]  ... v[639]    ]
...         [ ...     ... ...       ]
Window 300: [ v[95680]... v[95999]  ]
```
Each **Row** is now an independent 20ms window. This structure allows us to perform "Cycle-Wise" math on all 300 windows simultaneously.

### 3. The Averaging Logic (SNR Improvement)
This matrix structure allows for two distinct calculation modes:
- **Mode 1: Cycle-Wise (Splitting)**: Use `axis=1`. This calculates a value (like RMS or Peak) for every one of the 300 windows individually. **Note**: This mode captures sub-cycle shapes (like Crest Factor) that are invisible to low-frequency sensors.
- **Mode 2: Global Mean (Averaging)**: Use `np.mean(..., axis=0)`. This collapses the 300 windows into one "Average Cycle" with $1/\sqrt{300}$ less noise.

### 4. Theoretical Note: Why HF-Averaging is NOT "Low Frequency"
A common misconception is that averaging 6 seconds of data makes it "low frequency." In NILM, this is false:
- **LF (Low Frequency)**: Sampling at 1Hz only tells you the **Total Weight** of the power. You have already lost the wave shape.
- **HF (High Frequency) Averaging**: We sample at 16,000Hz to see the **Fingerprint** (Harmonics, Peaks). Averaging 300 copies of a fingerprint does not turn it into a weight; it just gives you a **Denoised High-Resolution Fingerprint**.
- **The Proof**: Features like `Crest Factor` or `Harmonics` cannot be calculated from LF data at all. Their existence in our output PROVES that high-frequency information has been preserved and refined.

---

## Layer 1: Time-Domain Features

### 1.1 RMS Values — `I`, `V_rms`

**Formula**:
$$V_{rms} = \sqrt{\frac{1}{M}\sum_{m=0}^{M-1} v[m]^2}, \quad I_{rms} = \sqrt{\frac{1}{M}\sum_{m=0}^{M-1} i[m]^2}$$

**Code** (`get_time_domain_features`):
```python
# 1. Calculate RMS for all 300 cycles in parallel
v_rms_per_cycle = np.sqrt(np.mean(np.square(v_cycles), axis=1))

# 2. Average to get the stable research-grade feature
v_rms = np.mean(v_rms_per_cycle)
```

**Physical Meaning**: The "effective" value of the waveform. `I` (i.e., `I_rms`) is the **#1 ranked feature** in the ELECTRIMACS paper because total current amplitude alone correctly identifies 85% of appliances.

---

### 1.2 Power Triangle — `P`, `Q`, `S`, `PF`

**Formulas** (Eq. 14–16, 21 from ELECTRIMACS):
$$P = \frac{1}{M}\sum_{m=0}^{M-1} v[m] \cdot i[m]$$
$$S = V_{rms} \times I_{rms}$$
$$Q = \sqrt{S^2 - P^2}$$
$$PF = P / S$$

**Code** (`get_time_domain_features`):
```python
# P, Q, S, PF are all calculated cycle-by-cycle (300 times) 
# to capture the true cross-product relationship within each window.
p_active_cycles = np.mean(v_cycles * i_cycles, axis=1)
p_active = np.mean(p_active_cycles)
```

**Physical Meaning**:
- `P` (Active Power): The real energy consumed per second (Watts). Kettle vs. fridge have very different `P`.
- `S` (Apparent Power): What the socket "sees" — the total load on the cable.
- `Q` (Reactive Power): Energy stored and released by inductors/capacitors (motors, fans). `Q` is near zero for heaters but high for washing machines.
- `PF`: How "efficiently" the appliance uses electricity. A heater has `PF ≈ 1.0`, a motor has `PF ≈ 0.7`.

---

### 1.3 Crest Factor — `Fcv`, `Fci`

**Formula** (Eq. 20 from ELECTRIMACS):
$$F_{ci} = \frac{\max|i[m]|}{I_{rms}}, \quad F_{cv} = \frac{\max|v[m]|}{V_{rms}}$$

**Code** (`get_time_domain_features`):
```python
# Implementation Detail (The 300-Cycle Averaging Logic):
# We find the peak for each of the 300 cycles and divide by that cycle's local RMS.
# Result = Mean(Peak_n / RMS_n) for n=1 to 300.
i_peak_cycles = np.max(np.abs(i_cycles), axis=1) # Get peak for each row
cf_i_cycles = i_peak_cycles / (i_rms_cycles + 1e-6)
crest_i = np.mean(cf_i_cycles) # Average across all cycles
```

**Physical Meaning**: Measures the "peakedness" of the signal. 
- **Theory**: CF is the ratio of the peak amplitude of a waveform to its RMS value.
- **Linear Loads** (Heaters): Current is a pure sine wave, so CF is consistently $\approx \sqrt{2} \approx 1.414$.
- **Non-linear Loads** (SMPS, computers): These draw current in short, high-intensity pulses, creating a massive peak but a relatively low RMS, resulting in a high CF (e.g., 3.0+).
- **Why Cycle-Averaging?**: By calculating CF per cycle and then averaging, we prevent a single transient noise spike in the 6s window from falsely skewing the entire signature.

---

## Layer 2: Frequency-Domain (FFT Harmonic Loop, k=1…15)

This is the **core of the ELECTRIMACS method**. For every harmonic order $k$ from 1 to 15, the code extracts four features: `Pk`, `Qk`, `Ik`, `Sk`.

### Step 1: Cycle-Averaging FFT — The "Why k=1..15?" Theory

#### A. The FFT Frequency Decomposition

When you give FFT a signal, it decomposes it into **all possible sine waves** simultaneously. For a signal with $M$ samples at sampling rate $f_s$, the FFT outputs $M/2+1$ frequency bins. The frequency of bin $k$ is:

$$f_k = k \times \frac{f_s}{M}$$

This means the output array contains energy at frequencies: $0, \frac{f_s}{M}, \frac{2f_s}{M}, \frac{3f_s}{M} \dots$

#### B. The "Magic" of Using One Cycle

The key design choice is: we reshape the signal into chunks of exactly **$M = f_s / f_0 = 320$ samples** (one 50Hz cycle). This makes $\frac{f_s}{M} = f_0 = 50$ Hz, so:

$$f_k = k \times 50 \text{ Hz}$$

This means **array index $k$ maps perfectly to the $k$-th harmonic of the mains frequency**:

| Array Index `i_fft[k]` | Physical Frequency | Power System Name |
| :--- | :--- | :--- |
| `k = 0` | 0 Hz | DC offset (should be ≈ 0) |
| `k = 1` | 1 × 50 = **50 Hz** | **Fundamental** (the mains wave itself) |
| `k = 2` | 2 × 50 = **100 Hz** | 2nd harmonic (even) |
| `k = 3` | 3 × 50 = **150 Hz** | 3rd harmonic (odd) |
| `k = 5` | 5 × 50 = **250 Hz** | 5th harmonic (odd) |
| `k = 7` | 7 × 50 = **350 Hz** | 7th harmonic (odd) |
| `k = 15` | 15 × 50 = **750 Hz** | 15th harmonic (limit) |

So `i_fft[3]` is NOT index 3 of an arbitrary array — it is literally **the amplitude and phase of the 150Hz component** of the current signal. No manual frequency selection needed.

#### C. Why Stop at k=15?

Two reasons:
1. **Physical**: Home appliance harmonics decay rapidly with $k$. Above the 15th harmonic (>750Hz), the energy is statistically negligible (the paper verified this empirically).
2. **Mathematical**: Odd harmonics (k=1,3,5,7...) are dominant in symmetric non-linear loads (most home appliances). Even harmonics (k=2,4,6...) appear only in asymmetric loads. The ELECTRIMACS paper shows even-harmonic features are "close to zero" for most appliances.

#### D. Cycle-Averaging Code — Where does k come from?

**Key insight**: `rfft()` does NOT compute one harmonic at a time. It computes **all harmonics simultaneously** and stores them all in one array. The loop over `k` comes **after**, when we simply read from that array.

```
Step A: rfft() on one cycle
    Input:  [320 raw samples of current]
    Output: [161 complex numbers]  ← each slot = one harmonic
              i_fft[0]  = 0 Hz   (DC, should be ≈ 0)
              i_fft[1]  = 50 Hz  ← fundamental (k=1)
              i_fft[2]  = 100 Hz ← 2nd harmonic (k=2)
              i_fft[3]  = 150 Hz ← 3rd harmonic (k=3)
              ...
              i_fft[15] = 750 Hz ← 15th harmonic (k=15)
              ...
              i_fft[160]= 8000Hz ← Nyquist limit

Step B: np.mean(axis=0)
    Average 300 copies of that 161-element array → still 161 elements
    (each slot now has less noise)

Step C: for k in range(1, 16): i_fft[k]
    ← This is the ONLY place k appears.
    ← We are just reading slots 1 through 15 from the already-computed array.
```

**Code** (`get_harmonic_analysis`):
```python
# Step A+B combined: FFT every cycle, then average
v_fft = np.mean(np.fft.rfft(v_cycles, axis=1), axis=0)
i_fft = np.mean(np.fft.rfft(i_cycles, axis=1), axis=0)
# Now i_fft is a 161-element array where i_fft[k] = k-th harmonic

# Step C: Extract harmonics 1..15 by indexing
for k in range(1, 16):
    i_fft[k]  # ← this IS the k-th harmonic (50k Hz component)
    v_fft[k]  # ← voltage k-th harmonic
```

After this, `i_fft[k]` is a **complex number** $= a + bj$ where:
- $|i\_fft[k]|$ = strength of the k-th harmonic current
- $\angle i\_fft[k]$ = phase shift of the k-th harmonic relative to cycle start

### Step 2: The Harmonic Loop (Extracting `I1–I15`, `P1–P15`, `Q1–Q15`, `S1–S15`)

Once the averaged complex arrays `v_fft` and `i_fft` are ready, the engine runs a loop to calculate the four dimensions of every harmonic.

**The Math of Complex Power**:
For each harmonic $k$, the complex power $\mathbf{S}_k$ is:
$$\mathbf{S}_k = V_k \cdot I_k^* = P_k + jQ_k$$
Where:
- $I_k^*$ is the **complex conjugate** of the current harmonic.
- $\text{Re}(\mathbf{S}_k)$ is the Active Harmonic Power ($P_k$).
- $\text{Im}(\mathbf{S}_k)$ is the Reactive Harmonic Power ($Q_k$).

**Why the `scale = 2.0 / M^2`?**
- One $1/M$ comes from the FFT normalization.
- Another $1/M$ comes from the RMS scaling.
- The $2.0$ factor accounts for the energy in the "negative" frequency half of the spectrum (since we use `rfft`).

**Python Implementation** (`get_harmonic_analysis`):
```python
# The 'scale' ensures we convert raw FFT products into physical units (Watts/Vars)
scale = 2.0 / (samples_per_cycle**2)

for k in range(1, 16):  # From fundamental (k=1) up to 15th harmonic
    # 1. Complex cross-product to get active and reactive components
    # v_fft[k] * np.conj(i_fft[k]) gives the phase-aware power
    S_k_complex = v_fft[k] * np.conj(i_fft[k]) * scale

    pk = np.real(S_k_complex)           # → 'P1', 'P2', ..., 'P15'
    qk = np.imag(S_k_complex)           # → 'Q1', 'Q2', ..., 'Q15'
    sk = np.abs(S_k_complex)            # → 'S1', 'S2', ..., 'S15'
    vk = np.abs(v_fft[k]) / np.sqrt(2)  # → 'V1', 'V2', ..., 'V15'
    
    # 2. Harmonic Current Magnitude (RMS)
    # i_mags[k] was calculated as np.abs(i_fft[k]) * (2.0 / samples_per_cycle)
    ik = i_mags[k] / np.sqrt(2)         # → 'I1', 'I2', ..., 'I15'
    
    # Save to feature dictionary
    f[f'P{k}'], f[f'Q{k}'], f[f'I{k}'], f[f'S{k}'], f[f'V{k}'] = pk, qk, ik, sk, vk
```

**Physical Meaning per feature**:
- **`I3`, `I5`**: Non-linear loads (like LED bulbs or laptop chargers) create massive spikes at these odd frequencies. This is a primary "fingerprint".
- **`P3`, `Q7`**: Many motor-driven loads (fridges, fans) shift the phase of their harmonics. By calculating $P$ and $Q$ at *each* harmonic, we capture the "lag" or "lead" of every frequency component, allowing for near-perfect appliance separation.
- **`V1–V15`**: While voltage is grid-dependent, harmonic voltage drops can reveal high-current load behavior during transients.

---

## Layer 3: Aggregate Distortion Metrics

After processing individual harmonics, the code calculates the "big picture" of signal pollution.

### 3.1 Harmonic Aggregates — `IH`, `PH`, `QH`, `SH`, `VH`

**Formulas** (Eq. 8, 12, 13 from ELECTRIMACS):
$$I_H = \sqrt{\sum_{k=2}^{15} I_k^2}, \quad P_H = \sum_{k=2}^{15} P_k, \quad Q_H = \sum_{k=2}^{15} Q_k$$

**Code**:
```python
# Accumulated inside the loop above for k >= 2:
ih_sq_sum += ik**2
ph_sum    += pk
qh_sum    += qk

# After loop:
f['IH'] = np.sqrt(ih_sq_sum)   # Total harmonic current
f['PH'] = ph_sum                # Total harmonic active power
f['QH'] = qh_sum                # Total harmonic reactive power
f['SH'] = f['VH'] * f['IH']    # Total harmonic apparent power
```

**Physical Meaning**: `IH` is how much of the total current comes from harmonics (distortion) rather than the fundamental. A clean heater has `IH ≈ 0`. A switching power supply has high `IH`.

---

### 3.2 Distortion Power — `D`, `DI`, `DV`, `SN`

**Formulas** (Eq. 16–19 from ELECTRIMACS):
$$D = \sqrt{S^2 - P^2 - Q^2}$$
$$D_I = S_1 \times THD_I, \quad D_V = S_1 \times THD_V$$
$$S_N = \sqrt{D_I^2 + D_V^2 + S_H^2}$$

**Code**:
```python
f['D']    = np.sqrt(max(0, S_total**2 - P_total**2 - Q_total**2))  # Distortion power
f['THDI'] = f['IH'] / (f['I1'] + 1e-6)   # Current harmonic distortion rate
f['DI']   = f['S1'] * f['THDI']           # Current distortion power
f['DV']   = f['S1'] * f['THDV']           # Voltage distortion power
f['SN']   = np.sqrt(f['DI']**2 + f['DV']**2 + f['SH']**2)  # Non-fundamental apparent power
```

**Physical Meaning**: 
- `D` (Distortion Power): The "wasted" power caused purely by waveform distortion. High `D` means the appliance is introducing harmonics into the grid.
- `DI`: How much current distortion is damaging the fundamental apparent power.
- `SN`: A comprehensive "how non-linear is this appliance?" score. `DI` is ranked **7th** in ELECTRIMACS top features.

---

## Layer 4: Wavelet-Domain Analysis (DWT)

### 4.1 Deep Dive: Understanding DWT (The Microscope Analogy)

If FFT is like a **Prism** (splitting light into colors), then DWT is like a **Microscope** with multiple lenses of different magnification.

#### A. Level vs. Harmonic (k)
Unlike FFT, which uses infinite sine waves, DWT uses small, brief waves called **Wavelets**. Instead of harmonic orders $k$, it uses **Levels (Scales)**:

- **High-Magnification Lens (Level 4 - Detail)**: Zooms in on the tiny, high-frequency "shivers" and "sparks" in the current.
- **Low-Magnification Lens (Level 0 - Approximation)**: Looks at the big, slow changes in the signal (the 50Hz bulk).

#### B. Why is "Energy" a Feature?
In the code `np.sum(np.square(c))`, we calculate the **Energy** of each level. 
- A **Heater** is very smooth; it will have almost **zero energy** in Level 4 (the high-freq detail).
- A **Vacuum Cleaner** motor creates massive electrical arcing (sparks) during operation. These sparks show up as huge energy bursts in **Level 4**.

#### C. FFT vs. DWT: The Time-Frequency Trade-off
- **FFT** tells you **"What"** frequencies are there, but it is "blind" to *when* they happen within the 6s window.
- **DWT** tells you **"How much"** high-frequency activity occurred. Because wavelets are local in time, they are the perfect tool for capturing **transients** (the exact moment of switching).

### 4.2 Mathematical Foundation: The Mallat Algorithm

The DWT is computed using the **Mallat Algorithm**, which passed the signal through a series of filters.

#### A. The DWT Formula
The wavelet coefficients $C(j, k)$ are calculated as the inner product of the signal $x(n)$ and the wavelet function $\psi$ at scale $j$ and position $k$:

$$C(j, k) = \sum_{n} x(n) \psi_{j,k}(n)$$

Where energy for each level $j$ is the sum of squared coefficients:
$$E_j = \sum_{k} |C(j, k)|^2$$

#### B. The Filtering Process (Successive Decomposition)
For `level=4`, the signal goes through 4 stages of filtering:
1.  **Stage 1**: Signal $\rightarrow$ [Low-pass] $\rightarrow$ $A_1$; [High-pass] $\rightarrow$ **$D_1$** (Highest freq details)
2.  **Stage 2**: $A_1 \rightarrow$ [Low-pass] $\rightarrow$ $A_2$; [High-pass] $\rightarrow$ **$D_2$**
3.  **Stage 3**: $A_2 \rightarrow$ [Low-pass] $\rightarrow$ $A_3$; [High-pass] $\rightarrow$ **$D_3$**
4.  **Stage 4**: $A_3 \rightarrow$ [Low-pass] $\rightarrow$ **$A_4$** (Lowest freq); [High-pass] $\rightarrow$ **$D_4$**

**Final Features (5 Energies)**:
- `DWT_E0` ($A_4$): The "Average" baseline of the signal.
- `DWT_E1` ($D_4$): Mid-low frequency transients.
- `DWT_E2` ($D_3$): Mid-high frequency transients.
- `DWT_E3` ($D_2$): High frequency transients.
- `DWT_E4` ($D_1$): Ultra-high frequency transients (the "sharpest" spikes).

### 4.3 Python Implementation (`get_wavelet_features`)

```python
# 'db4' stands for Daubechies 4-tap filters
coeffs = pywt.wavedec(i_t, 'db4', level=4)

# pywt.wavedec returns: [A4, D4, D3, D2, D1]
for i, c in enumerate(coeffs):
    f[f'DWT_E{i}'] = np.sum(np.square(c))
```

**Physical Meaning**: Two appliances may have the same steady-state `P` and `I`, but their turn-on transient signatures are unique. `DWT_E4` captures the highest-frequency "jolt" that occurs in the first few milliseconds of switching.

---

## Layer 5: Orchestration & Precision

### 5.1 Orchestration — `compute_hf_features`
The final feature vector is combined and rounded for storage efficiency and data stability.

**Precision Control**:
- **Watts/VARs**: Rounded to 2 decimal places (e.g., `P_active`).
- **Amps/Volts**: Rounded to 4-6 decimal places (e.g., `I_rms`).
- **Ratios/PF**: Rounded to 4 decimal places (e.g., `PF`, `THDI`).

---

---

## Layer 6: Integration with Multi-Domain Diffusion Models

The extracted 80-dimensional feature vector serves as the **Contextual Conditioning Signal** for the Multivariate Diffusion Transformer (DiT).

### 7.1 The Conditioning Mechanism
In the Diffusion process, the model learns to reverse noise into clean appliance power signals. By injecting these multi-domain features:
- **FFT Features** provide the "Steady-State Identity" (e.g., this is a Fridge running).
- **DWT Features** provide the "Event Boundary" (e.g., the Fridge just kicked in at index 450).
- **Distortion Features** provide the "Non-linear Signature" (e.g., this is an LED bulb, not a simple heater).

### 7.2 Why 80 Features?
While 80 features might seem high, the **Transformer architecture** is designed to handle high-dimensional embeddings efficiently. These features allow the model to:
1.  **Disambiguate overlapping loads**: Distinguish between two appliances with similar total power but different harmonic signatures.
2.  **Improve Temporal Precision**: Use wavelet energy to pinpoint exact start/stop times, reducing the "smearing" effect often seen in low-frequency NILM.

---

## Complete Feature Reference Table

| Feature Domain | Key Columns | Source Function | Scientific Role |
| :--- | :--- | :--- | :--- |
| **Time (Basic)** | `V_rms`, `I_rms`, `P_active`, `Q_reactive`, `S_apparent`, `PF` | `get_rms`, `get_power_metrics` | Global energy consumption |
| **Time (Peaks)** | `Fci`, `Fcv` | `get_power_metrics` | Waveform spikiness/sharpness |
| **Harmonics** | `P1..15`, `Q1..15`, `I1..15`, `S1..15`, `V1..15` | `get_harmonic_analysis` (Loop) | Spectral "fingerprint" of the load |
| **Aggregates** | `PH`, `QH`, `IH`, `VH`, `SH` | `get_harmonic_analysis` | Cumulative pollution metrics |
| **Distortion** | `D`, `SN`, `DI`, `DV`, `THDI`, `THDV` | `get_harmonic_analysis` | Non-linearity and grid impact |
| **Transients** | `DWT_E0`, `DWT_E1`, `DWT_E2`, `DWT_E3`, `DWT_E4` | `get_wavelet_features` | Switching-on "spark" signature |

> [!TIP]
> This multi-resolution feature set ensures high discriminative power for modern NILM architectures, particularly for generative models like Diffusion Transformers.

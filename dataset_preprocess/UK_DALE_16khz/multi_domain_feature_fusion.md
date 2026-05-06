# Multi-Domain NILM Data Pipeline: High/Low-Frequency Fusion Strategy

## 1. The Core Philosophy: Feature Extraction vs. Raw Data Retention
The fundamental principle of establishing a Multi-Domain NILM dataset is strictly adhering to **Dimensionality Reduction**.
* **The Problem:** Capturing High-Frequency (HF) data at $16\text{ kHz}$ generates $96,000$ raw electrical sampling points every $6$ seconds. Directly pushing millions of sequential `.flac` points into Deep Learning arrays (like NILMFormer or CNNs) will immediately trigger the Curse of Dimensionality and massive Out-Of-Memory (OOM) failures.
* **The Solution:** We utilize an *offline feature extractor*. Instead of storing the $96,000$ points, the data pipeline runs signal processing math over the waveform to compress it into $5$ to $10$ highly-descriptive, scalar physical constants (features).
* **Crucial Step:** Once the high-frequency features are extracted, the raw $96,000$ coordinate points are permanently **discarded from RAM/DataFrames**. This shrinks terabytes of audio data down to highly manageable megabyte-sized tabular datasets.

---

## 2. Universal Time Alignment (The 6-Second Clock Grid)
**Challenge:** Disparate resolutions exist across domains and datasets. For instance, UK-DALE Low-Frequency (LF) Mains power samples every $1\text{s}$, UK-DALE submeters sample every $6\text{s}$, and REDD submeters sample every $3\text{s}$.
**Solution:** Establish an absolute global timeline constrained rigidly to a **6-second resolution**.

To match misaligned time readings (e.g., a submeter logging an event at $t=5\text{s}$ against a uniform grid aiming for $t=6\text{s}$), we employ **Pandas Resampling and Forward-Fill (Interpolation) (`.resample('6S').ffill()`)**. 
* If a kettle changes state at `10:00:05`, the interpolation safely carries that exact power signature into the uniform `10:00:06` grid. This seamlessly aligns cross-dataset timestamps without null (`NaN`) conflicts or signal distortion.

---

## 3. The Extraction & Matching Workflow (SOP)
The full sequential methodology operates as follows:

1. **Establish the Base Window:** Define a universal $6$-second epoch (e.g., From `10:00:00` to `10:00:06`).
2. **Capture Macroscopic (LF) Data:** Read the interpolated Low-Frequency Average Power metrics (W and VA) for Mains, along with the Ground Truth submeter state for that identical epoch.
3. **Capture & Slice Microscopic (HF) Target:** Utilize the UNIX timestamp to locate the specific `.flac` file block. Extract the precise temporal slice containing $96,000$ raw voltage/current values.
4. **Execute Feature Engineering Engine:** Run mathematical transformations (e.g., Fast Fourier Transform) across the slice to generate representative features.
5. **Output the Multi-Domain Feature Row:** Compile macroscopic and microscopic observations into a single tabular array element:
<br>

| Unix Timestamp | LF Mains (W) | Submeter (Ground Truth) | HF Feature: 3rd Harmonic | HF Feature: Crest Factor |
| :--- | :--- | :--- | :--- | :--- |
| `1374447600` | $2540\text{ W}$ | `Kettle: ON` | `0.85` | `1.41` |

---

## 4. Recommended High-Frequency Descriptive Features
To successfully squish a dense electrical sine-wave into powerful NILM fingerprints, the framework should calculate the following steady-state and transient characteristics:

* **Current Harmonics (FFT Analysis):** Extracting the amplitude coefficients of the fundamentally odd electrical harmonics (principally the 3rd, 5th, and 7th). Rectified switching power supplies (Laptops/TVs) produce extremely noisy harmonic combinations, while pure resisters (Kettles) are heavily isolated to the fundamental 1st frequency.
* **Total Harmonic Distortion (THD):** A singular mathematical constant summarizing how significantly a device warps the perfect $50\text{Hz}$ sinusoidal current.
* **V-I Trajectory Features (Phase Shift):** Utilizing the lag between Voltage and Current zero-crossings. You can capture the 'Area Enclosed' within the V-I curve array to inherently split devices into highly inductive, capacitive, or purely resistive profiles immediately.
* **Crest Factor & Form Factor:** Captures the "peakiness" of the waveform. Helpful in identifying internal capacitor smoothing behavior common in modern consumer electronics.

---

## 5. Visual Architecture (Flowchart)

```mermaid
%%{init: {'theme': 'default', 'themeVariables': {'background': '#ffffff', 'primaryColor': '#ffffff', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#f4f4f4', 'lineColor': '#333', 'fontFamily': 'arial'}}}%%
flowchart TD
    classDef lowFreq fill:#e8f4f8,stroke:#0277bd,stroke-width:2px,color:#000
    classDef highFreq fill:#fbe9e7,stroke:#d84315,stroke-width:2px,color:#000
    classDef process fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#000
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef output fill:#ede7f6,stroke:#512da8,stroke-width:2px,color:#000

    %% Phase 1: Raw Data Loading
    subgraph Phase1 [Phase 1: Raw Data Loading]
        direction TB
        LF_Mains[Low-Frequency Mains<br>Active Power 1 Hz .dat]:::lowFreq
        LF_Sub[Low-Frequency Submeters<br>Appliance Status 1/6 Hz .dat]:::lowFreq
        HF_Raw[High-Frequency V-I<br>Raw Audio 16 kHz .flac]:::highFreq
    end

    %% Phase 2: Time Alignment
    subgraph Phase2 [Phase 2: Global 6-Second Time Alignment]
        direction TB
        Sync[Create Global Unified Timeline<br>Strict 6s Epochs]:::process
        Interpolate[Forward-Fill & Resample<br>Align Irregular Timestamps]:::process
        LF_Aligned[Aligned LF Feature Matrix<br>Timestamp | Mains W | Status]:::lowFreq
        
        Sync --> Interpolate --> LF_Aligned
    end

    %% Phase 3: High-Frequency Extraction
    subgraph Phase3 [Phase 3: High-Frequency Feature Extraction]
        direction TB
        MatchTime[Pass 6s Timestamps to Audio Slicer]:::process
        SliceHF[Extract 6s Audio Slice<br>96,000 coordinate points]:::process
        MathEngine[Feature Extraction Math Engine<br>FFT, THD, V-I Trajectory]:::process
        HFScalar[Compact HF Fingerprints<br>H3, H5, Crest Factor, Area]:::highFreq
        
        MatchTime --> SliceHF --> MathEngine --> HFScalar
    end

    %% Phase 4: Dimensionality Reduction & Fusion
    subgraph Phase4 [Phase 4: Fusion & Dimensionality Reduction]
        direction TB
        Combine[Feature Vector Assembly<br>Concat LF + HF Features]:::process
        DropRaw[Delete Raw 96k Waveform<br>Drop .flac to free memory]:::decision
        FinalMatrix[(Enhanced Dataset Matrix<br>~10 MB / Week .csv)]:::output
        
        Combine --> DropRaw --> FinalMatrix
    end

    %% Connections
    LF_Mains & LF_Sub --> Sync
    LF_Aligned -. "Triggers matching<br>timestamp search" .-> MatchTime
    HF_Raw --> SliceHF
    
    LF_Aligned --> Combine
    HFScalar --> Combine
```


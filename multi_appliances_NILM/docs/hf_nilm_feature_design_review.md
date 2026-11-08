# High-Frequency NILM for Appliance Power Estimation: Feature Preparation and Input Design

Literature review compiled 2026-10-30. Scope: papers that use high-frequency (HF, ≥ 1 kHz) voltage/current to estimate appliance power or state, with emphasis on (1) how raw waveforms are turned into model inputs, (2) how input channels are designed, and (3) how outputs and labels are structured. Details are taken from the papers' own method sections; items that could not be confirmed are marked.

---

## 1. Paper-by-paper summary

### 1.1 Power estimation (regression) on UK-DALE / REDD HF data

| Paper | Data and label rate | Preprocessing and framing | Input representation and channels | Model and output |
|---|---|---|---|---|
| Schirmer & Mporas 2021, *IEEE TETCI* ([10.1109/TETCI.2021.3086226](https://doi.org/10.1109/TETCI.2021.3086226)) | UK-DALE House 1, 16 kHz V/I, 1–7 Dec 2014; appliance labels 1 Hz | Frames of W = 16,000 samples (1 s), one label per frame; spectrogram sub-frames of 320 samples (one 50 Hz cycle), 50 % overlap | Baselines: current, voltage, or V-I spectrograms. Proposed: Double Fourier Integral Analysis (DFIA), built from the product of current and voltage samples in the frame and then a 2D Fourier transform; magnitude and phase used as channels. DFIA is resized to spectrogram size, reducing voltage harmonics more than current harmonics because voltage varies less | 2D CNN (3 conv layers, 8 filters, kernel 3), **one model per appliance**, 8 appliances; output bounded by aggregate power. +7.2 % accuracy over spectrograms |
| Schirmer & Mporas 2022, *Sensors* 22:7200 ([10.3390/s22197200](https://doi.org/10.3390/s22197200)) | REDD Houses 3 and 5 HF (16.5 kHz) + AMPds2; appliance labels down-sampled to 1 min | Median filter (length 21); HF down-sampled ×5 to 3.3 kHz (keeps the first 27 harmonics); HF frame = 55 samples (one cycle); mean–std normalisation from training data | Six W × W images from a (current, voltage) frame pair: V-I trajectory (binary grid), DFIA, PQ plane (outer product of current and voltage), recurrence plot, Gramian Angular Field, Markov Transition Field | CNN feature extractor + DNN regressor. PQ best at low noise and fastest; V-I trajectories most robust under high noise; gains stop increasing above about 1.2 kHz sampling |
| Kolosov, Robinson, Schirmer & Mporas 2025, *Sensors* 25:5280 ([10.3390/s25175280](https://doi.org/10.3390/s25175280)) | REDD House 3 HF; embedded 15 kHz smart-meter prototype | Frame = 275 samples (one REDD cycle) for the HF domain, or 55 samples after ×5 sub-sampling; seq2point buffer of 10 consecutive frames (stride 1) | Per frame and per signal (V, I, P): 9 time-domain statistics (min, max, variance, standard deviation, 25/50/75 % percentiles, kurtosis, skewness) and 9 spectral statistics (amplitude, centroid, spread, skewness, kurtosis, entropy, flatness, crest, slope). Input tensor: Frame × Feature. 5 feature domains × 4 signal sets × 5 normalisations = 100 scenarios | CNN regression (seq2point). Voltage-only features worst (about 65 %); current time–frequency 83.8 %; V+I+P time–frequency best (85.9 %); time-domain power gives the best accuracy-per-parameter |
| Kamyshev et al. 2025, COLD, *IEEE PowerTech* ([10.1109/PowerTech59965.2025.11180295](https://doi.org/10.1109/PowerTech59965.2025.11180295)) | UK-DALE House 1, 16 kHz; 6 s submeter labels | 6 s steady-state segments only: a z-score event detector removes segments after on/off events (possible HF–submeter misalignment); 100k segments; loads < 10 W → "other" | **Current only**: STFT, 20 ms window, 10 ms hop, 50 frequency bins → 600 × 50 spectrogram, decibel scale, global z-score (mean −19.29 dB, std 7.21 dB) | 10 Transformer encoder layers (d = 100); adaptive average pooling collapses time after the first layer (justified by steady state); softmax head outputs **power shares summing to 1** (energy conservation) for 42 devices jointly |
| Moghimian Hoosh et al. 2026, NILMbench, *Energy* ([10.1016/j.energy.2026.141678](https://doi.org/10.1016/j.energy.2026.141678); preprint [10.36227/techrxiv.173834807.74888368/v1](https://doi.org/10.36227/techrxiv.173834807.74888368/v1)) | Preprint: FLEP (synthetic aggregates from PLAID, 30 kHz). Published code: UK-DALE 16 kHz, House 1 → House 2 | FITPS (frequency-invariant transformation of periodic signals): each mains cycle resampled between voltage zero crossings into an L × T matrix (cycles × samples per cycle); UK-DALE frames 6 s = 2 × 96,000 samples plus 11 aggregate-power context samples | Benchmarks five input designs: DFIA magnitude/phase images (Schirmer), Fryze active/non-active distance matrices (Faustine), STFT (COLD), raw current chunks (DeepDFML), colour V-I trajectories (ResNet-18) | All produce softmax power shares; energy conservation checked; outputs below 10 W set to off; modified Jaccard metric (MJ) |
| Lange & Bergés 2018, Variational BOLT, *AAAI* ([10.1609/aaai.v32i1.11342](https://doi.org/10.1609/aaai.v32i1.11342)) | REDD 16 kHz | Instantaneous power waveform per voltage cycle, aligned at voltage zero crossings; waveforms averaged between detected events to reduce sequence length | One vector of N samples per cycle (additive power waveform) | Factorial HMM with a recurrent variational posterior; infers up to 6 of 10 unsupervised components active at once |
| Wu & Wang 2019, *Energies* 12:1572 ([10.3390/en12081572](https://doi.org/10.3390/en12081572)) | UK-DALE Houses 1, 2, 5, 16 kHz down-sampled to 2 kHz (+ BLUED); events located from 6 s submeters | 7 s current window (14,000 samples) around each switching event | Complex STFT spectrogram 224 × 100 × 2 (real and imaginary channels), split at the event into before and after blocks (224 × 50 × 2 each) | "Concatenate CNN" (e.g., DenseNet-121 embedding) contrasting before vs after to suppress background loads; event classification, energy by ON/OFF pairing |

### 1.2 Feature design studies on other HF datasets (useful for channel design)

| Paper | Data | Feature preparation | Model and output |
|---|---|---|---|
| Saraswat, Lundstrom & Salapaka, *Scalable Hybrid Classification-Regression Solution for High-Frequency NILM* (arXiv:2208.10638) | 8-appliance NREL test bed, 200 kHz bandwidth | Per 60 Hz cycle of aggregate current: RMS (1); magnitudes of the fundamental and 3rd–13th odd harmonics, both raw and L2-normalised (14); db7 8-level wavelet coefficients (12 selected); wave-shape ratios between specific waveform points. Normalised harmonics and wavelet/wave-shape features were needed for generalisation | Random forest classifier per load; DNN per load regressing RMS current, with classifier outputs as extra inputs (hybrid); one prediction every cycle |
| *Capturing High-Frequency Harmonic Signatures for NILM: Building a Dataset for Load Disaggregation*, *Sensors* 25:4601 ([10.3390/s25154601](https://doi.org/10.3390/s25154601)) | Own lab dataset, 8 kHz | On-chip DFT over 0.5 s windows (about one sample every 2 s): I_rms, V_rms, power factor, apparent and active power, current harmonic magnitudes h1–h32 (no phase); the group's earlier work found odd harmonics up to the 15th sufficient | Seq2point regression per appliance (RF, CNN, SVR); RF best |
| Faustine & Pereira 2020, *Energies* 13:4154 ([10.3390/en13164154](https://doi.org/10.3390/en13164154)) | PLAID aggregated, 30 kHz | One-cycle current aligned at voltage zero crossing; Fryze decomposition into active current (proportional to voltage) and non-active current (remainder); each converted to a Euclidean distance matrix | 2-channel image → CNN, multi-label classification (softmax). Best model in NILMbench on UK-DALE (as "FaustineCNN") |
| Nolasco, Lazzaretti & Mulinari 2022, DeepDFML-NILM, *IEEE Sensors J.* ([10.1109/JSEN.2021.3127322](https://doi.org/10.1109/JSEN.2021.3127322)) | LIT-SYN, 15,360 Hz | Raw aggregate current, 12,800-sample windows (50 cycles) divided into 5 grid cells (YOLO-style) | 5 × (Conv1D, LeakyReLU, MaxPool) shared trunk; heads for event type, event location, and multi-label loads |
| Hassan, Javed & Arshad 2014, *IEEE TSG* ([10.1109/TSG.2013.2271282](https://doi.org/10.1109/TSG.2013.2271282)) | REDD HF | Events separated, cycle-by-cycle V-I snapshots | Wave-shape features from the V-I trajectory (looping direction, area, asymmetry, mean curve, self-intersection, etc.) compared with PQ and harmonic features; several classifiers |
| Gerasimov et al. 2025, *IEEE PowerTech* ([10.1109/PowerTech59965.2025.11180424](https://doi.org/10.1109/PowerTech59965.2025.11180424)) | PLAID | z-score event detection; one-cycle appliance signature estimated from before/after cycles | Fourier features including harmonic phase cosines; XGBoost with SHAP; classification |

---

## 2. Design patterns across the literature

### 2.1 Four families of input representation

| Family | What the model receives | Papers | Strengths | Weaknesses |
|---|---|---|---|---|
| Raw waveform | 1D current (and voltage) samples | DeepDFML; NILMbench UK-DALE frames | No hand-crafted bias | Very large inputs (96,000 samples per 6 s); needs much data |
| Cycle-aligned waveform | One vector per mains cycle, aligned at voltage zero crossings (FITPS) | V-BOLT, Faustine, NILMbench | Phase-consistent, compact, keeps waveform shape | Loses information between cycles unless stacked |
| 2D image per frame | Spectrogram, DFIA, V-I trajectory, distance matrix, PQ plane, recurrence/GAF/MTF | COLD, Wu & Wang, Schirmer 2021/2022, Faustine, ResNet-18 baseline | Exploits 2D CNNs and Transformers; DFIA and PQ encode V–I interaction | One frame at a time; image size limits temporal context |
| Compact feature vector per frame | Electrical quantities: RMS, power, power factor, harmonic magnitudes (raw and normalised), THD, wavelet and spectral statistics | Saraswat; Kolosov; Sensors 2025 harmonic dataset; Hassan (WS features); Gerasimov | Small, interpretable, cheap on edge devices, allows long sequences | Fine waveform detail lost; feature choice matters |

### 2.2 Channel design findings

| Question | Evidence |
|---|---|
| Voltage, current, or power? | Voltage alone is weakly informative because it is regulated by the utility (Kolosov: about 65 % accuracy vs 82–86 % with current or power). COLD uses current only. |
| Does combining V, I, and P help? | Yes, modestly: Kolosov's best setup combined V, I, and P time–frequency features (85.9 % vs 83.8 % for current alone). |
| How should V–I interaction be encoded? | DFIA (Schirmer 2021) and the PQ plane (Schirmer 2022) multiply current and voltage samples, capturing sideband harmonics and phase relations that separate spectrograms miss. Fryze decomposition (Faustine) splits current into a part in phase with voltage and the remainder, as two channels. |
| Magnitude only or phase too? | Magnitude-only harmonic sets are common (Sensors 2025 hardware gives no phase). Schirmer 2021 uses DFIA magnitude and phase channels; Gerasimov found harmonic-phase cosines among the most important features by SHAP. Wu & Wang keep real and imaginary STFT parts as channels. |
| How many harmonics? | Odd harmonics up to about the 13th–15th are usually sufficient (Saraswat; Sensors 2025 earlier work). Schirmer 2022 keeps 27 harmonics at 3.3 kHz and finds little gain above about 1.2 kHz sampling. |
| Scale of features | L2-normalised harmonics (shape independent of load size) improved generalisation in Saraswat. COLD uses decibel scale plus global z-score. Kolosov compared five normalisations; z-score and no normalisation were generally competitive, while 8-bit scaling hurt some setups. |

### 2.3 Temporal context

| Context length | Papers |
|---|---|
| One cycle (≈ 17–20 ms) | Schirmer 2022, Saraswat, Faustine, V-BOLT (per-cycle observations) |
| ≈ 1 s | Schirmer 2021 (16,000 samples); NILMbench preprint (60 cycles) |
| 6–7 s | COLD (6 s), NILMbench UK-DALE (6 s + about ±30 s aggregate context), Wu & Wang (7 s) |
| Short sequences of frames | Kolosov (10 frames, seq2point); Sensors 2025 harmonic dataset (seq2point windows) |
| Minutes to hours | **None found** in HF NILM; only low-frequency NILM models use long windows |

### 2.4 Output and label design

| Choice | Papers |
|---|---|
| One model per appliance | Schirmer 2021; Saraswat (one DNN per load); Sensors 2025 harmonic dataset |
| Joint output with energy conservation (softmax power shares) | COLD; NILMbench baselines |
| Hybrid: classifier outputs fed to regressor | Saraswat |
| Event classification + ON/OFF pairing for energy | Wu & Wang |
| Label alignment strategy | Schirmer 2021: one label per 1 s frame. COLD: drop segments after events. NILMbench: steady-state training frames, nearest submeter reading within 10 s. Wu & Wang: locate events from submeters, then classify waveform windows |

---

## 3. Implications for this project

Our HF branch computes about 50 interpretable features per 6 s block (e.g., `P_active`, `I_rms`, power factor, `THDI`, harmonic magnitudes, crest factor) and feeds 1024-step sequences to a joint multi-appliance model. Relative to the literature:

| Aspect | Position of this project | Suggested additions supported by literature |
|---|---|---|
| Representation family | Compact feature vector (Section 2.1, family 4), like Saraswat, Kolosov, and the Sensors 2025 harmonic dataset | Cite these as precedent for interpretable compact features on edge-compatible meters |
| Channels | Mostly current and power derived | Add an ablation by signal (current-only vs current + power vs + voltage), following Kolosov |
| Harmonic scaling | Harmonic magnitudes (check whether absolute or relative) | Add L2-normalised harmonics or ratios to the fundamental (Saraswat) |
| Phase information | Not confirmed in current feature list | Add harmonic phase relative to voltage (e.g., cosines), supported by Gerasimov's SHAP results and DFIA/Fryze designs |
| V–I interaction | Power factor captures only the fundamental relation | Consider Fryze active/non-active current RMS, or a small cycle-level V–I summary |
| Temporal context | 1024 steps (≈ 102 min) | This is the clearest gap: no HF NILM paper found uses minute-to-hour context |
| Output | Joint power + state per appliance per step | Compare with softmax energy-conservation heads (COLD/NILMbench) as a baseline or ablation |
| Label alignment | Fix nearest-reading alignment (see `nilmbench_vs_our_hf_pipeline.md`) | Report how transitions are handled, since COLD and NILMbench exclude them |

---

## 4. References

1. P. A. Schirmer and I. Mporas, "Double Fourier Integral Analysis Based Convolutional Neural Network Regression for High-Frequency Energy Disaggregation," *IEEE Trans. Emerg. Topics Comput. Intell.*, 2021. doi:10.1109/TETCI.2021.3086226
2. P. A. Schirmer and I. Mporas, "2D Transformations of Energy Signals for Energy Disaggregation," *Sensors*, 22(19):7200, 2022. doi:10.3390/s22197200
3. D. Kolosov, M. Robinson, P. A. Schirmer, and I. Mporas, "An Instrumental High-Frequency Smart Meter with Embedded Energy Disaggregation," *Sensors*, 25(17):5280, 2025. doi:10.3390/s25175280
4. I. Kamyshev, S. Moghimian Hoosh, D. Kriukov, E. Gryazina, and H. Ouerdane, "COLD: Concurrent Loads Disaggregator for Non-Intrusive Load Monitoring," *IEEE PowerTech*, 2025. doi:10.1109/PowerTech59965.2025.11180295
5. S. Moghimian Hoosh, I. Kamyshev, J. Penuela, F. Mahmood, T. Al-Ansari, and H. Ouerdane, "NILMbench: A novel benchmark for high-frequency NILM regression models," *Energy*, 360, 2026. doi:10.1016/j.energy.2026.141678
6. H. Lange and M. Bergés, "Variational BOLT: Approximate Learning in Factorial Hidden Markov Models with Application to Energy Disaggregation," *AAAI*, 2018. doi:10.1609/aaai.v32i1.11342
7. Q. Wu and F. Wang, "Concatenate Convolutional Neural Networks for Non-Intrusive Load Monitoring across Complex Background," *Energies*, 12(8):1572, 2019. doi:10.3390/en12081572
8. G. Saraswat, B. Lundstrom, and M. V. Salapaka, "Scalable Hybrid Classification-Regression Solution for High-Frequency Nonintrusive Load Monitoring," arXiv:2208.10638.
9. "Capturing High-Frequency Harmonic Signatures for NILM: Building a Dataset for Load Disaggregation," *Sensors*, 25(15):4601, 2025. doi:10.3390/s25154601
10. A. Faustine and L. Pereira, "Multi-Label Learning for Appliance Recognition in NILM Using Fryze-Current Decomposition and Convolutional Neural Network," *Energies*, 13(16):4154, 2020. doi:10.3390/en13164154
11. L. da Silva Nolasco, A. E. Lazzaretti, and B. M. Mulinari, "DeepDFML-NILM: A New CNN-Based Architecture for Detection, Feature Extraction and Multi-Label Classification in NILM Signals," *IEEE Sensors J.*, 22(1):501–509, 2022. doi:10.1109/JSEN.2021.3127322
12. T. Hassan, F. Javed, and N. Arshad, "An Empirical Investigation of V-I Trajectory Based Load Signatures for Non-Intrusive Load Monitoring," *IEEE Trans. Smart Grid*, 5(2):870–878, 2014. doi:10.1109/TSG.2013.2271282
13. G. Gerasimov et al., "Toward Explainable NILM: Real-Time Event-Based NILM Framework for High-Frequency Data," *IEEE PowerTech*, 2025. doi:10.1109/PowerTech59965.2025.11180424

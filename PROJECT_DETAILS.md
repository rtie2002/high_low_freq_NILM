# Project Details: Multi-Domain High & Low Frequency Hybrid NILM

## 1. Project Overview
This project aims to develop a hybrid Non-Intrusive Load Monitoring (NILM) model that combines **low-frequency data** with **high-frequency data**. By leveraging the complementary nature of both data types, the model aims to achieve superior performance in both load disaggregation and state classification compared to models utilizing only low-frequency features.

---

## 2. Key Objectives & Workflow

### Phase 1: Dataset Construction and LF-HF Alignment
The first operational step is to build a reliable fused dataset before model training.
* **Task:** Extract high-frequency (HF) features from UK-DALE 16 kHz voltage-current `.flac` files, align them with low-frequency (LF) aggregate and appliance submeter readings on a fixed 6-second grid, and export one fused CSV per appliance.
* **Goal:** Produce clean training tables containing `readable_time`, LF aggregate power, HF engineered features, appliance power target, and ON/OFF state labels.

### Phase 2: High-Frequency Feature Selection
High-frequency electricity data contains rich signature patterns, but it also spans multiple domains and features (e.g., time domain, frequency domain, time-frequency wavelets, transient features).
* **Task:** Perform rigorous **feature selection** on the extracted HF features to identify the most informative and non-redundant feature subsets for each appliance.
* **Goal:** Reduce dimensionality, remove duplicated/noisy signatures, improve model generalization, and make the final thesis explanation stronger.
* **Selection Strategy:** Use correlation filtering, mutual information ranking, model-based importance, and ablation testing to select stable features for both regression and classification tasks.

### Phase 3: Baseline Models
Before building the final hybrid model, establish clear baselines.
* **LF-only Baseline:** Train a model using only low-frequency aggregate power.
* **HF-only Baseline:** Train a model using only selected high-frequency features.
* **Concat Baseline:** Train a simple model using LF aggregate plus selected HF features.
* **Goal:** Prove whether HF features add measurable value before introducing more complex fusion architectures.

### Phase 4: Multi-Pipeline Model Architecture
Since different high-frequency features originate from different domains (e.g., V-I trajectories, harmonics, high-frequency current waveforms), they possess distinct structural characteristics.
* **Architecture Design:** Create dedicated processing pipelines (or architectural branches, e.g., separate CNN/RNN/Transformer branches) customized for each feature domain.
* **Fallback Strategy:** If separate multi-pipeline processing is computationally unrealistic or overly complex, a unified fusion/concatenation (Concat) approach can be utilized as a baseline.
* **Data Fusion:** Integrate the processed high-frequency feature representations with the low-frequency sequence data before the final task heads.

### Phase 5: Multi-Task Learning (Disaggregation & Classification)
The model must output two distinct predictions simultaneously:
1. **Load Disaggregation (Regression Task):**
   * **Objective:** Predict the continuous power consumption of individual appliances.
   * **Optimization Metric:** Driven by **Mean Absolute Error (MAE)**.
2. **State Classification (Classification Task):**
   * **Objective:** Detect the operational states (e.g., ON/OFF, multi-state) of the appliances.
   * **Optimization Metric:** Driven by **F1-Score**.

---

## 3. Evaluation Baseline
* **Primary Benchmark:** The proposed hybrid model must consistently **outperform (beat)** baseline models that are trained exclusively on **low-frequency features**.
* **Metrics for Comparison:**
  * Lower MAE for disaggregation.
  * Higher F1-Score for state classification.

See `PROJECT_PLANNING.md` for the executable research plan.

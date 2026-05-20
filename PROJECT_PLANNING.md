# Project Planning: Multi-Domain NILM With HF Feature Selection

## 1. Current Project Direction

This project aims to build a hybrid NILM system that combines:

* Low-frequency aggregate power features.
* High-frequency voltage-current features extracted from UK-DALE 16 kHz waveform data.
* Multi-task outputs for appliance power disaggregation and ON/OFF state classification.

The immediate next step is **not** to build a complex deep model first. The first research step after dataset fusion is **high-frequency feature selection**, because the current feature extractor produces many overlapping features.

---

## 2. Why Feature Selection Comes First

The current HF feature engine extracts features from several domains:

* Time-domain power and morphology: `V_rms`, `I_rms`, `P_active`, `S_apparent`, `PF`, `Fcv`, `Fci`.
* Shape statistics: `I_skew`, `I_kurt`, `V_skew`, `I_std`, `V_std`.
* Harmonics: `I1`, `I3`, `I5`, `I7`, `I9`, `I11`, `I13`, `I15`, and voltage equivalents.
* Distortion metrics: `IH`, `VH`, `THDI`, `THDV`.
* Band-power features: `I_BP_low`, `I_BP_mid`, `I_BP_high`, `V_BP_low`.
* Spectral descriptors: `I_spec_entropy`, `I_env_0` to `I_env_7`.
* Time-frequency wavelet features: `DWT_E0` to `DWT_E4`.

Many of these features are likely redundant. For example, `I_rms`, `I1`, `I_BP_low`, and `P_active` may carry strongly overlapping information. Directly feeding all features into the final model may increase noise, overfitting, training cost, and explanation difficulty.

Feature selection should therefore become the first formal research phase.

---

## 3. Literature-Backed HF Feature Selection Strategy

### 3.1 Literature Basis

The feature selection design should be supported by both general machine learning literature and NILM-specific studies.

| Paper | Venue | Feature Selection Method | Useful Idea for This Project |
| :--- | :--- | :--- | :--- |
| Guyon & Elisseeff, "An Introduction to Variable and Feature Selection" | JMLR, 2003 | General filter, wrapper, embedded feature selection taxonomy | Use feature selection to improve prediction, reduce cost, and improve interpretability. |
| Peng, Long & Ding, "Feature Selection Based on Mutual Information..." | IEEE TPAMI, 2005 | mRMR: minimum redundancy, maximum relevance | Main mathematical basis for selecting informative but non-duplicated HF features. |
| Sadeghianpourhamami et al., "Comprehensive feature selection for appliance classification in NILM" | Energy and Buildings, 2017 | Systematic feature elimination | Use all candidate electrical features first, then remove weak features through validation rather than choosing manually. |
| Li et al., "A Feature Engineering-Based NILM Framework..." | IEEJ Transactions on Electrical and Electronic Engineering, 2024 | Two-stage mRMR + Random Forest feature selection | Closest NILM precedent for this project: combine filter-based and model-based feature selection. |
| Souza et al., "Selection of features from power theories to compose NILM datasets" | Advanced Engineering Informatics, 2022 | Collinearity analysis + machine learning validation | Use correlation/collinearity analysis before model training to remove redundant power features. |
| Cannas et al., "Selection of Features Based on Electric Power Quantities for NILM" | Applied Sciences, 2021 | NCA and MRMR ranking, increasing feature-count validation | Evaluate performance by adding features progressively according to ranking. |
| "An Information-Theoretic Analysis of High-Frequency Load Disaggregation" | Entropy, 2026 | mRMR-ranked harmonics + Random Forest regression | Use information-theoretic feature ranking and RF validation for high-frequency NILM. |

### 3.2 Literature Collection Rule

* Download and store only open-access or legally available full-text PDFs.
* For paywalled papers, store only the official DOI, citation, abstract summary, and official URL.
* Do not rely on unofficial redistribution copies when a legal source is unavailable.
* Keep a small bibliography table in the project planning instead of copying long source text.

### 3.3 Core Method Choice

The main method for this project is:

```text
Correlation filtering + mRMR + Random Forest ranking + stability selection + ablation validation
```

This is preferred over metaheuristic feature selection because it is easier to explain, easier to reproduce, and better supported by NILM literature.

GGO, FSFS, genetic algorithms, and deep feature selectors may be kept as optional comparison methods, not the core method.

---

## 4. Planned Research Workflow

### Phase 1: LF-HF Dataset Construction

**Input**

* UK-DALE low-frequency mains and appliance submeter `.dat` files.
* UK-DALE 16 kHz voltage-current `.flac` files.

**Process**

* Extract HF features every 6 seconds.
* Resample LF aggregate and appliance power to the same 6-second grid.
* Fuse both domains by timestamp.
* Save one appliance-specific CSV containing:
  * `readable_time`
  * HF feature columns
  * `aggregate`
  * `{appliance}_power`
  * `on_off`

**Existing Code**

* `dataset_preprocess/high_frequency_data_extract/high_frequency_data_extract.py`
* `dataset_preprocess/high_frequency_data_extract/hf_feature.py`
* `dataset_preprocess/ukdale_processing.py`

**Expected Output**

Fused appliance CSV files such as:

* `kettle_house2_*.csv`
* `fridge_house2_*.csv`
* `washingmachine_house2_*.csv`
* `microwave_house2_*.csv`
* `dishwasher_house2_*.csv`

---

### Phase 2: HF Feature Selection

**Input**

For each appliance CSV:

* `X_hf`: all HF feature columns.
* `y_reg`: appliance power target, e.g. `kettle_power`.
* `y_cls`: appliance state label, `on_off`.

Exclude these columns from HF feature selection:

* `readable_time`
* `aggregate`
* `{appliance}_power`
* `on_off`

The feature-selection implementation should produce a reproducible folder per appliance:

```text
feature_selection_outputs/
  kettle/
    feature_cleaning_report.csv
    correlation_drop_report.csv
    rank_mrmr_cls.csv
    rank_mrmr_reg.csv
    rank_rf_cls.csv
    rank_rf_reg.csv
    stability_report.csv
    selected_features.json
  fridge/
  washingmachine/
  microwave/
  dishwasher/
  selected_features_global.json
```

**Step 2.0: Feature Cleaning**

* Remove constant or near-constant features.
* Remove columns with NaN/Inf values beyond a fixed threshold.
* Replace remaining invalid values using training-fold statistics only.
* Standardize numeric features for mutual information, NCA, Lasso, and distance-based methods.
* Save `feature_cleaning_report.csv` with original feature count, removed columns, retained columns, and reason for removal.

Default thresholds:

```text
near_constant_variance_threshold = 1e-8
max_invalid_ratio = 0.05
```

**Step 2.1: Correlation and Collinearity Filtering**

* Compute absolute Pearson and Spearman correlation among HF features.
* If either `abs(Pearson) > 0.95` or `abs(Spearman) > 0.95`, treat the pair as redundant.
* Keep the feature with higher target relevance; if target relevance is similar, keep the more physically interpretable feature.
* Save `correlation_drop_report.csv` with dropped feature, retained feature, correlation value, and reason.

Expected redundancy groups to check carefully:

* `I_rms`, `I_std`, `I1`, `I_BP_low`
* `V_rms`, `V_std`, `V1`, `V_BP_low`
* `P_active`, `S_apparent`, `PF`
* `THDI`, `IH`, higher-order harmonic magnitudes
* `I_env_*` versus broad band-power features

**Step 2.2: mRMR Ranking**

Compute separate mRMR rankings for both tasks:

* Classification: maximize mutual information with `on_off`, penalize redundancy with already selected features.
* Regression: maximize mutual information with `{appliance}_power`, penalize redundancy with already selected features.

Selection score:

```text
score(f) = relevance(f, target) - mean(redundancy(f, selected_features))
```

Outputs:

* `rank_mrmr_cls.csv`: features ranked for ON/OFF state classification.
* `rank_mrmr_reg.csv`: features ranked for appliance power regression.

Each ranking file should include:

```text
feature, rank, relevance_score, redundancy_penalty, final_score, feature_domain
```

**Step 2.3: Random Forest Importance**

Use simple, explainable models to estimate feature importance:

* `RandomForestClassifier` for `on_off`.
* `RandomForestRegressor` for appliance power.
* Record impurity importance and permutation importance.
* Prefer permutation importance for final reporting because it is less biased toward high-cardinality or high-variance features.

Outputs:

* `rank_rf_cls.csv`
* `rank_rf_reg.csv`

Each ranking file should include:

```text
feature, rank_impurity, impurity_importance, rank_permutation, permutation_importance, permutation_std, feature_domain
```

Default RF settings:

```text
n_estimators = 300
max_depth = None
min_samples_leaf = 2
class_weight = "balanced" for classification
random_state = 42
```

**Step 2.4: Multi-Task Feature Union**

Because the project has both power regression and ON/OFF classification targets, feature selection must not optimize only one target.

Create:

```text
F_cls = top_k features from fused mRMR_cls + RF_cls ranking
F_reg = top_k features from fused mRMR_reg + RF_reg ranking
F_final = union(F_cls, F_reg)
```

Default:

```text
k = 15 per task
target_final_feature_count = 20 to 30
```

If `F_final` exceeds 30 features, reduce by rank-fusion score and stability frequency.

Rank-fusion score:

```text
fused_rank_score = 0.35 * normalized_mrmr_rank
                 + 0.35 * normalized_rf_permutation_rank
                 + 0.20 * normalized_target_relevance_rank
                 + 0.10 * normalized_domain_priority_rank
```

Domain priority should not hard-code a result; it should only break ties in favor of physically meaningful features:

```text
P_active / I_rms / harmonics / THDI / wavelet energy / spectral envelope
```

**Step 2.5: Stability Selection**

Repeat feature ranking across:

* Different appliances.
* Different time-based folds.
* Different days where possible.

Use time-based folds, not random row splits, to avoid leakage from adjacent windows.

Default:

```text
n_time_folds = 5
stability_keep_threshold = 0.60
```

For every feature:

```text
stability_frequency = selected_fold_count / total_fold_count
```

Keep features selected in at least 60% of folds unless ablation proves that a lower-stability feature is important for a specific appliance.

Output:

* `stability_report.csv`
* `selected_features.json`
* `selected_features_global.json`

**Step 2.6: Feature Domain Labels**

Every output report should label each feature domain:

```text
time_domain: V_rms, I_rms, P_active, S_apparent, PF, Fcv, Fci
shape_statistics: I_skew, I_kurt, V_skew, I_std, V_std
harmonics: I1, V1, I3, V3, ... I15, V15
distortion: IH, VH, THDI, THDV
band_power: I_BP_low, I_BP_mid, I_BP_high, V_BP_low
spectral_envelope: I_env_0 ... I_env_7
wavelet: DWT_E0 ... DWT_E4
```

This makes later ablation and thesis explanation easier.

**Step 2.7: Final Selected Feature Sets**

Produce appliance-specific and global selected feature lists:

* `selected_features_kettle`
* `selected_features_fridge`
* `selected_features_washingmachine`
* `selected_features_microwave`
* `selected_features_dishwasher`
* `selected_features_global`

The global feature set should contain features that are repeatedly useful across multiple appliances.

---

### Phase 3: Baseline Experiments

The purpose of this phase is to prove that HF information is useful before building a complex fusion model.

**Baseline A: LF-only**

* Input: low-frequency `aggregate` sequence only.
* Output: appliance power and ON/OFF state.
* Purpose: main benchmark to beat.

**Baseline B: HF-only**

* Input: selected HF features only.
* Output: appliance power and ON/OFF state.
* Purpose: test whether HF signatures alone contain appliance information.

**Baseline C: LF + HF Concat**

* Input: `aggregate` plus selected HF features.
* Output: appliance power and ON/OFF state.
* Purpose: simple fusion baseline.

Expected claim:

```text
LF-only < HF-only or LF+HF concat < multi-branch LF-HF fusion
```

The exact ranking may differ by appliance, but the final hybrid model should beat LF-only in MAE and F1.

---

### Phase 4: Multi-Branch LF-HF Fusion Model

Build a model with separate branches for low-frequency and high-frequency information.

**Recommended Architecture**

```text
LF branch:
  aggregate sequence
  -> TCN / BiLSTM / Transformer encoder
  -> LF embedding

HF branch:
  selected HF feature sequence
  -> MLP / Transformer encoder / grouped feature encoder
  -> HF embedding

Fusion:
  concat(LF embedding, HF embedding)
  -> shared dense layers

Output heads:
  regression head -> appliance_power
  classification head -> on_off
```

**Loss Function**

```text
total_loss = regression_loss + lambda_cls * classification_loss
```

Recommended starting point:

* Regression loss: MAE or Huber loss.
* Classification loss: Binary Cross Entropy.

Optional consistency constraint:

```text
if predicted on_off is near 0, predicted appliance_power should also be near 0
```

---

### Phase 5: Ablation Study

The ablation study is essential for proving the value of each feature domain.

Run experiments with:

* LF only.
* LF + all HF features.
* LF + correlation-filtered HF features.
* LF + mRMR-selected HF features.
* LF + mRMR + RF selected HF features.
* LF + selected HF features.
* LF + time-domain HF features only.
* LF + harmonics only.
* LF + distortion and band-power only.
* LF + spectral envelope only.
* LF + wavelet features only.
* LF + final selected multi-domain feature set.

Report:

* MAE for appliance power disaggregation.
* RMSE and either NDE or CVRMSE for regression robustness.
* F1-score for ON/OFF state classification.
* Precision, recall, and MCC for classification.
* Selected feature count, stability frequency, and feature-selection runtime.

Acceptance criteria:

* `LF + selected HF` must outperform `LF-only` on MAE and/or F1.
* `LF + selected HF` should match or outperform `LF + all HF`.
* If `LF + all HF` wins for a specific appliance, document which appliance needs all-domain features and why.
* The final selected HF set should be smaller than the full HF set unless full-domain retention is empirically justified.

---

## 5. Expected Thesis Contribution

The planned contribution should be framed as:

> A feature-selection-driven multi-domain NILM framework that fuses low-frequency aggregate power with selected high-frequency voltage-current signatures for joint appliance disaggregation and state classification.

The strongest claim should not be only that more HF features improve NILM. The stronger claim is:

> Carefully selected HF features improve NILM more reliably than using all HF features blindly.

---

## 6. Immediate TODO List

1. Confirm that fused LF-HF CSV generation works for all target appliances.
2. Create a legal literature inventory table with DOI, venue, method, and local/open-access status.
3. Create a feature selection script for appliance-specific HF feature ranking.
4. Generate cleaning, correlation, mRMR, Random Forest, and stability reports.
5. Save selected feature lists as reproducible JSON files.
6. Train LF-only baseline.
7. Train LF + all HF baseline.
8. Train LF + selected HF concat baseline.
9. Train multi-branch LF-HF model.
10. Run ablation experiments by HF feature domain.
11. Compare all methods using MAE, RMSE/CVRMSE, F1-score, precision, recall, and MCC.

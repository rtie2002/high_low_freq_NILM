# Project Planning: Multi-Domain NILM With HF Feature Selection

Chinese companion version: `PROJECT_PLANNING_ZH.md`.

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

---

## 7. Detailed Explanation of the Feature Selection Plan

This section explains the idea in plain technical language, with simple mathematical examples. The goal is to make the feature selection stage reproducible and easy to justify in a thesis.

### 7.1 Big Picture

The current HF extractor produces many electrical signatures. Some features describe power level, some describe waveform shape, some describe harmonics, and some describe time-frequency transients.

The problem is:

```text
More features does not always mean better NILM.
```

If many features are duplicated, noisy, or irrelevant, the model may overfit. For example, if `I_rms`, `I_std`, `I1`, and `I_BP_low` all describe nearly the same current magnitude, using all of them can make the model unnecessarily complex.

The goal is to find a compact feature subset:

```text
F_final = useful HF features with high target relevance and low redundancy
```

This selected subset is then fused with low-frequency aggregate power for the final NILM model.

---

### 7.2 Dataset View

For each appliance CSV, the table looks like:

| readable_time | V_rms | I_rms | P_active | I3 | THDI | DWT_E1 | aggregate | kettle_power | on_off |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2013-07-22 01:00:00 | 240.4 | 0.58 | 107.3 | 0.33 | 0.74 | 0.057 | 105.9 | 1.0 | 0 |
| 2013-07-22 01:00:06 | 240.5 | 0.58 | 107.1 | 0.33 | 0.74 | 0.058 | 105.8 | 1.0 | 0 |

For feature selection:

```text
X_hf = [V_rms, I_rms, P_active, I3, THDI, DWT_E1, ...]
y_reg = kettle_power
y_cls = on_off
```

Do not include these as candidate HF features:

```text
readable_time
aggregate
kettle_power
on_off
```

Why exclude `aggregate`? Because `aggregate` is the LF input, not an HF feature. We want to know which high-frequency signatures are useful independently before fusion.

---

### 7.3 Stage 0: Feature Cleaning

Before ranking features, remove obviously bad columns.

#### 7.3.1 Constant or Near-Constant Features

If a feature barely changes, it cannot help the model.

Example:

| window | feature_A |
| ---: | ---: |
| 1 | 0.001 |
| 2 | 0.001 |
| 3 | 0.001 |
| 4 | 0.001 |

Variance:

```text
Var(feature_A) ≈ 0
```

So `feature_A` should be removed.

Default threshold:

```text
near_constant_variance_threshold = 1e-8
```

#### 7.3.2 Invalid Values

If a feature contains too many `NaN`, `Inf`, or invalid values, it is unreliable.

Default:

```text
drop feature if invalid_ratio > 0.05
```

Meaning: if more than 5% of rows are invalid, remove the feature.

Output:

```text
feature_cleaning_report.csv
```

Example report:

| feature | action | reason |
| :--- | :--- | :--- |
| V_rms | keep | valid |
| I_env_7 | drop | invalid_ratio > 0.05 |
| V_skew | drop | near_constant |

---

### 7.4 Stage 1: Correlation and Collinearity Filtering

This step removes duplicated information.

#### 7.4.1 Pearson Correlation

Pearson correlation measures linear relationship:

```text
r(x, y) = cov(x, y) / (std(x) * std(y))
```

If:

```text
abs(r) > 0.95
```

then two features are almost duplicated.

#### 7.4.2 Simple Example

Suppose:

| sample | I_rms | I_std |
| ---: | ---: | ---: |
| 1 | 0.50 | 0.50 |
| 2 | 0.60 | 0.60 |
| 3 | 0.70 | 0.70 |
| 4 | 0.80 | 0.80 |

Then:

```text
corr(I_rms, I_std) = 1.0
```

They carry the same information. Keep only one.

#### 7.4.3 Which One To Keep?

Use this rule:

```text
1. Keep the feature with higher target relevance.
2. If relevance is similar, keep the more physically interpretable feature.
```

Example:

```text
I_rms and I_std are highly correlated.
I_rms has clearer electrical meaning.
Keep I_rms, drop I_std.
```

Expected duplicated groups:

```text
I_rms / I_std / I1 / I_BP_low
V_rms / V_std / V1 / V_BP_low
P_active / S_apparent / PF
THDI / IH / higher-order harmonics
```

Output:

```text
correlation_drop_report.csv
```

Example:

| dropped_feature | kept_feature | pearson | spearman | reason |
| :--- | :--- | ---: | ---: | :--- |
| I_std | I_rms | 0.998 | 0.997 | duplicated current magnitude |
| V_std | V_rms | 0.999 | 0.999 | duplicated voltage magnitude |

---

### 7.5 Stage 2: mRMR Ranking

mRMR means:

```text
minimum Redundancy Maximum Relevance
```

It chooses features that:

```text
1. are relevant to the target
2. are not redundant with already selected features
```

#### 7.5.1 Mutual Information Idea

Mutual information measures how much knowing one variable reduces uncertainty about another variable.

General form:

```text
I(X;Y) = sum_x sum_y p(x,y) log( p(x,y) / (p(x)p(y)) )
```

Interpretation:

```text
I(feature; target) is high
=> feature contains useful information about the target
```

For this project:

```text
I(I3; on_off)
```

means: how much the 3rd harmonic helps identify ON/OFF state.

```text
I(P_active; kettle_power)
```

means: how much active power helps predict kettle power.

#### 7.5.2 mRMR Score

For a candidate feature `f`:

```text
score(f) = relevance(f, target) - redundancy(f, selected_features)
```

More explicitly:

```text
score(f) = I(f; y) - (1 / |S|) * sum I(f; s)
```

Where:

```text
f = candidate feature
y = target, either on_off or appliance_power
S = already selected feature set
I(f; y) = mutual information between feature and target
I(f; s) = mutual information between candidate feature and already selected feature
```

#### 7.5.3 Simple Example

Assume we want to select features for `on_off`.

| feature | MI with on_off | redundancy with selected | mRMR score |
| :--- | ---: | ---: | ---: |
| P_active | 0.80 | 0.00 | 0.80 |
| I_rms | 0.78 | 0.75 | 0.03 |
| THDI | 0.45 | 0.10 | 0.35 |
| DWT_E1 | 0.40 | 0.05 | 0.35 |

Even though `I_rms` has high relevance, it is very redundant with `P_active`. Therefore, mRMR may prefer `THDI` or `DWT_E1` because they add new information.

This is the key reason mRMR is useful.

#### 7.5.4 Two Targets

This project has two targets:

```text
y_cls = on_off
y_reg = appliance_power
```

So produce two rankings:

```text
rank_mrmr_cls.csv
rank_mrmr_reg.csv
```

A feature can be important for classification but less important for regression.

Example:

```text
DWT_E1 may help detect switching transients, so it helps on_off.
P_active may help estimate continuous power, so it helps appliance_power.
```

---

### 7.6 Stage 3: Random Forest Importance

mRMR is a filter method. It looks at statistical dependency before training a complex model. Random Forest is a model-based method. It checks which features are actually useful inside a predictive model.

Use:

```text
RandomForestClassifier -> on_off
RandomForestRegressor  -> appliance_power
```

#### 7.6.1 Why Random Forest?

Random Forest is useful here because:

* It handles nonlinear relationships.
* It works well with tabular feature data.
* It gives feature importance.
* It is easier to explain than a deep neural network.

#### 7.6.2 Impurity Importance

Impurity importance measures how much a feature reduces decision-tree impurity.

But it can be biased, so it should not be the only importance score.

#### 7.6.3 Permutation Importance

Permutation importance is more intuitive.

Idea:

```text
1. Train model normally.
2. Measure validation performance.
3. Shuffle one feature column.
4. Measure how much performance drops.
5. Larger drop = more important feature.
```

Example:

| feature | F1 before shuffle | F1 after shuffle | importance |
| :--- | ---: | ---: | ---: |
| P_active | 0.90 | 0.65 | 0.25 |
| THDI | 0.90 | 0.82 | 0.08 |
| V_rms | 0.90 | 0.89 | 0.01 |

Here, `P_active` is most important.

Outputs:

```text
rank_rf_cls.csv
rank_rf_reg.csv
```

---

### 7.7 Stage 4: Multi-Task Feature Union

Because NILM has two objectives, we should not select features for only one task.

Create:

```text
F_cls = features useful for ON/OFF classification
F_reg = features useful for appliance power regression
F_final = F_cls union F_reg
```

Default:

```text
top_k = 15 features per task
final target = 20 to 30 features
```

#### 7.7.1 Example

Suppose:

```text
F_cls = [DWT_E1, THDI, I3, I_kurt, Fci]
F_reg = [P_active, I_rms, I1, PF, S_apparent]
```

Then:

```text
F_final = [DWT_E1, THDI, I3, I_kurt, Fci, P_active, I_rms, I1, PF, S_apparent]
```

This final set supports both:

```text
state detection + power prediction
```

#### 7.7.2 Rank Fusion

If too many features are selected, combine rankings:

```text
fused_rank_score = 0.35 * normalized_mrmr_rank
                 + 0.35 * normalized_rf_permutation_rank
                 + 0.20 * normalized_target_relevance_rank
                 + 0.10 * normalized_domain_priority_rank
```

Lower score means better rank.

This avoids trusting only one method.

---

### 7.8 Stage 5: Stability Selection

A feature should not be selected only because it worked in one random split.

Use time-based folds:

```text
Fold 1 = day/time block 1
Fold 2 = day/time block 2
Fold 3 = day/time block 3
Fold 4 = day/time block 4
Fold 5 = day/time block 5
```

Do not use random row split, because adjacent NILM windows are highly similar. Random row split can leak near-identical windows into train and test.

#### 7.8.1 Stability Frequency

```text
stability_frequency(feature) = selected_fold_count / total_fold_count
```

Example:

| feature | selected folds | stability |
| :--- | ---: | ---: |
| P_active | 5/5 | 1.00 |
| THDI | 4/5 | 0.80 |
| DWT_E3 | 3/5 | 0.60 |
| V_skew | 1/5 | 0.20 |

Default rule:

```text
keep feature if stability >= 0.60
```

So keep:

```text
P_active, THDI, DWT_E3
```

Drop:

```text
V_skew
```

unless ablation shows it is useful for a specific appliance.

---

### 7.9 Stage 6: Ablation Validation

Feature ranking alone is not enough. The final proof must come from experiments.

Compare:

```text
LF only
LF + all HF features
LF + correlation-filtered HF
LF + mRMR-selected HF
LF + mRMR + RF selected HF
LF + selected time-domain HF only
LF + selected harmonics only
LF + selected spectral envelope only
LF + selected wavelet only
```

The important thesis result should look like:

```text
LF + selected HF > LF only
LF + selected HF >= LF + all HF
```

If this happens, the claim is strong:

```text
Selected HF signatures improve NILM more reliably than blindly using all HF features.
```

#### 7.9.1 Example Result Table

| Method | Feature Count | MAE lower better | F1 higher better |
| :--- | ---: | ---: | ---: |
| LF only | 1 | 42.0 | 0.71 |
| LF + all HF | 52 | 36.5 | 0.78 |
| LF + mRMR HF | 20 | 34.0 | 0.81 |
| LF + mRMR + RF HF | 24 | 32.8 | 0.84 |

This would support the proposed feature selection method.

---

### 7.10 How To Explain This In The Thesis

A concise thesis explanation:

> The high-frequency feature extractor produces a rich but redundant multi-domain representation. To prevent overfitting and improve interpretability, a hybrid feature selection strategy is introduced. First, invalid and collinear features are removed. Second, mRMR is used to select features with high target dependency and low inter-feature redundancy. Third, Random Forest permutation importance validates feature usefulness in nonlinear predictive models. Finally, time-based stability selection and ablation experiments identify appliance-specific and global feature subsets for LF-HF NILM fusion.

The main novelty is not just using feature selection. The stronger idea is:

```text
multi-task, literature-backed, stability-aware feature selection for LF-HF NILM fusion
```

---

# Mini NILM Feature Validation Plan

## 1. Purpose

This document defines the planned mini NILM experiment for validating high-frequency feature selection.

The goal is not only to rank features statistically. The goal is to prove that a selected feature subset can support useful NILM prediction.

The main research question is:

```text
Can selected high-frequency aggregate features support multi-appliance state detection and appliance power estimation?
```

The proposed mini NILM model should predict:

```text
1. which appliances are ON
2. how much power each appliance consumes
```

This is more realistic than training one separate model for each appliance, because real NILM deployment receives one aggregate signal and must infer all active appliances at the same time.

## 2. Motivation

The current high-frequency feature extractor produces many features from each aggregate voltage-current window.

Examples:

```text
P_active
I_rms
S_apparent
PF
I3
I5
THDI
DWT_E1
I_spec_entropy
```

Many features are useful, but many are also redundant. For example:

```text
P_active
I_rms
S_apparent
I1
DWT_E0
```

can all follow the same load magnitude pattern.

Therefore, feature selection is needed to find a compact subset that is:

```text
1. relevant to appliance targets
2. not strongly redundant
3. useful inside an actual NILM model
```

mRMR is useful for the first two points. The mini NILM model is needed for the third point.

## 3. Why A Multi-Appliance Mini NILM Model

A single-appliance model predicts only one target:

```text
aggregate features -> kettle_power
aggregate features -> fridge_power
aggregate features -> microwave_power
```

This is simple, but it does not fully match real NILM deployment.

In deployment, the system should answer:

```text
At this time window, which appliances are ON?
What is the power of each appliance?
```

Therefore, the mini NILM validation model should be multi-appliance:

```text
aggregate HF features -> all appliance states + all appliance powers
```

For our appliance set:

```text
kettle
fridge
microwave
dishwasher
washingmachine
```

the model outputs:

```text
state outputs:
kettle_on, fridge_on, microwave_on, dishwasher_on, washingmachine_on

power outputs:
kettle_power, fridge_power, microwave_power, dishwasher_power, washingmachine_power
```

This is also more efficient because one model shares feature representation across appliances.

## 4. Main Model Design

The planned mini NILM model has two tasks:

```text
Task 1: multi-label ON/OFF classification
Task 2: multi-output power regression
```

The simplest implementation uses two classical models:

```text
Model A: multi-output ON/OFF classifier
Model B: multi-output power regressor
```

The final appliance power prediction is produced by gating the regression output with the ON/OFF output.

Hard gating:

```text
if predicted_on_i == 1:
    final_power_i = predicted_power_i
else:
    final_power_i = 0
```

Soft gating:

```text
final_power_i = P(on_i) * predicted_power_i
```

Soft gating is preferred because it uses the confidence of the ON/OFF detector.

## 5. Architecture Diagram

```text
Aggregate HF feature window
        |
        v
Selected feature subset
        |
        +-----------------------------+
        |                             |
        v                             v
Multi-output classifier        Multi-output regressor
        |                             |
        v                             v
P(on) for each appliance       raw power for each appliance
        |                             |
        +-------------+---------------+
                      |
                      v
        final_power = P(on) * raw_power
                      |
                      v
Multi-appliance NILM prediction
```

## 6. Recommended First Implementation

Use scikit-learn models first because they are easy to debug.

Classifier:

```text
MultiOutputClassifier(RandomForestClassifier)
```

Regressor:

```text
MultiOutputRegressor(RandomForestRegressor)
```

Suggested settings:

```text
RandomForestClassifier:
n_estimators = 200
class_weight = balanced
random_state = 42
n_jobs = -1

RandomForestRegressor:
n_estimators = 200
random_state = 42
n_jobs = -1
```

Reason for using Random Forest first:

```text
1. handles nonlinear feature interactions
2. works with mixed feature scales
3. needs little preprocessing
4. is easier to debug than deep learning
5. is strong enough for feature validation
```

Later, after the experiment is stable, the same design can be upgraded to a neural network:

```text
shared backbone
state detection head
power regression head
```

## 7. Input And Target Definition

Input X:

```text
aggregate high-frequency features
```

Example:

```text
P_active
I_rms
S_apparent
PF
I3
I5
I7
THDI
DWT_E1
I_spec_entropy
```

State target Y_state:

```text
kettle_on
fridge_on
microwave_on
dishwasher_on
washingmachine_on
```

Power target Y_power:

```text
kettle_power
fridge_power
microwave_power
dishwasher_power
washingmachine_power
```

Important:

```text
appliance_power columns are targets, not input features.
```

The model should not use target appliance power as input to predict itself.

## 8. Dataset Strategy

Use two dataset views.

```text
1. ON-buffered data
2. Full timeline data
```

They serve different purposes.

## 9. ON-Buffered Data

Example:

```text
feature_selection/dataset/on_only_wk30_wk31
```

This data contains target appliance ON windows plus a small OFF buffer.

Use this data for:

```text
1. mRMR ranking
2. active-state feature usefulness analysis
3. early mini NILM sanity testing
```

It answers:

```text
When an appliance is active or near active, which features explain its power?
```

Limitation:

```text
It is not enough to prove full NILM deployment behavior.
```

## 10. Full Timeline Data

The full timeline should contain continuous aggregate feature rows.

It includes:

```text
long OFF periods
target appliance ON periods
other appliance activity
overlapping appliance operation
background aggregate variation
```

Use this data for:

```text
1. final mini NILM validation
2. state detection testing
3. false positive OFF testing
4. deployment-style evaluation
```

This is necessary because real NILM sees the full aggregate signal, not only selected ON windows.

## 11. Why Full Data Is Hard

In full timeline NILM, most appliances are OFF most of the time.

This creates imbalance:

```text
OFF samples >> ON samples
```

If this is not handled, the model may learn:

```text
always predict OFF
always predict 0 W
```

This can give good overall MAE but poor NILM behavior.

Therefore, evaluation must separate:

```text
ON performance
OFF performance
overall performance
```

## 12. Feature Selection Strategy

The feature validation should compare multiple feature sets.

Feature sets:

```text
P_active only
all features
mRMR top 3
mRMR top 5
mRMR top 10
mRMR top 15
mRMR top 20
random-k features
correlation top-k features
```

This allows us to check whether mRMR is actually useful.

A good feature subset should:

```text
1. beat P_active only
2. beat random-k features
3. match or approach all features with fewer inputs
4. reduce false positive power during OFF periods
5. produce sensible prediction timelines
```

## 13. Multi-Appliance mRMR Strategy

mRMR is currently appliance-specific.

For a multi-appliance model, we can combine rankings.

Plan:

```text
1. Run mRMR for each appliance power target.
2. Run mRMR for each appliance ON/OFF target if full ON/OFF labels are available.
3. Take top-k features from each appliance/task.
4. Merge them into a global selected feature set.
5. Remove duplicates.
6. Optionally rank by frequency of selection across appliances.
```

Example:

```text
kettle top 10
fridge top 10
microwave top 10
dishwasher top 10
washingmachine top 10
```

Merged set:

```text
global_mrmr_top_features
```

This global set is then used by the multi-appliance mini NILM model.

## 14. Training Pipeline

For each feature set:

```text
1. Load full timeline dataset.
2. Select input feature columns.
3. Build Y_state for all appliances.
4. Build Y_power for all appliances.
5. Split by time.
6. Train multi-output ON/OFF classifier.
7. Train multi-output power regressor.
8. Predict ON probabilities.
9. Predict raw powers.
10. Compute final gated power predictions.
11. Save metrics.
12. Save prediction CSV.
13. Save prediction plots.
```

Time split:

```text
week 30 -> train
week 31 -> test
```

or:

```text
first 70 percent -> train
last 30 percent -> test
```

Avoid using random split as the main result.

## 15. Prediction Formula

Let:

```text
p_on[i] = predicted probability that appliance i is ON
p_raw[i] = predicted raw power for appliance i
```

Then:

```text
p_final[i] = p_on[i] * p_raw[i]
```

Optionally clip negative predictions:

```text
p_final[i] = max(0, p_final[i])
```

If using hard ON/OFF:

```text
p_final[i] = p_raw[i] if p_on[i] >= threshold_i else 0
```

Thresholds can be:

```text
0.5 by default
or tuned on validation data for best F1
```

## 16. Metrics

Report metrics per appliance and averaged across appliances.

State detection metrics:

```text
precision
recall
F1-score
false positive rate
false negative rate
```

Power regression metrics:

```text
MAE_all
RMSE_all
MAE_on
RMSE_on
MAE_off
false_positive_power_off_mean
false_positive_power_off_95th_percentile
R2_on
```

Why separate metrics are needed:

```text
MAE_all can look good if the appliance is OFF most of the time.
MAE_on checks active power estimation.
MAE_off checks false positive power during OFF periods.
F1 checks ON/OFF detection quality.
```

## 17. Baselines

The mini NILM model must be compared against baselines.

Baseline 1:

```text
zero baseline
always predict all appliances OFF and 0 W
```

Baseline 2:

```text
mean baseline
predict mean training power for each appliance
```

Baseline 3:

```text
ON mean baseline
if true ON is known, predict mean ON power
```

This is not deployable but useful as an upper sanity reference for regression.

Baseline 4:

```text
P_active only model
```

Baseline 5:

```text
all features model
```

Baseline 6:

```text
random-k feature model
```

Baseline 7:

```text
correlation top-k feature model
```

The mRMR-selected feature set should perform better than random-k and should approach all-feature performance with fewer features.

## 18. Prediction Visualization

For every important model, save a prediction CSV.

Columns:

```text
readable_time
true_kettle_power
pred_kettle_power
true_fridge_power
pred_fridge_power
true_microwave_power
pred_microwave_power
true_dishwasher_power
pred_dishwasher_power
true_washingmachine_power
pred_washingmachine_power
kettle_on_true
kettle_on_prob
...
```

Plots should show:

```text
true power
predicted power
ON/OFF shading
aggregate P_active
selected features if needed
```

Visual inspection should check:

```text
1. Does the model detect appliance ON events?
2. Does predicted power follow the true power shape?
3. Does the model return to near zero when appliance is OFF?
4. Does the model create false positive power spikes?
5. Does the model simply copy aggregate power?
6. Does the model miss short appliance events?
```

Metrics alone are not enough.

## 19. Trust Criteria

A feature subset is trustworthy only if:

```text
1. it is selected or supported by mRMR
2. it beats random-k subsets
3. it beats simple correlation-only selection
4. it beats P_active-only baseline
5. it approaches all-feature performance with fewer features
6. it gives good ON/OFF F1
7. it gives low MAE_on
8. it gives low false positive power during OFF
9. its prediction timeline is visually sensible
```

If the mini NILM model itself fails the baselines, feature validation should not be trusted.

## 20. Expected Outputs

Recommended output folder:

```text
feature_selection/results/mini_nilm_multi/
```

Files:

```text
metrics_summary.csv
per_appliance_state_metrics.csv
per_appliance_power_metrics.csv
feature_set_comparison.csv
baseline_comparison.csv
```

Prediction files:

```text
predictions/
    mrmr_top10_predictions.csv
    all_features_predictions.csv
    p_active_only_predictions.csv
```

Plots:

```text
plots/
    kettle_true_vs_pred.png
    fridge_true_vs_pred.png
    microwave_true_vs_pred.png
    dishwasher_true_vs_pred.png
    washingmachine_true_vs_pred.png
```

## 21. Experiment Claim Boundary

The correct claim is:

```text
ON-buffered data validates active-state feature usefulness.
Full timeline data validates deployment-style NILM behavior.
The mini NILM model is used only as a validation model after it beats simple baselines.
```

Avoid claiming:

```text
mRMR features are final NILM features
```

unless they pass the full pipeline.

Preferred wording:

```text
The proposed feature-selection pipeline first ranks high-frequency aggregate descriptors using mRMR. Candidate feature subsets are then evaluated using a multi-appliance mini NILM model that jointly performs appliance state detection and power estimation. The selected subsets are compared against zero, mean, P_active-only, random-k, correlation-based, and all-feature baselines. Final feature usefulness is determined from both numerical metrics and true-vs-predicted timeline inspection under full-timeline NILM conditions.
```

## 22. Future Upgrade

After the Random Forest mini NILM is stable, a neural multi-task model can be tested.

Neural design:

```text
input selected HF features
shared MLP or temporal backbone
state detection head
power regression head
```

Loss:

```text
total_loss = alpha * weighted_BCE + beta * weighted_power_MAE
```

This would allow end-to-end multi-task learning.

However, the Random Forest version should be built first because it is easier to debug and better suited for initial feature validation.


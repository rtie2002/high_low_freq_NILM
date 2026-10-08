# Whole-House Data Split: Rationale, Statistics, and House Selection

## 1. Decision summary

This experiment uses a **whole-house split**:

- **Training:** UK-DALE 1; REFIT 3, 5, 9, and 11.
- **Validation:** UK-DALE 5 and REFIT 2.
- **Final test:** UK-DALE 2 and REFIT 20.
- Each training house contributes one continuous 5-week block.
- Each validation and test house contributes its established 8-week block.
- Normalization statistics are fitted on the five training houses only.

"Whole-house" does **not** mean that the model is trained using only one house.
It means that a house is treated as one indivisible domain: windows from the
same house must not be divided between training and validation/test.

## 2. Why a whole-house split is necessary

For house \(h\), the aggregate signal can be written as

\[
x_h(t)=\sum_{i=1}^{5}y_{h,i}(t)+r_h(t),
\]

where \(y_{h,i}(t)\) is the power of target appliance \(i\), and \(r_h(t)\) is
the residual background from unmodelled appliances, standby loads, measurement
noise, and imperfect channel alignment.

### 2.1 Windows from one house are not independent

Randomly splitting windows from the same house creates strong dependence
between training and validation data. Both sides contain the same:

- appliance models and characteristic power levels;
- background appliances and standby load;
- occupancy and usage patterns;
- sensor calibration and missing-data pattern;
- combinations of appliances that commonly operate together.

The validation set can therefore contain a recognisable **house fingerprint**.
A low validation loss may then show that the model remembers the house, rather
than learning appliance patterns that transfer to an unseen house.

### 2.2 It measures the correct generalisation problem

A within-house random split estimates performance when the house has already
been observed:

\[
P(y\mid x,\;h\text{ seen during training}).
\]

A whole-house split instead tests the practical NILM objective:

\[
P(y\mid x,\;h\text{ unseen during training}).
\]

This is the relevant setting for deploying one trained model to a new home.

### 2.3 It exposes background-domain shift

The residual distributions below differ substantially. For example, the mean
residual is 212 W in UK-DALE 1 but 892 W in REFIT 5. If windows from both ends
of one house are placed in train and validation, this domain shift is hidden.
Holding out complete houses makes failures caused by unfamiliar background
power visible.

### 2.4 It supports honest model selection

The validation houses are used for architecture selection, early stopping,
threshold calibration, and hyperparameter selection. The final test houses
must remain untouched until the method is fixed. Otherwise repeated inspection
of UK-DALE 2 or REFIT 20 gradually turns the test set into another validation
set.

### 2.5 Why two validation houses rather than one

One validation house can give a noisy and house-specific conclusion. Two
validation houses provide:

- one held-out UK-DALE domain and one held-out REFIT domain;
- different background distributions;
- protection against choosing a model that works only for a single house.

Model selection should therefore use **house-macro metrics**, followed by the
per-house and per-appliance results. Pooling all validation rows alone can let
the easier or larger house dominate the conclusion.

## 3. Scope and statistical definitions

The tables cover every house currently available in this project with the
required five target-appliance channels:

- UK-DALE houses 1, 2, and 5;
- REFIT houses 2, 3, 5, 9, 11, and 20.

They do not claim to describe every house in the complete public UK-DALE and
REFIT releases. Statistics describe the **selected experimental intervals**,
not the full lifetime of each original recording.

Definitions:

- Sampling interval: 8 seconds.
- Effective days: `number of available rows * 8 / 86,400`. It may be shorter
  than 35 or 56 calendar days because the original recordings contain gaps.
- Residual background:

\[
r_h(t)=\max\!\left(x_h(t)-\sum_{i=1}^{5}y_{h,i}(t),0\right).
\]

- One event: one contiguous run of state label 1. A recording gap starts a new
  sequence, so events are not joined across gaps.
- ON hours: number of samples with state label 1 multiplied by 8 seconds.

## 4. Interval and background statistics

| Dataset | House | Role | Start | End | Rows | Effective days | Mean residual (W) | Median residual (W) | Residual >=800 W (%) |
|:--|--:|:--|:--|:--|--:|--:|--:|--:|--:|
| UK-DALE | 1 | Train | 2013-09-21 | 2013-10-25 | 367,079 | 33.99 | 212.14 | 158.11 | 1.18 |
| REFIT | 3 | Train | 2014-12-01 | 2015-01-04 | 361,139 | 33.44 | 703.69 | 393.00 | 29.33 |
| REFIT | 5 | Train | 2014-12-20 | 2015-01-23 | 373,493 | 34.58 | 891.91 | 577.00 | 28.35 |
| REFIT | 9 | Train | 2014-11-12 | 2014-12-16 | 377,683 | 34.97 | 510.53 | 165.00 | 15.26 |
| REFIT | 11 | Train | 2014-12-30 | 2015-02-02 | 377,651 | 34.97 | 213.30 | 131.00 | 5.76 |
| UK-DALE | 5 | Validation | 2014-07-07 | 2014-08-31 | 597,780 | 55.35 | 434.46 | 321.40 | 8.09 |
| REFIT | 2 | Validation | 2014-08-31 | 2014-10-25 | 590,213 | 54.65 | 294.56 | 112.00 | 5.01 |
| UK-DALE | 2 | Test | 2013-06-07 | 2013-08-01 | 604,131 | 55.94 | 160.24 | 114.45 | 0.33 |
| REFIT | 20 | Test | 2014-05-29 | 2014-07-23 | 599,536 | 55.51 | 280.23 | 219.00 | 3.15 |

### Complete residual-background distribution

| Dataset | House | Role | 0-100 W (%) | 100-200 W (%) | 200-400 W (%) | 400-800 W (%) | >=800 W (%) |
|:--|--:|:--|--:|--:|--:|--:|--:|
| UK-DALE | 1 | Train | 16.75 | 47.76 | 29.34 | 4.98 | 1.18 |
| REFIT | 3 | Train | 0.17 | 14.40 | 36.33 | 19.77 | 29.33 |
| REFIT | 5 | Train | 1.51 | 0.44 | 21.95 | 47.75 | 28.35 |
| REFIT | 9 | Train | 8.26 | 46.73 | 12.27 | 17.48 | 15.26 |
| REFIT | 11 | Train | 38.20 | 36.65 | 12.27 | 7.12 | 5.76 |
| UK-DALE | 5 | Validation | 0.60 | 4.95 | 69.70 | 16.66 | 8.09 |
| REFIT | 2 | Validation | 41.32 | 35.64 | 16.11 | 1.92 | 5.01 |
| UK-DALE | 2 | Test | 39.36 | 36.95 | 20.69 | 2.67 | 0.33 |
| REFIT | 20 | Test | 0.56 | 37.99 | 52.10 | 6.20 | 3.15 |

The training set deliberately contains a broad background range. This is
important, but it does not make train and test distributions identical. The
model must still learn appliance evidence rather than using aggregate level as
a shortcut.

## 5. Appliance event statistics

| Dataset | House | Role | Kettle | Fridge | Dishwasher | Washing machine | Microwave |
|:--|--:|:--|--:|--:|--:|--:|--:|
| UK-DALE | 1 | Train | 169 | 1,785 | 70 | 255 | 173 |
| REFIT | 3 | Train | 288 | 524 | 70 | 73 | 105 |
| REFIT | 5 | Train | 354 | 2,148 | 182 | 230 | 431 |
| REFIT | 9 | Train | 257 | 1,153 | 38 | 13 | 38 |
| REFIT | 11 | Train | 302 | 452 | 7 | 14 | 33 |
| UK-DALE | 5 | Validation | 161 | 2,389 | 78 | 222 | 34 |
| REFIT | 2 | Validation | 608 | 1,133 | 76 | 54 | 161 |
| UK-DALE | 2 | Test | 221 | 1,709 | 46 | 30 | 211 |
| REFIT | 20 | Test | 254 | 1,378 | 28 | 31 | 136 |

### Appliance ON duration

All values are hours.

| Dataset | House | Role | Kettle | Fridge | Dishwasher | Washing machine | Microwave |
|:--|--:|:--|--:|--:|--:|--:|--:|
| UK-DALE | 1 | Train | 5.0 | 347.7 | 18.0 | 50.3 | 4.0 |
| REFIT | 3 | Train | 13.1 | 379.7 | 49.7 | 60.3 | 2.0 |
| REFIT | 5 | Train | 9.7 | 386.8 | 85.5 | 49.2 | 18.1 |
| REFIT | 9 | Train | 10.4 | 394.4 | 70.1 | 16.3 | 1.4 |
| REFIT | 11 | Train | 10.2 | 222.1 | 7.9 | 19.2 | 0.8 |
| UK-DALE | 5 | Validation | 4.7 | 482.1 | 33.1 | 55.7 | 1.0 |
| REFIT | 2 | Validation | 19.0 | 527.3 | 114.2 | 69.9 | 4.1 |
| UK-DALE | 2 | Test | 9.4 | 656.1 | 39.6 | 20.8 | 7.0 |
| REFIT | 20 | Test | 6.9 | 668.6 | 21.2 | 33.2 | 5.5 |

### Split totals

| Role | Rows | Effective house-days | Kettle events | Fridge events | Dishwasher events | Washing-machine events | Microwave events |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Train | 1,857,045 | 171.95 | 1,370 | 6,062 | 367 | 585 | 780 |
| Validation | 1,187,993 | 110.00 | 769 | 3,522 | 154 | 276 | 195 |
| Test | 1,203,667 | 111.45 | 475 | 3,087 | 74 | 61 | 347 |

These totals show why accuracy alone is unsuitable: fridge samples and events
are abundant, while dishwasher, washing-machine, and microwave activity is
sparse and varies strongly by house.

## 6. Why each house was assigned to its role

### Training houses

| House | Reason for inclusion |
|:--|:--|
| UK-DALE 1 | Provides the training-side UK-DALE domain, many fridge cycles, and useful coverage of all five appliances under a relatively moderate background. |
| REFIT 3 | Adds a difficult high-background domain: mean residual 704 W and 29.33% of samples above 800 W. |
| REFIT 5 | Adds the most extreme mean residual, 892 W, while also providing the strongest microwave and dishwasher event coverage among the training houses. |
| REFIT 9 | Adds a different medium/high-background distribution and sparse washing-machine/microwave behaviour. This discourages dependence on one easy activity pattern. |
| REFIT 11 | Adds a lower-background REFIT condition and genuinely rare dishwasher, washing-machine, and microwave activity. It tests whether training remains robust to house imbalance. |

For each training house, candidate 5-week windows were searched every two
days **inside its already established 8-week interval**. A candidate required
at least 80% sample coverage and minimum appliance activity. Among valid
candidates, the selected window maximised the weakest appliance event-coverage
ratio. This prevents the selected five weeks from containing abundant fridge
activity but almost no rare-appliance examples.

### Validation houses

| House | Reason for selection |
|:--|:--|
| UK-DALE 5 | Provides a completely unseen UK-DALE house and a more difficult residual background than UK-DALE 2. It prevents model selection from being based only on a clean UK-DALE condition. |
| REFIT 2 | Provides a completely unseen REFIT house. Its mean residual of 295 W is close to REFIT 20's 280 W, while its detailed residual distribution and appliance behaviour remain different. It is therefore relevant without leaking the final test house. |

The validation set is intentionally neither the easiest nor the most extreme
pair. It contains enough events for every appliance and represents both source
datasets, making it suitable for early stopping and architecture selection.

### Final test houses

| House | Reason for keeping it as test |
|:--|:--|
| UK-DALE 2 | The established unseen UK-DALE test house. It is comparatively clean and tests whether the model preserves waveform quality under a simpler background. |
| REFIT 20 | The established unseen REFIT test house. It has a different appliance population and is the more difficult cross-dataset/cross-house evaluation target. |

These two houses must not be used to choose loss weights, architecture,
thresholds, checkpoints, or post-processing rules.

## 7. Practical evaluation rules

1. Select checkpoints and hyperparameters using validation houses only.
2. Report house-macro AP first, then per-house/per-appliance AP, calibrated F1,
   MAE, ON-MAE, and high-background FPR.
3. Keep UK-DALE 2 and REFIT 20 as separate test scenarios; do not report only a
   pooled score.
4. Compute normalization from training houses only.
5. Preserve `dataset` and `house` columns even in the combined training CSV so
   that batching and error analysis remain traceable by house.
6. Do not compare the new corrected-source results directly with old runs that
   used an earlier prepared dataset without rerunning the baseline.

## 8. Audit sources

All numerical values in this document come from:

- `datasets/mixed_ukdale_refit_5w_house_split/selection_summary.csv`
- `datasets/mixed_ukdale_refit_5w_house_split/selection_meta.json`
- `scripts/prepare_mixed_ukdale_refit_5week_house_split.py`

The generated rows are re-extracted from the current corrected per-house
8-second CSV files. They are not copied from the old prepared mixed dataset.

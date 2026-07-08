# Multi-Appliance NILM Dataset Selection Notes

This note summarizes the 10 multi-appliance NILM papers you provided and turns them into a practical recommendation for **which dataset to choose first** and **which experiment scenarios to design first**.

## Quick recommendation

If your goal is to build a **credible, reproducible multi-appliance baseline** in your current low-frequency pipeline, the safest first choice is:

**1. UK-DALE as the primary benchmark**

- Most papers in your list use `UK-DALE` for multi-appliance work.
- It supports both:
  - `same-house / seen-house` experiments
  - `cross-house / unseen-house` experiments
- It has an extra appliance (`kettle`) that is frequently used in multi-appliance papers, so it is easier to compare with prior work.
- Its 6 s low-frequency setup is common in the literature and easier to align with your current framework than high-frequency event-driven papers.

**2. REDD as the second benchmark**

- REDD is also common, especially for 4-appliance setups:
  `dishwasher`, `fridge`, `microwave`, `washer dryer / washing machine`.
- It is useful for comparison with `MATNilm`, `Conv-NILM-Net`, and several older multi-label papers.
- But it is less stable as a first benchmark because many papers use different house splits, different preprocessing, or only short training windows.

**3. Do not start from high-frequency PLAID-style papers**

- Those papers are useful scientifically, but they are not the best first benchmark for your current low-frequency multi-appliance pipeline.
- They often assume event extraction, current/voltage waveform features, or image-like features from high-frequency measurements.

## Paper-by-paper summary

| Paper | Multi-appliance? | Main task | Dataset(s) | Length / sampling | Cross-house? | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `Multi-Target Energy Disaggregation using Convolutional Neural Networks` | Yes | Multi-target regression | `ENERTALK`, `REDD` | ENERTALK resampled to `1 s`; REDD downsampled to `3 s` | Yes | Uses combined REDD houses `1+2+3` for training and tests on houses `4/5/6`; also cross-domain tests. |
| `On time series representations for multi-label NILM` | Yes | Multi-label classification | `REDD`, `UK-DALE` | Low-frequency; repository indicates `5 min` and `1 h` model-selection settings | Yes | Focus is representation choice rather than a new deep architecture; useful for feature/representation benchmarking. |
| `UNet-NILM_A Deep Neural Network for Multi-tasks Appliances State Detection and Power Estimation in NILM` | Yes | Multi-task state detection + power regression | `UK-DALE` | mains `1 Hz`, appliance `1/6 Hz`, resampled to `6 s`; input window `100` | No | Uses `house 1`, Jan-Mar 2015, 5 appliances; good same-house benchmark, not a strong generalization benchmark. |
| `Variational Regression for Multi-Target Energy Disaggregation` | Yes | Multi-target regression + state prediction | `UK-DALE`, `REFIT` | sampling period `6 s`; input window `200` | Yes | Best paper in your list for explicit benchmark design: same-house, cross-house, and cross-dataset scenarios are all discussed clearly. |
| `A non-intrusive load monitoring system using multi-label classification approach` | Yes | Multi-label state classification | Private Thailand house dataset | `1 min` | No | Good example of low-frequency multi-label classification, but not suitable as your main public benchmark. |
| `Attention-Based Deep Learning Approach for Nonintrusive and Simultaneous State Detection of Multiple Appliances in Smart Buildings` | Yes | Multi-label state detection + derived energy estimation | `REFIT` | `8 s`; best input length around `72` samples (`9.6 min`), tested up to `90` | Yes | Strong modern cross-house paper: train on `Building 2`, test on unseen portion of same building plus unseen `Building 9` and `20`. |
| `Conv-NILM-Net, a causal and multi-appliance model for energy source separation` | Yes | Multi-appliance source separation | `REDD`, `UK-DALE` | REDD at `1 Hz`; UK-DALE at `1/6 Hz`; input is `1 day` (`86400` or `14400` points) | Partly | Multi-appliance regression model; strong for causal real-time source separation, but preprocessing/training setup is unusual. |
| `MATNilm_Multi-appliance-task Non-intrusive Load Monitoring with Limited Labeled Data` | Yes | Multi-task regression + classification with augmentation | `REDD`, `UK-DALE` | REDD input/output `864/64`; UK-DALE `464/64`; REDD sampled at `3 s`, UK-DALE at `6 s` | Yes | Very relevant to your work because it explicitly studies `limited labeled data` and unseen-house testing. |
| `Multi-Label Learning for Appliance Recognition in NILM Using Fryze-Current Decomposition and Convolutional Neural Network` | Yes | Multi-label appliance recognition | `PLAID` | high-frequency `30 kHz`; one-cycle current/voltage event windows | No | High-frequency event-based classification; not aligned with low-frequency continuous disaggregation pipelines. |
| `Multilabel_Appliance_Classification_With_Weakly_Labeled_Data_for_Non-Intrusive_Load_Monitoring` | Yes | Multi-label state classification with weak supervision | `UK-DALE`, `REFIT` | UK-DALE downsampled to `6 s`; REFIT downsampled to `8 s`; bag length `2550` | Yes | Important if you want a weak-label scenario; strong benchmark for unseen-house and mixed-dataset training. |

## What the literature is really telling you

Across these papers, the literature splits into **three different problem families**:

### 1. Low-frequency multi-appliance regression / multitask NILM

Representative papers:

- `UNet-NILM`
- `MATNilm`
- `Variational Regression`
- `Conv-NILM-Net`
- `Multi-Target Energy Disaggregation using CNN`

Common properties:

- Input is aggregate active power.
- Output is appliance power, appliance states, or both.
- Datasets are usually `UK-DALE`, `REDD`, sometimes `REFIT`.
- Sampling is usually `3 s`, `6 s`, or `8 s`.

This family is the **closest match to your current framework**.

### 2. Low-frequency multi-label state classification

Representative papers:

- `Attention-Based Deep Learning...`
- `Multilabel Appliance Classification With Weakly Labeled Data...`
- `A non-intrusive load monitoring system using multi-label classification approach`
- `On time series representations for multi-label NILM`

Common properties:

- Output is mostly `ON/OFF` state instead of full power trace.
- Often easier to optimize than direct regression.
- Good if your first objective is a robust multi-appliance detector rather than full disaggregation.

This family is useful if you want a **simpler first milestone**.

### 3. High-frequency event / waveform classification

Representative paper:

- `Multi-Label Learning for Appliance Recognition in NILM Using Fryze-Current Decomposition and CNN`

Common properties:

- Uses current/voltage waveforms at `30 kHz`.
- Requires event-centric preprocessing and different feature extraction.
- Not directly comparable to the low-frequency disaggregation papers.

This family should be treated as a **separate research track**, not mixed into your first benchmark.

## Best dataset choice for your project

## Option A: Best first benchmark

**Dataset:** `UK-DALE`

**Why**

- Appears repeatedly in the strongest modern multi-appliance papers.
- Supports both multitask regression and multi-label classification baselines.
- Supports `seen-house`, `unseen-house`, and even `cross-dataset` transfer setups.
- Easier to compare against `UNet-NILM`, `MATNilm`, `Variational Regression`, and weak-label papers.

**Suggested appliance set**

- `kettle`
- `fridge`
- `dishwasher`
- `washing machine`
- `microwave`

This is the most common 5-appliance set across the papers.

## Option B: Best second benchmark

**Dataset:** `REDD`

**Why**

- Important for reproducing `MATNilm` and comparing to older baselines.
- Common 4-appliance setup:
  - `dishwasher`
  - `fridge`
  - `microwave`
  - `washer dryer`

**Caution**

- REDD results are often sensitive to house split and preprocessing.
- Some papers use `houses 2-6 train, house 1 test`, while others use very limited training windows or combined-house settings.

## Option C: Add REFIT only after the first baseline is stable

**Dataset:** `REFIT`

**Why**

- Strong for generalization and weak-label evaluation.
- Used by the attention-based and weakly supervised classification papers.

**Caution**

- Good for stress-testing generalization, but not ideal as your first reproducibility target if your current models are still unstable.

## Recommended scenario design

You should not start with only one experiment. The papers suggest a **three-stage benchmark ladder**.

### Scenario 1: Same-house / same-domain baseline

Purpose:

- Check whether the model can learn at all.
- Remove cross-house difficulty from the first debugging stage.

Recommended dataset:

- `UK-DALE house 1`

Recommended split:

- train / val / test on different time ranges from the same house

Why first:

- If this fails, your model or preprocessing is broken.
- Many papers first show strong same-house performance before harder settings.

### Scenario 2: Cross-house within the same dataset

Purpose:

- Measure whether the model generalizes to unseen homes.

Recommended dataset:

- `UK-DALE`: train on `houses 1,3,4,5`; test on `house 2`
- or `REDD`: train on `houses 2-6`; test on `house 1`

Why second:

- This is the most standard and defensible benchmark in the papers.
- It is much more meaningful than same-house only.

### Scenario 3: Limited labeled data

Purpose:

- Match your research direction more closely.
- Test whether multi-appliance modeling helps under label scarcity.

Recommended dataset:

- `REDD` or `UK-DALE`

Recommended design:

- one day labeled training data
- unseen-house test
- compare:
  - full-data baseline
  - one-day baseline
  - one-day + augmentation / weak labels

Why third:

- `MATNilm` shows this is a valid research scenario.
- This is a good place to contribute something new, after you first confirm basic reproducibility.

### Scenario 4: Cross-dataset transfer

Purpose:

- Test robustness, not initial reproducibility.

Recommended datasets:

- train on `UK-DALE`, test on `REFIT`
- optionally train on `UK-DALE`, test on `REDD`

Why last:

- This is the hardest setting.
- Several papers show cross-dataset behavior is much less stable.

## What you should pick first

If you want the **most practical order** for your current project, I recommend:

1. `UK-DALE`, 5 appliances, low-frequency continuous disaggregation/classification
2. Start with `same-house`
3. Then move to `cross-house`
4. Only after that, add `limited labeled data`
5. Use `REDD` as the second dataset for paper comparison
6. Keep `REFIT` for generalization / weak-label experiments

## Final recommendation

If I had to choose **one first benchmark and one first scenario** for your codebase:

- **Dataset first:** `UK-DALE`
- **Scenario first:** `cross-house within UK-DALE`, after a quick same-house sanity run

Reason:

- It is the cleanest compromise between reproducibility, comparability to prior work, and relevance to your current low-frequency multi-appliance framework.
- It avoids the event-based high-frequency branch.
- It matches the strongest modern papers in your list.

## Practical benchmark template

Use this as your first benchmark template:

- **Dataset:** `UK-DALE`
- **Appliances:** `kettle`, `fridge`, `dishwasher`, `washing machine`, `microwave`
- **Sampling:** `6 s`
- **Scenario A:** same-house sanity check
- **Scenario B:** train on seen houses, test on unseen house
- **Metrics:** `MAE`, `SAE`, `F1`
- **Optional classification metrics:** `F1-micro`, `F1-macro`, `exact match`
- **Later extension:** one-day labeled training + augmentation / weak labels

## What not to mix in the first paper-style benchmark

Avoid mixing these into the first comparison:

- `PLAID` high-frequency event data with low-frequency UK-DALE/REDD pipelines
- state-only papers with power-regression papers without clearly separating the task
- same-house and cross-house results in one undifferentiated table
- custom private-house datasets as your main evidence

That usually leads to unfair or confusing conclusions.

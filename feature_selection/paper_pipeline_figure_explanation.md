# Report: How The Paper Designs Its NILM Feature-Selection Pipeline

This report explains the pipeline figure and the related method in:

```text
Comparative Evaluation of Non-Intrusive Load Monitoring Methods
Using Relevant Features and Transfer Learning
```

The explanation follows the paper structure:

```text
Section 3: electrical feature computation
Section 4: feature selection
Section 5: classifier training and evaluation
Section 5.4: transfer learning validation
```

The paper is mainly a **classification** study. It predicts appliance identity:

```text
input waveform/features -> appliance class label
```

It does not predict continuous appliance power. That is an important difference from our current `mRMR.py`, whose default target is `appliance_power`.

---

## 1. Overall Idea Of The Figure

The figure shows a supervised NILM appliance-identification pipeline.

It has two parts:

```text
1. Training step
2. Test and operating step
```

During training, the method has access to:

```text
voltage waveform
current waveform
known appliance class label
```

During testing, the method only receives:

```text
new voltage/current waveform
```

Then it predicts:

```text
appliance class label
```

The full paper pipeline can be summarized as:

```text
voltage/current waveform
-> compute electrical features
-> select useful feature subset
-> train classifier
-> evaluate appliance identification
```

The feature-selection block is the central idea of the paper. The authors want to show that NILM models should not blindly use all extracted features. Instead, a smaller subset of meaningful electrical features can improve accuracy, reduce overfitting, reduce computation cost, and improve interpretability.

---

## 2. Detailed Pipeline Walkthrough Following The Figure

This section follows the exact blocks in the figure and connects each block to the paper section.

The figure is arranged like this:

```text
Training step:

input voltage/current
-> Section 3: computation of p electrical features
-> Section 4: feature selection of d < p optimal features
-> Section 5: class modeling / classifier training
-> trained model

Test and operating step:

new input voltage/current
-> computation of d selected features
-> HEA identification / classifier prediction
-> predicted class label
```

The most important idea is that **Section 3 creates the candidate feature pool**, **Section 4 chooses the smaller useful subset**, and **Section 5 proves whether that subset works in classification**.

---

### 2.1 Training Step, Block 1: Input Voltage And Current Measurements

This block belongs to the beginning of the pipeline before Section 3.

The paper starts from measured high-frequency waveforms:

```text
v(t) = voltage waveform
i(t) = current waveform
```

These waveforms come from individual appliances, not full-house aggregate signals. For example, the waveform may come from:

```text
one fridge
one kettle
one lamp
one microwave
one fan
```

The paper mainly uses **steady-state** waveform periods. That means the appliance has already settled after switching ON. The paper avoids using transient switching behavior as the main signature because transient shape can change depending on external factors.

Example:

```text
If a kettle turns on, the first few waveform cycles may contain switching effects.
After that, the waveform becomes more stable.
The paper extracts features from the stable part.
```

This is the input to Section 3.

---

### 2.2 Training Step, Block 2: Section 3, Computation Of p Electrical Features

This block is labeled in the figure as:

```text
Section 3
Computation of p electrical features
```

Here, the raw waveform is converted into numerical features.

The paper first computes Fourier coefficients from the voltage and current waveform. Then it derives electrical quantities from the fundamental and harmonic components.

The paper computes 90 electrical features in total:

```text
I, I1, I2, ..., I15, IH
P, P1, P2, ..., P15, PH
Q, Q1, Q2, ..., Q15, QH
S, S1, S2, ..., S15, SH, SN
THDI
D, DI, DV
Fp, Fp1, Fp2, ..., Fp15
FCI
```

Feature meaning:

```text
I  = RMS current
Ik = RMS current at harmonic k
IH = total harmonic current

P  = total active power
Pk = active power at harmonic k
PH = total harmonic active power

Q  = total reactive power
Qk = reactive power at harmonic k
QH = total harmonic reactive power

S  = apparent power
THDI = current total harmonic distortion
FCI = current crest factor
```

Then the paper applies the **additivity criterion**. This is still part of Section 3.2.

The additivity criterion asks:

```text
Can this feature from an aggregate signal be interpreted as the sum
of the feature contributions from individual appliances?
```

This matters for NILM because a house-level signal is a mixture:

```text
aggregate = appliance 1 + appliance 2 + appliance 3 + ...
```

If a feature is additive, then when one appliance switches ON, the aggregate feature changes by approximately that appliance's contribution.

The paper finds:

```text
90 computed electrical features
-> 34 additive features kept
```

So in the figure:

```text
p = 34
```

The output of Section 3 is a feature matrix:

```text
X = n samples x p features
```

For the paper:

```text
X = n samples x 34 additive features
```

Example row:

```text
sample 1 = [P, P1, P3, Q, Q1, Q3, PH, QH, ...]
```

The class label for that row might be:

```text
y = microwave
```

---

### 2.3 Training Step, Side Input: Class Labels y

The figure shows class labels entering the feature-selection block.

This means feature selection is supervised for several methods.

The class labels are:

```text
y = known appliance identity
```

Example:

```text
sample 1 -> kettle
sample 2 -> fridge
sample 3 -> microwave
```

These labels allow the feature-selection methods to ask:

```text
Which features separate appliance classes?
```

For Mutual Information, the question is:

```text
How much information does feature P3 contain about the class label?
```

For LDA, the question is:

```text
Does this feature help separate class clusters?
```

For sequential forward selection, the question is:

```text
Does adding this feature improve classifier accuracy?
```

---

### 2.4 Training Step, Block 3: Section 4, Feature Selection Of d < p Optimal Features

This is the central block in the figure.

The input is:

```text
p = 34 additive candidate features
class labels y
```

The output is:

```text
d selected features
```

where:

```text
d < p
```

This means the paper reduces:

```text
34 candidate features
-> smaller selected subset
```

The paper does this because not all 34 additive features are equally useful. Some may be weak, noisy, repeated, or unnecessary for classification.

The paper compares several feature-selection methods:

```text
1. KNN-based sequential forward feature selection
2. LDA-based sequential forward feature selection
3. Mutual Information
4. PCA
5. LDA
6. DNN-based feature scoring
```

Each method produces a different selected subset.

Example from the proposed dataset:

```text
KNN sequential forward FS: 12 selected features
LDA sequential forward FS: 18 selected features
MI: 20 selected features
PCA: 17 selected features
LDA: 14 selected features
DNN: 27 selected features
```

So Section 4 is not one pipeline. It creates several alternative pipelines.

For example:

```text
Pipeline A:
34 additive features -> MI -> 20 selected features

Pipeline B:
34 additive features -> KNN sequential forward FS -> 12 selected features

Pipeline C:
34 additive features -> DNN feature scoring -> 27 selected features
```

These selected feature subsets are then passed to Section 5.

---

### 2.5 Example: How Mutual Information Selects Features

Mutual Information is the closest paper method to our mRMR idea.

It scores each feature according to how informative it is about the class label.

Simple example:

```text
Feature P1:
MI(P1, appliance_class) = high

Feature P14:
MI(P14, appliance_class) = low
```

Then `P1` is ranked higher than `P14`.

The paper sorts features by relevance score and selects the subset before the largest drop in relevance.

In Table 2, on the proposed dataset, MI selects 20 features:

```text
P1, P, P5, Q, Q1, QH, PH,
P7, P15, P9, Q5, Q7, P13, Q3,
Q9, P11, Q13, P3, Q15, Q11
```

Interpretation:

```text
These 20 features carry strong information about appliance class labels.
```

Why many selected features are harmonic features:

```text
Different appliances may have similar total power P,
but different harmonic power patterns.
```

Example:

```text
Appliance A:
P = 800 W, P3 = high, Q5 = high

Appliance B:
P = 820 W, P3 = low, Q5 = low
```

If the classifier only uses `P`, it may confuse A and B.

If it uses:

```text
P, P3, Q5
```

then it can distinguish them better.

---

### 2.6 Example: How Sequential Forward Selection Selects Features

Sequential forward selection is a wrapper method.

It uses classifier performance directly during feature selection.

The algorithm is:

```text
1. Start with no selected features.
2. Try each possible first feature.
3. Pick the feature giving the best classifier accuracy.
4. Keep that feature.
5. Try adding each remaining feature one by one.
6. Pick the next feature that improves accuracy most.
7. Repeat until adding more features no longer improves accuracy.
```

Example:

```text
Start:
S = {}

Try:
S + P1 -> 80% accuracy
S + Q1 -> 76% accuracy
S + P7 -> 78% accuracy

Select:
S = {P1}

Next try:
{P1, Q1} -> 86%
{P1, P7} -> 89%
{P1, Q3} -> 84%

Select:
S = {P1, P7}
```

This continues until accuracy stops improving.

In Table 2, KNN-based sequential forward selection chooses 12 features:

```text
P1, Q1, P7, Q3, Q, P, P3,
PH, P5, Q5, QH, Q9
```

Interpretation:

```text
For KNN classification on the proposed dataset,
these 12 features give the best useful subset before extra features stop helping.
```

This method is classifier-dependent. If the classifier changes, the selected subset can change.

That is why the LDA-based sequential forward method selects a different 18-feature subset.

---

### 2.7 Training Step, Block 4: Section 5, Class Modeling / Classifier Training

After Section 4 selects features, Section 5 trains a classifier.

The input to Section 5 is:

```text
d selected features
class labels y
```

The paper evaluates four classifiers:

```text
KNN
LDA
DNN
Random Forest
```

The classifier learns:

```text
selected features -> appliance class
```

Example:

```text
[P1, P, P5, Q, Q1, QH, PH, ...] -> kettle
[P1, P, P5, Q, Q1, QH, PH, ...] -> fridge
```

This creates many full pipelines.

Example full pipelines:

```text
Section 3: compute 34 additive features
Section 4: MI selects 20 features
Section 5: train KNN classifier
```

```text
Section 3: compute 34 additive features
Section 4: KNN sequential forward FS selects 12 features
Section 5: train Random Forest classifier
```

```text
Section 3: compute 34 additive features
Section 4: DNN feature scoring selects 27 features
Section 5: train LDA classifier
```

The output of this block is:

```text
trained model
```

---

### 2.8 Test And Operating Step, Block 1: New Voltage And Current Measurements

The lower half of the figure is the test-time pipeline.

At test time, the system receives a new voltage/current waveform:

```text
new v(t), new i(t)
```

The true class label is unknown.

The trained system must predict it.

---

### 2.9 Test And Operating Step, Block 2: Computation Of d Selected Features

At test time, the system does not need to use all 34 candidate features.

It uses the same selected feature subset from Section 4.

For example, if MI selected:

```text
P1, P, P5, Q, Q1, QH, PH, ...
```

then the test-time feature vector must contain those same selected features.

This keeps training and testing consistent:

```text
training features = selected d features
testing features = same selected d features
```

This is why the lower block says:

```text
Computation of d selected features
```

instead of:

```text
Computation of p electrical features
```

The benefit is:

```text
less computation
less noisy input
same feature space as the trained classifier
```

---

### 2.10 Test And Operating Step, Block 3: HEA Identification / Classifier Prediction

The final block is:

```text
HEA identification
(classifier prediction)
```

HEA means:

```text
Home Electrical Appliance
```

The trained classifier receives the selected features and outputs:

```text
predicted class label
```

Example:

```text
selected features from unknown appliance
-> trained KNN classifier
-> predicted label = microwave
```

In the figure, the prediction is written as:

```text
y_hat
```

The paper evaluates whether `y_hat` matches the true appliance label.

---

### 2.11 Section 5 Evaluation: How The Paper Checks If The Pipeline Works

Section 5 validates all these pipelines using 8-fold cross-validation.

The dataset is split into 8 parts:

```text
7 parts for training
1 part for testing
```

This repeats 8 times.

The paper measures:

```text
Accuracy
F-measure
Recall
Precision
Accuracy / number of features
```

The metric:

```text
Accuracy / number of features
```

is used because the paper wants high accuracy with a compact feature subset.

Example result:

```text
MI + KNN on PLAID:
20 features
99.13% accuracy
```

This is useful because it is almost as accurate as larger feature subsets but more compact.

---

### 2.12 Where Our mRMR Replaces The Paper Block

Our mRMR belongs exactly in the Section 4 block:

```text
Feature selection of d < p optimal features
```

The paper's Section 4 alternatives are:

```text
MI
PCA
LDA
KNN sequential forward FS
LDA sequential forward FS
DNN feature scoring
```

Our replacement is:

```text
mRMR
```

So our adapted pipeline is:

```text
Section 3 style:
compute HF features

Section 4 replacement:
use mRMR to rank/select features

Section 5 style:
train downstream NILM model and evaluate
```

The key improvement over the paper's MI block is:

```text
MI checks feature relevance only.
mRMR checks relevance and redundancy.
```

Example:

```text
P_active, I_rms, S_apparent, I1
```

may all be individually useful, but they may repeat similar load magnitude information.

mRMR tries to avoid selecting too many repeated magnitude features. It prefers a subset that contains different kinds of information:

```text
magnitude feature
harmonic feature
distortion feature
spectral feature
wavelet feature
```

This is why mRMR is suitable for our HF feature set.

---

## 3. Section 3: Electrical Feature Computation

Section 3 explains the datasets and the electrical features.

The paper uses high-frequency voltage and current measurements from individual appliances.

The raw signals are:

```text
v(t) = voltage waveform
i(t) = current waveform
```

The paper mainly focuses on **steady-state appliance operation**. It does not mainly learn from ON/OFF switching transients.

The reason is that transient signals can be unstable. They may change because of:

```text
switching timing
grid impedance
supply voltage distortion
sampling frequency
appliance switching mechanism
```

So the paper extracts steady-state periods and computes interpretable electrical features from them.

---

## 3. Section 3.1: Datasets Used

The paper uses two datasets.

### 3.1.1 PLAID Dataset

PLAID is a public high-frequency appliance dataset recorded in the USA.

Important details:

```text
sampling rate: 30 kHz
location: Pittsburgh, Pennsylvania, USA
grid: 60 Hz
data type: individual appliance voltage/current measurements
```

The paper uses 11 appliance categories:

```text
air conditioner
compact fluorescent lamp
fridge
hairdryer
laptop
microwave
washing machine
bulb
vacuum
fan
heater
```

After extracting steady-state periods, the paper obtains:

```text
71 appliance classes
36,720 individuals / recordings
```

Here, a class is not just a broad appliance category. It can represent a specific appliance type or brand, such as one specific incandescent light bulb model.

### 3.1.2 Proposed Dataset

The paper also introduces its own public dataset.

Important details:

```text
location: France
grid: 50 Hz
sampling rates: 250 kHz for some recordings, 50 kHz for others
data type: individual appliance voltage/current measurements
```

The proposed dataset contains:

```text
24 appliance categories
35 appliance types
61 appliance classes after considering different power levels
488 individuals / recordings
```

For appliances with different operating modes or power levels, each power level can be treated as a separate appliance class.

Example:

```text
Fan level 1
Fan level 2
Fan level 3
```

may be treated as different classes because their electrical signatures are different.

---

## 4. Section 3.2: From Waveform To 90 Electrical Features

Section 3.2 explains how the paper converts voltage/current waveforms into features.

The paper first computes Fourier coefficients from the sampled waveform. From these coefficients, it computes electrical quantities for the fundamental and harmonic components.

The original feature pool contains:

```text
90 electrical features
```

These features include:

| Feature Family | Meaning |
|---|---|
| `I`, `Ik`, `IH` | RMS current, current harmonics, total harmonic current |
| `P`, `Pk`, `PH` | active power, active harmonic powers, total harmonic active power |
| `Q`, `Qk`, `QH` | reactive power, reactive harmonic powers, total harmonic reactive power |
| `S`, `Sk`, `SH`, `SN` | apparent power and related apparent harmonic powers |
| `THDI` | current total harmonic distortion |
| `D`, `DI`, `DV` | distortion powers |
| `Fp`, `Fpk` | global and harmonic power factors |
| `FCI` | current crest factor |

Example:

```text
P1 = active power at the fundamental harmonic
P3 = active power at the 3rd harmonic
Q5 = reactive power at the 5th harmonic
PH = total harmonic active power
QH = total harmonic reactive power
```

This matters because different appliances create different harmonic patterns.

For example:

```text
a simple resistive heater may be mostly described by fundamental active power
an appliance with power electronics may create stronger odd harmonics
```

So harmonic features can help distinguish appliances that have similar total power but different electrical behavior.

---

## 5. Section 3.2: Additivity Criterion

After computing 90 features, the paper does not use all of them.

It keeps only features that satisfy the **additivity criterion**.

The idea is:

```text
feature(aggregate signal) = sum of feature(individual appliance signals)
```

In NILM, this is useful because real household measurements are aggregate signals:

```text
aggregate current = current from appliance 1 + current from appliance 2 + ...
```

If a feature is additive, then when an appliance turns ON, the aggregate feature changes by approximately the amount contributed by that appliance.

This makes the feature useful for appliance identification from aggregate measurements.

The paper keeps:

```text
34 additive features
```

So the feature pipeline becomes:

```text
raw waveform
-> 90 computed electrical features
-> 34 additive candidate features
```

In the figure notation:

```text
p = 34
```

because `p` is the number of candidate features used for feature selection.

---

## 6. Section 4: Feature Selection

Section 4 is the key section for our work.

The paper defines feature selection as the process of finding a smaller feature subset:

```text
F' subset of F
```

where:

```text
F  = all candidate features
F' = selected feature subset
```

The goal is:

```text
select d useful features from p candidate features
```

with:

```text
d < p
```

Since the paper uses `p = 34` additive features, feature selection chooses a smaller subset such as:

```text
12 features
14 features
20 features
27 features
```

depending on the method.

The reason for feature selection is:

```text
avoid curse of dimensionality
reduce overfitting
remove non-discriminating features
reduce computational cost
keep features interpretable
improve classification performance
```

The paper separates feature-selection methods into two families:

```text
filter methods
wrapper methods
```

---

## 7. Filter Methods In The Paper

Filter methods select features before classifier training.

They score or rank features based on statistical criteria.

They are usually faster because they do not repeatedly train a classifier during feature selection.

The paper uses these filter-style methods:

```text
PCA
LDA
Mutual Information
DNN-based feature scoring
```

### 7.1 Mutual Information

Mutual Information measures how much information a feature contains about the appliance class label.

Conceptually:

```text
high MI(feature, class label)
= feature is useful for distinguishing appliance classes
```

Example:

If `P1` has high MI, it means:

```text
knowing P1 helps identify which appliance class produced the waveform
```

The paper ranks features by relevance score, then selects the subset before the largest drop in relevance.

In the proposed dataset, MI selects 20 features:

```text
P1, P, P5, Q, Q1, QH, PH,
P7, P15, P9, Q5, Q7, P13, Q3,
Q9, P11, Q13, P3, Q15, Q11
```

In PLAID, MI also selects 20 features:

```text
P3, PH, P1, P, Q1, Q, Q9,
Q7, P7, P5, Q5, QH, Q3, P9,
Q11, P11, Q13, Q15, P15, P13
```

Notice that both MI selections include many active and reactive harmonic powers.

This suggests that:

```text
harmonic active/reactive powers carry strong appliance identity information
```

### 7.2 PCA

PCA is a dimensionality-reduction/filter method.

It looks for directions of high variance or dispersion in the feature space.

The paper uses PCA as one compared feature-selection method.

In the proposed dataset, PCA selects 17 features.

In PLAID, PCA keeps 31 features, meaning it removes only:

```text
P14, Q, Q1
```

The interpretation is:

```text
PCA may keep many features if many dimensions contribute to data variance
```

However, PCA is not directly based on appliance labels in the same way as MI.

So a feature can have high variance but not necessarily be the best appliance discriminator.

### 7.3 LDA Feature Selection

LDA is supervised because it uses class labels.

Its goal is to find features that improve class separation.

In the proposed dataset, LDA selects 14 features:

```text
P7, Q, Q1, P9, P3, Q5, PH,
Q7, P11, P5, Q9, P, P1, Q11
```

In PLAID, LDA selects 12 features:

```text
P3, PH, P15, Q13, Q5, QH, Q15,
P9, P7, Q3, Q9, Q7
```

Again, the selected features are strongly related to harmonics.

### 7.4 DNN-Based Feature Scoring

The paper proposes a DNN-based feature-selection method.

The DNN has an input neuron for each feature.

The idea is:

```text
if an input feature has strong learned weights in the first layer,
then the DNN considers it important
```

So the authors train the neural network, then use the first-layer weights as feature relevance scores.

In the proposed dataset, DNN selects 27 features.

In PLAID, DNN selects 17 features.

The DNN method can capture nonlinear relationships, but it may also depend heavily on the amount of training data and training stability.

---

## 8. Wrapper Method: Sequential Forward Selection

Wrapper methods select features by training/evaluating a classifier during feature selection.

The paper proposes a sequential forward feature-selection method.

The idea is:

```text
start with no selected features
try each candidate feature
choose the one that gives the best classifier accuracy
repeat by adding one feature at a time
stop when adding more features no longer improves accuracy
```

The algorithm is:

```text
1. Start with empty selected set S = {}
2. For every candidate feature f not in S:
      test classifier accuracy using S + f
3. Select the feature that gives the highest accuracy
4. Add that feature to S
5. Repeat until accuracy no longer improves
```

The paper tests this wrapper method with:

```text
KNN classifier
LDA classifier
```

So there are two forward-selection pipelines:

```text
KNN-based sequential forward feature selection
LDA-based sequential forward feature selection
```

Example from the proposed dataset:

KNN-based sequential forward selection chooses 12 features:

```text
P1, Q1, P7, Q3, Q, P, P3,
PH, P5, Q5, QH, Q9
```

This means:

```text
using these 12 features gives the best KNN accuracy
before additional features stop helping
```

LDA-based sequential forward selection chooses 18 features:

```text
P1, Q, P, Q9, Q3, P3, P2,
P10, Q4, P4, P6, P9, Q8, P13,
P8, Q15, Q5, Q11
```

This means:

```text
the best feature subset depends on the classifier
```

That is a key idea in wrapper methods.

---

## 9. Section 4.2: How The Paper Decides The Number Of Features

The paper does not use a fixed `top_k` for every method.

For score-based methods such as MI, PCA, LDA, and DNN:

```text
features are sorted by descending relevance score
the subset is selected before the largest drop in relevance score
```

This is why MI selects 20 features in both datasets, while DNN selects 27 in one dataset and 17 in the other.

For sequential forward selection:

```text
features are added until classifier accuracy no longer improves
```

This is why KNN sequential forward selection selects:

```text
12 features in the proposed dataset
25 features in PLAID
```

The selected number is data-dependent and method-dependent.

---

## 10. Section 4.2: What The Selected Features Tell Us

Tables 2 and 3 show that several features appear repeatedly across methods and datasets.

Commonly selected features include:

```text
P
P1
PH
Q
Q1
QH
P3
P5
P7
P9
Q3
Q5
Q7
Q9
Q11
```

The paper observes that:

```text
P, P1, PH, Q, Q1, QH often appear regardless of the feature-selection method
```

This means fundamental active/reactive power and harmonic active/reactive power are stable appliance descriptors.

The paper also observes that MI and LDA often select odd-order harmonic features.

Example:

```text
P3, P5, P7, P9, P11, P13, P15
Q3, Q5, Q7, Q9, Q11, Q13, Q15
```

Why this matters:

Many home appliances contain power electronics or nonlinear loads. These can introduce odd harmonics into current and power waveforms. Therefore, odd-order harmonics can help identify appliance type.

Simple example:

```text
Appliance A and Appliance B may both consume around 1000 W.
If we only use total active power P, they may look similar.
But Appliance A may have strong P3 and Q5,
while Appliance B may have weak harmonics.
Then harmonic features help separate them.
```

This is why feature selection is useful:

```text
it can discover which harmonic descriptors carry discriminative appliance information
```

---

## 11. Section 5: Classifier Training

After feature selection, the paper trains classifiers using the selected feature subsets.

The classifiers are:

```text
KNN
LDA
DNN
Random Forest
```

This creates many pipelines.

Example pipelines:

```text
34 additive features
-> MI feature selection
-> KNN classifier
-> evaluation
```

```text
34 additive features
-> KNN sequential forward feature selection
-> KNN classifier
-> evaluation
```

```text
34 additive features
-> PCA feature selection
-> Random Forest classifier
-> evaluation
```

```text
34 additive features
-> DNN feature scoring
-> DNN classifier
-> evaluation
```

So the paper is not one single model. It is a comparative framework with multiple feature-selection/classifier combinations.

---

## 12. Section 5.2: Evaluation Procedure

The paper uses 8-fold cross-validation.

The data is split into 8 parts:

```text
7 parts for training
1 part for testing
```

This is repeated until every part has been used as the test set.

The split ratio is:

```text
87.5% training
12.5% testing
```

The paper evaluates classification performance using:

```text
Accuracy
F-measure / F1-score
Recall
Precision
Accuracy / number of features
```

The last metric is important:

```text
Accuracy / number of features
```

It rewards compact feature subsets.

Example:

```text
99% accuracy using 20 features
```

may be considered better than:

```text
99.1% accuracy using 34 features
```

if the small subset is easier to compute and interpret.

---

## 13. Section 5.3: Results And Meaning

The paper shows that feature selection improves appliance classification.

For the proposed dataset, the best average F-measure across classifiers is from MI:

```text
MI average F-measure = 92.96%
```

For PLAID, MI is also the best on average:

```text
MI average F-measure = 85.80%
```

This is important for our work because it supports the usefulness of information-based feature selection in NILM.

On PLAID, the best single accuracy reported is:

```text
KNN sequential forward FS + KNN classifier
25 features
99.19% accuracy
```

But MI is very close:

```text
MI + KNN classifier
20 features
99.13% accuracy
```

So MI gives nearly the same accuracy with fewer features.

This supports the idea that:

```text
good feature selection can reduce the feature count
without losing much accuracy
```

---

## 14. Section 5.4: Transfer Learning Test

The paper also tests whether selected features generalize across datasets.

The idea is:

```text
select features on Dataset A
use that selected subset on Dataset B
check classification performance
```

They test both directions:

```text
features selected from proposed dataset -> classify PLAID
features selected from PLAID -> classify proposed dataset
```

This is called cross transfer learning in the paper.

The purpose is to check whether the selected features are not only good for one dataset but also robust across:

```text
different grid frequencies
different recording protocols
different appliance brands
different appliance taxonomies
```

The results show that selected feature subsets can transfer well.

For example:

```text
MI features selected from the proposed dataset
-> PLAID classification with KNN
-> 99.12% accuracy
```

This is strong evidence that selected electrical features can contain general appliance-discriminative information.

---

## 15. How To Understand Tables 2 And 3

Tables 2 and 3 are feature-selection result tables.

They do not show the final classifier performance directly.

They answer:

```text
Which features did each feature-selection method choose?
```

For each dataset, the table gives:

```text
method
number of selected features
selected feature names
```

Example from Table 2:

```text
Method: MI
Number selected: 20
Selected: P1, P, P5, Q, Q1, QH, PH, ...
```

Interpretation:

```text
MI ranked these 20 features as most informative for appliance class labels.
```

Example from Table 3:

```text
Method: KNN sequential forward FS
Number selected: 25
Selected: PH, Q, P1, P3, QH, Q5, P5, ...
```

Interpretation:

```text
When features were added one by one,
these 25 features gave the best KNN classification behavior on PLAID.
```

The tables show that different methods choose different subsets because they optimize different criteria.

---

## 16. Detailed Example: Why Feature Selection Helps

Assume the candidate feature set contains:

```text
P
P1
Q
Q1
P3
Q3
P5
Q5
THDI
```

Two appliances may have similar total active power:

```text
Appliance A: P = 800 W
Appliance B: P = 820 W
```

If the classifier only uses `P`, it may confuse them.

But their harmonics may be different:

```text
Appliance A: high P3, high Q5
Appliance B: low P3, low Q5
```

Then a selected feature subset like:

```text
P, P3, Q5
```

can distinguish them better than:

```text
P only
```

This explains why the paper often selects harmonic active/reactive power features.

---

## 17. Where Our mRMR Fits

Our `mRMR.py` fits into the paper's Section 4 feature-selection block.

In the paper figure, our method replaces:

```text
Feature selection of d < p optimal features
```

The paper compares:

```text
MI
PCA
LDA
Sequential Forward Selection
DNN feature scoring
```

Our method adds:

```text
mRMR
```

So the modified pipeline becomes:

```text
voltage/current windows
-> compute HF features
-> mRMR feature selection
-> selected feature subset
-> train NILM model
-> evaluate
```

---

## 18. Why mRMR Is A Natural Extension Of The Paper

The paper's MI method selects features by relevance:

```text
high mutual information with appliance class label
```

But MI alone does not explicitly check whether selected features repeat the same information.

Example:

```text
P, P1, S, I_rms
```

may all be strongly related to appliance magnitude.

Plain MI may rank several of them highly because each one is individually useful.

mRMR improves this idea by checking:

```text
1. relevance to the target
2. redundancy with already selected features
```

So mRMR asks:

```text
Is this feature useful?
Does it add new information beyond features already selected?
```

This is especially useful in our project because our HF feature list contains many overlapping descriptors:

```text
I_rms
I_std
P_active
S_apparent
I1
DWT_E0
I_BP_low
```

Many of these may describe similar load magnitude information.

mRMR can reduce this duplication.

---

## 19. Paper Pipeline vs Our Current Pipeline

The paper pipeline:

```text
steady-state individual appliance waveform
-> 90 electrical features
-> 34 additive features
-> feature selection
-> appliance classification
```

Target:

```text
y = appliance class
```

Our current pipeline:

```text
ON-period plus two OFF-step buffer
-> HF feature extraction
-> feature selection using mRMR
-> downstream NILM model
```

Current default target:

```text
y = appliance_power
```

So our current method is not identical to the paper.

The paper does:

```text
classification of appliance identity
```

Our current `mRMR.py` mainly does:

```text
feature ranking for appliance power regression
```

But the feature-selection block idea is strongly related.

---

## 20. Suggested Thesis Framing

A careful way to describe the relationship is:

```text
Houidi et al. demonstrate that NILM appliance identification benefits from
interpretable electrical features and feature selection. Their pipeline computes
a pool of additive high-frequency electrical features, applies several feature
selection methods, and validates the selected subsets using classifiers and
cross-dataset transfer tests.

Inspired by this design, our work treats feature selection as a dedicated block
between HF feature extraction and NILM model training. Instead of using only
univariate Mutual Information ranking, we use mRMR to select features that are
both target-relevant and mutually non-redundant. This is suitable for our
high-frequency NILM feature set because many extracted descriptors are
correlated or overlapping.
```

---

## 21. Main Takeaway

The paper's feature-selection design is:

```text
compute many physics-based HF electrical features
keep NILM-suitable additive features
compare several feature-selection methods
train classifiers on each selected subset
evaluate accuracy and compactness
test whether selected features transfer across datasets
```

Our mRMR method can be viewed as a replacement or extension of the paper's Section 4 feature-selection block.

The expected benefit is:

```text
better feature compactness
less redundancy
more interpretable selected HF feature subsets
```

But final superiority must be shown experimentally by training and evaluating downstream NILM models.

# Complete Feature Selection Pipeline: mRMR + NILM Wrapper Validation

## 1. Current Decision

For now, the `on_off` target is disabled as the main mRMR setting.

The main target is appliance power:

```text
kettle_power
fridge_power
microwave_power
dishwasher_power
washingmachine_power
```

Reason:

```text
The current ON-only buffered dataset is not suitable for ON/OFF feature selection,
because it contains only ON rows plus nearby buffer OFF rows.
```

Therefore, the current mRMR stage should answer:

```text
Which HF features are relevant to appliance power while avoiding repeated information?
```

It should not answer:

```text
Which features best classify ON vs OFF?
```

That can be done later using a full or balanced ON/OFF dataset.

## 2. Why Stage 01 Alone Is Not Enough

Stage 01 is a correlation-based redundancy filter.

It checks whether two features are highly similar:

```text
|Pearson(feature_i, feature_j)| > threshold
or
|Spearman(feature_i, feature_j)| > threshold
```

If two features are highly correlated, Stage 01 keeps one and drops the other.

This is useful for identifying duplicated information, but it has an important limitation:

```text
High correlation does not mean two features contain exactly the same physical information.
```

Example:

```text
I_rms and P_active may be highly correlated,
but P_active also contains phase / power factor information.
```

So Stage 01 should be treated as:

```text
preliminary redundancy analysis
```

not final feature selection.

## 3. Why mRMR Is Added

mRMR means:

```text
minimum Redundancy Maximum Relevance
```

mRMR improves over pure correlation filtering because it considers two things:

```text
1. Relevance to the target
2. Redundancy with already selected features
```

For the current power-target setting:

```text
relevance = mutual information(feature, appliance_power)
redundancy = mutual information(feature, selected feature)
```

So mRMR does not simply remove features because they are correlated.

It asks:

```text
Is this feature informative for appliance power?
Is it different from the features already selected?
```

## 4. mRMR Equations

### 4.1 Relevance

For each feature `F_i`, relevance is measured using mutual information with target `Y`.

```text
R_i = I(Y; F_i)
```

where:

```text
Y = appliance_power
F_i = one HF feature
```

Mutual information:

```text
I(Y; F_i) = sum_y sum_f p(y, f) log( p(y, f) / (p(y)p(f)) )
```

Higher value means:

```text
the feature contains more information about appliance power
```

### 4.2 Redundancy

For a candidate feature `F_i`, redundancy is measured against the already selected feature subset `S`.

```text
Q_S,i = mean I(F_i; F_j), for all F_j in S
```

where:

```text
S = features already selected by mRMR
```

Mutual information between two features:

```text
I(F_i; F_j) = sum_fi sum_fj p(fi, fj) log( p(fi, fj) / (p(fi)p(fj)) )
```

Higher value means:

```text
the candidate feature repeats information already selected
```

### 4.3 mRMR Score

The mRMR score is:

```text
mRMR_score = R_i / Q_S,i
```

In practice:

```text
mRMR_score = relevance / redundancy
```

The best next feature is:

```text
F_i* = argmax R_i / Q_S,i
```

This means:

```text
select features with high appliance-power relevance
and low redundancy with selected features
```

## 5. Current mRMR Script Logic

The current script:

```text
feature_selection/mRMR.py
```

does this:

```text
1. Read appliance CSV
2. Use HF feature columns
3. Use appliance_power as target
4. Estimate mutual information relevance
5. Estimate pairwise feature redundancy
6. Rank features using mRMR
7. Save mRMR ranking
```

The output is a ranking:

```text
rank 1 = strongest mRMR feature
rank 2 = next best feature after considering redundancy
...
```

However, mRMR alone still does not prove the final best feature subset.

It gives:

```text
a supervised feature ranking
```

not:

```text
final NILM model validation
```

## 5.1 Formula-To-Code Explanation In `mRMR.py`

This section maps the mRMR mathematical formula directly to the current code.

### Step 1: Define The Data Matrix

In the paper:

```text
D = {x_i,k}
```

where:

```text
i = feature index
k = sample / record index
```

In our NILM case:

```text
rows    = HF time windows
columns = HF features
```

Example features:

```text
I_rms, P_active, S_apparent, THDI, DWT_E0, ...
```

In `mRMR.py`, this is prepared in:

```python
prepare_xy()
```

The code creates:

```python
X = df[feature_cols]
y = df[target_col]
```

For the current setting:

```text
X = HF features
y = appliance_power
```

Example:

```text
y = kettle_power
```

### Step 2: Relevance Term

The paper defines relevance using mutual information:

```text
R_i = I(Y; F_i)
```

where:

```text
Y   = target
F_i = candidate feature
```

The mutual information equation is:

```text
I(Y; F_i) = sum_y sum_f p(y, f) log( p(y, f) / (p(y)p(f)) )
```

Meaning:

```text
How much information does feature F_i provide about target Y?
```

In our code, this is implemented in:

```python
relevance_scores()
```

Current power-target code uses:

```python
scores = mutual_info_regression(
    X_scaled, y, discrete_features=False, random_state=random_state
)
```

This estimates:

```text
I(appliance_power; feature)
```

Example:

```text
I(kettle_power; P_active)
I(kettle_power; I_rms)
I(kettle_power; THDI)
```

Higher relevance means:

```text
the feature is more informative for appliance power
```

### Step 3: Redundancy Term

The paper defines redundancy between features using mutual information:

```text
I(F_i; F_j)
```

For a candidate feature `F_i`, redundancy with the selected subset `S` is:

```text
Q_S,i = mean I(F_i; F_j), for F_j in S
```

Meaning:

```text
How much information does candidate feature F_i repeat from features already selected?
```

In our code, pairwise redundancy is first computed in:

```python
pairwise_redundancy()
```

The code estimates feature-to-feature mutual information:

```python
mi = mutual_info_regression(
    X_scaled[other_features],
    y_feature,
    discrete_features=False,
    random_state=random_state,
)
```

This estimates:

```text
I(feature_i; feature_j)
```

Then inside:

```python
run_mrmr()
```

the redundancy of a candidate feature is calculated as:

```python
red = float(redundancy.loc[feature, selected].mean())
```

This means:

```text
red = average redundancy between candidate feature and already selected features
```

### Step 4: mRMR Score

The paper selects the feature that maximizes:

```text
F_i* = argmax R_i / Q_S,i
```

In words:

```text
select the feature with high target relevance and low redundancy
```

In our code, this is implemented in:

```python
run_mrmr()
```

The actual score line is:

```python
score = rel / (red + eps) if selected else rel
```

where:

```text
rel = relevance MI with appliance_power
red = mean redundancy MI with selected features
eps = small number to avoid division by zero
```

For the first feature:

```python
score = rel
```

because no feature has been selected yet, so redundancy is zero.

For later features:

```python
score = relevance / redundancy
```

### Step 5: Forward Selection

mRMR selects features sequentially.

The code starts with:

```python
selected = []
remaining = list(X.columns)
```

At each rank:

```python
for feature in remaining:
    calculate relevance
    calculate redundancy with selected features
    calculate mRMR score
```

Then it selects the candidate with the highest score:

```python
feature, score, rel, red = max(candidates, key=lambda x: x[1])
```

After selection:

```python
selected.append(feature)
remaining.remove(feature)
```

So the ranking is built one feature at a time:

```text
rank 1 = best relevance
rank 2 = best relevance/redundancy after rank 1
rank 3 = best relevance/redundancy after rank 1 and rank 2
...
```

### Step 6: Output Meaning

The output CSV contains:

```text
rank
feature
mrmr_score
relevance_mi_to_target
mean_redundancy_mi_to_selected
```

Interpretation:

```text
rank = feature selection order
feature = selected HF feature
mrmr_score = relevance / redundancy
relevance_mi_to_target = information with appliance_power
mean_redundancy_mi_to_selected = repeated information with selected features
```

Important:

```text
A feature with high relevance can still appear lower in the ranking
if it is highly redundant with already selected features.
```

Example:

```text
DWT_E0 may have high relevance to appliance power,
but if P_active or I_rms already contains similar information,
mRMR may rank DWT_E0 lower.
```

### Step 7: Current Target Setting

The current default target in `mRMR.py` is:

```text
power
```

So the script uses:

```python
mutual_info_regression()
```

not:

```python
mutual_info_classif()
```

This means the current mRMR ranking is based on:

```text
MI(feature; appliance_power)
```

not:

```text
MI(feature; on_off)
```

This is suitable for the current ON-only-buffer analysis because the goal is active-state appliance power relevance.

## 5.2 LaTeX Formula And Exact Code Mapping

This section shows the mRMR equations in LaTeX style and the exact code used in `mRMR.py`.

### 5.2.1 Data Matrix

Mathematical notation:

$$
X =
\begin{bmatrix}
x_{1,1} & x_{2,1} & \cdots & x_{n,1} \\
x_{1,2} & x_{2,2} & \cdots & x_{n,2} \\
\vdots  & \vdots  & \ddots & \vdots  \\
x_{1,K} & x_{2,K} & \cdots & x_{n,K}
\end{bmatrix}
$$

where:

$$
n = \text{number of HF features}
$$

$$
K = \text{number of samples / time windows}
$$

For this project:

$$
F_i \in \{I_{rms}, P_{active}, S_{apparent}, THDI, DWT\_E0, ...\}
$$

$$
Y = \text{appliance power}
$$

Example:

$$
Y = kettle\_power
$$

Code in `mRMR.py`:

```python
feature_cols = [f for f in HF_FEATURES if f in df.columns]
X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

y = df[target_col]
y = y.fillna(0.0).astype(float)
```

Explanation:

```text
X contains all HF features.
y contains the appliance power target.
```

### 5.2.2 Relevance Formula

Mathematical notation:

$$
R_i = I(Y; F_i)
$$

where:

$$
I(Y; F_i)
=
\sum_y \sum_f p(y,f)
\log \left(
\frac{p(y,f)}{p(y)p(f)}
\right)
$$

Meaning:

```text
R_i measures how much information feature F_i gives about appliance power Y.
```

If:

$$
R_i \text{ is high}
$$

then:

```text
feature F_i is highly relevant to appliance power.
```

Code in `mRMR.py`:

```python
def relevance_scores(X: pd.DataFrame, y: pd.Series, target: str, random_state: int) -> pd.Series:
    X_scaled = StandardScaler().fit_transform(X)
    if target == "on_off":
        scores = mutual_info_classif(
            X_scaled, y, discrete_features=False, random_state=random_state
        )
    else:
        scores = mutual_info_regression(
            X_scaled, y, discrete_features=False, random_state=random_state
        )
    return pd.Series(scores, index=X.columns, name="relevance_mi")
```

For the current setting:

```python
scores = mutual_info_regression(...)
```

because:

```text
target = appliance_power
```

So the code estimates:

$$
I(appliance\_power; F_i)
$$

### 5.2.3 Redundancy Formula

Mathematical notation:

$$
Q_{S,i}
=
\frac{1}{|S|}
\sum_{F_j \in S}
I(F_i; F_j)
$$

where:

$$
S = \text{already selected feature subset}
$$

and:

$$
I(F_i; F_j)
=
\sum_{f_i} \sum_{f_j} p(f_i,f_j)
\log \left(
\frac{p(f_i,f_j)}{p(f_i)p(f_j)}
\right)
$$

Meaning:

```text
Q_S,i measures how much the candidate feature F_i repeats information
already contained in selected features.
```

If:

$$
Q_{S,i} \text{ is high}
$$

then:

```text
candidate feature F_i is highly redundant.
```

Important:

```text
This redundancy formula only shows the comparison between a candidate feature
and the already selected features.
```

It does not show the target comparison.

The target comparison is in the relevance formula:

$$
R_i = I(Y; F_i)
$$

So mRMR has two separate parts:

$$
\text{relevance} = R_i = I(Y; F_i)
$$

$$
\text{redundancy} = Q_{S,i} = \frac{1}{|S|}\sum_{F_j \in S} I(F_i;F_j)
$$

Then both are combined in the mRMR score:

$$
\text{mRMR score}
=
\frac{R_i}{Q_{S,i}}
$$

For NILM:

$$
R_i = I(appliance\_power; F_i)
$$

and:

$$
Q_{S,i} = \text{average } I(F_i; selected\ features)
$$

Example after `P_active` has been selected:

$$
S = \{P\_active\}
$$

For candidate `I_rms`:

$$
R_{I\_rms} = I(kettle\_power; I\_rms)
$$

$$
Q_{S,I\_rms} = I(I\_rms; P\_active)
$$

So:

$$
\text{mRMR}(I\_rms)
=
\frac{I(kettle\_power; I\_rms)}
{I(I\_rms; P\_active)}
$$

This means:

```text
I_rms is selected only if it is relevant to kettle_power
and not too redundant with P_active.
```

Code in `mRMR.py`:

```python
def pairwise_redundancy(X: pd.DataFrame, random_state: int) -> pd.DataFrame:
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    red = pd.DataFrame(0.0, index=X.columns, columns=X.columns)
    for feature in X.columns:
        y_feature = X_scaled[feature]
        other_features = [c for c in X.columns if c != feature]
        mi = mutual_info_regression(
            X_scaled[other_features],
            y_feature,
            discrete_features=False,
            random_state=random_state,
        )
        red.loc[feature, other_features] = mi
    return red
```

This computes:

$$
I(F_i; F_j)
$$

for every feature pair.

Then inside `run_mrmr()`:

```python
red = float(redundancy.loc[feature, selected].mean())
```

This implements:

$$
Q_{S,i}
=
\frac{1}{|S|}
\sum_{F_j \in S}
I(F_i; F_j)
$$

### 5.2.4 mRMR Selection Formula

Mathematical notation:

$$
F_i^*
=
\arg\max_{F_i \in G \setminus S}
\frac{R_i}{Q_{S,i}}
$$

where:

$$
G = \text{all features}
$$

$$
S = \text{already selected features}
$$

Meaning:

```text
Choose the next feature with high relevance and low redundancy.
```

Code in `mRMR.py`:

```python
for feature in remaining:
    rel = float(relevance[feature])
    if selected:
        red = float(redundancy.loc[feature, selected].mean())
    else:
        red = 0.0

    score = rel / (red + eps) if selected else rel
    candidates.append((feature, score, rel, red))

feature, score, rel, red = max(candidates, key=lambda x: x[1])
```

This line:

```python
score = rel / (red + eps) if selected else rel
```

implements:

$$
\frac{R_i}{Q_{S,i}}
$$

The small value `eps` prevents division by zero:

```python
eps = 1e-9
```

For the first selected feature:

$$
S = \emptyset
$$

so:

```python
score = rel
```

because there is no selected feature yet, therefore redundancy cannot be calculated.

### 5.2.5 Forward Selection Update

After choosing the best candidate:

```python
selected.append(feature)
remaining.remove(feature)
```

Mathematical meaning:

$$
S \leftarrow S \cup \{F_i^*\}
$$

$$
G \leftarrow G \setminus \{F_i^*\}
$$

So the process repeats:

```text
rank 1: best feature by relevance
rank 2: best relevance/redundancy after rank 1
rank 3: best relevance/redundancy after rank 1 and rank 2
...
```

### 5.2.6 Output Columns

The code saves:

```python
rows.append(
    {
        "rank": rank,
        "feature": feature,
        "mrmr_score": score,
        "relevance_mi_to_target": rel,
        "mean_redundancy_mi_to_selected": red,
    }
)
```

Column meaning:

| Column                           | Formula meaning   | Explanation                                         |
| ----------------------------------| -------------------| -----------------------------------------------------|
| `rank`                           | selection order   | mRMR ranking position                               |
| `feature`                        | $F_i^*$           | selected feature                                    |
| `mrmr_score`                     | $R_i / Q_{S,i}$   | high relevance and low redundancy                   |
| `relevance_mi_to_target`         | $R_i = I(Y; F_i)$ | information with appliance power                    |
| `mean_redundancy_mi_to_selected` | $Q_{S,i}$         | average repeated information with selected features |

### 5.2.7 Simple Interpretation Example

Suppose:

$$
R_{P\_active} = 0.68
$$

and it is selected first.

Then for `I_rms`:

$$
R_{I\_rms} = 0.62
$$

but:

$$
Q_{S,I\_rms} = I(I\_rms; P\_active) = 0.80
$$

Then:

$$
mRMR(I\_rms) = \frac{0.62}{0.80} = 0.775
$$

This means:

```text
I_rms is relevant to appliance power,
but it may be redundant with P_active.
```

Another feature may have slightly lower relevance but much lower redundancy, so mRMR may rank it higher.

That is the key idea:

```text
mRMR does not only ask "is this feature useful?"
It also asks "is this feature giving new information?"
```

## 5.3 Full Worked Example With 4 Features

This example shows the complete mRMR calculation from beginning to final ranking.

Assume the target is:

```text
Y = kettle_power
```

Assume we have four candidate features:

```text
F1 = P_active
F2 = I_rms
F3 = THDI
F4 = PF
```

The mRMR score is:

$$
\text{score}(F_i)
=
\frac{R_i}{Q_{S,i}}
$$

where:

$$
R_i = I(Y; F_i)
$$

and:

$$
Q_{S,i}
=
\frac{1}{|S|}
\sum_{F_j \in S}
I(F_i; F_j)
$$

### 5.3.1 Given Relevance Values

Assume mutual information between each feature and the target has already been calculated:

| Feature | Relevance \(R_i = I(Y;F_i)\) |
| --- | ---: |
| `P_active` | 0.80 |
| `I_rms` | 0.75 |
| `THDI` | 0.45 |
| `PF` | 0.30 |

Interpretation:

```text
P_active has the highest individual relevance to kettle_power.
I_rms is also highly relevant.
THDI and PF are less individually relevant.
```

### 5.3.2 Given Feature-Feature MI Values

Assume the pairwise mutual information between features is:

| Feature pair | Mutual information |
| --- | ---: |
| \(I(P\_active; I\_rms)\) | 0.70 |
| \(I(P\_active; THDI)\) | 0.20 |
| \(I(P\_active; PF)\) | 0.25 |
| \(I(I\_rms; THDI)\) | 0.15 |
| \(I(I\_rms; PF)\) | 0.20 |
| \(I(THDI; PF)\) | 0.10 |

Interpretation:

```text
P_active and I_rms are highly redundant.
THDI is less redundant with P_active and I_rms.
```

### 5.3.3 Round 1: Select First Feature

At the beginning:

$$
S = \emptyset
$$

No feature has been selected yet, so redundancy cannot be calculated.

Therefore, the first feature is selected using only relevance:

| Feature | Relevance |
| --- | ---: |
| `P_active` | 0.80 |
| `I_rms` | 0.75 |
| `THDI` | 0.45 |
| `PF` | 0.30 |

The highest relevance is:

```text
P_active = 0.80
```

So:

```text
Rank 1 = P_active
```

The selected subset becomes:

$$
S = \{P\_active\}
$$

### 5.3.4 Round 2: Select Second Feature

Remaining features:

```text
I_rms, THDI, PF
```

Now each candidate is compared with the selected feature:

$$
S = \{P\_active\}
$$

For `I_rms`:

$$
Q_{S,I\_rms}
=
I(I\_rms; P\_active)
=
0.70
$$

$$
\text{score}(I\_rms)
=
\frac{0.75}{0.70}
=
1.071
$$

For `THDI`:

$$
Q_{S,THDI}
=
I(THDI; P\_active)
=
0.20
$$

$$
\text{score}(THDI)
=
\frac{0.45}{0.20}
=
2.250
$$

For `PF`:

$$
Q_{S,PF}
=
I(PF; P\_active)
=
0.25
$$

$$
\text{score}(PF)
=
\frac{0.30}{0.25}
=
1.200
$$

Summary:

| Candidate | Relevance | Redundancy with \(S\) | mRMR score |
| --- | ---: | ---: | ---: |
| `I_rms` | 0.75 | 0.70 | 1.071 |
| `THDI` | 0.45 | 0.20 | 2.250 |
| `PF` | 0.30 | 0.25 | 1.200 |

Highest score:

```text
THDI = 2.250
```

So:

```text
Rank 2 = THDI
```

The selected subset becomes:

$$
S = \{P\_active, THDI\}
$$

### 5.3.5 Round 3: Select Third Feature

Remaining features:

```text
I_rms, PF
```

Now redundancy is the average MI with both selected features.

For `I_rms`:

$$
Q_{S,I\_rms}
=
\frac{
I(I\_rms; P\_active)
+
I(I\_rms; THDI)
}{2}
$$

$$
Q_{S,I\_rms}
=
\frac{0.70 + 0.15}{2}
=
0.425
$$

$$
\text{score}(I\_rms)
=
\frac{0.75}{0.425}
=
1.765
$$

For `PF`:

$$
Q_{S,PF}
=
\frac{
I(PF; P\_active)
+
I(PF; THDI)
}{2}
$$

$$
Q_{S,PF}
=
\frac{0.25 + 0.10}{2}
=
0.175
$$

$$
\text{score}(PF)
=
\frac{0.30}{0.175}
=
1.714
$$

Summary:

| Candidate | Relevance | Redundancy with \(S\) | mRMR score |
| --- | ---: | ---: | ---: |
| `I_rms` | 0.75 | 0.425 | 1.765 |
| `PF` | 0.30 | 0.175 | 1.714 |

Highest score:

```text
I_rms = 1.765
```

So:

```text
Rank 3 = I_rms
```

The selected subset becomes:

$$
S = \{P\_active, THDI, I\_rms\}
$$

### 5.3.6 Round 4: Select Final Feature

Only one feature remains:

```text
PF
```

Its redundancy is:

$$
Q_{S,PF}
=
\frac{
I(PF; P\_active)
+
I(PF; THDI)
+
I(PF; I\_rms)
}{3}
$$

$$
Q_{S,PF}
=
\frac{0.25 + 0.10 + 0.20}{3}
=
0.183
$$

Its mRMR score is:

$$
\text{score}(PF)
=
\frac{0.30}{0.183}
=
1.639
$$

So:

```text
Rank 4 = PF
```

### 5.3.7 Final mRMR Ranking

| Rank | Feature | Relevance | Redundancy | mRMR score |
| ---: | --- | ---: | ---: | ---: |
| 1 | `P_active` | 0.80 | 0.000 | 0.800 |
| 2 | `THDI` | 0.45 | 0.200 | 2.250 |
| 3 | `I_rms` | 0.75 | 0.425 | 1.765 |
| 4 | `PF` | 0.30 | 0.183 | 1.639 |

Final selected order:

```text
1. P_active
2. THDI
3. I_rms
4. PF
```

### 5.3.8 Interpretation

`I_rms` has high relevance:

```text
I(kettle_power; I_rms) = 0.75
```

However, it is also highly redundant with `P_active`:

```text
I(I_rms; P_active) = 0.70
```

Therefore, mRMR does not select `I_rms` immediately after `P_active`.

Instead, it selects `THDI` second because `THDI` adds more different information:

```text
I(THDI; P_active) = 0.20
```

This shows the purpose of mRMR:

```text
Select features that are useful for the target,
but avoid selecting features that repeat the same information.
```

## 6. Why A Wrapper Model Is Still Needed

mRMR uses target information, but it is still a filter method.

It does not train a NILM model.

Therefore, it cannot directly answer:

```text
Does this selected feature subset improve appliance power prediction?
```

To answer that, we need a wrapper/model-validation stage.

The wrapper stage trains a NILM baseline model using different feature subsets and evaluates prediction performance.

## 7. NILM Wrapper Model Concept

For appliance power prediction, the wrapper model should be a regression model.

Possible models:

```text
SVR
Random Forest Regressor
LightGBM Regressor
XGBoost Regressor
MLP Regressor
```

The goal is not necessarily to create the final best NILM model immediately.

The goal is:

```text
use a strong baseline model to test whether selected features preserve prediction performance
```

## 7.1 How mRMR Combines With RFE Inside The Training Loop

In the paper, mRMR is not only used once before the wrapper.

Instead, mRMR is combined with RFE during each recursive elimination step.

The idea is:

```text
RFE provides model-based importance.
mRMR provides relevance-redundancy information.
The final feature ranking combines both.
```

### 7.1.1 Combined Ranking Equation

For a candidate feature `F_i`, the combined score can be written as:

$$
r_i =
\beta \cdot M_i
+
(1 - \beta) \cdot
\frac{R_i}{Q_{S,i}}
$$

where:

$$
M_i = \text{model-based importance of feature } F_i
$$

$$
R_i = I(Y; F_i)
$$

$$
Q_{S,i}
=
\frac{1}{|S|}
\sum_{F_j \in S}
I(F_i; F_j)
$$

and:

$$
\beta \in [0, 1]
$$

Meaning:

```text
beta controls the balance between wrapper model importance and mRMR score.
```

If:

```text
beta = 1
```

then:

```text
the method becomes pure RFE
```

If:

```text
beta = 0
```

then:

```text
the method becomes pure mRMR
```

If:

```text
beta = 0.5
```

then:

```text
model importance and mRMR contribute equally
```

### 7.1.2 Classification Version In The Paper

The paper uses SVM-RFE for classification.

For SVM:

$$
M_i = |w_i|
$$

where:

```text
w_i is the SVM weight for feature F_i.
```

So the paper's idea is:

$$
r_i =
\beta |w_i|
+
(1 - \beta)
\frac{R_i}{Q_{S,i}}
$$

Then the weakest feature is removed:

$$
F_{drop}
=
\arg\min r_i
$$

### 7.1.3 Power Regression Version For NILM

For our current NILM setting, the target is appliance power:

```text
Y = appliance_power
```

So the wrapper model should be regression-based.

Instead of SVM-RFE classification, we can use:

```text
SVR-RFE
```

or another regression model with feature importance, such as:

```text
LightGBM Regressor
Random Forest Regressor
XGBoost Regressor
```

For SVR with a linear kernel:

$$
M_i = |w_i|
$$

For tree-based regressors:

$$
M_i = \text{feature importance}
$$

or:

$$
M_i = \text{permutation importance}
$$

The same combined score idea can still be used:

$$
r_i =
\beta \cdot M_i
+
(1 - \beta)
\frac{R_i}{Q_{S,i}}
$$

Then remove the lowest-ranked feature.

### 7.1.4 Full Training Loop

The complete loop is:

```text
Input:
    X = HF features
    y = appliance_power
    G = all features
    beta = balance between model importance and mRMR

Initialize:
    current_features = G
    best_score = infinity
    best_subset = G

Repeat:
    1. Train regression model using current_features

    2. Evaluate validation performance
       Example metric: RMSE or MAE

    3. Save current subset if performance is best

    4. Compute model importance M_i for every feature

    5. Compute mRMR relevance:
           R_i = MI(appliance_power; F_i)

    6. Compute mRMR redundancy:
           Q_S,i = mean MI(F_i; F_j)
           where F_j are other features in current subset

    7. Compute combined ranking:
           r_i = beta * M_i + (1 - beta) * R_i / Q_S,i

    8. Remove weakest feature:
           drop feature with smallest r_i

    9. Repeat until minimum feature number is reached

Output:
    best_subset
    performance curve
    removed-feature order
```

### 7.1.5 Difference From Current `mRMR.py`

Current `mRMR.py` only does:

```text
mRMR filter ranking
```

It does not yet do:

```text
train model
evaluate validation performance
combine model importance with mRMR
remove weakest feature recursively
```

So current `mRMR.py` is Stage 02:

```text
mRMR supervised filter ranking
```

The complete hybrid method would be Stage 03:

```text
mRMR + regression RFE wrapper loop
```

### 7.1.6 Why This Is Stronger

mRMR alone answers:

```text
Which features are relevant to appliance power and not redundant?
```

Wrapper alone answers:

```text
Which features help the model prediction?
```

Combined mRMR + RFE answers:

```text
Which features help the model while also avoiding redundant information?
```

This is more suitable for final feature selection because it connects statistical relevance with actual NILM prediction performance.

## 8. Complete Pipeline Logic

The complete feature-selection pipeline should be:

```text
Start
  |
  v
Prepare dataset
  |
  v
Choose target = appliance_power
  |
  v
Run mRMR ranking
  |
  v
Create candidate feature subsets
  |
  v
Train NILM regression baseline
  |
  v
Evaluate prediction performance
  |
  v
Choose smallest subset with stable / best performance
  |
  v
Check stability across dataset cases
  |
  v
Final selected feature group
```

## 9. Candidate Feature Subsets

Do not choose a fixed number such as `top_k = 20` as the final answer.

Instead, mRMR should rank all features or a large enough list.

Then evaluate several candidate subsets:

```text
top 5
top 10
top 15
top 20
top 25
top 30
all features
```

Example:

| Feature subset | Meaning |
| --- | --- |
| all features | baseline, no feature selection |
| top 5 mRMR | very compact feature set |
| top 10 mRMR | compact feature set |
| top 15 mRMR | moderate feature set |
| top 20 mRMR | larger selected set |
| top 30 mRMR | conservative selected set |

Then compare model performance.

## 10. Model Evaluation

For appliance power regression, use metrics such as:

```text
MAE
RMSE
NDE
SAE
R2
```

The key comparison is:

```text
all features vs selected features
```

Example decision rule:

```text
If top 15 features gives similar RMSE to all 50 features,
choose top 15 because it is simpler and less redundant.
```

Another example:

```text
If performance keeps improving until top 30,
then top 20 is too aggressive.
```

## 11. Stability Check

After choosing candidate subsets, compare stability across dataset cases:

```text
full_wk30_only
full_wk30_wk31
on_only_wk30_wk31
```

The selected features do not need to be exactly identical, but the important feature groups should be stable.

Example stable group:

```text
current magnitude / power group:
I_rms, P_active, S_apparent, I1, DWT_E0, I_BP_low
```

The representative feature may change, but the physical group should remain meaningful.

## 12. Recommended Final Pipeline

Recommended thesis pipeline:

```text
Stage 01:
    Correlation-based redundancy analysis

Stage 02:
    mRMR feature ranking using appliance_power

Stage 03:
    NILM regression wrapper validation

Stage 04:
    Stability check across weeks / dataset settings

Stage 05:
    Final feature subset or final feature groups
```

## 13. Why This Is Better

This avoids the weakness of each individual method.

| Method | Strength | Weakness |
| --- | --- | --- |
| Stage 01 correlation | Finds obvious redundancy | May drop useful correlated features |
| mRMR | Uses target and avoids redundancy | Does not train NILM model |
| Wrapper model | Tests actual prediction performance | More computationally expensive |

Together:

```text
Stage 01 explains redundancy structure.
mRMR ranks features by target relevance and non-redundancy.
Wrapper validation proves whether the selected subset works for NILM prediction.
```

## 14. Important Interpretation

Do not say:

```text
dropped features are useless
```

Say:

```text
dropped or lower-ranked features may be redundant under this dataset and target.
```

Also say:

```text
final usefulness is confirmed by wrapper model validation.
```

This is safer and more scientifically defensible.

## 15. Thesis Wording

Possible thesis wording:

```text
After the preliminary correlation-based redundancy analysis, mRMR was used as a supervised filter method to rank high-frequency features according to their mutual information with appliance power while penalizing redundancy with already selected features. Since mRMR does not directly evaluate NILM prediction performance, the ranked feature subsets were subsequently validated using a regression-based NILM baseline. The final feature subset was selected based on the trade-off between prediction accuracy and feature compactness, with stability checked across dataset settings.
```

Short version:

```text
mRMR provides target-aware feature ranking, while the wrapper NILM model validates whether the selected features preserve appliance power prediction performance.
```

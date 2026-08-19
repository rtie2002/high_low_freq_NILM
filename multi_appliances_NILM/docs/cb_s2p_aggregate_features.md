# CB-S2P Aggregate Feature Explanation

Paper:

```text
State disaggregation for Non-Intrusive Load Monitoring (NILM) with calibrated
bagged Seq2Point 1D convolutional neural networks
```

This note explains the input features used in the paper and why they are useful
for NILM.

## Main Idea

The paper does not feed only raw aggregate power into the model.

Instead, it constructs several feature channels from the aggregate mains signal:

```text
raw aggregate
first-order difference
rolling mean
rolling standard deviation
```

So the input at time `t` is:

```math
\mathbf{x}_t =
\left[
x_t,\ \Delta x_t,\ \mu_t^{(w)},\ \sigma_t^{(w)}
\right]
\tag{1}
```

where:

| Feature | Meaning |
|---|---|
| `x_t` | raw aggregate power at time `t` |
| `Delta x_t` | power change from previous timestep |
| `mu_t^(w)` | local rolling mean over a window of length `w` |
| `sigma_t^(w)` | local rolling standard deviation over a window of length `w` |

The idea is:

```text
raw power tells the model the current magnitude
delta power tells the model where rise/fall events happen
rolling mean tells the model the local background level
rolling std tells the model whether the local region is stable or fluctuating
```

This is important because appliance waveforms can change across houses and
datasets. If the model sees only raw aggregate power, it may memorize fixed
amplitude or fixed waveform width. These engineered aggregate features help the
model focus more on local shape, event changes, and temporal context.

## Feature 1: Raw Aggregate Power

```math
x_t
```

This is the original mains power reading at timestep `t`.

In NILM:

```text
x_t = sum of all active appliances + background/unknown load
```

Meaning:

```text
raw aggregate gives the model the absolute power level
```

Effect:

```text
useful for high-power appliances such as kettle, microwave, dishwasher
```

Limitation:

```text
raw aggregate alone is ambiguous because many appliances can overlap
```

Example:

```text
aggregate rises from 300 W to 2400 W
```

The raw value says power is high, but it does not directly tell which appliance
caused the rise. That is why the paper adds more features.

## Feature 2: First-Order Difference

The paper defines:

```math
\Delta x_t = x_t - x_{t-1}
\tag{2}
```

Meaning:

```text
Delta x_t measures the change of aggregate power between two consecutive points.
```

Interpretation:

| Pattern | Meaning |
|---|---|
| large positive `Delta x_t` | possible appliance turn ON |
| large negative `Delta x_t` | possible appliance turn OFF |
| near-zero `Delta x_t` | stable region |

Why this helps:

```text
ON/OFF events are often easier to detect from power rise/fall than from absolute power.
```

For your MultiNILM problem, this is very relevant. You noticed that appliance
amplitude and width change across domains, but the rise/fall shape still gives
clues about whether something turns ON or OFF.

Example:

```text
t-1: aggregate = 300 W
t  : aggregate = 2300 W
Delta x_t = 2000 W
```

This strong positive delta may indicate a kettle or microwave ON event.

For fridge:

```text
t-1: aggregate = 320 W
t  : aggregate = 420 W
Delta x_t = 100 W
```

The amplitude is smaller, but the delta still marks a possible compressor ON
transition.

## Feature 3: Rolling Mean

The paper uses a local mean:

```math
\mu_t^{(w)}
=
\frac{1}{w}
\sum_{i=0}^{w-1}
x_{t-i}
\tag{3}
```

where:

| Symbol | Meaning |
|---|---|
| `w` | rolling window size |
| `x_{t-i}` | previous aggregate samples inside the local window |
| `mu_t^(w)` | average aggregate level around time `t` |

Meaning:

```text
rolling mean gives the model a local baseline.
```

Why this helps:

The same absolute power value can mean different things depending on the local
background.

Example:

```text
House A background: 100 W
Fridge ON:          200 W

House B background: 500 W
Fridge ON:          600 W
```

The fridge still adds about `100 W`, but the raw aggregate level is very
different. Rolling mean helps the model understand:

```text
what is normal around this local region?
is current power higher than the local baseline?
```

Effect:

```text
helps cross-house generalization because the model sees relative local context
instead of only absolute aggregate magnitude
```

## Feature 4: Rolling Standard Deviation

The paper uses local variation:

```math
\sigma_t^{(w)}
=
\sqrt{
\frac{1}{w}
\sum_{i=0}^{w-1}
\left(
x_{t-i} - \mu_t^{(w)}
\right)^2
}
\tag{4}
```

Meaning:

```text
rolling standard deviation measures how unstable or fluctuating the local window is.
```

Interpretation:

| Local pattern | Rolling std |
|---|---|
| stable background | low |
| steady fridge ON | low or moderate |
| kettle spike / microwave burst | high near transition |
| washing machine / dishwasher cycle | higher because power changes by stage |

Why this helps:

Some appliances are not identified only by amplitude. Their local variability is
also important.

Example:

```text
fridge:
small rise, then relatively stable compressor region

dishwasher:
multi-stage cycle with heating, pump, idle, heating again

washing machine:
irregular cycle, spin, drain, heating
```

Rolling std gives the model a signal about whether the local aggregate is:

```text
smooth / stable
or
bursty / changing
```

This helps reduce the problem where the model memorizes one fixed waveform
width.

## Window Sizes

The paper reports rolling statistics using fixed local window sizes:

```text
w in {5, 7} minutes
```

This means rolling mean and rolling std are computed over short local temporal
contexts.

For NILM, this is useful because appliance events are local:

```text
turn-on edge
turn-off edge
short activation
local fluctuation
```

The model does not need only the full long sequence. It also needs small local
statistics around each timestep.

## Standardization

After constructing features, the paper standardizes them using training-set
statistics:

```math
\tilde{\mathbf{x}}_t
=
diag(\sigma_{train})^{-1}
\left(
\mathbf{x}_t - \mu_{train}
\right)
\tag{5}
```

Meaning:

```text
subtract training mean
divide by training standard deviation
```

Important:

```text
mean/std are computed only from the training block
validation and test use the same train statistics
```

This avoids data leakage.

Why standardization is needed:

```text
raw aggregate may be hundreds or thousands of watts
delta may be positive/negative and sparse
rolling mean has a different scale
rolling std has another scale
```

Without standardization, the model may over-focus on the numerically largest
feature and ignore smaller but useful features.

## Seq2Point Window Input

The paper then builds a Seq2Point input window:

```math
\mathbf{X}_t
=
\left[
\tilde{\mathbf{x}}_{t-\lfloor W/2 \rfloor},
\ldots,
\tilde{\mathbf{x}}_{t+\lfloor W/2 \rfloor}
\right]
\in
\mathbb{R}^{W \times C}
\tag{6}
```

where:

| Symbol | Meaning |
|---|---|
| `W` | temporal input window length |
| `C` | number of feature channels |
| `X_t` | model input window centered around time `t` |

The appendix reports:

```text
W = 128
C = 4
```

So each model input is:

```text
128 timesteps x 4 feature channels
```

The four channels are:

```text
raw mains
first-order difference
rolling mean
rolling standard deviation
```

## Why These Features Help NILM

These features solve different weaknesses of raw aggregate input.

| Problem | Helpful Feature | Reason |
|---|---|---|
| appliance amplitude changes across houses | rolling mean | gives local baseline |
| ON/OFF edge is small | delta power | highlights rise/fall |
| waveform width changes | rolling mean/std | gives local context instead of fixed template |
| noisy aggregate | rolling mean | smooths local trend |
| bursty/multi-stage appliance | rolling std | captures local fluctuation |
| model overfits raw watt scale | standardization | balances feature scale |

## Relation To Your MultiNILM-Fractional Model

Your current model uses:

```text
raw aggregate + 8 fractional derivative channels
```

That is already a feature front-end.

But the paper's feature idea is slightly different:

```text
raw aggregate
delta aggregate
local mean
local standard deviation
```

The fractional channels help with multi-order temporal change. The CB-S2P
features help with local event and baseline context.

A useful combined design for your model could be:

```text
raw aggregate
delta aggregate
absolute delta aggregate
rolling mean
rolling std
fractional alpha channels
```

This would let the model see:

```text
absolute power
rise/fall edge
local background
local variability
multi-scale fractional memory
```

## Practical Interpretation

For your problem:

```text
UK-DALE and REFIT appliance widths/amplitudes change slightly.
```

If the model sees only raw waveform, it may memorize:

```text
fridge usually lasts this long
kettle usually has this width
dishwasher usually has this amplitude
```

But if the model sees aggregate-derived features, it can learn:

```text
there is a rise here
this region is above local baseline
this signal is stable or unstable
this event has appliance-like shape even if duration changes
```

So the main contribution of these features is not making the model larger. The
main contribution is giving the model more physically meaningful views of the
same aggregate signal.

## Short Summary

The paper uses four aggregate-only input features:

```text
x_t                 raw aggregate power
Delta x_t           first-order power change
mu_t^(w)            rolling local mean
sigma_t^(w)         rolling local standard deviation
```

These features help the model detect ON/OFF changes, local baseline shifts, and
local variability. For your MultiNILM, this idea is useful because it may reduce
memorization of fixed waveform width and make the model focus more on event
shape and local context.

# Context-Aware Windowing In SGN And MATNilm

**Key idea:** both SGN and MATNilm use a longer aggregate input window than the appliance output window, so the model can use surrounding context when predicting the center target segment.

## Mathematical Formulation

For an aggregate signal \(x_t\) and appliance target \(y_t\):

```text
Input  : X_t = x[t-w : t+s+w-1]
Output : Y_t = y[t   : t+s-1]
```

where:

```text
s = target output length
w = extra context length on each side
input length = s + 2w
output length = s
```

## Window Diagram

```text
Time  ------------------------------------------------------------>

Input aggregate window X_t
        |<------ past context ------>|<-- target -->|<----- future context ----->|
        |------------ w ------------|------ s ------|------------- w ------------|
        x[t-w]                    x[t]           x[t+s-1]                    x[t+s+w-1]

Predicted appliance output Y_t
                                   |<-- target -->|
                                   |------ s ------|
                                   y[t]        y[t+s-1]
```

## Paper Settings

| Paper | Dataset | Input Length | Output Length | Context |
|---|---:|---:|---:|---:|
| SGN | REDD | 864 | 64 | 400 + 400 |
| SGN | UK-DALE | 432 | 32 | 200 + 200 |
| MATNilm | REDD | 864 | 64 | 400 + 400 |
| MATNilm | UK-DALE | 464 | 64 | 200 + 200 |

## Why It Matters

The model does not predict appliance states or power from an isolated instant. It observes the signal before, during, and after the target period, which helps capture appliance transitions, operation duration, and overlapping appliance behavior.

## Relation To This Project

For a future multi-label, multi-appliance NILM model, this means:

```text
aggregate context window -> appliance ON/OFF labels + appliance power outputs
```

The context-aware setup is suitable for predicting multiple appliances at the same time because appliance activity is temporally dependent and often overlaps.


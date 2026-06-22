# MATNilm Architecture Explained

Paper:

```text
MATNilm: Multi-appliance-task Non-intrusive Load Monitoring with Limited Labeled Data
```

## 1. Core Idea

MATNilm is a **multi-appliance, multi-task NILM model**.

It receives one aggregate household power sequence and predicts, at the same
time:

```text
1. appliance power consumption      -> regression task
2. appliance ON/OFF operating state -> classification task
```

Unlike single-appliance NILM models, MATNilm does not train one independent
model for each appliance. Instead, it uses one shared framework to model several
appliances together.

The key motivation is:

```text
appliance outputs are not independent
```

For example, the aggregate power is shared across appliances, and appliance
usage can be temporally related. A washing machine cycle may last for a long
period, while kettle and microwave events are short. Some appliances may also
overlap or rarely appear together.

## 2. Input And Output Formulation

Let the aggregate signal be:

```text
x_t
```

Let the power target of appliance `i` be:

```text
y_t^i
```

Let the ON/OFF state target of appliance `i` be:

```text
o_t^i
```

For `n` target appliances, the model predicts:

```text
power outputs:
Y_t = [y_t^1, y_t^2, ..., y_t^n]

state outputs:
O_t = [o_t^1, o_t^2, ..., o_t^n]
```

In sequence form, the model learns:

```text
aggregate input sequence
        ->
multi-appliance power sequence + multi-appliance ON/OFF sequence
```

## 3. Context-Aware Window

MATNilm follows a context-aware sliding-window setting.

The aggregate input window is longer than the output target window:

```text
Input  : X_t = x[t-w : t+s+w-1]
Output : Y_t = y[t   : t+s-1]
```

where:

```text
s = target output length
w = extra context length on each side
```

Therefore:

```text
input length = s + 2w
output length = s
```

This gives the model:

```text
past context + target window + future context
```

while predicting only:

```text
target window
```

Example from the paper:

```text
UK-DALE:
input length  = 464
output length = 64
context       = 200 before + 200 after

464 = 200 + 64 + 200
```

This setup is strong for offline or delayed NILM evaluation, but it is not
strict real-time because it uses future context.

## 4. High-Level Architecture

MATNilm has three main parts:

```text
1. shared encoder
2. multi-appliance decoder
3. regression and classification heads
```

Architecture flow:

```text
Aggregate input window
        |
        v
Shared encoder
        |
        v
Shared hidden representation
        |
        v
Multi-appliance decoder with 2DMA
        |
        v
Appliance-specific branches
        |
        +--> power regression head
        |
        +--> ON/OFF classification head
```

## 5. Meaning Of `m` And `n`

This is the part that is easy to confuse.

In MATNilm:

```text
n = number of target appliances
m = number of stacked decoder blocks
```

So:

```text
n is not the number of decoder blocks
n is the number of appliance branches
```

If the model predicts five appliances:

```text
kettle
fridge
microwave
dishwasher
washing machine
```

then:

```text
n = 5
```

If the decoder contains three repeated decoder blocks:

```text
m = 3
```

The structure is:

```text
decoder block 1
decoder block 2
...
decoder block m
```

Inside each decoder block, there are representations for all `n` appliances.

Simple diagram:

```text
                  n appliance branches
              appliance 1  appliance 2  ...  appliance n
                  |            |                  |
Decoder block 1   |            |                  |
                  v            v                  v
Decoder block 2   |            |                  |
                  v            v                  v
       ...
                  v            v                  v
Decoder block m   |            |                  |
                  v            v                  v
```

Therefore:

```text
n depends on how many appliances you want to disaggregate.
m is a model-depth hyperparameter.
```

## 6. Shared Encoder

The encoder maps the aggregate input sequence into a shared hidden
representation.

Mathematically:

```text
H = Encoder(X)
```

where:

```text
X = aggregate input window
H = shared latent representation
```

The encoder can be based on convolutional layers or LSTM layers. In the paper,
MAT-Conv uses the convolutional encoder, while MAT-LSTM uses the LSTM encoder.

The purpose of the encoder is to learn common information from the aggregate
signal before the model separates it into appliance-specific branches.

## 7. Appliance-Specific Decoder Branches

After the shared encoder, MATNilm uses one branch per appliance.

For appliance `i`, the hidden representation is:

```text
H^i
```

Each appliance branch learns appliance-specific information, but the branches
are not fully isolated because 2DMA allows information exchange:

```text
across time
across appliances
```

This is the main advantage over training independent single-appliance models.

## 8. Two-Dimensional Multi-Head Attention: 2DMA

2DMA means:

```text
Two-Dimensional Multi-Head Attention
```

The two dimensions are:

```text
1. temporal dimension
2. appliance dimension
```

So each decoder block performs attention in two directions:

```text
temporal attention:
    learns relationships across time for the same appliance

appliance-wise attention:
    learns relationships across appliances at the same time step
```

## 9. Temporal Attention

Temporal attention asks:

```text
For one appliance, which time steps are important?
```

For appliance `i`, the model attends over its own sequence representation:

```text
H^i = [h_1^i, h_2^i, ..., h_s^i]
```

The attention mechanism compares time steps inside the same appliance branch.

Conceptually:

```text
kettle at time t may depend on kettle behavior before and after t
washing machine at time t may depend on longer cycle patterns
```

Mathematically, using multi-head attention:

```text
TemporalAttn(H^i) = MultiHeadAttention(Q=H^i, K=H^i, V=H^i)
```

With residual connection and layer normalization:

```text
H_temp^i = LayerNorm(H^i + TemporalAttn(H^i))
```

This helps the model capture appliance operation patterns over time.

## 10. Appliance-Wise Attention

Appliance-wise attention asks:

```text
At one time step, which other appliance branches are relevant?
```

At time `t`, collect the hidden states of all `n` appliances:

```text
Z_t = [h_t^1, h_t^2, ..., h_t^n]
```

Then attention is applied across appliances:

```text
ApplianceAttn(Z_t) = MultiHeadAttention(Q=Z_t, K=Z_t, V=Z_t)
```

With residual connection and layer normalization:

```text
Z_app,t = LayerNorm(Z_t + ApplianceAttn(Z_t))
```

This lets the model learn interactions such as:

```text
some appliances overlap
some appliances rarely operate together
some appliance patterns follow others
aggregate power must be distributed among appliance branches
```

## 11. Multi-Head Attention Formula

For query `Q`, key `K`, and value `V`, scaled dot-product attention is:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

For multi-head attention:

```text
head_j = Attention(QW_j^Q, KW_j^K, VW_j^V)
```

and:

```text
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
```

where:

```text
h     = number of attention heads
d_k   = key/query dimension
W     = learnable projection matrices
```

Using multiple heads lets the model attend to different relationship patterns
at the same time.

## 12. Decoder Block Structure

Each decoder block contains:

```text
1. temporal attention
2. appliance-wise attention
3. feed-forward layer
4. residual connections
5. layer normalization
```

A simplified decoder block:

```text
Input appliance representations
        |
        v
Temporal multi-head attention
        |
        v
Appliance-wise multi-head attention
        |
        v
Feed-forward layer
        |
        v
Output appliance representations
```

This block is repeated `m` times:

```text
Decoder = Block_1 -> Block_2 -> ... -> Block_m
```

## 13. Regression And Classification Heads

For each appliance `i`, MATNilm has two output heads:

```text
1. regression head
2. classification head
```

The regression head predicts appliance power:

```text
p_t^i = predicted power of appliance i
```

The classification head predicts appliance ON/OFF state:

```text
o_t^i = predicted ON/OFF probability of appliance i
```

The final appliance power output can be viewed as a gated prediction:

```text
final power = regression output x ON/OFF output
```

Conceptually:

```text
if appliance is OFF -> predicted power should be near zero
if appliance is ON  -> regression output estimates power level
```

## 14. Loss Function

MATNilm combines regression loss and classification loss across all appliances.

For appliance `i`:

```text
regression loss     = MSE between true power and predicted power
classification loss = BCE between true ON/OFF and predicted ON/OFF
```

Total loss:

```text
L = sum_i ( L_power^i + L_on^i )
```

where:

```text
L_power^i = MSE(y^i, predicted_y^i)
L_on^i    = BCE(o^i, predicted_o^i)
```

This makes the model learn both tasks jointly:

```text
power estimation
state detection
```

## 15. Why MATNilm Is Important For This Project

MATNilm is highly relevant because your project also focuses on:

```text
multi-appliance NILM
multi-label ON/OFF classification
multi-output power regression
one aggregate input
non-event-based datasets such as UK-DALE / REFIT
```

The difference is:

```text
MATNilm focuses on deep architecture + sample augmentation.
Your current work tests whether extra high-frequency electrical features improve the baseline.
```

So your research can use MATNilm as a strong deep-learning reference, while
ExtraTrees can be used first as a classical baseline to test feature usefulness.

## 16. One-Slide Summary

```text
MATNilm is a multi-appliance, multi-task NILM architecture.
It uses one aggregate input sequence to predict both appliance power
and appliance ON/OFF states for multiple appliances.

n = number of appliances / appliance branches
m = number of stacked decoder blocks

The 2DMA module applies:
1. temporal attention across time
2. appliance-wise attention across appliance branches

This lets the model learn both appliance temporal behavior and
cross-appliance interactions.
```


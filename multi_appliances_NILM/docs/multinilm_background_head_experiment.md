# MultiNILM background-head experiment

## Decision

The microwave-only residual adapter is removed from the active experiment. It
reduced microwave AP and increased high-background false positives. The new
experiment returns to the relational five-appliance baseline and adds only one
auxiliary output: unobserved household/background power.

Experiment ID:

```yaml
background_swap_8w_relation_background_head
```

## Why this change is needed

The failed adapter treated short aggregate-power changes as microwave evidence.
This is not reliable at 8 s resolution because unobserved household loads can
look almost identical to the target appliances.

Evidence from the previous experiment:

| Observation | Result |
|---|---:|
| REFIT fridge false-positive samples with no other target appliance ON | 95.7% |
| Median residual power during REFIT fridge false positives | 222 W |
| Median aggregate change during those false positives | 1.2 W/sample |
| Median oracle candidate power for true REFIT microwave ON | 1.62 kW |
| Median oracle candidate power for false REFIT microwave ON | 1.44 kW |
| Validation microwave AP: baseline / failed adapter | 0.436 / 0.385 |

The fridge failure is mainly a stable unobserved load, not a missing transient
feature. The microwave failure is an unobserved high-power load whose amplitude
overlaps the real microwave. A local edge detector therefore cannot solve the
underlying ambiguity.

The residual target is sufficiently usable in the current dataset. Only
0.21--0.49% of samples have `aggregate - sum(target appliances) < 0`; these
small alignment/meter inconsistencies are clipped to zero.

## Architecture before: failed microwave adapter

```mermaid
flowchart TD
    X[Aggregate] --> FE[Fractional frontend]
    FE --> ENC[Shared multiscale encoder + TCN]
    ENC --> H[Five appliance feature heads]
    H --> REL[Relation attention]
    X --> LOC[Local transient expert]
    LOC --> ADD[Microwave residual addition]
    REL --> ADD
    ADD --> MW[Microwave power + state]
    REL --> OTH[Other four power + state outputs]
```

Problem: the local feature was added after appliance encoding and relation
attention without a learned feature-space projection. More fundamentally, the
local branch reacted to unknown-load edges as though they were microwave edges.

## Architecture after: supervised background head

```mermaid
flowchart TD
    X[Aggregate] --> FE[Fractional frontend]
    FE --> ENC[Shared multiscale encoder + TCN]
    ENC --> H[Five appliance feature heads]
    H --> REL[Relation attention across five targets]
    REL --> OUT[Five appliance power + state outputs]
    ENC --> BG[1x1 background power head]
    OUT --> REC[Aggregate reconstruction loss]
    BG --> REC
```

Important boundaries:

- The background output is not a sixth reported appliance.
- It has no ON/OFF classifier.
- It does not participate in relation attention.
- It is used during training to compete for power that does not belong to the
  five target appliances.
- Evaluation outputs and the five-appliance metrics are unchanged.

## Targets and equations

Let aggregate power be `x(t)` and the five true target powers be `y_i(t)`.
The supervised background target is

\[
b(t)=\max\left(0,\;x(t)-\sum_{i=1}^{5}y_i(t)\right).
\]

The model predicts five appliance powers `y_hat_i(t)` and background `b_hat(t)`.
The two new losses are

\[
L_{\mathrm{bg}}=
\operatorname{SmoothL1}\left(\hat b,b\right),
\]

\[
L_{\mathrm{rec}}=
\operatorname{SmoothL1}\left(
x,\hat b+\sum_{i=1}^{5}\hat y_i
\right).
\]

Both are evaluated in aggregate-normalized units after the physical sum is
formed in watts. The total loss is

\[
L_{\mathrm{total}}=
L_{\mathrm{NILM}}
+0.10L_{\mathrm{bg}}
+0.05L_{\mathrm{rec}}.
\]

`L_NILM` and all existing appliance loss settings are unchanged, so this is a
single interpretable architecture/loss ablation. The two auxiliary terms are
outside the existing dynamic power/state balancing.

## What changed in code

- `model/MultiNILM.py`
  - removed the active microwave-only residual route;
  - added an optional one-layer background head from shared context features;
  - retained the five appliance heads and relation attention;
  - kept the normal `(power, state_logits)` model return interface.
- `model/MultiNILM_loss.py`
  - constructs the background target directly from each training batch;
  - correctly converts appliance-specific normalized outputs to watts before
    reconstruction;
  - adds weighted SmoothL1 background and reconstruction losses.
- `evaluation/plots.py`
  - shows background and reconstruction losses in the fourth loss panel.
- Existing model YAML
  - uses the new experiment ID and enables only the background head;
  - no new model configuration file was created.

## How to judge this experiment

Do not judge it only from total validation loss. Compare against
`background_swap_8w` using:

1. fridge and microwave validation AP;
2. fridge FPR in the 200--400 W and 400--800 W residual-background bins;
3. microwave FPR in the `>=800 W` bin;
4. fridge OFF-MAE and microwave ON-MAE;
5. target-focused REFIT 20 and UK-DALE 2 waveforms;
6. `train/val_loss_background` and `train/val_loss_reconstruction` in
   `loss_detail.csv`.

The experiment succeeds only if the background losses learn while fridge and
microwave false positives decrease without a material drop in their AP.

## Training command

Run from the `multi_appliances_NILM` directory:

```powershell
& "C:\Users\Raymond Tie\anaconda3\python.exe" main.py `
  --mode train_evaluate `
  --model multinilm_fractional `
  --experiment config/experiment_mixed_ukdale_refit_8w.yaml `
  --model-config config/models/multinilm_fractional_relational_background_swap.yaml
```

## Research basis

This implementation is a small project-specific ablation, not a faithful copy
of another architecture. It follows the physical NILM decomposition
`aggregate = modelled appliances + unobserved residual`. Conv-NILM-Net also
formulates NILM as multi-source separation with an additive residual/noise term.
Recent residual-aware multi-appliance work explicitly argues that forcing only
the selected appliances to reconstruct aggregate power is incorrect when
unobserved loads exist, and instead combines a residual branch with aggregate
reconstruction.

- Conv-NILM-Net: https://arxiv.org/abs/2208.02173
- RAPC-Net: https://www.mdpi.com/2076-3417/16/17/8866

---

## Result: background head rejected

The completed run showed that this auxiliary side task did not solve the target
problem. Its prediction was never consumed by the five appliance heads, so it
could only regularise the shared encoder indirectly.

At epoch 150, the weighted auxiliary contribution was only

\[
0.10(0.237)+0.05(0.216)=0.0345,
\]

compared with validation `L_NILM = 109.657` (about 0.03%). Increasing that
weight is not a clean remedy: it would allocate more shared capacity to an
output that is still unused during appliance inference.

| Metric | Relation baseline | Dual expert | Background head |
|---|---:|---:|---:|
| Validation microwave AP | 0.436 | **0.467** | 0.420 |
| REFIT microwave AP | 0.488 | **0.591** | 0.494 |
| UK-DALE microwave AP | 0.792 | **0.805** | 0.763 |
| REFIT fridge AP | **0.745** | 0.726 | 0.714 |
| Validation fridge FPR at residual >=800 W | 0.691 | **0.667** | 0.716 |
| REFIT fridge FPR at residual >=800 W | 0.688 | **0.488** | 0.587 |

The active configuration therefore disables the background head and restores
the exact relation dual-expert architecture. The old result directory remains
unchanged as a negative ablation record.

## Next controlled experiment: hard-negative mining

Experiment ID:

```yaml
background_swap_8w_relation_dual_expert_hard_negative
```

### Why this targets the observed failure

The ordinary false-positive term averages over every true-OFF sample. Easy OFF
samples dominate that average, while the small subset of unknown loads that
look like a fridge or microwave receive little influence. The new term selects
the highest-scoring true-OFF samples separately for fridge and microwave.

For selected appliance `i`, let `H_i` be the top 5% eligible OFF logits. A
two-sample (16 s) guard is removed on each side of true ON regions to avoid
training aggressively on timestamp and threshold alignment errors:

\[
H_i=\operatorname{TopK}_{5\%}
\{s_{i,t}:z_{i,t}=0,\ t\notin\text{ON guard}\},
\]

\[
L_{\mathrm{HN}}=\frac{1}{2}
\sum_{i\in\{\mathrm{fridge},\mathrm{microwave}\}}
\frac{1}{|H_i|}\sum_{s\in H_i}\operatorname{softplus}(s).
\]

It is deliberately outside the existing dynamic state balancing:

\[
L_{\mathrm{total}}=L_{\mathrm{existing}}
+\lambda_{\mathrm{HN}}(e)L_{\mathrm{HN}}.
\]

This prevents the new term from silently reducing the effective BCE and FP
weights. The schedule is zero for epochs 1--20, ramps linearly during epochs
21--30, and then stays at `lambda_HN = 0.05`.

### Architecture and loss boundary

```mermaid
flowchart TD
    X[Aggregate] --> FE[Fractional frontend]
    X --> LE[Local expert]
    FE --> GE[Global multiscale encoder + TCN]
    GE --> GATE[Per-appliance local/global gates]
    LE --> GATE
    GATE --> REL[Relation attention across five appliances]
    REL --> OUT[Five power and state outputs]
    OUT --> BASE[Existing power + balanced state loss]
    OUT --> HN[Top 5% true-OFF logits<br/>fridge and microwave only]
    BASE --> TOTAL[Total loss]
    HN --> TOTAL
```

No new input feature, appliance output, threshold rule, data split, or test
post-processing is introduced. Therefore any change can be attributed to the
hard-negative training signal.

### Required success criteria

The experiment is accepted only if all of the following hold against the same
dual-expert reference:

1. fridge and microwave validation AP do not materially decrease;
2. high-residual-background FPR decreases on validation and both test houses;
3. recall decreases by no more than 2--3 percentage points;
4. the other three appliances do not show a systematic AP drop;
5. long false-ON segments visibly decrease in target-focused waveforms.

`loss_detail.csv` now records the raw train/validation hard-negative loss, its
effective scheduled weight, and per-appliance hard-negative losses. Lower FPR
alone is insufficient: a model that simply predicts OFF more often fails the
AP/recall criteria.

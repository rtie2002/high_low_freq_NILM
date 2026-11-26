# Reference-Conditioned Joint Multi-Appliance NILM
## Technical Design and Implementation Specification — Version 2.0

**Document type:** Research engineering specification / implementation report  
**Date:** 26 November 2026  
**Research status:** Proposed model and training protocol; the event-filtering / reference-waveform preparation step has been reported as completed by the researcher. **Stage 1 encoder, reference bank, conditioned Stage 2, and Stage 3 are not asserted to be implemented or validated.**  
**Primary datasets:** UK-DALE and REFIT  
**Target sampling interval:** 8 seconds (researcher reports the *current prepared events* have already been converted to a constant 8-second grid)  
**Target appliances (canonical order):** kettle, fridge, dishwasher, washing machine, microwave  
**Stage 2 aggregate window:** 1,024 samples = 8,192 s = 136 min 32 s  
**Implementation philosophy:** Reuse the existing multi-appliance/MART-NILM preprocessing and model wherever feasible; add an independently pretrained appliance-signature pathway and test its incremental value under rigorous house-disjoint evaluation.

> **Important distinction:** This is a **design specification**, not an account of already completed experiments. Numerical configurations below are **initial experiment choices** unless identified otherwise. Scientific claims of improvement, unseen-appliance generalization, or inverse-optimization capability must not be made before appropriate experiments.

---

## Contents

1. Executive summary and design decisions
2. Research objective, hypotheses, and scientific limitations
3. Physics, problem definition, and meaning of “inverse”
4. Stage boundaries and end-to-end data flow
5. Global data contract, appliance identities, and quality metadata
6. House-disjoint experimental splits and channel availability
7. Prepared reference-event dataset: schema, quality control, and event boundaries
8. Stage 1 dataset and variable-length batching
9. Stage 1 reference-encoder network
10. Stage 1 representation-learning objective
11. Stage 1 training, optimization, diagnostics, and acceptance criteria
12. Stage 1.5 reference-bank construction
13. Stage 2 reuse of existing MART-NILM data
14. Stage 2 network design: minimal integration first
15. Stage 2 loss functions and label masking
16. Stage 2 sampling, training, and optional partial labels
17. Stage 3 cross-house inference and reconstruction
18. Evaluation, metrics, baselines, and ablations
19. Repository layout, interfaces, example configuration, and pseudocode
20. Data integrity, anti-leakage tests, and automated checks
21. Implementation roadmap and go/no-go gates
22. Risks, open questions, and later research extensions
23. Reference literature and audit notes

---

# 1. Executive summary and design decisions

### 1.1 Core hypothesis

A complete independent appliance operating event contains a recognizable operating signature. Signatures learned from multiple households may capture category-level patterns that are useful for extracting the corresponding appliance from a new household's mixed aggregate power.

\[
R_i^{(j)}\xrightarrow{E_\theta}e_i^{(j)}\in\mathbb R^{d},\qquad d=128.
\]

A frozen reference encoder produces category-level signatures from complete events originating **only in training households**:

\[
\mathbf E_{\mathrm{bank}}=[e_1;\dots;e_5]\in\mathbb R^{5\times128}.
\]

A joint network predicts all five appliance powers and states **in one aggregate pass**:

\[
(\widehat{\mathbf P},\widehat{\mathbf S})=H_\phi(Y,\mathbf E_{\mathrm{bank}}),
\quad Y\in\mathbb R^{1\times1024},\quad
\widehat{\mathbf P},\widehat{\mathbf S}\in\mathbb R^{5\times1024}.
\]

### 1.2 Decisions already reached in the discussion

| Subject | Baseline decision | Rationale / condition |
|---|---|---|
| Source waveforms | Existing filtered, complete ON events; **current version already sampled every 8 s** | Do not re-extract or resample again without evidence of a problem. |
| Stage 1 event length | **Variable-length** | Keep kettle events short and washer/dishwasher cycles complete. |
| Stage 1 training | **One shared encoder** trained on examples from **all five classes** | Inputs remain **individual submetered** appliance traces, never sums. |
| Stage 1 batch handling | Length-based grouping and padding + mask; or per-group forward passes through the same weights | Avoid treating padding as real OFF. |
| Stage 1 output | One 128-dimensional event embedding | Same embedding dimension regardless of event duration. |
| Bank | Fixed category-level training-house bank, initially mean of `M=5` representative event embeddings | Save once and use in both Stage 2 and Stage 3; `M` is ablated. |
| Stage 2 data | Reuse previous **MART-NILM** 8-second aggregate/target/state dataset | No new raw parser and no forced full-cycle window. |
| Stage 2 length | `T=1024` samples | A fixed window represents household time, **not** event duration. |
| Stage 2 predictions | Five appliance powers and five state probabilities **jointly** | Existing MART backbone and cross-appliance interaction can be retained. |
| Conditioning | Insert a light reference-conditioned feature modulation at a defined point | Compare with equally sized appliance-ID conditioning. |
| Stage 2 labels | Full five-channel training first; masked partial-label houses later | Unmonitored ≠ OFF. |
| Stage 3 | Held-out-house aggregate + saved bank; no new test-home reference waveforms | Test submeters reserved for scoring only. |
| Novelty test | Reference bank vs appliance ID vs MART baseline | A static bank could otherwise become merely a learned class code. |
| Physics constraint | No hard `sum(targets)=mains` equality | Unknown household loads and meter mismatch remain. |
| Inverse problem terminology | **Amortized, reference-conditioned inverse estimation** | No neural-adjoint or test-time latent inverse optimization in v1. |

### 1.3 What is confirmed versus planned

**Reported completed:** Original preprocessing pipeline; prior MART-NILM dataset/model; extraction/filtering of separate appliance ON events; researcher reports latest event sampling is constant at 8 s.

**To implement / integrate:** Event manifest and QA against latest resampled files; variable-length dataset; shared reference encoder; loss and training; reference bank and provenance; integration of conditioning into MART; training experiments; full house-disjoint inference and evaluation.

**Not established:** That class-level embeddings improve MAE/F1 over an appliance-ID control, that a mean vector preserves complex whole-cycle temporal information, or that the model supports unseen appliance classes.

---

# 2. Research objective, hypotheses, and scientific limitations

## 2.1 Research question

**Can whole-cycle representations learned from isolated, submetered appliance events transfer to joint multi-appliance disaggregation in unseen houses, improving source attribution and rare-event detection beyond strong event-aware NILM baselines?**

## 2.2 Hypotheses to test independently

**H1 — Category-level signature learning.** Full operating events from different devices in the same class are statistically more similar in a learned space than events from other classes. This is a *distribution-level* hypothesis, not a promise for every pair of waveforms.

\[
\mathbb E[\mathrm{sim}(E(R_i^a),E(R_i^b))]\;>\;
\mathbb E[\mathrm{sim}(E(R_i^a),E(R_j^c))],\quad i\ne j.
\]

**H2 — Useful transfer to mixture separation.** Adding pretrained signatures helps predict true appliance power from the aggregate, beyond adding the same number of generic learned appliance-ID parameters.

**H3 — Joint attribution.** Allowing the five output targets to share features and interact is preferable to five independent conditional runs for at least some overlapping-load scenarios; it is not an identifiability guarantee.

**H4 — Target-absent discrimination.** Training includes genuine target-absent examples so the extractor does not hallucinate an appliance whenever it sees another class with a similar power amplitude.

## 2.3 Core scientific caveats

- A **speaker in VoiceFilter** is one physical identity speaking different utterances. Here, **washing machine** is a **category** covering different devices. Cross-device invariance is a modeling objective, not a given.
- A frozen class embedding carries **no new per-test-house observation**. If exactly five bank vectors never change, mathematically the model remains a function of aggregate alone with a fixed learned conditioning constant.
- Low-frequency active-power NILM has much less distinguishable information than an audio time-frequency spectrogram. Long temporal structure may be necessary but cannot guarantee unique source attribution.
- Stage 1 clean-event classification performance is **not proof** of Stage 2 disaggregation improvement.
- The current five-class training design does **not** establish zero-shot recognition of a new appliance class.

---

# 3. Physics, problem definition, and meaning of “inverse”

## 3.1 Forward measurement model

At aligned sample time `t`, household aggregate active power is approximately:

\[
Y_t=\sum_{i=1}^{5}P_{i,t}+B_t+\varepsilon_t.
\]

- `Y_t` (W): measured mains / aggregate active power.
- `P_{i,t}` (W): one of five monitored target appliances.
- `B_t` (W): power from **all other** household loads not represented as explicit targets; it is not assumed zero.
- `\varepsilon_t`: meter noise, alignment error, and modeling discrepancy.

At one instant the additive measurement operator has one output and at least five target unknowns:

\[
A=\begin{bmatrix}1&1&1&1&1\end{bmatrix},\qquad \operatorname{rank}(A)=1.
\]

This inverse problem is **underdetermined**. The model relies on learned temporal patterns and labelled training mixtures to choose plausible target assignments.

## 3.2 What is solved in this project

The model learns an **amortized inverse mapping**:

\[
H_{\phi}:\big(Y_{t:t+T},\mathbf E_{\mathrm{bank}}\big)
\longmapsto \big(\widehat{\mathbf P}_{t:t+T},\widehat{\mathbf S}_{t:t+T}\big).
\]

No optimization over test-window latent codes is required. This is **not** the earlier neural-adjoint method that minimized a meter reconstruction loss over generator latent variables. It is valid to describe the work as **neural inverse estimation** or **reference-guided joint load disaggregation**; do **not** describe it as test-time inverse optimization or as solving the underdetermination exactly.

## 3.3 What not to impose

\[
\sum_{i=1}^{5}\widehat P_{i,t}=Y_t
\]

is generally **false** because `B_t` is unknown. Do **not** introduce a hard equality meter loss. Optional over-attribution penalties belong to a later carefully masked experiment only.

---

# 4. Stage boundaries and end-to-end data flow

```text
                        Existing aligned processed UK-DALE / REFIT data
                                          |
                                Freeze house split
                                          |
                       +------------------+------------------+
                       |                                     |
               Clean appliance events                    MART-NILM
                 (training homes)                    aggregate windows
                       |                                     |
                       v                                     |
          STAGE 1: shared reference encoder                   |
          individual complete cycles; variable L              |
          classification + contrastive training               |
                       |                                     |
                       v                                     |
                 Frozen encoder                               |
                       |                                     |
                       v                                     |
          STAGE 1.5: build training-only bank                  |
                 5 x 128 vectors                              |
                       |                                     |
                       +------------------+------------------+
                                          |
                                          v
                            STAGE 2: joint extractor
                         Aggregate [B,1,1024] + bank
                             Existing MART backbone
                         + reference conditioning module
                                          |
                                          v
                           Five power/state sequences
                                          |
                                          v
                             STAGE 3: frozen test
                  unseen-house aggregate + SAME frozen bank
                        no test-home submeter as an input
```

### 4.1 Training/inference contract

| Stage | Model is optimized? | Observed signal | Label / supervision | Main persistent output |
|---|---|---|---|---|
| 0 | No | Processed house data | Manual channel-quality audit | Splits, verified adapters, QA |
| 1 | **Yes** | Single clean appliance event | Appliance category | Encoder checkpoint |
| 1.5 | **No** | Selected **training-house** complete events | Known appliance category for indexing | Bank tensor and manifest |
| 2 | **Yes** | Aggregate + bank | Five target power/state sequences where labelled | Joint extractor checkpoint |
| 3 | **No** | Test aggregate + same bank | Test target meter **for scoring only** | Full-length predictions and metrics |

---

# 5. Global data contract and identities

## 5.1 Canonical class order — immutable

```python
APPLIANCES = (
    "kettle",           # 0
    "fridge",           # 1
    "dishwasher",       # 2
    "washing_machine",  # 3
    "microwave",        # 4
)
NUM_TARGETS = 5
SAMPLE_INTERVAL_SECONDS = 8
STAGE2_WINDOW = 1024
EMBEDDING_DIM = 128
```

**Never** rely on alphabetical sort order from a DataFrame or file glob. Persist an explicit `class_to_index` map in each checkpoint/bank. Verify label and bank order in unit tests.

## 5.2 House identity

Identify households by both dataset and house number, e.g. `REFIT_H03`, `UKDALE_H03`. A house split in UK-DALE is independent of one in REFIT.

## 5.3 Canonical `HouseRecord` adapter

```python
@dataclass
class HouseRecord:
    dataset_id: str                      # "refit" or "ukdale"
    house_id: str                        # e.g. "REFIT_H03"
    timestamp_s: np.ndarray              # int64 [N], aligned UTC 8s grid
    aggregate_w: np.ndarray              # float32 [N]
    targets_w: np.ndarray                # float32 [5, N]
    target_valid: np.ndarray             # bool [5, N]
    aggregate_valid: np.ndarray          # bool [N]
    contiguous_segment_id: np.ndarray   # int32 [N]
    channel_metadata: dict
```

`targets_w[i,t]` may contain a placeholder when `target_valid[i,t] == False`, **but no loss can use it**. A nonmonitored appliance must *never* be silently filled with zero and treated as OFF. For a genuinely monitored appliance with a valid zero reading, zero **is** a meaningful OFF target.

## 5.4 Recommended invariants

1. Timestamp grid: `timestamp_s[k+1] - timestamp_s[k] == 8` **within** each valid contiguous segment.
2. `timestamp_s` strictly increases, no duplicate rows after alignment.
3. Power units are physical watts or documented conversions to watts; no per-event unit-variance normalization.
4. No sample or event is allowed to cross a missing-data discontinuity.
5. All training-fitted normalization and calibration parameters are persisted and never refitted on validation/test houses.
6. Retain raw W labels for reported MAE/SAE even if model inputs are scaled.
7. Store the raw channel mapping, including combined/suspicious channels and time-varying reassignment.

---

# 6. House-disjoint experimental split

## 6.1 Proposed **provisional** first fold

| Dataset | Test (unseen) | Validation (unseen for optimization) | Training pool |
|---|---|---|---|
| REFIT | **H2** | **H5** | Audited remainder; full-label candidates H3/H9/H11/H20, H15 to verify |
| UK-DALE | **H2** | **H5, clean channels only** | Audited remainder; H1 full-label candidate, H3 usable kettle channel |

This plan originated in earlier discussion and must be **revalidated against the current channel inventory and current MART splits** before finalizing. UK-DALE H5 includes a washer–dryer channel, which must not be silently relabelled as a pure washing machine. Combined or reassigned channels in either dataset require exclusion, splitting by valid time range, or an explicitly separate task.

**One held-out full-label house per dataset satisfies the initial experiment but is a limited generalization estimate.** The paper-quality study should later rotate test houses where eligible and clearly state the small number of independent devices.

## 6.2 Split timing and consistency

The **same house split** must govern:

- Stage 1 reference encoder fit and early stopping.
- Stage 1.5 bank source events.
- Stage 2 MART aggregate/target windows and optimization.
- Stage 3 evaluation and any calibration.

If older MART-NILM windows use a different split, **filter or regenerate Stage 2 window indices**; do not keep the incompatible split simply because the tensors already exist. No normalization fitting, supervised event selection, synthetic source mixing, or bank construction may use test homes.

## 6.3 Channel coverage versus full-label houses

Stage 1 uses *any* clean training-house appliance submeter; the house need not contain all five classes. Stage 2 first trains on full-five-label windows for a clean joint benchmark, then optionally adds windows with some known targets and a proper loss mask. A label missing for part of a “fully equipped” house makes that **window** partially labelled.

---

# 7. Prepared reference-event data and QA

## 7.1 Status: reference waveforms are already prepared

The researcher reports that reference ON events have already been detected, retained with their full cycles, and **converted to regular 8-second sampling**. Do not automatically re-run raw-REFIT/UKDALE parsers or re-apply interpolation. The next task is to **validate the exact newest event file** and generate indexed examples without destroying its whole-cycle structure.

### 7.2 Earlier uploaded `events.csv` snapshot versus latest prepared events

A directly inspectable older upload at the time this report was produced has schema:

```csv
event_id,dataset,house,channel,event_on,above_threshold,source_idx,timestamp,unix_time,appliance_power_w,aggregate_power_w
```

A full streaming audit of that **particular upload** found approximately **2,279,177 data rows**, **2,628 unique event IDs**, **REFIT dishwasher only**, from **8 houses**. Its within-event adjacent timestamps were **not** uniformly 8 s (frequent 7, 6, 1, 8, 5, 2 s, etc.). Thus this older attached artifact does **not** verify the researcher's newer claim that **the current prepared files** have already been regularized to 8 s. The current files may be different. **Do not overwrite or unnecessarily resample the newer prepared data based on this older snapshot.** Before coding, connect the loader to the actual latest event output and verify its grid and all five class files.

Also, the `event_on` and `above_threshold` flags in this legacy schema must not be confused with file-row membership in a full event: rows with `event_on=0` can be legitimate before/after context or internal pauses. Preserve those rows if they are inside the event boundaries.

## 7.3 Canonical Stage 1 record

A single reference sample is a *complete independent appliance event* `R_{i,j}` in real physical watts, along with a class label and provenance.

\[
R_{i,j}=[P_i(t_j^{start}),\ldots,P_i(t_j^{end})]\in\mathbb R^{L_{i,j}},
\qquad \Delta t=8\,\mathrm{s}.
\]

The number of points `L_{i,j}` differs both across classes and **within** a class. **This is intended.** No global reference-window length is imposed.

Examples (illustrative, not measured):

| Target | Hypothetical complete-event duration | Points at 8 s |
|---|---:|---:|
| Kettle A | 88 s | 11 |
| Kettle B | 160 s | 20 |
| Microwave A | 200 s | 25 |
| Fridge compressor run | 1,200 s | 150 |
| Dishwasher | 6,400 s | 800 |
| Washer | 8,000 s | 1,000 |
| Longer washer cycle | 10,400 s | 1,300 |

**Do not** stretch 11 kettle samples to imitate 1,000 samples, or crop 1,300 washer samples down to 1,024 as a default.

## 7.4 Event semantics

- Kettle: one complete boiling/activation episode.
- Microwave: one complete activation (short pauses depending on validated device-level rule).
- Fridge: one compressor operating event; do not call this a complete daily refrigerator usage cycle.
- Dishwasher and washing machine: one complete multi-phase program, **including real internal idle periods**.
- “Complete” means observed beginning and end inside a contiguous valid recording, not merely a long run of `above_threshold == 1`.
- Preserve original power values during real OFF intervals; **padding is not OFF**, and **missing data are not OFF**.

Use the pre-existing successful event filtering logic. Keep the event thresholds (`τ_i`), gap-merge rules (`G_i`), and minimum durations (`D_i`) in a versioned config. Review long cycles and extreme gap-merges because two adjacent real cycles could be incorrectly joined.

## 7.5 Pre/post OFF context

The current reference waveforms may already include context; do not append context twice. If future extraction needs an explicit reproducible convention, a **proposed** initial rule is:

\[
C_{pre}=C_{post}=\min\left(C^{max}_i,\max(1,\lfloor0.10 L_{cycle}\rfloor)\right),
\]

with proposed caps of 4 samples (kettle/microwave), 8 (fridge), 16 (washer/dishwasher) per side. Only add **actual contiguous valid** before/after readings, never manufactured zeros, intervals from another event, or missing-data interpolation. This is a configurable hypothesis, not an established optimal amount. Internal true OFF phases remain fully preserved.

## 7.6 Stage 1 event manifest — **one row per event, not per timestep**

```csv
event_id,dataset_id,house_id,appliance,channel,source_path,segment_id,source_start_idx,source_end_idx_exclusive,on_start_idx,on_end_idx_exclusive,samples_8s,duration_s,pre_context_samples,post_context_samples,peak_w,p90_w,quality_flag,is_complete,split
```

`source_start_idx`, `source_end_idx_exclusive` refer to the latest 8-second source array. The stage-1 event length equals `end-start` and the slice convention is **start inclusive, end exclusive**. If events are stored as contiguous groups in a large event file instead of pointers to source arrays, additionally store `event_file_path`, `event_first_row`, `event_last_row_exclusive` and verify nonoverlap.

Example illustrative manifest row:

```csv
REFIT_H03_K_00001,refit,REFIT_H03,kettle,A5,data/processed/refit/h03.parquet,10,10120,10137,10121,10135,17,136,1,2,1940,1895,ok,true,train
```

**Important:** The entry is illustrative; do not assume REFIT H03 kettle actually maps to `A5`. Use the verified channel mapping.

## 7.7 Event audit before training

Generate `stage1_event_statistics.csv` summarizing counts, duration percentiles, number of houses/devices, missingness and context fractions, by `(dataset, house, class)` and `(dataset, class)`:

- Event count (actual independent events; **not** number of augmented windows).
- Duration median, P5/P25/P75/P95/P99 and longest valid event.
- Signal peak and P90 ON power distributions.
- Actual 8-second increment consistency.
- Events with too few samples, truncated boundary, or long internal recording gaps.
- Proportion of reference points that are genuine OFF context / internal OFF.
- Channel quality and potential shared-circuit contamination.

Manually inspect random and extreme waveforms for each class, particularly complicated washer/dishwasher cycles. Keep accepted/rejected reasons in a versioned audit log.

---

# 8. Stage 1 dataset, batching, and augmentation

## 8.1 DataLoader output

```python
@dataclass
class ReferenceSample:
    power_w: torch.Tensor   # float32 [L]; L varies by event
    appliance_id: int       # 0..4
    event_id: str
    house_id: str
    dataset_id: str
    duration_s: float
    real_length: int
```

The event loader reads a complete waveform, applies **Stage 1-specific** deterministic preprocessing, and optionally train-only augmentation. Validation/test use no random augmentation.

### 8.2 Variable-length batching: two supported strategies

**Recommended first implementation — group by class/approximate length.** Sample 5 classes × 8 independent events; within each class group, pad only to that group's longest `L`. Run each padded group through **the same shared encoder instance**, gather 5 × 8 fixed-dimensional embeddings, concatenate them, compute one global supervised-contrastive loss, then call `backward()` once.

```text
8 kettle events:    L = 10..25    -> pad to 25   -> E_shared -> [8,128]
8 fridge events:    L = 80..190   -> pad to 190  -> E_shared -> [8,128]
8 washer events:    L = 500..1300 -> pad to 1300-> E_shared -> [8,128]
8 dishwasher:       ...                           -> E_shared -> [8,128]
8 microwave:        ...                           -> E_shared -> [8,128]
                                                     |
                                        concat embeddings -> [40,128]
                                                     |
                                           CE + SupCon -> one update
```

Alternative: pad all `[B,1,Lmax]` with valid masks and process one forward pass. This is simpler but can waste substantial memory if short kettle events share a batch with very long washing cycles.

**Padding implementation requirement:** Valid masks must propagate through downsampling and all pooling stages. Prefer GroupNorm or masked-safe normalization rather than unmasked BatchNorm pooling over many padded positions. Padded zeros can still affect convolution near the valid boundary; avoid training a representation that inadvertently learns group-specific padding artifacts. For difficult cases, process each event or each exact-length group independently and accumulate encoded tensors before the loss.

## 8.3 Sampling protocol

Choose `class -> house -> independent event`, not a random sample time; otherwise fridge and very long cycles dominate. Preserve cross-house positive pairs whenever possible. Overlapping crops of the **same physical event** are not independent observations and should not be the sole positives. No house is allowed to cross split boundaries.

**Initial balanced batch:** `5 × 8 = 40` independent events, except when rare classes need sampling with replacement of *different training events* (track this explicitly). Consider separate group sizes if classes have very different data counts. The actual number of events per class is determined by the audit, not prescribed as 1,000.

## 8.4 Data augmentation: conservative and optional

\[
\widetilde R(t)=a\,R(t)+\eta(t),\qquad a\sim U(0.98,1.02).
\]

Small additive noise scaled to observed meter noise and mild boundary jitter preserving the complete physical cycle may be tested. Never permute washer phases, massively speed-stretch an event, or normalize each waveform to unit variance in a way that erases informative absolute power. Synthetic views are **not independent physical cycles**. Apply only to training homes, never to held-out validation/test events.

---

# 9. Stage 1 architecture: shared variable-length signature encoder

## 9.1 Mandatory input/output contract

\[
E_\theta: \big(R\in\mathbb R^{L},\; L\ge L_{min}\big)
\longrightarrow e\in\mathbb R^{128}.
\]

`L` can change per sample; sampling interval remains 8 s. The dimension is fixed. One shared parameter set `θ` serves **all five classes**. No aggregate is present in Stage 1.

## 9.2 Input feature channels

Given raw power `P_t` in watts, the following initial **train/config-frozen** transforms retain physical scale:

\[
x_{1,t}=\log\left(1+\frac{\max(P_t,0)}{100\;\mathrm W}\right),
\]

\[
x_{2,t}=\operatorname{asinh}\left(\frac{P_t-P_{t-1}}{100\;\mathrm W}\right).
\]

Set the first discrete difference to 0; record negative-measurement rules, sensor noise and saturation in config. The 100 W scale is an engineering starting point, not a tuned scientific constant. Add a compact metadata vector:

\[
m(R)=\left[\log\left(1+\frac{8L}{60}\right),\quad
\log\left(1+\frac{P_{90}(R)}{100\,\mathrm W}\right)\right].
\]

This makes duration and typical magnitude explicit and testable. To avoid learning *only* trivial class duration, run waveform-free duration/amplitude baselines and embedding diagnostics.

## 9.3 Recommended v1 encoder blocks

| Stage | Operation | Example shape for one event | Notes |
|---|---|---|---|
| Input | `power`, `Δpower` | `[2,L]` | Keep valid-length mask. |
| Stem | `Conv1d(2,64,k=7,p=3)`, activation | `[64,L]` | Small events remain meaningful. |
| Residual block 1 | `64 -> 128`, stride 2 | `[128,ceil(L/2)]` | GroupNorm / GELU. |
| Residual block 2 | `128 -> 128`, stride 2 | `[128,ceil(L/4)]` | Preserve mask. |
| Temporal context | 2–4 residual dilated temporal blocks | `[128,L']` | Receptive field should be documented. |
| Pool | Masked mean/max + ordered 4-bin pooling | Fixed length | Preserve rough cycle ordering. |
| Metadata fusion | Concatenate duration and P90 | Fixed length | Test physical feature ablation. |
| Embedding | Small MLP | `[128]` | Deployed feature. |
| Projection (train only) | `MLP(128->64)` + L2 norm | `[64]` | SupCon space. |
| Classification (train only) | `Linear(128->5)` | `[5]` | CE supervision. |

The exact number of temporal blocks and dilation schedule must be chosen to accommodate long events and validated for **very short** events (e.g., `L≈10`); 4×/8× downsampling must not reduce short events to an invalid one-timestep representation. Record per-layer receptive field; the example architecture is not yet implemented.

### 9.4 Ordered masked pooling

Let `F=ConvNet(R)` with shape `[C,L']` and valid timestep mask `m_t`. The global mean is:

\[
z_{mean,c}=\frac{\sum_{t=1}^{L'}m_tF_{c,t}}{\max(1,\sum_tm_t)}.
\]

The masked max excludes invalid timesteps (`-inf` before max, with explicit all-invalid rejection). Divide the **valid sequence** into four equal-order temporal bins and pool each bin separately:

\[
z_{ord}=[\mathrm{mean}(F_{Q_1}),\ldots,\mathrm{mean}(F_{Q_4})].
\]

Use masked adaptive binning defined by true input length, not by batch padding length. For exceptionally short events with fewer than four post-downsampling samples, use a documented fallback (interpolate feature-bin boundaries / reuse valid samples, or reduce downsampling) rather than producing NaN. Concatenate global/ordered features with metadata and map through an MLP to `e∈R^128`.

**Design rationale:** Global pooling alone could erase the phase ordering relevant to washers and dishwashers; the ordered bins preserve coarse early/middle/late structure without hand-labelled phases.

---

# 10. Stage 1 losses and representation learning

## 10.1 Proposed default objective

\[
\boxed{\mathcal L_{\mathrm{S1}}=\mathcal L_{\mathrm{SupCon}}+0.3\mathcal L_{\mathrm{CE}}.}
\]

The combination is an **initial proposal**, not an experimental conclusion. GE2E is the original VoiceFilter speaker-encoder loss, while this project tentatively uses supervised contrastive learning plus classification; these are **not the same algorithm**.

### 10.2 Supervised contrastive loss

For normalized projected embedding `z_a`, positive index set `\mathcal P(a)` contains **other independent events of the same appliance class** (ideally from other training homes):

\[
\mathcal L_{SupCon} =
-\frac{1}{|\mathcal I|}\sum_{a\in\mathcal I}
\frac{1}{|\mathcal P(a)|}\sum_{p\in\mathcal P(a)}
\log\frac{\exp(z_a^\top z_p/\tau)}
{\sum_{k\ne a}\exp(z_a^\top z_k/\tau)};
\quad \tau=0.10.
\]

Exclude anchors that have no positive pairs; do not allow NaNs from empty positive sets. With 8 events per class, positives should normally exist. A class-balanced batch prevents all negatives from being the same few classes.

### 10.3 Cross-entropy head

\[
\mathcal L_{CE}=-\frac1B\sum_{a=1}^{B}\log\operatorname{softmax}(C(e_a))_{c_a}.
\]

The classification head is only a **training scaffold**; Stage 2 receives the feature embedding `e`, not the five class logits.

### 10.4 Avoid feature collapse

If SupCon is too strong, all washing-machine cycle embeddings may become nearly identical and lose information on stage duration/amplitude. Compare:

- CE-only encoder.
- SupCon + CE.
- GE2E-inspired objective (optional comparator).
- With/without raw physical metadata.

Evaluate both class retrieval **and downstream Stage 2 metrics**. The best class discrimination is not necessarily the best conditional source extraction.

---

# 11. Stage 1 optimization and evaluation

## 11.1 Initial configuration

| Parameter | Initial v1 value |
|---|---|
| Encoder embedding | 128 |
| Contrastive projection | 64 |
| Classes per batch | 5 |
| Independent events / class / batch | 8 |
| Effective batch | 40 |
| Loss | SupCon + 0.3 CE |
| Temperature | 0.10 |
| Optimizer | AdamW |
| LR | `3e-4` |
| Weight decay | `1e-4` |
| Epoch budget | 100 |
| Early stopping patience | 15 validation epochs |
| Gradient norm clip | 1.0 |
| Seeds | 0, 1, 2 |
| Precision | BF16 AMP if supported; else FP32 |
| Data workers | Tune for host/disk speed |

Use train-only scales. Dynamic augmentation only in train. Save best validation checkpoint, latest checkpoint and a resume state with optimizer/scheduler/scaler and RNG states.

## 11.2 Validation checks

- House-disjoint class classification accuracy and macro-F1.
- Cross-house nearest-neighbor retrieval for the five categories.
- Confusion matrix for heater-like classes (e.g., dishwasher heating vs kettle).
- Same-class/different-house similarity versus different-class similarity.
- Statistics of embedding norm and within-class variation.
- Duration-only and peak-only class baseline; assess whether waveform gives added signal.
- Robustness to short and long complete cycles.
- Visual embedding plots are diagnostic only, not primary scientific evidence.

**Checkpoint selection uses validation homes; never test homes.** Holdout test-home clean events may be scored once after model selection, but cannot be fed into the fixed production bank.

## 11.3 Stage 1 acceptance gate

A successful Gate 1 requires valid event decoding, correct cross-house data provenance, non-NaN loss, non-trivial held-out-house representation performance, stable embeddings, and a reproducible `reference_encoder_best.pt`. High classification accuracy alone is not sufficient to start a paper claim.

---

# 12. Stage 1.5: reference bank and its provenance

## 12.1 Distinguish encoder learning from embedding construction

Stage 1 **learns the function** `E_θ`. Stage 1.5 uses frozen `E_θ*` to compute **specific reusable vectors**. The vectors are not automatically obtained merely because the encoder weights were trained.

Select `M` independent complete cycles per appliance **from training homes only**, preferably varied in duration and true device/house. Start with `M=5`, but compare `M∈{1,3,5,10,20}` and possibly more if performance has not saturated.

\[
e_i^{bank}=\operatorname{Pool}_{j=1\ldots M}E_{\theta^*}(R_{i,j}).
\]

Initial pool is arithmetic mean. If the learned embedding is explicitly L2 normalized, a mean-then-renormalize policy is possible, but apply it consistently at Stage 2 and Stage 3. **If physical magnitude/duration are encoded in vector norm, blindly normalizing the whole vector could erase information.** An alternative is to normalize shape features and concatenate a separately scaled metadata vector.

## 12.2 Separate reference selection from query targets

For a cleanest Stage 2 experiment, create a deterministic **train-house reference-event pool** that does not reuse the *exact* target event as the Stage 2 query's reference. This avoids trivial memorization or waveform copying when testing dynamic-reference variants. A fixed training-only category bank with event overlap is not equivalent to explicit target-label leakage across houses, but independent reference/query events make the intended interpretation much cleaner.

## 12.3 Bank artifact specification

```python
reference_bank = {
    "embeddings": torch.FloatTensor(5, 128),
    "appliance_order": APPLIANCES,
    "encoder_checkpoint_sha256": "...",
    "split_sha256": "...",
    "normalizer": {...},
    "pooling": "mean",
    "events_per_class": 5,
    "reference_event_ids": {name: ["..."] for name in APPLIANCES},
    "version": "v1",
}
```

Artifacts:

```text
data/reference_bank/reference_bank_v1.pt

data/reference_bank/reference_bank_v1_manifest.csv
```

The manifest should include `class, event_id, house_id, dataset_id, device_id_if_known, duration_s, peak_w, encoder_hash, split_hash`. Rebuilding with the same seed/config/data must reproduce exactly the same bank.

## 12.4 Fixed versus dynamic reference experiments

- **Fixed bank:** same `[5,128]` in every Stage 2 and Stage 3 call; implementation baseline.
- **Dynamic training bank:** select different *training-house* events/subsets across Stage 2 steps, with no exact-event overlap with the query; use frozen encoder to compute/cached event embeddings. At Stage 3 use the frozen deterministic bank.
- **Multi-prototype bank:** keep washer/dishwasher diverse cycle representations instead of averaging them; a later extension only after fixed-bank results.

**Critical risk:** A fixed category bank could function like five learned class IDs. Stage 2 comparison against ID-conditioned MART is mandatory.

---

# 13. Stage 2 data: reuse existing MART-NILM dataset

## 13.1 Primary decision

**Do not build Stage 2 from the reference `events.csv`.** Those clips are event-centered and do not preserve all natural target-absent periods, unknown backgrounds, overlaps, or original household time distribution.

**Reuse the existing MART-NILM aggregate-window Dataset / DataLoader** built from aligned continuous UK-DALE and REFIT, subject to a consistent house split and label-validity audit. The Stage 2 dataset is not required to have every appliance active in a window. The *model output* still contains all five targets at every timestamp.

## 13.2 Window and tensor contract

\[
T=1024,\quad \Delta t=8\,\mathrm s,\quad T\Delta t=8192\,\mathrm s.
\]

A full-labelled batch:

```python
aggregate_w    # float32 [B, 1, 1024]
target_power_w # float32 [B, 5, 1024]
target_state   # float32 [B, 5, 1024]
label_valid    # bool    [B, 5, 1024]
references     # float32 [B, 5, 128], broadcast from frozen bank
```

The bank can be loaded in the training process and broadcast per batch, **not** duplicated to disk for each sample.

Stage 1 input length and Stage 2 window length need **not** match:

\[
R_{K}\in\mathbb R^{12}\to e_K\in\mathbb R^{128},\quad
R_{W}\in\mathbb R^{900}\to e_W\in\mathbb R^{128},
\]

while Stage 2 always uses `Y∈R^1024` and predicts `P∈R^{5×1024}`. Stage 2's windows represent *household time*, not normalized copies of complete appliance cycles.

## 13.3 Old MART data reuse checklist

- Check target order matches canonical `APPLIANCES` exactly.
- Check all existing windows use `8 s` and are aligned across mains/submeters.
- Check `T=1024` and label arrays have the stated shape.
- Check the old `train/val/test` house split matches **this** project's Stage 1 split.
- Check the old loss balancing normalizers were fit on **training houses only**.
- Check background treatment and data gaps; do not bridge discontinuities.
- Check that zero means genuine OFF, and unavailable channels can be masked.
- Keep the same recorded/synthetic sampling policy across scientific comparison groups.

If incompatibilities are found, change the **adapter/window index**, not the whole project architecture. `MARTDatasetAdapter` should translate old loader outputs into the canonical `JointWindowBatch` structure.

## 13.4 Partial-label support — separate experiment

A house with only fridge, dishwasher and washer valid channels contributes valid losses to those three targets, but all five are **still predicted**. For instance:

\[
m_{h,:}=[0,1,1,1,0].
\]

Kettle/microwave are *unobserved*, not zero. The main v1 baseline can restrict Stage 2 to full-five-target training windows; masked partial labels can be introduced only as an ablation to test the value of training-house diversity.

---

# 14. Stage 2 model: minimally modify MART first

## 14.1 Why not rebuild the extractor immediately?

The existing MART-NILM already uses a shared temporal encoder, joint multi-appliance power/state prediction, and cross-appliance feature exchange. Replacing the whole architecture with a new CNN + attention + decoder would confound any gain due to reference signatures with changes in backbone size, optimization or architecture. **V1 must preserve the existing MART backbone wherever possible.**

### 14.2 Proposed injection point

At an intermediate aggregate feature tensor `Z_Y` with channel dimension `C`, use a small shared MLP to map the `i`th reference embedding into FiLM modulation:

\[
(\gamma_i,\beta_i)=\operatorname{MLP}_{FiLM}(e_i),\quad
Z_i=(1+\gamma_i)\odot Z_Y+\beta_i.
\]

`1+γ` enables near-identity initialization. `γ_i,β_i∈R^C` are shared transformations across targets, not separate per-class networks. Broadcast across the temporal length and create five conditioned target features. Feed these to MART's existing cross-appliance interaction/decoders or to the closest compatible existing query-feature interface.

**Critical engineering note:** The exact insertion depends on the current MART code. Do not arbitrarily replace its appliance-specific decoders or state-gating. First inspect `forward()` tensor shapes and compare two placements (before vs after shared feature exchange) only if needed. If the existing decoder expects no target-axis feature, add a thin adapter rather than restructuring everything.

### 14.3 Tensor example

```python
Y               # [B, 1, 1024]
Z = F_mart(Y)     # [B, C, T_prime]
E               # [B, 5, 128]
film = MLP(E)    # [B, 5, 2*C]
gamma, beta = film.chunk(2, dim=-1)
Z_cond = (1 + gamma[..., None]) * Z[:, None, :, :] \
       + beta[..., None]                       # [B, 5, C, T_prime]
# Existing MART joint interaction -> existing power/state heads.
```

This code is illustrative: adapt to actual MART layout, dtype and normalization. Initialize FiLM final layer weights/bias to zeros (or otherwise near zero) so the initial mapping minimally perturbs pretrained/familiar features.

### 14.4 Joint output contract

\[
(\widehat{\mathbf P},\widehat{\mathbf S})=H_\phi(Y,E),\quad
\widehat{\mathbf P}\in\mathbb R_{\ge0}^{B\times5\times1024},\quad
\widehat{\mathbf S}\in[0,1]^{B\times5\times1024}.
\]

Output all five targets **simultaneously**. The presence of a Kettle event does not require fridge, microwave or washer to be ON. A household window can contain 0–5 active monitored targets plus unobserved residual loads. Long cycles may cross several 1024-point windows; keep their true timestamps and labels and reconstruct by overlap-add at inference.

### 14.5 Stronger joint model is an optional later iteration

Only if the minimal integration demonstrates that signatures carry useful information, consider replacing fixed appliance heads with a **shared variable-query decoder** and introducing lightweight cross-query attention over `[B,T',N,C]` (attend across `N` queries at each coarse timestep, not over an enormous flattened `N×T` token grid). Variable number `N` is not a validated v1 capability and must be tested separately. **Do not claim the initial MART-compatible model can add an unseen air conditioner without retraining.**

### 14.6 Why no direct VoiceFilter STFT mask?

VoiceFilter predicts a spectrogram mask; its input has frequency/time structure. Eight-second active-power NILM has only one scalar per time point. V1 should use existing MART's direct power/state heads. A nonnegative temporal mask `m_i(t)Y(t)` can be explored as a comparator if strictly nonnegative aligned mains conditions hold, but it is not a necessary architectural inheritance.

---

# 15. Stage 2 supervision and loss design

## 15.1 Baseline rule: retain prior MART loss first

For scientifically interpretable comparisons, the initial Stage 2 experiment should use **the same loss, state labels, focal-event sampling and bounded appliance-wise loss balancing as the existing MART baseline**, adding only reference conditioning. This ensures gains are not explained by a different loss recipe.

If the existing code cannot easily support partial labels, start on full-label training homes, **without** guessing missing values. Move to masked partial-label training after a working full-label result. Keep loss variants in explicit configs and ablation tables.

## 15.2 Generic masked losses (for the later partial-label variant)

Index batch `b`, target `i`, timestep `t`. Define:

- `p[b,i,t]`: true appliance power (W) if label is valid.
- `s[b,i,t]∈{0,1}`: true state if label is valid.
- `\ell[b,i,t]`: predicted ON logits.
- `\hat a[b,i,t]≥0`: predicted ON amplitude, if architecture exposes a separate amplitude branch.
- `\hat p[b,i,t]≥0`: predicted final gated or direct power.
- `m[b,i,t]∈{0,1}`: both target meter and mains/sample validity.
- `q_i>0`: fixed training-set per-target loss scale, optionally `max(50 W, P95_ON_train_i)`.

Let `\mathcal V_i={(b,t):m=1}` and `\mathcal V_i^{ON}={(b,t):m=1,s=1}`.

**Masked all-point power L1:**

\[
\mathcal L_{P,i}=\frac{\sum_{b,t}m_{b,i,t}|\hat p_{b,i,t}-p_{b,i,t}|}
{q_i\max(1,\sum_{b,t}m_{b,i,t})}.
\]

**State loss:**

\[
\mathcal L_{S,i}=\frac{\sum_{b,t}m_{b,i,t}\,\mathrm{BCEWithLogits}(\ell_{b,i,t},s_{b,i,t};w_i^+)}
{\max(1,\sum_{b,t}m_{b,i,t})}.
\]

Provisional positive weight `w_i^+=clip(N_OFF_i/N_ON_i, 1,20)`, computed from **training** data only. If MART already uses a superior established balancing rule, preserve it in the primary experiment.

**ON-state power amplitude loss:**

\[
\mathcal L_{ON,i}=\frac{\sum_{b,t}m_{b,i,t}s_{b,i,t}|\hat a_{b,i,t}-p_{b,i,t}|}
{q_i\sum_{b,t}m_{b,i,t}s_{b,i,t}},
\]

but **skip the term** for targets with no valid ON samples. If the deployed head has no separate amplitude, define it using `\hat p` with the appropriate interpretation rather than inventing a nonexistent head.

**OFF leakage:**

\[
\mathcal L_{OFF,i}=\frac{\sum_{b,t}m_{b,i,t}(1-s_{b,i,t})\hat p_{b,i,t}}
{q_i\sum_{b,t}m_{b,i,t}(1-s_{b,i,t})},
\]

and skip the term if the denominator has no valid OFF examples.

Average each defined loss **only over classes with applicable valid targets**. Do not allow all-invalid batches to backpropagate an artificial zero and skew batch statistics; skip them safely.

## 15.3 Optional revised objective — **not** the initial fairness baseline

\[
\mathcal L_{S2}=1.0\,\mathcal L_P+1.0\,\mathcal L_S
+0.5\,\mathcal L_{ON}+0.2\,\mathcal L_{OFF}
+\lambda_{excess}\,\mathcal L_{excess}.
\]

Start `λ_excess=0`. Coefficients above are engineering candidates, **not** required if existing MART loss differs.

An optional consistency penalty is:

\[
\mathcal L_{excess}=
\frac1{TP_0^2}\sum_t
\left[\max\left(0,\sum_{i=1}^5\hat p_{i,t}-y_t-\delta\right)\right]^2.
\]

This penalty is appropriate only for reliable time alignment, nonnegative consumption, and a justified tolerance `δ`. In earlier attached legacy event data some target power readings exceeded concurrent mains readings, so a rigid or unmasked excess loss could be harmful. Unknown residual load does **not** justify an equality loss.

## 15.4 Temporal event labels

Reuse the existing MART state-label rules per appliance: power threshold `τ^P_i`, merge gaps `G_i`, minimum run `D_i`, valid-span boundary handling, and documented spike removal. Stage 1 event boundaries and Stage 2 ON labels are related but not identical objects: Stage 1 can include real internal OFF pauses in a *complete cycle* while Stage 2 still predicts momentary ON/OFF states.

---

# 16. Stage 2 sampling and training

## 16.1 Sample windows, not complete events

A Stage 2 sample is one contiguous `T=1024` window of household aggregate and five aligned target traces. It may contain a tiny kettle event, a partial washer cycle, multiple fridge cycles, no monitored appliance active at all, or overlapping appliances. These are all valid when labels/coverage are correct.

Train sampler can overrepresent rare target activation (MART's existing focal-event strategy), but **validation and test must retain natural prevalence**. Include genuine target-absent and high-confusion negative windows; otherwise rare appliances will be hallucinated.

## 16.2 Full-label primary experiment

Use verified five-label houses/windows, existing MART sampling and objective, identical train/validation/test homes and same seeds as comparators. This is the cleanest first integration with joint five-target losses.

## 16.3 Partial-label auxiliary experiment

Once the baseline runs, add training homes lacking some target submeters. The model still predicts all five targets, but losses for unknown channels are masked out. Provisional full-label:partial-label window sampling `70:30` is only a starting point to validate, not a rule. Avoid allowing a single overrepresented house to dominate; mix across houses and datasets.

## 16.4 Synthetic mixing

Keep the previous MART focal-event background-stratified mixing **unchanged** in the first scientific comparison if it was part of the baseline protocol. Any later increase in synthesis should be ablated separately. In synthetic windows, construct all five target labels and unknown residual backgrounds consistently; do not inject a known target onto a background that already contains the same target and then incorrectly label it.

## 16.5 Optimization

**Preferred:** Copy the already validated MART-NILM optimizer, scheduler, 3 seeds, learning-rate reductions, and balancing parameters into the new experiment configuration for fair comparison.

**Fallback stand-alone initial settings, if needed:** AdamW `lr=3e-4`, weight decay `1e-4`, batch size `8` for `[B,5,1024]` on available GPU, max `150` epochs, min `40` epochs, early stopping patience `15`, gradient clipping `1`, BF16 AMP where supported; optional ReduceLROnPlateau, factor `0.5`. This is not a mandate to replace the MART regimen.

Frozen Stage 1 encoder and bank mean Stage 2 updates only `φ` (MART backbone, conditioning module, existing heads). Do not accidentally detach Stage 2 conditioning weights from the gradient graph. A cached bank does not require gradients.

### Stage 2 sanity / unit-overfit tests

- Overfit 4–16 clean windows to verify loss and gradient path.
- Check a target-present example and target-absent example for each class.
- Check predicted power units (W), nonnegativity, and ON-state thresholds.
- Check class-to-reference ordering using known examples.
- Check loss masking on full and partial houses.
- Check identical data, split and sampling between baseline and conditioned experiments.

---

# 17. Stage 3 inference

## 17.1 Inputs

- **Aggregate from a held-out house**, aligned 8 s, as required by the pretrained MART input pipeline.
- **Exactly the same saved `reference_bank_v1.pt`** used in Stage 2 (for the fixed-bank experiment).
- Stage 2 checkpoint and **training-only** scaling/state calibration.
- **No** new test-house ON reference, no new test-house submeter input, no gradient descent.

## 17.2 Sliding prediction

Break each continuous valid aggregate segment into windows `T=1024`, initial inference stride `S=512` (68 min 16 s). For a house sequence shorter than T, pad input with an explicit valid mask / boundary policy and crop output. Do not slide across gaps.

For overlapping windows `k`, combine predicted time-aligned powers by weighted overlap-add:

\[
\hat p_i(t)=
\frac{\sum_k w_{k,t}\hat p_i^{(k)}(t)}
{\sum_k w_{k,t}},
\quad \sum_kw_{k,t}>0.
\]

Use nonzero coverage weights at edges (e.g., uniform weighting as a safe starting point) and test exact timestamp reconstruction. State logits/probabilities may be overlapped consistently before thresholding; document choice.

## 17.3 Test label use

Held-out submeter data may be loaded in a **separate evaluator after predictions have been generated** to calculate metrics. They must not influence reference selection, model inference, normalization, early stopping or manual hyperparameter selection. For deployment, omit the evaluator completely.

---

# 18. Metrics, baselines and ablations

## 18.1 Essential metrics

Report by **dataset × test house × appliance**, followed by macro averages:

- **MAE (W)** over all valid aligned samples.
- **ON-MAE (W)** over valid ground-truth ON times.
- **Macro state F1**, Precision and Recall.
- **False positive rate / active-time false alarm rate** on target-absent periods, with the exact definition disclosed.
- **Event-level F1** using an a priori onset/offset matching tolerance.
- **Blockwise SAE** / energy-error metric with fixed block size and precise formula.
- PR-AUC, if probabilities are available and positive state support is sufficient.
- Inference runtime, memory, checkpoint size, number of true training events.

No rebalancing of test time; all long natural OFF periods remain. Tune probability thresholds on validation houses only.

## 18.2 Comparisons required to identify causal contribution

| Experiment | Main change | Joint outputs? | Question |
|---|---|---|---|
| **B0** | Original MART-NILM | Yes | Strong baseline. |
| **B1** | MART + equally sized learned appliance-ID conditioning | Yes | Is any gain simply conditioning? |
| **B2** | MART + **fixed pretrained** reference bank | Yes | Does clean-event pretraining add value? |
| **B3** | MART + dynamic train-reference subsets (frozen encoder) | Yes | Does varying the actual reference matter? |
| **B4** | B3 + partial-label houses | Yes | Does extra device/house diversity help? |
| **B5** | B3 + optional added losses/synthesis | Yes | Incremental gains beyond strong v1. |

Keep the backbone, parameter budget, data split, event injection sampler, optimizer, training seeds and evaluation identical as far as practicable. Log parameter counts, number of optimization steps and compute time.

### 18.3 Stage 1-specific ablations

- CE-only vs CE + SupCon vs optional GE2E.
- Full complete cycle vs cropped partial cycle (reference representation quality and Stage 2 performance).
- Power-only vs power+delta vs power+delta+duration metadata.
- Global pooling vs ordered temporal pooling.
- Number of genuine independent training cycles and distinct train houses.
- Reference bank size `M=1,3,5,10,20`, fixed selection seed.
- Mean bank vs representative medoid bank (optional).
- Randomly shuffled wrong-class bank test (must generally harm or change target assignment if conditioning is used; results must be interpreted with frozen model).

### 18.4 Go/no-go criterion

If fixed/dynamic pretrained references cannot consistently improve upon **B1 ID-conditioned MART** on held-out-house MAE, ON-MAE, and rare-event false positive/recall under equivalent conditions, do **not** claim the Stage 1 encoder is advantageous. Explore temporally resolved reference matching as a separate research hypothesis, not as an unnoticed architectural addition.

---

# 19. Repository structure, interfaces, and illustrative pseudocode

## 19.1 Suggested organization

```text
Reference_Conditioned_Inverse_NILM/
├── README.md
├── pyproject.toml
├── configs/
│   ├── common.yaml
│   ├── splits.yaml
│   ├── stage1.yaml
│   ├── reference_bank.yaml
│   ├── stage2.yaml
│   └── inference.yaml
├── data/
│   ├── processed/                        # links to EXISTING processed data
│   │   ├── UKDALE/
│   │   └── REFIT/
│   ├── splits/house_split_v1.json
│   ├── audits/channel_audit.csv
│   ├── audits/stage1_event_statistics.csv
│   ├── stage1/
│   │   ├── latest_events/               # actual 8s event output or path config
│   │   ├── train_events.csv
│   │   ├── val_events.csv
│   │   └── test_events.csv
│   ├── stage2/
│   │   ├── train_windows.csv            # may point to legacy MART arrays
│   │   ├── val_windows.csv
│   │   └── test_windows.csv
│   └── reference_bank/
│       ├── reference_bank_v1.pt
│       └── reference_bank_v1_manifest.csv
├── src/reference_nilm/
│   ├── common/
│   │   ├── configuration.py
│   │   ├── checkpoints.py
│   │   ├── seed.py
│   │   └── metrics.py
│   ├── data/
│   │   ├── channel_mapping.py
│   │   ├── house_adapter.py
│   │   ├── event_manifest.py
│   │   ├── reference_dataset.py
│   │   ├── reference_collate.py
│   │   ├── class_house_sampler.py
│   │   ├── mart_dataset_adapter.py
│   │   └── joint_window_dataset.py
│   ├── stage1/
│   │   ├── encoder.py
│   │   ├── pooling.py
│   │   ├── losses.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── build_reference_bank.py
│   ├── stage2/
│   │   ├── reference_conditioning.py
│   │   ├── mart_integration.py
│   │   ├── losses.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── stage3/
│       ├── inference.py
│       ├── overlap_add.py
│       └── evaluate.py
├── scripts/
│   ├── 00_verify_data_and_splits.py
│   ├── 01_index_reference_events.py
│   ├── 02_train_stage1.py
│   ├── 03_build_bank.py
│   ├── 04_verify_mart_adapter.py
│   ├── 05_train_stage2.py
│   ├── 06_evaluate_cross_house.py
│   └── 07_compare_ablations.py
├── tests/
│   ├── test_sampling_grid.py
│   ├── test_event_integrity.py
│   ├── test_no_house_leakage.py
│   ├── test_padding_and_length.py
│   ├── test_reference_bank_order.py
│   ├── test_mart_adapter.py
│   ├── test_invalid_target_mask.py
│   ├── test_stage1_encoder_shapes.py
│   ├── test_joint_output_shapes.py
│   ├── test_reference_changes_predictions.py
│   ├── test_absent_target.py
│   └── test_overlap_add.py
├── checkpoints/
│   ├── stage1/
│   └── stage2/
├── runs/
└── reports/
```

Every file should have **one clear responsibility**, type hints for public interfaces, docstrings giving tensor shapes and units, and mathematical explanations adjacent to nontrivial loss/conditioning code. Do not put the entire project into a single gigantic `train.py`.

## 19.2 Configuration example

```yaml
project: Reference_Conditioned_Joint_NILM
seed: 0
appliances: [kettle, fridge, dishwasher, washing_machine, microwave]
sampling_interval_s: 8
stage1:
  source: data/stage1/latest_events
  input_channels: 2
  embedding_dim: 128
  projection_dim: 64
  variable_length: true
  events_per_class_batch: 8
  batch_classes: 5
  pooling: masked_mean_max_ordered4
  loss: supcon_plus_ce
  supcon_temperature: 0.1
  lambda_ce: 0.3
  optimizer: adamw
  lr: 0.0003
  weight_decay: 0.0001
  max_epochs: 100
  patience: 15
  power_scale_w: 100
reference_bank:
  train_houses_only: true
  events_per_class: 5
  pooling: mean
  strict_reference_query_disjoint: true
stage2:
  source: existing_mart_dataset
  window_length: 1024
  num_targets: 5
  reference_dim: 128
  freeze_stage1_encoder: true
  conditioning: film
  preserve_mart_backbone: true
  preserve_original_mart_loss: true
  preserve_original_mart_sampling: true
  use_partial_label_houses: false  # primary comparator first
  default_full_to_partial_ratio: [0.7, 0.3]
  max_epochs_fallback: 150
  batch_size_fallback: 8
inference:
  window_length: 1024
  stride: 512
  reference_bank_path: data/reference_bank/reference_bank_v1.pt
  use_test_house_reference: false
```

## 19.3 Reference event loader pseudocode

```python
def __getitem__(self, k):
    row = self.event_manifest.iloc[k]
    assert row.split == self.split
    power_w = load_exact_event(row)   # float32 [L]; 8s grid, includes internal pauses
    assert len(power_w) == row.samples_8s
    assert len(power_w) >= self.min_valid_length
    features, metadata = transform_reference(power_w, self.train_fitted_params)
    return features, metadata, row.appliance_id, row.event_id, row.house_id
```

`collate` pads to group-specific max length and emits valid masks. A missing/invalid event is rejected in the audit; the training loop should not silently truncate it.

## 19.4 Stage 1 step pseudocode

```python
class_groups = sampler.sample_groups(classes=5, events_per_class=8)
embeddings, logits, labels = [], [], []
for class_id, batch in class_groups.items():
    x, valid_mask, metadata = collate_group(batch)
    e, z, class_logits = reference_encoder(x, valid_mask, metadata)
    embeddings.append(z)
    logits.append(class_logits)
    labels.extend([class_id] * len(batch))

z = torch.cat(embeddings, dim=0)    # [40,64]
logits = torch.cat(logits, dim=0)  # [40,5]
loss = supcon_loss(z, labels) + 0.3 * cross_entropy(logits, labels)
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(reference_encoder.parameters(), 1.0)
optimizer.step()
```

Use AMP/scaler as appropriate and avoid duplicate event IDs within a batch when possible. This is architecture-agnostic *pseudocode*, not an executable full script.

## 19.5 Bank construction pseudocode

```python
encoder.eval()
with torch.inference_mode():
    for class_id in range(5):
        event_ids = deterministic_train_only_selection(
            class_id=class_id, count=M, across_houses=True
        )
        e = torch.stack([encoder.encode(event_by_id(i)) for i in event_ids])
        bank[class_id] = bank_pool(e)  # mean initially
save_bank(bank, class_order=APPLIANCES, event_ids=event_ids_by_class)
```

Do not let the selection function read validation/test house event IDs, even if present in the file library.

## 19.6 Stage 2 step pseudocode

```python
batch = next(mart_train_loader)
y, p_true, state_true, valid = adapter(batch)
E = frozen_bank[None, :, :].expand(y.size(0), -1, -1)
p_pred, s_logits = conditioned_mart(y, E)
loss = mart_loss(p_pred, s_logits, p_true, state_true, valid)
loss.backward()
optimizer.step()
```

Production code must handle optimizer zeroing, AMP, clipping, checkpointing, normalization and dataloader iteration. The frozen bank stays in the same device/dtype as the conditioned model and is not retrained by accident.

---

# 20. Tests and integrity requirements

### 20.1 Required automated tests

| Test | Pass condition |
|---|---|
| Constant 8-second event grid | Within every valid latest event, adjacent timestamps differ by 8 s. |
| Event boundary integrity | Whole cycle and valid internal OFF pauses preserved; no accidental across-event concatenation. |
| Variable-length encoder | `L=12,20,150,900,1300` -> embeddings of exactly `[128]` without NaNs. |
| Padding invariance | Changing padding length/values outside mask does not materially change embeddings. |
| Source reference provenance | No validation/test home contributes to Stage 1 fitting or bank. |
| Reference/query disjointness | Bank event IDs are disjoint from any same exact Stage 2 supervised query-event instance where required. |
| Class ordering | Bank row 3 is washing machine iff target channel 3 is washing machine. |
| MART adapter | `Y:[B,1,1024]`, `P/S/mask:[B,5,1024]`; units and timestamps aligned. |
| Partial label | Unknown target never receives a zero-power or zero-state loss. |
| All labels invalid | Batch is skipped safely, no NaNs, no misleading gradient. |
| Joint output shape | `[B,5,1024]` power and state. |
| Conditioning sensitivity | Switching/scrambling valid reference rows changes model output appropriately when conditioning learned. |
| Nonnegative power | No negative predicted power after final activation. |
| Overlap-add | Reconstructed time series has exact original valid timestamps and no unweighted uncovered samples. |
| Determinism | Same frozen encoder, bank selection seed and manifest reproduce bank checksums. |
| Inference isolation | Test stage never accesses submeter except inside offline scoring routine. |

### 20.2 Logging and artifact provenance

Each experiment logs configuration file/hash, Git commit, dataset version/checksum, house split hash, event count by class and house, bank manifest/checksum, class order, seed, optimizer configuration, selected checkpoint, validation metrics, test metrics once, device/GPU, runtime and library versions. This lets an ablation be reproduced without guessing which bank was used.

### 20.3 Safety from misleading evaluation

Check for target power readings exceeding mains frequently, shared channels, negative powers, gaps, time-offset inconsistencies, and long all-zero spans. Keep a visible QA report for every exclusion. Do not infer an unmonitored target's absence from zero-filled convenience arrays. For target-sparse appliances, report precision/false positives on naturally OFF-dominated continuous test sequences.

---

# 21. Implementation roadmap and acceptance gates

| Gate | Concrete task | File/artifact | Acceptance criterion |
|---|---|---|---|
| **G0** | Verify latest *8 s* references, verify all five classes, freeze common house split | `house_split_v1.json`, QA, event manifest | Consistent splits/units/order; quality log; no test leakage. |
| **G1** | Implement variable-length reference loader and collator | `reference_dataset.py`, `reference_collate.py` | Mixed-length complete events form safe batches. |
| **G2** | Implement shared CNN/TCN + SupCon/CE | `encoder.py`, `losses.py`, `stage1_best.pt` | Finite training, cross-house validation, numerical tests. |
| **G3** | Build fixed train-only reference bank | `reference_bank_v1.pt`, manifest | `[5,128]`, exact source provenance, repeatability. |
| **G4** | Adapt existing MART dataset to canonical Stage 2 tensors | `mart_dataset_adapter.py` | Existing windows validated, no split mismatch. |
| **G5** | Implement tiny conditioning adapter in MART | `reference_conditioning.py` | `[B,5,128]` accepted; joint `[B,5,1024]` outputs. |
| **G6** | Train baseline and conditioned matched experiments | runs B0/B1/B2 | Validation comparison under same data/seed/loss. |
| **G7** | Add dynamic references / partial labels only if warranted | runs B3/B4 | Benefit isolated; masks and disjointness tests pass. |
| **G8** | Frozen cross-house evaluation and reporting | metrics, predictions, plots | One truly held-out test home per dataset plus limitations stated. |

**Immediate next implementation task:** **G0/G1** — connect the loader to the researcher's newest resampled event export (rather than the old irregular `events.csv` upload), verify full-cycle integrity, generate an event manifest, and implement a variable-length `Dataset` + masked length-bucketed `collate`. **Do not redo raw preprocessing.**

Suggested initial development sequence:

```text
latest 8s reference events -> manifest -> batch tests -> Stage 1 encoder
-> frozen bank -> MART dataset adapter -> FiLM integration
-> ID-conditioned control -> house-disjoint evaluation
```

---

# 22. Risks and later extensions

## 22.1 Most important experimental risk: reference equals class ID

If Stage 2 always sees the same five vectors, the fixed bank may simply function as five learned class codes. For any static bank:

\[
H_\phi(Y, E^{bank})\equiv F_{\phi,E^{bank}}(Y).
\]

This does **not** make the model wrong, but it limits claims about the value of reference enrollment. Compare against a learned ID embedding of the same dimension and an equivalent conditioning module. Dynamic reference subsets provide an additional test of whether the waveform itself matters.

## 22.2 Stage 1 may learn duration more than shape

Kettle events are short and washer events are long. A duration-only classifier may perform well without learning subtle waveforms. Include trivial-feature controls and validation retrieval from other houses.

## 22.3 Mean pooling may erase long-cycle structure

Whole-cycle CNN + ordered pooling preserves *some* temporal information, but compression to one `[128]` vector can still lose phase detail. If simple bank results are weak, consider a temporal prototype sequence and cross-correlation/attention against aggregate features as a **distinct future hypothesis**; it is not a minor tweak.

## 22.4 House diversity matters

Thousands of events from one fridge do not equal thousands of devices. Track unique training homes/devices per class, and ablate distinct-house count independently of event count.

## 22.5 Full-label versus partial-label joint training

Using only full-five-target houses maximizes complete-label supervision but sacrifices household/device diversity. Masked partial-label training may help; it may also introduce an imbalance or unknown-source attribution ambiguity. Run both; do not assume 70:30 is best.

## 22.6 Unknown background and physical ambiguity

Reference conditioning does not eliminate `rank(A)=1` ambiguity. The model must reject similar interfering devices and should not be evaluated only on target-present excerpts. Avoid claiming exact source recovery from one active power channel under arbitrary overlaps.

## 22.7 Unseen appliance classes / variable N

In v1 all five appliance categories are known at training. A single shared encoder and query-conditioning layer are *structurally compatible* with larger target catalogs, but unseen-class extraction cannot be claimed until there is a class-disjoint benchmark with many different source types and explicit reference enrollment.

## 22.8 Additional data augmentation

Existing MART focal-event background-stratified augmentation is a strong baseline. Reference-stage independent-cycle augmentation should be conservative. Any new synthesis, physics loss, multi-prototype reference memory or encoder fine-tuning must enter as one-at-a-time ablations after reference-vs-ID value has been demonstrated.

---

# 23. References and evidence notes

### 23.1 Primary VoiceFilter inspiration

Wang, Q., Muckenhirn, H., Wilson, K., Sridhar, P., Wu, Z., Hershey, J., Saurous, R. A., Weiss, R. J., Jia, Y., & Lopez Moreno, I. (2019). **VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking.** *INTERSPEECH 2019*, 2728–2732. DOI: `10.21437/Interspeech.2019-1101`.

**Directly supported by that paper:** two separately trained networks; the reference-speaker encoder trained with GE2E on clean speech; 256-dimensional d-vectors; a target extraction network consuming both a speaker embedding and a mixture spectrogram; clean target reference is a different utterance from the target used for separation; multi-target speaker extraction can be repeated for different speaker references.

**Not directly demonstrated by VoiceFilter:** NILM, 8-second power data, 5-way joint appliance prediction, a general category-level bank for physically different devices, SupCon+CE appliance embeddings, or strict unseen-house performance. These are **our proposed design extensions**.

### 23.2 Research datasets

- **UK-DALE:** public household aggregate and individual appliance energy/power data. Verify the channel list, resampling alignment and any combined monitoring channels from the released dataset metadata and the project's existing mapping.
- **REFIT:** public household aggregate and appliance power recordings. Validate sampling irregularities/real 8-second conversion, device labels and plug reassignment against the existing prepared dataset and metadata.

### 23.3 Prior project foundation

- The researcher's existing **MART-NILM** joint multi-appliance pipeline is the intended Stage 2 backbone and strong baseline. Its exact implementation (layers, scaling, loss coefficients and sampling) must be inspected in the actual code before integrating conditioning. Do not represent conceptual shapes as observed code details.
- The earlier **Neural-Adjoint Inverse Optimization** work is distinct. This new project does not implement latent optimization, generator-based posterior inversion or a learned physical forward model in v1.
- A preexisting earlier planning file, `Reference_Conditioned_Joint_NILM_Implementation_Blueprint.md`, is superseded in this document where conversations clarified **constant 8-second current reference events** and **prefer MART reuse over an entirely new attention extractor**.

### 23.4 Interpretation statement

The purpose of this specification is to make implementation reproducible and scientifically falsifiable. The central result to establish is not “Stage 1 recognizes five clean appliance classes,” but whether the **additional complete-cycle reference representation leads to a measurable joint-disaggregation gain beyond a strong baseline and an equally conditioned ID model, under unchanged house-disjoint evaluation**.

---

## Final engineering contract (one-page summary)

```text
SAMPLE FREQUENCY:      1 sample / 8 seconds (CURRENT prepared data)
TARGET ORDER:          kettle / fridge / dishwasher / washing_machine / microwave

STAGE 1:
    Input:             R_i [1, L_i], L_i variable; single appliance, full event
    Sampling:          8 s; variable event duration, intact internal pauses
    Model:             ONE shared CNN/TCN encoder + masked/ordered pooling
    Output:            e_i [128]
    Training:          5-class CE + SupCon (proposed), house-disjoint
    No aggregate:      TRUE

STAGE 1.5:
    Input:             training-only reference events + frozen encoder
    Output:            E_bank [5,128] + event-ID/house provenance
    Initial M:         5 independent cycles per class; run count ablation
    Test home bank:    PROHIBITED in strict cross-house evaluation

STAGE 2:
    Dataset:           existing MART-NILM continuous aggregate windows
    Input:             Y [B,1,1024], E_bank [B,5,128]
    Labels:            P_true/S_true [B,5,1024], validity mask
    Model:             existing MART joint architecture + small reference FiLM
    Output:            power/state for all 5 appliances together
    Training:          preserve existing MART loss/sampling initially
    Missing channels:  MASK; never assume OFF

STAGE 3:
    Input:             held-out house aggregate + SAME saved bank
    Training:          none
    Reference wav:     none needed at test time for fixed-bank protocol
    Inference:         1024 point sliding windows, stride 512 initially
    Output:            continuous 5-appliance power/state estimates
    Ground truth:      TEST SCORING ONLY

ESSENTIAL EVIDENCE:
    MART baseline vs ID-conditioned MART vs pretrained reference MART
    all with identical data splits, sampling, losses and fair compute
    report MAE / ON-MAE / macro-F1 / false positives by held-out house

IMMEDIATE ACTION:
    Verify latest 8-second event files -> manifest -> variable-length
    Dataset/collator -> shared Stage 1 reference encoder.
```

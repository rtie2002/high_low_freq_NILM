# MART-NILM: multi-appliance literature positioning

## 1. Executive conclusion

The 15 papers in the local literature folder do not all solve the same problem.
They fall into four groups:

1. **Multi-label state detection:** one model predicts which appliances are on,
   but it does not reconstruct dense appliance power waveforms.
2. **Multi-target power regression:** one model predicts several appliance
   powers, while state metrics are derived by thresholding the regression.
3. **Joint power and state learning:** power and operating state are explicit,
   coupled outputs.
4. **Transfer, weak-label, or limited-label learning:** the main contribution is
   reduced annotation or target-domain adaptation rather than ordinary
   multi-appliance disaggregation.

This distinction matters. A high F1 score from an event-level, high-frequency,
synthetic, or state-only experiment is not directly comparable with dense power
and state prediction from a recorded low-frequency whole-house aggregate.

MART-NILM should therefore **not** be sold as the first multi-appliance network,
the first joint power--state model, the first attention model, or the first use
of recomposed aggregates. All four claims would be contradicted by prior work.
Its strongest defensible position is:

> **MART-NILM studies joint five-appliance power and state estimation under a
> strict house-held-out, real-aggregate protocol, and addresses two failure
> mechanisms that arise in this setting: sparse event--background coverage and
> cross-appliance power--state loss coupling.**

The architecture is the shared vehicle for this study. The principal novelty
should be placed in the problem formulation, the two targeted training
strategies, and the controlled evidence supporting them.

---

## 2. What the 15 papers actually claim

The table below uses the authors' own experimental setting. `P` denotes dense
power estimation and `S` denotes operating-state prediction.

| # | Study | Actual task and evaluation setting | How the paper sells the work | Gap left for MART-NILM |
|---:|---|---|---|---|
| 1 | Buddhahai *et al.* (2018), DOI 10.1016/j.scs.2018.02.002 | Multi-label **S**; one custom house; 1-min current, active/reactive power, and power factor | A systematic machine-learning design process: feature selection, classifier selection, learning curves, and RAkEL with a decision tree | No dense power waveform, no public cross-house protocol, and more electrical channels than a normal active-power smart meter |
| 2 | Dash and Sahoo (2024), DOI 10.1109/JESTIE.2023.3333308 | Multi-label **S** on REFIT; train building 2; test an unseen part of building 2 and unseen buildings 9 and 20 | A lightweight causal/dilated TCN with multi-head attention, long receptive field, unseen-building tests, ablation, and complexity analysis | State only; long all-off subsequences are removed; appliance energy is approximated rather than densely regressed |
| 3 | Conv-NILM-Net (2023), arXiv:2208.02173 | Multi-target **P**; causal Conv-TasNet-style source separation; REDD within-building cross-validation and an artificially formed UK-DALE aggregate | Real-time causality, one network for several sources, and a very small model | No explicit state head; its UK-DALE aggregate does not retain the complete recorded household residual; the main protocol is not a disjoint train/validation/test-house protocol |
| 4 | MATNilm (2024), arXiv:2307.14778 | Joint **P+S**; REDD and UK-DALE; full-data and one-day limited-label scenarios; operation-profile augmentation | Accurate multi-appliance learning from very little labeled data through a shared hierarchy, two-dimensional temporal/appliance attention, and online augmentation | Limited-label learning is the primary problem; validation is not a separate held-out house; augmentation is not designed to equalize residual-background exposure |
| 5 | Liu and Fan (2026), arXiv:2609.18315 | Joint **P+S**; REDD, UK-DALE, and REFIT; recorded unseen-house tests; training-house tails used for validation | Label-preserving residual-background recomposition, reliability-gated prediction consistency, and the FLAME mixture-of-experts architecture | This is the closest overlap. MART-NILM cannot claim the first background recomposition. Its distinction must be **focal event coverage across residual strata**, a separate held-out validation protocol, and appliance-wise task balancing |
| 6 | Faustine and Pereira (2020), DOI 10.3390/en13164154 | Multi-label **S** from 30-kHz PLAID current/voltage event snippets; synthetic combinations of up to three appliances | Fryze-current decomposition and an image-like current representation for simultaneous appliance recognition | Different sensing hardware, event-level classification, and no continuous low-frequency power disaggregation |
| 7 | Ayub and El-Alfy (2020) | Multi-target **P** with direct point-to-point CNNs on ENERTALK and REDD; primarily random or combined-data splits, with extra transfer tests | Direct multi-target regression, lower resource cost than separate models, and cross-appliance/cross-domain transfer demonstrations | The primary protocol is not complete house-held-out validation and test; state is obtained by thresholding power; no explicit real-residual or multi-task optimization treatment |
| 8 | Tanoni *et al.* (2023), DOI 10.1109/TSG.2022.3191908 | Multi-label **S**; UK-DALE and REFIT; held-out test houses; synthetic bags composed from 1--4 activations plus a recorded residual | Multi-instance learning from coarse weak labels, reducing annotation cost; CRNN with clip smoothing | State only; evaluation uses recomposed bags rather than uninterrupted recorded test aggregates; background is sampled but not stratified for sparse focal events |
| 9 | Nalmpantis and Vrakas (2020), DOI 10.1007/s00521-020-04916-5 | Multi-label **S** on UK-DALE/REDD, mainly same-house temporal evaluation | Compact time-series representations such as PAA, SAX, SFA, BOSS, WEASEL, and Signal2Vec for edge deployment | No power waveform, no joint power--state task, and no strict unseen-house validation/test protocol |
| 10 | SAMNet (2022), DOI 10.1109/TSG.2021.3139395 | Joint **P+S for one target appliance at a time**; UK-DALE and REFIT cross-house tests; REDD transfer test | Latency-free causal TCN experts, task attention, learned state boundary, and dynamic regression/classification balancing | Despite being multi-task, it is not one joint five-appliance model. Its single power--state ratio does not address coupling among several appliances; this is the direct motivation for MART-NILM's appliance-wise balance |
| 11 | Singhal *et al.* (2019), DOI 10.1109/TSG.2018.2815763 | Multi-label **S** on REDD and Pecan Street; Pecan uses an 80/20 house split | New supervised deep dictionary-learning and transform-learning formulations for multi-label classification | State only; daily/minute-averaged inputs; no dense power regression or modern house-held-out joint-task protocol |
| 12 | Li *et al.* (2023), DOI 10.1016/j.apenergy.2022.120223 | One-to-many **P**; pretrain on REFIT and fine-tune on target REFIT/REDD/UK-DALE or a measured non-residential room | One transferable model for all appliances, fewer target measurements, fewer parameters, and practical deployment | Target-domain labeled data and fine-tuning are used; it is adaptation rather than zero-shot unseen-house generalization; no explicit state output |
| 13 | Sun *et al.* (2025), DOI 10.1109/TIM.2025.3541652 | Joint multi-appliance **P+S** transfer across REFIT, REDD, and UK-DALE | Shared/specific CNN--Transformer sequence model, SGN power gate, and transfer of both regression and classification tasks | The contribution is target-domain transfer and fine-tuning; task losses use a fixed global coefficient, while scale imbalance is mainly treated through target normalization |
| 14 | UNet-NILM (2020), DOI 10.1145/3427771.3427859 | Joint multi-appliance **P+S** on UK-DALE house 1; artificial aggregate formed from selected appliances plus a television | One-dimensional U-Net, multi-label state estimation, multi-target quantile regression, and uncertainty estimates | Same-house experiment and artificial aggregate; no unseen-house test, no full recorded residual, and no separate validation-house protocol |
| 15 | V.M.Regressor (2023), DOI 10.3390/s23042051 | Multi-target power with auxiliary state supervision; train UK-DALE 1 and test UK-DALE 1/2 and REFIT 2/5; ten seeds | Variational inference for better multi-target generalization, small deployment cost, and structured stress-test scenarios | Only one source training house; no background-coverage intervention; no independent validation houses; architecture complexity rather than task coupling is the main contribution |

### Important correction to the present manuscript

SAMNet is a **multi-task** model that couples state and power for one target
appliance; it is not a single joint model that outputs all five appliances.
It is valuable as the origin of the dynamic power--state balance, but it should
not be presented as an ordinary multi-appliance architecture in the same sense
as UNet-NILM, MATNilm, or MART-NILM.

---

## 3. Common publication strategies in the literature

The successful papers do not sell a collection of layers. They select one
dominant practical bottleneck and align the method and experiments with it:

| Dominant bottleneck | Representative papers | Evidence used to support the claim |
|---|---|---|
| One model instead of one model per appliance | Conv-NILM-Net, UNet-NILM, Li *et al.*, V.M.Regressor | Parameter count, model size, training/inference time, single- versus multi-target comparison |
| Real-time or latency-free operation | Conv-NILM-Net, SAMNet, Dash and Sahoo | Causal architecture, receptive-field analysis, latency or complexity table |
| Scarce labels | Tanoni *et al.*, MATNilm, Li *et al.*, Sun *et al.* | Label-fraction curves, one-day training, target-domain fine-tuning, transfer experiments |
| Better signal representation | Buddhahai *et al.*, Faustine and Pereira, Nalmpantis and Vrakas | Feature/representation comparison and classifier ablation |
| Cross-house robustness | V.M.Regressor, Dash and Sahoo, Liu and Fan | Explicit same-house, unseen-house, and cross-dataset scenarios |
| Background dependence | Liu and Fan | Paired recomposition, consistency ablation, unseen-house results |

MART-NILM should follow the same discipline. Its central bottleneck is not
generic model capacity. It is **reliable joint learning from a full aggregate
when sparse target events are embedded in unseen household backgrounds and the
five appliances produce unequal optimization signals**.

---

## 4. Defensible advantages of MART-NILM

### 4.1 Strong experimental protocol

The current five-week protocol uses:

- training: UK-DALE 1 and REFIT 3, 5, 9, 11;
- validation: UK-DALE 5 and REFIT 2;
- test: UK-DALE 2 and REFIT 20;
- recorded validation/test aggregates, including unmonitored loads;
- one joint model for five appliances, power and state;
- three seeds and separate reporting for the two test houses.

This is stronger than same-house random splitting, target-house fine-tuning,
artificial monitored-load sums, or validation taken from a source training
house. The exact defensible statement is:

> Training, validation, and test households are disjoint, and the final model
> is evaluated on recorded whole-house aggregates from held-out homes in both
> UK-DALE and REFIT.

Do not call this the first cross-house experiment; several papers have unseen
test houses. The distinction is the **separate held-out validation houses plus
recorded residuals in two datasets under one fixed joint protocol**.

### 4.2 Focal-event background-stratified recomposition

Prior work already recomposes appliance signals:

- Tanoni *et al.* create random activation bags with recorded noise;
- MATNilm generates training windows from operation profiles;
- Liu and Fan replace residual background while preserving all targets and add
  a consistency loss.

MART-NILM's narrower novelty is the **sampling objective**:

1. choose the focal appliance uniformly;
2. require a real active trace of that appliance;
3. sample the residual background across predefined power strata;
4. sample the remaining target traces independently;
5. use the construction only for training.

This is a coverage intervention, not a generic anti-overfitting method and not
a claim that the synthetic joint distribution is physically exact. The
factorial ablation supports this interpretation: focal mixing alone improves
validation ON MAE from 357.8 to 301.6 W and macro-F1 from 0.605 to 0.649, while
overall MAE changes from 21.16 to 21.68 W. It improves event visibility at a
small average-error cost; the complete training strategy recovers the average
error.

### 4.3 Bounded appliance-wise power--state balancing

SAMNet motivates dynamic balancing between a regression task and a
classification task, but SAMNet trains one target appliance at a time. In a
five-appliance model, a single global ratio couples unrelated errors: a large
microwave power error can change the classification weight applied to the
fridge.

MART-NILM computes a detached, bounded balance for each appliance separately.
The contribution is not simply “dynamic loss weighting”; it is the conversion
of a single-target task balance into a **bounded appliance-wise balance that
prevents cross-appliance reweighting**.

The factorial experiment supports this claim. Appliance-wise balancing alone
improves validation MAE, ON MAE, SAE, and macro-F1 relative to fixed joint
training. Combining it with focal-event mixing gives 20.79 W MAE, 315.41 W ON
MAE, 15.02 W SAE, and 0.687 macro-F1, compared with 21.16 W, 357.81 W, 15.94 W,
and 0.605 for fixed joint training.

### 4.4 Evidence is broader than a single final score

The current evidence includes:

- a three-seed main result on two held-out test houses;
- module-level architecture ablation;
- a three-seed `2 x 2` factorial training-strategy ablation;
- per-appliance and background-conditioned analyses;
- gradient-scale auditing at the shared encoder;
- qualitative waveform inspection.

This is a stronger experimental story than reporting only aggregate MAE.

---

## 5. Claims that should not be made

1. **Do not claim the first multi-appliance network.** Multi-target and
   multi-label models precede MART-NILM by many years.
2. **Do not claim the first joint power--state network.** UNet-NILM, MATNilm,
   and the 2025 transfer model already predict both.
3. **Do not claim the first cross-appliance attention.** MATNilm explicitly
   models appliance and temporal relations.
4. **Do not claim the first synthetic mixing or background swap.** Tanoni,
   MATNilm, and especially Liu and Fan are direct prior work.
5. **Do not claim the smallest or fastest model.** Conv-NILM-Net and the
   lightweight state detectors have stronger efficiency claims.
6. **Do not claim that focal-event mixing improves every appliance.** The
   validation ablation shows no meaningful microwave F1 gain.
7. **Do not claim that the noisy-fridge failure is solved.** The waveform audit
   still shows false fridge plateaus under difficult REFIT backgrounds.
8. **Do not claim state of the art before the common-protocol baselines are
   complete.** Published numbers use incompatible splits, aggregates, targets,
   and metrics.

---

## 6. Recommended related-work table

### 6.1 Main-paper table

The present columns “Scenario / Split / Architecture / Contribution” make the
table text-heavy and overemphasize familiar components such as CNN, TCN, and
attention. A more useful table is:

| Study | Scope | Output | Evaluation aggregate | Generalization protocol | Primary focus |
|---|---|---|---|---|---|

Recommended definitions:

- **Scope:** one target or joint multiple targets;
- **Output:** S, P, or P+S;
- **Evaluation aggregate:** recorded, artificially summed, recomposed bag, or
  event snippet;
- **Generalization protocol:** same house, unseen house, cross-dataset, or
  target fine-tuning;
- **Primary focus:** one short phrase only.

The main paper should include the most comparable low-frequency methods:

1. UNet-NILM;
2. Conv-NILM-Net;
3. V.M.Regressor;
4. Li *et al.* one-to-many transfer;
5. Tanoni *et al.*;
6. MATNilm;
7. Dash and Sahoo;
8. Sun *et al.* 2025;
9. Liu and Fan 2026;
10. MART-NILM.

SAMNet should be discussed in the multi-task optimization paragraph rather
than described as a direct joint multi-appliance competitor. The remaining
older, high-frequency, or representation-oriented papers can be summarized in
prose or placed in an appendix taxonomy. This preserves coverage of all 15
papers without making the main table unreadable.

### 6.2 MART-NILM row

The row should say:

| MART-NILM | Joint five appliances | P+S | Recorded whole-house validation/test aggregate | Disjoint train/validation/test houses in UK-DALE and REFIT | Focal event--background coverage and appliance-wise task balance |

The row should not list every architecture block. Those blocks belong in the
method figure and architecture ablation.

---

## 7. Recommended paper story

### One-sentence thesis

> Joint multi-appliance NILM fails not only because source separation is hard,
> but also because sparse appliance events are observed under limited household
> backgrounds and unequal appliance tasks compete through a shared optimizer.

### Introduction structure

1. **NILM and practical value.** Define aggregate-to-appliance estimation.
2. **Why joint multi-appliance learning.** One forward pass, shared temporal
   evidence, and simultaneous operating context; avoid claiming constant cost
   as the number of outputs still matters.
3. **Why existing results do not fully establish household transfer.** Separate
   same-house, synthetic-aggregate, target-fine-tuned, state-only, and truly
   held-out real-aggregate settings.
4. **Two failure mechanisms.** Event--background coverage and cross-appliance
   power--state loss coupling.
5. **Method overview.** MART-NILM backbone plus the two training strategies.
6. **Three contribution bullets.** State only claims supported by the current
   experiments.

### Related-work structure

1. **Multi-appliance state and power estimation.** Separate state-only,
   power-only, and joint P+S methods.
2. **Generalization, weak supervision, and data recomposition.** Discuss
   transfer, limited labels, synthetic bags, and residual replacement.
3. **Joint-task optimization.** Use SAMNet as the single-target antecedent and
   explain why multi-appliance training requires appliance-wise control.
4. End with a short positioning paragraph; do not repeat the introduction.

### Method structure

1. Problem and real residual definition.
2. MART-NILM shared temporal and appliance-specific relational model.
3. Joint state-conditioned power output.
4. Power and state objectives.
5. Bounded appliance-wise balance.
6. Focal-event background-stratified recomposition.
7. A short training/inference summary clarifying that recomposition and balance
   add no inference-time parameters.

### Experiment structure

1. Datasets, five-week disjoint-house split, and state-label provenance.
2. Common-protocol baselines and implementation details.
3. Metrics and equal-house aggregation.
4. Main results, with UK-DALE 2 and REFIT 20 reported separately.
5. Computational cost.
6. Architecture ablation on validation houses.
7. `2 x 2` training-strategy ablation on validation houses.
8. Test-house waveform examples and failure analysis.

For a clean scientific protocol, architecture and training decisions should be
supported by the validation houses. The two test houses should be reserved for
the frozen complete model and frozen baselines. If test-house ablations are
retained, they must be explicitly described as a post-hoc report-only audit and
must not be used to revise the model.

---

## 8. Contribution text suitable for the paper

The following is deliberately conservative and defensible:

> The contributions are threefold. First, we formulate joint five-appliance
> power and state estimation under a house-held-out protocol in which training,
> validation, and test homes are disjoint, and validation and test retain the
> recorded whole-house aggregate, including unmonitored loads. Second, we
> introduce focal-event background-stratified recomposition, which exposes a
> real active trace of each appliance to residual backgrounds spanning several
> power ranges during training, without modifying validation, testing, or the
> inference model. Third, we introduce bounded appliance-wise power--state
> balancing, which scales each appliance's classification loss using its own
> detached regression-to-classification ratio and thereby avoids globally
> reweighting all state tasks in response to one appliance's error. Three-seed
> module and factorial ablations distinguish the contributions of the network
> blocks and the two training strategies.

Do not put final numerical superiority in the contribution list until UNet-NILM
and MATNilm have been rerun under exactly the same split and metrics.

---

## 9. How to interpret the present results

The current result is credible but not yet sufficient for a state-of-the-art
claim:

- UK-DALE 2: MAE 7.451 W and macro-F1 0.876;
- REFIT 20: MAE 10.798 W and macro-F1 0.716;
- equal-house test summary: MAE 9.125 W and macro-F1 0.796;
- 1.373 M trainable parameters and a 5.335 MiB checkpoint.

The training factorial gives the clearest method evidence: relative to fixed
joint training, the complete strategy improves validation macro-F1 from 0.605
to 0.687 and MAE from 21.164 to 20.791 W. The architecture ablation gives its
strongest support to temporal dilation and appliance-specific decoding. The
multi-scale stem and relation attention have smaller, appliance-dependent
effects and should not be described as universally beneficial.

The REFIT microwave F1 and noisy-background fridge waveforms remain failure
cases. They should be shown honestly in the failure analysis. The paper can
still be publishable if its claim is **a rigorously evaluated training framework
for difficult joint real-aggregate NILM**, rather than a claim that every
appliance or background condition has been solved.

---

## 10. Work required before submission

1. Run UNet-NILM and MATNilm under the exact same five-week house split,
   normalization, metrics, and three seeds.
2. Fill the baseline and computational-cost rows; do not compare published
   numbers directly.
3. Resolve and document the provenance of every ON/OFF label. The paper must
   match the preprocessing code and the stored CSV labels exactly.
4. Use validation ablations for design claims and reserve test houses for the
   frozen comparison.
5. Keep the two test houses separate in the main table; use an equal-house mean
   only as a clearly labeled summary.
6. Retain waveform examples for kettle, fridge, dishwasher, washing machine,
   and microwave, including difficult REFIT cases.
7. Present focal mixing as event/background coverage, not generic
   regularization, and state explicitly that it does not improve microwave F1
   in the current validation experiment.
8. Present appliance-wise balancing as the main optimization contribution and
   link it explicitly to the limitation of applying single-target task balance
   to a joint multi-appliance model.

---

## 11. Local evidence base

All 15 papers were reviewed from:

`C:\Users\Raymond Tie\Desktop\PhD\Literature Review\NILM\multi-appliances`

The key DOI/arXiv identifiers are included in Section 2 so that the comparison
can be audited against the source papers. The current MART-NILM numerical
evidence comes from `multi_appliances_NILM/docs/multinilm_experiment_log.md` and
the manuscript tables in `paper/multi-applainces_paper.tex`.

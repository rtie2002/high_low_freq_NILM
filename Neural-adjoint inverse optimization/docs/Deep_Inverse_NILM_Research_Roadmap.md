# Deep Inverse NILM — Research Direction and Codex Handoff

**Status:** Research planning / feasibility validation. **Not an implementation authorization.**  
**Updated:** 2026-11-11  
**Audience:** Codex and research collaborators  
**Language:** English for unambiguous implementation handoff.

**Core research topic:** Deep learning-based inverse modeling for multi-appliance NILM.

> **One-sentence mission:** Develop a three-paper research program for **multi-appliance NILM as an ill-posed inverse problem**, progressing from **identifiability-aware inverse optimization**, to **structured generative source priors**, to **diffusion posterior inference**. At inference, **only household aggregate power is observed**. Do **not** silently substitute an ordinary aggregate-to-appliance supervised predictor for the intended forward-model-based inversion.

---

## 0. Read this first: decisions, non-negotiable constraints

1. **Preserve the user's original research concept:** learn appliance-specific **forward source dynamics** from training appliance power; compose their outputs using the known physical aggregation operator; **freeze the learned source components**; at test time recover appliance contributions by **optimizing latent/source variables through the frozen forward system**.
2. The physical measurement is additive. **Do not train a neural network merely to learn summation** and expect it to yield useful source-specific input gradients.
3. **Do not train `aggregate -> appliance power` as the proposed core model.** A direct NILM predictor is allowed as an **independent baseline** only, or by explicit later research decision. The earlier Phase B implementation is *not* the intended forward model.
4. During **inference**, the model has **only aggregate power** (plus parameters/statistics learned from training). It does **not** receive true appliance power, true ON/OFF labels, event locations, operating stages, or test-house residuals derived from appliance ground truth.
5. During **training**, submetered appliance-power sequences from UK-DALE/REFIT are available. If useful, proxy states may be *derived* from power for auxiliary training/metrics, **but are not mandatory** and must never be treated as observed inference inputs.
6. **Paper 1 must have a concrete, testable optimization novelty** beyond standard sparse coding, temporal dictionaries, regularized gradient descent, neural-adjoint, and learned iterative reconstruction. Proposed novelty is a **hypothesis**, not a confirmed literature-first invention.
7. **Paper 2 must have its own substantive mechanism** beyond swapping the Paper 1 dictionary for a VAE/GAN. **Paper 3 must have its own mechanism** beyond putting diffusion and an aggregate-consistency loss together.
8. **Background/unmonitored loads are essential.** Do not assign all unexplained aggregate energy to a freely optimized nonnegative background channel. Negative signed residuals are possible because of meter alignment, sampling, and other errors.
9. **Do not immediately code a complete new architecture.** The next task is a targeted literature/technical audit and a minimal identifiability experiment; proceed phase by phase with evidence and short checkpoint reports.
10. Keep code **readable and compact**: clear file responsibilities, straightforward mathematical comments, documented tensor shapes/units, avoid agent-generated monoliths and unnecessary abstraction. Do not change the old NILM repository.

---

## 1. Project locations and existing work

**Original research/data and NILM baseline repository (read-only):**

```text
C:\Users\Raymond Tie\Desktop\PhD\Code\multi-domain NILM\high_low_freq_NILM\multi_appliances_NILM
```

**New research repository:**

```text
C:\Users\Raymond Tie\Desktop\PhD\Code\multi-domain NILM\high_low_freq_NILM\Neural-adjoint inverse optimization
```

The assistant cannot directly access these local Windows paths; **Codex must inspect the actual files locally before relying on interfaces**.

### Previously implemented — retain, but do not confuse with the new main method

| Previous phase | Files / result | Correct interpretation now |
|---|---|---|
| **A — Physics Projection** | `power_projection.py`, `tests/test_power_projection.py`; 6 tests passed | Box-constrained power-balance correction of already proposed source power values. **Not the Neural-Adjoint input inversion core**. Keep as an optional ablation/alternative optimization component. |
| **B — Initial Inverse Estimator** | `initial_estimator.py`, `initial_pipeline.py`, `train_initial.py`, `evaluate_initial.py`, `configs/initial.yaml`, `tests/test_initial_estimator.py`; combined 9 tests passed | A lightweight **direct supervised NILM baseline** mapping aggregate to appliance power/state/background. **Not a learned forward model**. Keep for fair comparisons; do not automatically continue its training as a prerequisite of Paper 1. |
| Smoke check | 2 epochs, 4 train batches/epoch, 2 validation batches, CPU; macro F1 ~0.139; background nearly constant | Pipeline-only diagnostics; **not** valid performance evidence or a journal result. |

Phase A projection solved, for proposals `q_j` and positive flexibilities `v_j`:

\[
\underset{0\le u_j\le U_j}{\operatorname{minimize}}
\sum_{j=1}^{6}\frac{(u_j-q_j)^2}{2v_j}
+\frac{(X-\sum_j u_j)^2}{2\sigma_\epsilon^2}.
\]

An exact bounded solution was tested via the monotone scalar equation and bisection. **It corrects aggregate balance; it cannot by itself identify the true source.** It is not a required component of the revised Paper 1.

### Existing data/protocol (verify against actual old code)

- UK-DALE and REFIT; predefined **house-disjoint** train/validation/test splits.
- Five targets, in order: `kettle`, `fridge`, `dishwasher`, `washingmachine`, `microwave`.
- Reported resampling interval: **8 seconds**; initial windows **1024 samples**, stride **512**; do not cross timestamp discontinuities.
- Original appliance targets may use **separate z-score normalization**, which **cannot be summed directly**. Convert to real watts or a **single common physically consistent scale** for any forward aggregation.
- Existing `NILMDataLoader`, timeline overlap reconstruction, evaluation metrics, state-processing logic should be reused where appropriate and verified rather than guessed.
- Earlier smoke test reported signed residual minimum around **−2631 W**, which requires a full prevalence/alignment audit before aggressive physical consistency penalties.

---

## 2. Scientific foundation: the genuine obstacle

At each time point, the measured mains power is approximately

\[
X_t=\sum_{i=1}^{N} P_{i,t}+B_t+\epsilon_t,
\qquad N=5.
\]

Here `B` is the contribution from unmonitored loads and `epsilon` captures measurement/alignment mismatch. More than one source combination can yield the same mains measurement: **the inverse problem is underdetermined and generally non-identifiable without assumptions**.

### Why a neural forward network trained on raw appliance powers is not enough

If the target is the *controlled sum* of the five submeters, then the exact forward operator is

\[
F(P)=\sum_i P_i,
\quad \frac{\partial F}{\partial P_i}=1,
\quad \frac{\partial}{\partial P_i}\frac12\|F(P)-X\|^2=F(P)-X.
\]

**Every source receives the same aggregate reconstruction gradient.** Even zero reconstruction error does not imply correct appliance attribution. A large CNN approximating this sum cannot generate missing information; differing learned derivatives might be surrogate artifacts. When targeting *real* mains and inputting only five appliances, the unobserved background also makes the forward mapping stochastic, not uniquely determined by those five inputs.

### Correct meaning of a *learned forward model* in this project

Learn **source production/dynamics before the sum**:

\[
P_i=G_{\theta_i}(a_i),
\qquad \hat X=F_\Theta(a,B)=\sum_i G_{\theta_i}(a_i)+B.
\]

- `a_i`: **unknown continuous source/activation representation** to be inferred from aggregate; **NOT an observed ON/OFF label**.
- `G_theta_i`: appliance-specific differentiable source dynamics learned from *training submetered power*, e.g. a small **convolutional temporal dictionary** or other interpretable dynamical source representation for Paper 1.
- The sum is an **exact known physical operator**, not what the neural network must learn.
- The **full forward system is learned in part** because `G_theta_i` are trained, though the final mixing operation is fixed.
- Learnability of `G` must not be conflated with guaranteed identifiability of `a` or `P`.

A simple first representation (for feasibility, **not claimed novel**) is:

\[
P_i(t)=\sum_m(D_{i,m}*a_{i,m})(t),
\]

with learned temporal atoms and **continuous** activation coefficients. Source-only autoencoders or sparse coding may be trained from individual appliance waveforms. At inference, freeze the learned dictionaries/decoders and optimize their coefficients from mains.

### Where useful, differentiated inverse gradients arise

For reconstruction residual `r = sum_i G_i(a_i) + B − X`, define the source Jacobian

\[
J_i(a_i)=\frac{\partial G_i(a_i)}{\partial a_i}.
\]

Then for a half-squared reconstruction objective (ignoring scaling and extra regularizers),

\[
\nabla_{a_i}\mathcal L_{\mathrm{rec}}=J_i^\top r.
\]

A learned source representation induces distinct source-specific gradient directions. For a linear dictionary this is `D_i^T r`. **This addresses the identical-gradient symptom, not mathematical uniqueness.** Two appliance response directions may still overlap significantly, and parameterization/normalization can affect comparisons.

---

## 3. Three-paper publication roadmap — preserve this separation

| Research dimension | **Paper 1 — Optimization mechanism** | **Paper 2 — Solution space** | **Paper 3 — Posterior distribution** |
|---|---|---|---|
| **Working theme** | **Identifiability-Aware Deep Inverse Optimization** | **Structured Generative Prior-Based Inversion** | **Ambiguity-Aware Diffusion Posterior Inference** |
| **Primary question** | How can an input-gradient inverse solver distinguish appliance contributions when the physical measurement is additive and ambiguous? | Can nonlinear source distributions improve realistic source attribution beyond a simple structural representation? | Can a diffusion posterior represent multiple plausible appliance allocations under source overlap and uncertain background? |
| **Representation** | Small, interpretable learned source dynamics / temporal dictionary | Nonlinear appliance-specific generative representations | Score/diffusion-based source distribution |
| **Inverse variable** | Continuous source activation coefficients `a` | Structured generative latent variables `z` | Source trajectories / posterior diffusion samples |
| **Mechanism to investigate** | **Confusability-aware/Jacobian-aware gradient optimization**, robust measurement likelihood | **Source-competitive structured latent inference** for overlapping appliances, without forbidding genuine simultaneity | **Ambiguity- and background-aware posterior guidance** with calibrated uncertainty |
| **Output** | One or several deterministic inversions | Realistic generative-constrained MAP-style solutions | Multiple plausible posterior samples, intervals, event probabilities |
| **Main comparison** | Adam/input inversion, sparse coding, dictionary, dynamics, FHMM, direct NILM | Paper 1, naive VAE/GAN latent inversion, source-separation generative baselines | Paper 2, ordinary diffusion posterior sampling, conditional diffusion NILM |
| **What is *not* novel alone** | Neural-Adjoint, temporal dictionary, learned gradient steps, aggregate physical loss | VAE/GAN source generators + latent optimization | Diffusion + aggregate guidance or more samples |

**Publication feasibility is conditional on stronger experiments and prior-art review.** Do not imply that three papers are already independently publishable or that a particular first-ever claim has been established.

### Paper 1 — candidate main mathematical program

**Working title:** *Identifiability-Aware Neural-Adjoint Optimization for Multi-Appliance NILM*.

**Train (source side, not direct aggregate-to-source):** use individual training appliance waveforms `P_i` to train compact, differentiable `G_theta_i(a_i)` (for example learned temporal atoms). Auxiliary source encoders may be used during training to infer codes from **submetered sources**, but **must not see true sources at test time**.

**Inference:** freeze all `G_theta_i` weights; use only test mains `X`, initialize unknown `a` from permitted training-derived prior/random starts, minimize

\[
\boxed{
 a^*=\arg\min_a\left\{
  \mathcal L_{\mathrm{measurement}}\!\left(X,\sum_iG_{\theta_i}(a_i)\right)
  +\lambda\mathcal R_{\mathrm{source}}(a)
 \right\};
 \quad \hat P_i=G_{\theta_i}(a_i^*).
}
\]

**Controlled mixtures:** measurement loss may be ordinary squared error, with no background.

**Real household mains:** explore a training-only residual/noise likelihood (rather than a free `B`):

\[
R_{\mathrm{train},t}=X_{\mathrm{train},t}-\sum_iP_{i,\mathrm{train},t},
\qquad
\mathcal L_{\mathrm{measurement}}
 =-\log p_R\!\left(X-\sum_iG_{\theta_i}(a_i)\right).
\]

The signed residual is **not ground-truth nonnegative background**. A residual likelihood can be informative but may be too permissive; stationarity across households is unproven. Avoid using held-out source labels to form inference inputs. Start with a simple robust residual model and explicitly check calibration and mismatch.

**Proposed, *not yet validated*, optimization novelty:** detect local competition between different source response directions and adapt the inverse update accordingly. Candidate source-sensitivity similarity:

\[
C_{ij}=\frac{|\langle g_i,g_j\rangle|}
{\|g_i\|_2\,\|g_j\|_2+\varepsilon},
\]

where `g_i`, `g_j` are chosen **columns/directions of the Jacobians**. For high-dimensional codes, compare Jacobian **subspaces** or normalized Gram matrices; do **not** call one scalar a complete identifiability proof.

A possible optimization **starting point** is the local inverse step

\[
\boxed{
\Delta a^*=\arg\min_{\Delta a}
\frac12\|r+J\Delta a\|_W^2
+\lambda\mathcal R_{\mathrm{source}}(a+\Delta a)
+\mu\|\Delta a\|_H^2,
}
\]

where `r = F_Theta(a)−X` and `J = [J_1 ... J_N]`. **Investigate** a mathematically defined dependence of `H`, update/trust region, or source competition on observed Jacobian confusability, under source priors and uncertainty. The formula above is a **candidate**, not a solved new algorithm, and ordinary Gauss–Newton/trust-region techniques are already known. It must show advantages **beyond** existing sparse coding and learned iterative methods.

**Critical requirements:** normalize source sensitivities to avoid favoring high-power appliances; account for activation amplitude scaling; permit real coincident appliance events; inspect sensitivity to initialization, code dimension, dictionary coherence, event duration and background.

### Paper 2 — separate technical research contribution

**Working title:** *Structured Generative-Prior Inverse Modeling for Multi-Appliance NILM*.

Replace simple source dictionaries with **nonlinear source models** `G_psi_i(z_i)`, trained exclusively from training appliance traces; e.g. continuous hierarchical activity/dynamics codes that are **inferred**, not supplied as real-life labels.

\[
\boxed{
 z^*=\arg\min_z
 -\log p_R\left(X-\sum_iG_{\psi_i}(z_i)\right)
 +\sum_i\lambda_i\mathcal R_i(z_i)
 +\gamma\mathcal R_{\mathrm{competition}}(z).
}
\]

The `competition` term is a **research placeholder**. Its design must be nontrivial, principled, and should **not prohibit simultaneous appliance use**. Show that improvements come from the proposed latent inference, not simply from increasing generator capacity. Distinguish this paper from existing GAN-prior audio separation, generative compressed sensing, source priors, and existing generative NILM.

### Paper 3 — separate technical research contribution

**Working title:** *Ambiguity-Aware Diffusion Posterior Inference for Multi-Appliance NILM*.

Learn a source distribution/score using **training** submetered appliance traces. The inference target is a **posterior distribution**, not a single best reconstruction:

\[
\boxed{p(P\mid X)\propto p_\theta(P)\,p_R(X-\textstyle\sum_i P_i).}
\]

Explore **source-confusability/background-aware** guidance that retains legitimate multiple posterior modes when measurements cannot distinguish sources. Report actual probabilistic validity/calibration, not only the diversity of samples. Compare with existing **BASIS**, diffusion posterior sampling, conditional diffusion NILM (e.g. DiffNILM), and recent adaptive-guidance methods. Merely increasing guidance weights is insufficient novelty.

---

## 4. Prior-art map — 20 papers supplied by the researcher

These are **required nearest-neighbor reading**, not evidence that the proposed innovations are unclaimed. Local filenames mirror the number in the list. Read the relevant methodology sections before drafting claims or code.

| # | Paper | Relevant existing idea / novelty boundary |
|---|---|---|
| 01 | *Benchmarking Deep Inverse Models over Time, and the Neural-Adjoint Method* | Train forward surrogate, freeze it, optimize inputs; random restarts and boundary loss. Not NILM-specific. |
| 02 | *Deep Sparse Coding for Non-Intrusive Load Monitoring* | Appliance-specific deep dictionaries and sparse-code inversion are prior art. |
| 03 | *Energy Disaggregation via Deep Temporal Dictionary Learning* | Temporal/nonlinear dictionary learning in NILM is prior art. |
| 04 | *A Dynamical Systems Approach to Energy Disaggregation* | **Especially important:** appliance input → dynamic power response; find unknown inputs from aggregate. Closely overlaps original conceptual framing. |
| 05 | *Approximate Inference in Additive Factorial HMMs with Application to Energy Disaggregation* | Difference signals, robust unmodelled component, constrained approximate inference. |
| 06 | *Signal Aggregate Constraints in Additive Factorial HMMs, with Application to Energy Disaggregation* | Expected per-appliance total-energy constraints and convex relaxation already used. |
| 07 | *Training Deep Neural Networks for the Inverse Design of Nanophotonic Structures* | Tandem forward/inverse architecture and nonuniqueness. |
| 08 | *Learned Primal-Dual Reconstruction* | Learned iterative reconstruction coupled with known forward and adjoint operators exists. |
| 09 | *Analyzing Inverse Problems with Invertible Neural Networks* | Multimodal inverse solutions, posterior and unrecoverable parameters. |
| 10 | *Compressed Sensing using Generative Models* | Optimize latent variables through a frozen generator to solve a linear inverse problem. |
| 11 | *Unsupervised Audio Source Separation Using Generative Priors* | Separate pre-trained GAN source priors and latent projected-gradient separation already exist. |
| 12 | *Source Separation with Deep Generative Priors* | **BASIS:** score-based priors and annealed Langevin posterior separation; bridges Papers 2 and 3. |
| 13 | *Nanophotonic Particle Simulation and Inverse Design Using Artificial Neural Networks* | Frozen NN forward surrogate + input backpropagation. |
| 14 | *Neural-Adjoint Method for the Inverse Design of All-Dielectric Metasurfaces* | Neural-adjoint inverse search and multi-start/boundary practices. |
| 15 | *Energy Disaggregation via Discriminative Sparse Coding* | **Especially important:** trains sparse codes/dictionaries toward disaggregation accuracy, not just source reconstruction. |
| 16 | *Non-Intrusive Load Monitoring Using Prior Models of General Appliance Types* | Prior appliance models adapted from aggregate; iterative separation. |
| 17 | *Enhancing Neural Non-Intrusive Load Monitoring with Generative Adversarial Networks* | Appliance waveform generators in neural NILM already explored. |
| 18 | *Deep Latent Generative Models for Energy Disaggregation* | VRNN latent generative NILM already exists. |
| 19 | *Physics-Informed Time-Aware Neural Networks for Industrial Nonintrusive Load Monitoring* | Physics-informed NILM is not itself novel. |
| 20 | *Neural Network-Aided NILM Disaggregation: Revealing Appliance Consumption Patterns with Iterative Subtraction* | Iterative source subtraction with CNN/LSTM already exists. |

**Highest-priority comparison set for Paper 1:** 01, 02, 03, 04, 05, 06, **08**, **15**. Include 16, 19, 20 when distinguishing physical constraints or iterative behavior. For Papers 2–3 prioritize 09–12, 17–18, plus fresh diffusion-inverse / diffusion-NILM literature. Verify up-to-date publication claims before writing a novelty statement.

**Specific novelty pitfalls from these papers:**

- **Forward source dynamics + inversion was not first conceived here** (04).
- **Structured dictionary optimization has significant NILM precedent** (02, 03, 15).
- **Trainable unrolled reconstruction is general inverse-problem prior art** (08).
- **GAN/VAE generator + latent source separation is prior art** (10, 11, 17, 18).
- **Score-based source posterior sampling is prior art** (12).
- **Power physics and iterative NILM are prior art** (05, 06, 19, 20).

---

## 5. Research validity: what must be demonstrated

**Identifiability:** For additive observations, local differences of source Jacobians do **not** ensure a unique inverse. A useful study will investigate the rank/conditioning of relevant tangent spaces, overlap of response directions, and ambiguity under realistic source variability. Do not claim a formal theorem unless assumptions are stated and proved.

**A clean evaluation ladder:**

1. **Controlled aggregate:** `X_controlled = sum_i P_i`, assembled from same-timestamp test source waveforms, no background. Verifies the source representation and inverse solver in an ideal but still underdetermined setting.
2. **Background-stress mixtures:** add measured or training-derived background from correct splits to controlled mixtures, with no test-source leakage at inference.
3. **Real mains:** evaluate measured `X_real`; investigate signed measurement mismatches and background proxy validity. This is necessary for NILM claims.

**Baseline ladder:**

- Direct existing NILM (MART-NILM or other well-trained network) for accuracy context, with identical splits and metrics.
- Naive raw-power sum inversion with Adam (demonstrates identifiability failure).
- Classical nonnegative sparse coding / dictionary inversion.
- Learned temporal dictionary/dynamics with **standard Adam / proximal gradient** and conventional regularization.
- Classical optimizer variations (multi-start, trust region/Gauss–Newton as relevant).
- Candidate confusability-adaptive optimizer (only after novelty is precisely specified).
- For Paper 2: simple VAE/GAN latent optimization under same parameter/runtime budget.
- For Paper 3: ordinary diffusion posterior sampling / conditional diffusion NILM.

**Report by appliance and averaged:** MAE (overall), ON MAE, macro-F1, SAE; reconstructed aggregate error, background false attribution, overlapping-event performance, cross-house performance, runtime, restarts, sensitivity to initialization/priors, parameter counts and compute. Compare same timestamps, sampling, labels, evaluation conventions, and seeds. Do not infer attribution accuracy from low aggregate error.

**Critical data audit:** For real house `r_signed = X_real − sum_i P_true_i`, report distribution/quantiles and frequency of negative residuals (especially `< −100, −500, −1000 W`), by dataset/house and around gaps. The observed minimum `~−2631 W` is an earlier **smoke diagnostic**, not a complete diagnosis. Investigate misalignment before making hard conservation claims. Never use `max(r_signed,0)` as if it were exact physical ground truth.

---

## 6. Proposed development order and go/no-go gates

**Do not re-enter the abandoned direct-CNN → residual-projection → repeated-refinement path unless the researcher specifically selects it as a comparator.** That earlier architecture is not the chosen research direction.

### R0 — Literature and architecture audit (**NOW**)

Codex should:

- Read the old repository interfaces and inspect new project without modifying existing code.
- Read methodologies/equations for papers **01, 02, 03, 04, 08, 15**, then papers 05/06/16 as background controls.
- Produce a compact **nearest-prior-art matrix**: exact algorithm, training supervision, test inputs, inverse optimization variable, objective, handling of background, and what remains unaddressed.
- Propose **2–3 precise candidate innovations** for Paper 1 that are algorithmically distinguishable from standard dictionary/sparse coding/trust-region, with a derivation and falsifiable mechanism.
- Clarify the actual learning objective for source generators: reconstructing each appliance alone is not equivalent to solving cross-source attribution; check for dictionary degeneracy and nonunique coefficients.
- **Stop and report; do not train any new full model or rewrite the existing project yet.**

**Gate:** Clearly define what is new mathematically before large implementation.

### R1 — Minimal source-model / Jacobian experiment

- Begin with **two source types** and small real windows (e.g. kettle vs fridge, and a deliberately confused pair).
- Learn a compact, **non-generative deep** source representation (e.g. a normalized nonnegative temporal dictionary) solely from training appliance power.
- Validate **source reconstruction fidelity** and OFF behavior without requiring true inference-time ON/OFF states.
- Visualize `J_i^T r`, normalized source sensitivities/Gram/subspace overlap and simple optimizer trajectories.
- Controlled sum only, compare against standard sparse coding and direct-power inversion.

**Gate:** Attributions must be meaningfully better than trivial methods. A prettier gradient or smaller aggregate residual is not enough.

### R2 — Paper 1 inverse optimizer prototype

- Implement the simplest feasible **input-only optimizer** through **frozen source models**; then test one mathematically explicit **confusability-aware** update against standard Adam and conventional regularizers.
- No aggregate-to-appliance estimator in the proposed inference path.
- Add training-only source priors and rigorously test overlap, initialization, and Jacobian scaling issues.

**Gate:** Proposed mechanism must consistently improve **source error**, not just aggregate matching, under controlled conditions.

### R3 — Real mains and background

- Audit residuals; design an explicit robust likelihood or other defensible background treatment; test real mains from UK-DALE and REFIT on house-disjoint splits.
- Perform removal/addition ablations of background treatment. Report failure modes and speed.

**Gate:** Evaluate scientific viability for Paper 1 publication, without claiming novelty from mere combination of existing modules.

### R4 — Paper 2 only after Paper 1 baseline is reproducible

- Investigate hierarchical/nonlinear source generative models and a **specific** novel multi-source latent inference algorithm. Test whether improvement is architectural or just parameter count/stronger prior.

### R5 — Paper 3 only after posterior question is precisely scoped

- Establish diffusion prior, physically grounded likelihood, posterior sampling and credible uncertainty metrics. Investigate source/background-confusability-specific guidance beyond ordinary DPS.

---

## 7. Questions Codex should answer in its next report

1. **Exact nearest existing paper:** Is there an established NILM algorithm that already uses essentially the proposed source-Jacobian-aware inverse updates, possibly under another name (reweighted sparse coding, coherent dictionary demixing, variable-metric proximal optimization, analysis/synthesis priors, Gauss–Newton, learned optimization)? Show equations and differences.
2. **Representation:** What is the **smallest** trainable appliance power-generating forward component that preserves temporal events and OFF periods without observed ON/OFF labels at test time? Why choose it over a classical dictionary?
3. **Confusability math:** How exactly will the method distinguish two correlated response Jacobians? What source prior breaks ties? What happens if two dictionaries are identical?
4. **Mathematical honesty:** What can and cannot be recovered from an additive single-channel signal? Are uniqueness/stability conditions plausible for the proposed source class? Provide counterexamples.
5. **Background:** How do we avoid (a) background absorbing all power and (b) forcing nonexistent exact balance when submeter/mains mismatch is severe?
6. **Generalization:** Does appliance-specific forward learning depend on house identity or usage distribution? What transfers to held-out houses?
7. **Novelty per paper:** What truly *new algorithmic operation* does each paper contribute? Name at least three most threatening baselines for each.
8. **Feasibility:** What is a minimal 2-appliance experiment, its required data, runtime, expected diagnostic plots, and stop/go criteria?

### Required immediate Codex output

A **design review only**, containing:

- A corrected 3-paper research direction (keep the **optimization → solution space → posterior** progression).
- A nearest-prior-art comparison grounded in actual files.
- Paper 1 candidate mathematical algorithm(s), with transparent novelty risks.
- A minimal R1 experimental protocol and file-change proposal.
- Explicit statement: **no new model training or coding has begun** unless the researcher separately authorizes it.

---

## 8. What not to repeat from the earlier conversation

There was considerable confusion because a project originally meant to follow **forward learning → frozen forward model → input optimization** drifted to **direct supervised NILM → physics projection → learned refinement**. Both are legitimate inverse-problem-related techniques, but **they are different experimental hypotheses**. Do not rename the latter as the former, and do not ask the researcher to re-litigate this established distinction.

**Chosen direction is fixed for planning purposes:** train source-side forward dynamics, enforce exact physical aggregation, freeze source models, **infer continuous source variables from measured aggregate by optimization**, and establish identifiability as the central Paper 1 question. Paper 2/3 increase source-model and posterior sophistication while each pursuing a distinct, verifiable contribution.

**Practical instruction:** Provide compact, staged proposals and explain new mathematical symbols in beginner-friendly terms before proposing heavy code. Ask for no redundant data already documented here. Report uncertainties instead of inventing novelty or performance.

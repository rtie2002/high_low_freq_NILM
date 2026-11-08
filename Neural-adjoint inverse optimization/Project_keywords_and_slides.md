## Deep Forward Modeling with Inverse Optimization

Project proposal for non-intrusive load monitoring (NILM): jointly estimate five target appliances from a whole-house main-meter signal containing additional, unlabeled loads. No experiments for this proposal have been completed.

### Four candidate approaches: goals, methods, and papers

These are related approaches to combining forward models and inverse inference, not four mutually exclusive versions of the same algorithm. Neural Adjoint directly matches a learned forward surrogate followed by input optimization. Deep unfolding can use a known physical operator, and an invertible network can generate inverse candidates without iterative input optimization. The detailed formulation and worked example later in this document describe the Neural Adjoint candidate only.

| Approach | Goal | Method | Representative paper |
| --- | --- | --- | --- |
| **Neural Adjoint** | Find inputs that produce a requested observation. | Train a forward surrogate, freeze its weights, then repeatedly optimize candidate inputs through it. | Ren et al., *Benchmarking Deep Inverse Models over Time, and the Neural-Adjoint Method*, **NeurIPS 2020**. [Paper](https://papers.nips.cc/paper_files/paper/2020/hash/007ff380ee5ac49ffc34442f5c2a2b86-Abstract.html) [1] |
| **Tandem neural network** | Train an inverse predictor whose outputs reproduce the requested response. | Pretrain and freeze a forward network. Train an inverse network through the forward network using response reconstruction loss. | Liu et al., *Training Deep Neural Networks for the Inverse Design of Nanophotonic Structures*, **ACS Photonics 2018**. [Paper](https://doi.org/10.1021/acsphotonics.7b01377) [4] |
| **Deep unfolding / learned iterative reconstruction** | Learn an efficient reconstruction procedure that incorporates a forward operator. | Unroll optimization iterations into network stages. Learn update components while retaining forward-model information. | Adler and Öktem, *Learned Primal-Dual Reconstruction*, **IEEE Transactions on Medical Imaging 2018**. [Paper](https://arxiv.org/abs/1707.06474) [7] |
| **Invertible neural network** | Represent multiple possible hidden inputs consistent with an observation. | Learn an invertible mapping between hidden inputs and observations augmented with latent variables. Sample the latent variables and evaluate the inverse mapping. | Ardizzone et al., *Analyzing Inverse Problems with Invertible Neural Networks*, **ICLR 2019**. [Paper](https://arxiv.org/abs/1808.04730) [8] |

| Approach | What is trained? | What happens for a new aggregate? |
| --- | --- | --- |
| Neural Adjoint | Forward-network weights | Keep weights fixed and iteratively update candidate appliance/background inputs. |
| Tandem network | Forward network first, then inverse-network weights | Run the inverse network to predict inputs. Optional later refinement is a separate addition. |
| Deep unfolding | Learned components of a finite sequence of reconstruction steps | Execute the trained reconstruction stages with the aggregate and forward operator. |
| Invertible neural network | Weights of an invertible model with forward and distribution-matching objectives | Combine the observation with sampled latent variables to obtain candidate inputs. |

The deployment descriptions are proposed NILM interpretations. The cited papers do not validate these four approaches for our five-target, unknown-background setting. For an invertible approach, dimensional augmentation and the training distributions must be designed explicitly; invertibility does not recover information lost in aggregation by itself.

**Selection guide:** prioritize Neural Adjoint if the supervisor means optimizing unknown device inputs through a frozen forward network. Prioritize tandem if the supervisor means training an inverse predictor through a forward model. Consider unfolding when learned iterative correction is central, and invertible models when representing several possible decompositions is central. These are research choices, not a proven performance ranking.

Every route must account for non-target loads and distinguish appliance accuracy from aggregate reconstruction. The forward component may be learned, known from physics, or a combination. A neural forward model that only approximates a known sum still needs an incremental-value justification. Tandem prediction and Neural Adjoint refinement can also be combined, but that hybrid requires its own comparison.

### Research question and novelty

**Research question:** Can constrained input optimization through a frozen differentiable forward model recover five target appliance trajectories under unmodeled background loads, with useful accuracy and computation cost relative to MARTNILM?

The inference procedure differs from direct aggregate-to-appliance prediction. Neural Adjoint is an established method [1]. Handling unmodeled observations also has NILM precedents [6]. Neither the general procedure nor adding a background component establishes novelty. A potential contribution would require a specific, validated model or inference technique for this five-appliance setting.

### Our actual data setting

| Component | Training | Deployment / test inference |
| --- | --- | --- |
| Whole-house aggregate | Observed | Observed |
| Kettle, refrigerator, dishwasher, washing machine, microwave | Individual power labels available | Individual powers unknown |
| Other appliances | No separate device labels | No separate device labels |
| Target-subtracted residual | Computable from training labels | Unknown; must be inferred or handled probabilistically |
| Held-out appliance ground truth | Used only in evaluation when available | Never supplied to the inverse optimizer |

The five targets are modeled together. Other devices remain present in the aggregate. We do not assume that the five target powers sum to the complete main-meter reading.

Repository context: [five-appliance data configuration](../multi_appliances_NILM/config/experiment_mixed_ukdale_refit_5w_house_split.yaml) and [MARTNILM model configuration](../multi_appliances_NILM/config/models/mart_nilm.yaml). The latter includes independent sampling of target appliance windows and residual background for training augmentation.

### Main and secondary keywords

| Level | Keyword | Meaning |
| --- | --- | --- |
| Framework | Deep forward modeling with inverse optimization | Learn a forward model, then search its input space |
| Main method | Neural Adjoint | Freeze forward-network weights and differentiate the loss with respect to unknown inputs [1] |
| Model property | Differentiable neural surrogate | A learned forward approximation that provides input gradients [1] |
| Scenario requirement | Background / residual modeling | Account for aggregate power outside the five targets; unmodeled-observation precedent in [6] |
| Supporting technique | Appliance-specific regularization | Restrict incorrect device allocations; an energy-constraint precedent appears in [3] |
| Supporting technique | Boundary constraints and multiple initializations | Restrict the search and assess initialization sensitivity [1] |

Latent-space optimization and generative priors [2, 5] are optional extensions. A tandem neural network [4] is an alternative involving a trained inverse network. None is required merely to define the basic input-optimization procedure.

### Notation and physical meaning

| Symbol | Meaning | Shape / unit |
| --- | --- | --- |
| $i=1,\ldots,5$ | Target appliance index | Integer |
| $t=1,\ldots,T$ | Sample index within a window | Integer |
| $p_i(t)$ | Active power of target appliance $i$ | Watts (W) |
| $P_{\mathrm{agg}}(t)$ | Whole-house active power at the main meter | W |
| $b(t)$ | Combined power of non-target appliances | W |
| $\varepsilon(t)$ | Measurement and alignment discrepancy | W |
| $\mathbf p_i$ | Appliance trajectory $[p_i(1),\ldots,p_i(T)]^\top$ | $\mathbb R^T$ |
| $\mathbf x$ | Stacked five-appliance trajectories | $\mathbb R^{5T}$ |
| $\mathbf y$ | Aggregate trajectory | $\mathbb R^T$ |
| $\mathbf r$ | Effective residual: background plus discrepancy | $\mathbb R^T$ |
| $\mathbf u=[\mathbf x^\top,\mathbf r^\top]^\top$ | All unknown inputs of the augmented forward model | $\mathbb R^{6T}$ |
| $\theta$ | Forward-network weights | Parameter vector |
| $m=1,\ldots,M$ | Training-window index | Integer |
| $k$ | Inverse-optimization iteration, not physical time | Integer |

All power trajectories are in W before any consistent scaling for optimization. With
$\mathbf x=[\mathbf p_1^\top,\ldots,\mathbf p_5^\top]^\top$,
the aligned active-power balance is

$$
P_{\mathrm{agg}}(t)
=
\sum_{i=1}^{5}p_i(t)+b(t)+\varepsilon(t).
$$

Define the sum operator $A=[I_T\ I_T\ I_T\ I_T\ I_T]$, where $I_T$ is the $T\times T$ identity matrix. Then

$$
\mathbf y=A\mathbf x+\mathbf r,
\qquad
\mathbf r=\mathbf b+\boldsymbol\varepsilon.
$$

The residual is one combined nuisance trajectory, not a sixth individually identified appliance. Although physical background consumption is nonnegative in a household without export, the effective residual can be negative because of discrepancies. Inspect negative residuals rather than silently clipping them. This formulation assumes compatible active-power channels and accounting for any generation or storage if present.

### Why the earlier five-input formulation needs revision

The mapping $F_\theta(\mathbf x)\approx\mathbf y$ omits background. The same five target powers can coexist with different other loads and therefore different whole-house aggregates. A deterministic model of the five powers alone cannot know the actual background realization.

For example, with all five targets off, the main meter might read 100 W or 800 W depending on other devices. Learning average correlations does not remove that missing information.

To preserve the supervisor's forward-training and inverse-optimization procedure, the candidate formulation below augments the unknown input with residual power:

$$
F_\theta(\mathbf x,\mathbf r)\approx\mathbf y.
$$

This is a proposed adaptation. It is not a model specified or validated for this scenario by Ren et al. [1].

### Connection to the Neural Adjoint paper

Ren et al. [1], Section 3, first learn a neural approximation $\hat f$ of a forward process. They then freeze the network and update candidate inputs. Their Eq. (4) has the form

$$
\mathbf q^{(k+1)}
=
\mathbf q^{(k)}
-
\alpha_k
\left.
\nabla_{\mathbf q}\ell\!\left(\hat f(\mathbf q),\mathbf y_{\mathrm{obs}}\right)
\right|_{\mathbf q=\mathbf q^{(k)}}.
$$

Here $\mathbf q$ is generic input notation used in this document to avoid confusing the paper's input with our five-appliance vector. The paper also introduces a boundary loss in Section 3.1.

| Paper mechanism [1] | Proposed five-appliance adaptation |
| --- | --- |
| Forward input | $\mathbf u$, containing five target trajectories and residual |
| Forward response | Whole-house aggregate $\mathbf y$ |
| Learned surrogate | $\hat f(\mathbf u)=F_{\hat\theta}(\mathbf x,\mathbf r)$ |
| Frozen weights during inversion | $\hat\theta$ |
| Optimized input | Both candidate appliance powers and residual |
| Boundary restriction | Allowed input domain, adapted to the data |
| Additional domain information | Appliance and residual regularizers requiring separate design |

The source supports the frozen-network input-gradient mechanism. It does not prove our background treatment, appliance regularizers, or NILM performance.

### Stage 1: prepare training pairs and train the forward model

For each training window, use the observed aggregate and five appliance labels:

$$
\mathcal D_{\mathrm{train}}
=
\left\{(\mathbf x_{\mathrm{lab}}^{(m)},\mathbf y^{(m)})\right\}_{m=1}^{M}.
$$

Compute the effective residual in aligned watt-space data:

$$
\mathbf r_{\mathrm{train}}^{(m)}
=
\mathbf y^{(m)}-A\mathbf x_{\mathrm{lab}}^{(m)}.
$$

This does not require labels for each other appliance. It estimates their combined contribution together with discrepancies. Deriving this quantity during training is allowed because the target labels are available there.

Build the augmented training input

$$
\mathbf u^{(m)}
=
\begin{bmatrix}
\mathbf x_{\mathrm{lab}}^{(m)}\\
\mathbf r_{\mathrm{train}}^{(m)}
\end{bmatrix},
$$

and train using

$$
\mathcal L_{\mathrm{forward}}(\theta)
=
\frac{1}{MT}\sum_{m=1}^{M}
\left\|F_\theta(\mathbf u^{(m)})-\mathbf y^{(m)}\right\|_2^2,
\qquad
\hat\theta\approx\underset{\theta}{\operatorname{arg\,min}}\,
\mathcal L_{\mathrm{forward}}(\theta).
$$

Only the network weights change. The approximation sign acknowledges that neural training does not guarantee a global minimum.

**Critical limitation:** the residual construction makes the exact relation $\mathbf y=A\mathbf x+\mathbf r$ true by definition. A network can simply learn addition. Low forward error therefore does not demonstrate appliance understanding or justify a neural surrogate. The exact sum must be a baseline.

Training labels and residuals can also help design appliance and background priors. The forward squared-error objective alone does not learn those priors. Their form, training procedure if learned, and influence during inversion remain open design decisions. Fit statistics and priors using training data only.

### Stage 2: freeze the model and infer five appliances plus residual

Given only the new aggregate $\mathbf y_{\mathrm{obs}}$, freeze $\hat\theta$ and initialize $\mathbf u^{(0)}$. Seek a solution of

$$
(\hat{\mathbf x},\hat{\mathbf r})
\in
\underset{(\mathbf x,\mathbf r)\in\mathcal U}
{\operatorname{arg\,min}}
\mathcal L_{\mathrm{inverse}}(\mathbf x,\mathbf r;\mathbf y_{\mathrm{obs}}),
$$

where one candidate objective is

$$
\begin{aligned}
\mathcal L_{\mathrm{inverse}}
={}&
\frac{1}{T}
\left\|F_{\hat\theta}(\mathbf x,\mathbf r)-\mathbf y_{\mathrm{obs}}\right\|_2^2\\
&+\lambda_{\mathrm{app}}R_{\mathrm{app}}(\mathbf x)
+\lambda_{\mathrm{bg}}R_{\mathrm{bg}}(\mathbf r).
\end{aligned}
$$

| Term | Role | Design status |
| --- | --- | --- |
| Observation mismatch | Explain the measured whole-house aggregate | Defined |
| $R_{\mathrm{app}}$ | Favor plausible target-appliance trajectories | Not yet specified or validated |
| $R_{\mathrm{bg}}$ | Restrict implausible background allocations | Not yet specified or validated |
| $\mathcal U$ | Encode allowed input ranges or other hard constraints | Requires data audit |
| $\lambda_{\mathrm{app}},\lambda_{\mathrm{bg}}\geq0$ | Balance the penalties | Select using validation data |

A constant or always-smooth background is not generally a valid assumption: non-target devices can have abrupt, high-power events. Penalizing residual magnitude alone can also force real background into target estimates.

For differentiable penalties and a simple box domain, a possible update is

$$
\mathbf u^{(k+1)}
=
\Pi_{\mathcal U}\left[
\mathbf u^{(k)}
-
\alpha_k
\nabla_{\mathbf u}
\mathcal L_{\mathrm{inverse}}(\mathbf u^{(k)};\mathbf y_{\mathrm{obs}})
\right].
$$

$\Pi_{\mathcal U}$ projects onto the allowed domain, such as clipping to box bounds. This projected, regularized update extends the paper's input-gradient rule. Non-smooth or discrete constraints require suitable optimization treatment. No global recovery guarantee is assumed.

Return the five appliance estimates. Retain the residual for diagnostics, not as an identified individual device. Never compute the test residual from held-out appliance ground truth inside the inference algorithm.

### Fundamental ambiguities and failure cases

| Failure | Why it happens | What must be tested |
| --- | --- | --- |
| Target inflation | Ignoring residual forces five targets to explain other loads | Errors during strong background events |
| Everything becomes background | With unrestricted residual, $\mathbf x=0,\mathbf r=\mathbf y_{\mathrm{obs}}$ fits the exact sum | Background-collapse rate |
| Target/background exchange | Similar-looking signals can be reassigned without changing the aggregate | Confusing device and background combinations |
| Surrogate exploitation | Optimization finds inputs where the network is inaccurate | Compare neural and exact physical reconstructions |
| Invalid prior | Incorrect assumptions exclude real behavior | Ablations across background regimes |
| Slow or unstable inversion | Many unknowns, local minima, and initialization sensitivity | Runtime, restarts, and variability |

For the exact sum model, any feasible change satisfying
$A\Delta\mathbf x+\Delta\mathbf r=0$
leaves the aggregate unchanged. Adding a residual makes the forward formulation complete but does not solve identifiability.

### Worked example: all five targets and other appliances

This one-sample example explains the procedure. Values are illustrative and not experiment results. The actual task uses complete windows.

**Step 1: collect the training channels.** Record the aggregate and the five target powers. Other devices do not need separate labels.

| Training sample | Kettle | Fridge | Dishwasher | Washing machine | Microwave | Aggregate |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0 W | 100 W | 0 W | 0 W | 0 W | 400 W |
| B | 0 W | 100 W | 0 W | 0 W | 1500 W | 2000 W |
| C | 0 W | 0 W | 0 W | 0 W | 0 W | 800 W |

**Step 2: derive training residuals.** The five-target sums are 100 W, 1600 W, and 0 W. Subtracting them from the aggregate gives residuals of 300 W, 400 W, and 800 W.

**Step 3: train the augmented forward model.** For sample B, the input contains five target powers and a 400 W residual. The training output is 2000 W. These rows illustrate the data format, not a sufficient training set. Assume the model perfectly represents summation for the remaining example.

**Step 4: freeze its weights.** The basic workflow trains one forward network. The inverse optimizer is an algorithm, not a second trained network. Learned priors or an optional MARTNILM initializer would introduce additional learned components and must be described separately.

**Step 5: receive a new aggregate of 2000 W.** The five device powers and residual are all unknown. For later evaluation only, suppose the hidden truth is:

| Component | Hidden truth |
| --- | --- |
| Kettle | 0 W |
| Refrigerator | 100 W |
| Dishwasher | 0 W |
| Washing machine | 0 W |
| Microwave | 1500 W |
| Other loads / residual | 400 W |
| Whole-house aggregate | 2000 W |

**Step 6: initialize candidate inputs.** One possible starting guess is refrigerator 100 W, microwave 1000 W, the other three targets 0 W, and residual 200 W. The forward model predicts 1300 W, leaving a 700 W deficit.

**Step 7: evaluate and update the candidates.** The optimizer uses aggregate mismatch plus the specified appliance and background penalties. It can change both the five target powers and the residual. Different initializations or priors can produce different results.

| Candidate | Five-target allocation | Residual | Predicted aggregate |
| --- | --- | --- | --- |
| Initial guess | Fridge 100 W, microwave 1000 W, others 0 W | 200 W | 1300 W |
| Correct allocation | Fridge 100 W, microwave 1500 W, others 0 W | 400 W | 2000 W |
| Background assigned to target | Fridge 100 W, microwave 1900 W, others 0 W | 0 W | 2000 W |
| Background collapse | All five targets 0 W | 2000 W | 2000 W |

These are candidate explanations, not a claimed gradient-descent trajectory. Aggregate error alone cannot select the correct row. Simple bounds may exclude some candidates but do not generally establish a unique answer.

**Step 8: stop and output the five target estimates.** Stop at the convergence tolerance or computation budget. A correct result in this example would be 100 W refrigerator power and 1500 W microwave power, with the other targets off. The algorithm is not guaranteed to recover that result.

**Step 9: evaluate using held-out ground truth.** Compare every target estimate with its label. Also inspect the inferred residual. A result with microwave 1900 W has a 400 W microwave error despite zero aggregate reconstruction error.

The procedure remains: train the forward model, freeze it, observe the aggregate, guess all unknown inputs, check and update them, and return appliance estimates. In this scenario the unknown inputs include residual power.

### Experiments needed before claiming feasibility

| Comparison | Purpose |
| --- | --- |
| Direct MARTNILM prediction | Existing five-target accuracy and speed reference |
| Exact sum plus appliance/background constraints | Establish whether neural forward modeling adds value |
| Learned forward surrogate with the same constraints | Isolate the effect of the surrogate |
| No-background version | Measure the cost of an incorrect closed-house assumption |
| Unrestricted-background version | Expose trivial reconstruction solutions |
| MARTNILM initialization followed by refinement | Test an optional hybrid separately |

Use identical data splits and target definitions. Tune on validation data. Evaluate across unseen houses and background load ranges, including sudden non-target events.

Report per-appliance power errors, state-detection performance, aggregate reconstruction, physical sum consistency, initialization stability, and runtime. Training-derived residuals can support diagnostics; test-derived residual ground truth remains evaluation-only.

A useful result must improve appliance recovery or another justified outcome, not merely reduce aggregate error. If exact summation performs equally well, a neural surrogate is not justified for this power-only formulation. Learning device-state-to-response dynamics is a possible redesign, but it changes the forward inputs and requires a separate formulation.

### Literature and reading order

| Priority | Paper | What it supports |
| --- | --- | --- |
| First | Ren et al., NeurIPS 2020 [1] | Frozen forward surrogate and input-gradient inversion |
| First | Kolter and Jaakkola, AISTATS 2012 [6] | Accounting for unmodeled observations in NILM |
| First | Zhong et al., NeurIPS 2014 [3] | Appliance energy constraints and source ambiguity |
| Optional generator route | Bora et al., ICML 2017 [2] | Generative constraints and latent-variable inversion |
| Optional generator route | Narayanaswamy et al., Interspeech 2020 [5] | Joint latent optimization for several sources |
| Alternative architecture | Liu et al., ACS Photonics 2018 [4] | Tandem forward and inverse networks |
| Alternative architecture | Adler and Öktem, IEEE Transactions on Medical Imaging 2018 [7] | Learned iterative reconstruction with a forward operator |
| Alternative architecture | Ardizzone et al., ICLR 2019 [8] | Invertible models and distributions of inverse solutions |

The five-appliance background-aware formulation above is our proposed adaptation. These papers do not establish that it works, that it improves MARTNILM, or that it is the first such NILM method.

### References

1. Ren, S., Padilla, W., and Malof, J. [Benchmarking Deep Inverse Models over Time, and the Neural-Adjoint Method](https://papers.nips.cc/paper_files/paper/2020/hash/007ff380ee5ac49ffc34442f5c2a2b86-Abstract.html). NeurIPS, 2020. [Full text](https://proceedings.neurips.cc/paper_files/paper/2020/file/007ff380ee5ac49ffc34442f5c2a2b86-Paper.pdf).
2. Bora, A., Jalal, A., Price, E., and Dimakis, A. G. [Compressed Sensing using Generative Models](https://proceedings.mlr.press/v70/bora17a.html). ICML, 2017, pp. 537–546.
3. Zhong, M., Goddard, N., and Sutton, C. [Signal Aggregate Constraints in Additive Factorial HMMs, with Application to Energy Disaggregation](https://proceedings.neurips.cc/paper_files/paper/2014/hash/bf64575b42486eb72efdfa0091be60a5-Abstract.html). NeurIPS, 2014.
4. Liu, D., Tan, Y., Khoram, E., and Yu, Z. [Training Deep Neural Networks for the Inverse Design of Nanophotonic Structures](https://doi.org/10.1021/acsphotonics.7b01377). ACS Photonics, 2018. [Open version](https://arxiv.org/abs/1710.04724).
5. Narayanaswamy, V., Thiagarajan, J. J., Anirudh, R., and Spanias, A. [Unsupervised Audio Source Separation Using Generative Priors](https://www.isca-archive.org/interspeech_2020/narayanaswamy20_interspeech.html). Interspeech, 2020, pp. 2657–2661.
6. Kolter, J. Z., and Jaakkola, T. [Approximate Inference in Additive Factorial HMMs with Application to Energy Disaggregation](https://proceedings.mlr.press/v22/zico12.html). AISTATS, 2012, pp. 1472–1482.
7. Adler, J., and Öktem, O. [Learned Primal-Dual Reconstruction](https://arxiv.org/abs/1707.06474). IEEE Transactions on Medical Imaging, 2018, 37(6), pp. 1322–1332. [DOI](https://doi.org/10.1109/TMI.2018.2799231).
8. Ardizzone, L., et al. [Analyzing Inverse Problems with Invertible Neural Networks](https://arxiv.org/abs/1808.04730). ICLR, 2019.

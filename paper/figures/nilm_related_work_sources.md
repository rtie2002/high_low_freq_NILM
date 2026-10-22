# Related-work figure: evidence and interpretation

## Current revision: nested appliance heads (2026-10-21)

The current editable source uses two levels: power/state coupling for one
appliance in (a)-(c), and feature sharing across appliances in (d)-(f). The
regenerator is now `draw_nilm_nested_concept.py`; the two older generators and
the historical panel mapping below describe superseded versions. Preserve manual
draw.io changes before regenerating. The current PNG/SVG exports match the
draw.io source. The older PDF export and manuscript caption have not been
updated in this drawing-only revision.

| Current panel | Primary evidence | Interpretation and omissions |
|---|---|---|
| (a) Parallel power-state heads | UNet-NILM, sections 2.2-2.3, equations (6)-(8), DOI 10.1145/3427771.3427859. | Shows the output objectives for appliance i. Actual UNet-NILM has vector outputs, per-appliance state softmax and quantile regression. No SGN product is added to a method that does not use one. |
| (b) State-gated power | Shin et al., Subtask Gated Networks for Non-Intrusive Load Monitoring, AAAI 2019, DOI 10.1609/aaai.v33i01.33011150; TransferNILM, section III-A, equations (13)-(15), DOI 10.1109/TIM.2025.3541652. | Explicit product of raw power and ON probability. Optional standby-power addition is omitted to isolate this common mechanism. MART's complete equation remains in its architecture figure. |
| (c) Expert-routed power-state heads | SAMNet, section III-B, Fig. 3 and section III-C, equation (15), DOI 10.1109/TSG.2021.3139395. | Experts feed task-specific attention fusion. Gates provide weights, and the final state tower gates raw power. Three experts are illustrative. SAMNet is one target appliance at a time. |
| (d) Shared/private branches | Transfer Learning for Multiappliance-Task Nonintrusive Load Monitoring, Fig. 2 and section III-A, DOI 10.1109/TIM.2025.3541652. | Shared aggregate processing followed by private appliance decoding. Dashed boxes contain paired gated heads. Branch internals are omitted. |
| (e) Appliance-specific expert routing | Liu and Fan (2026), arXiv:2609.18315, section III-A, one-to-many refinement stage; general precedent Ma et al., KDD 2018, DOI 10.1145/3219819.3220007. | Each appliance receives its private expert group and a shared expert through its own weighted fusion. E_i abbreviates the two private experts. The preceding many-to-one aggregation and residual self-gates are omitted. This is a routing motif, not a full FLAME reproduction. The NILM source is a preprint. |
| (f) Decoder feature interaction | MATNilm, section III-B, DOI 10.1109/TII.2023.3301026; MART-NILM decoder feature exchange; Vandenhende et al., DOI 10.1109/TPAMI.2021.3054719. | Cross-links explicitly exchange appliance features before paired heads. MATNilm already uses appliance and temporal attention. Gated heads are a composable example, not a claim that all interaction models have identical heads. |

The dashed enclosures make the nested organization visible: appliance-specific
features feed power and state branches; state probabilities condition final
power. This explanatory grouping is not a new formally established task class.
Routing weights (dashed arrows) differ from output power gates (product circles).
The three body mechanisms can be combined. Power-only, state-only and joint
power/state objectives can be introduced in prose without another figure row.

SGN primary source checked for this revision:
https://ojs.aaai.org/index.php/AAAI/article/view/3908 .

## Historical mapping (superseded)

The editable source is `nilm_related_work_concept.drawio`. All boxes, labels,
and connectors are native draw.io objects. The SVG and PDF are exports of that
same source. `build_nilm_related_work_concept.py` regenerates the initial layout;
manual changes in draw.io should be retained before rerunning that generator.

## Scope

The top row separates directly learned power, directly learned state, and joint
power/state objectives. It is not an exhaustive taxonomy of every NILM method.
The bottom row isolates three compatible sharing mechanisms, not a chronological
progression or a claim that these are mutually exclusive model families.

"Nested multi-task problem" is this manuscript's explanatory view: appliance
identity and prediction objective define two levels of organization. It is not a
claim of a new problem or a requirement for a hierarchical network topology.

## Panel-to-source mapping

| Panel | Evidence | What the schematic does and does not claim |
|---|---|---|
| (a) | Alami C. et al., Conv-NILM-Net, sections on source separation. https://arxiv.org/abs/2208.02173 | Direct multi-appliance power prediction. A thresholded state is optional post-processing, not a claimed original Conv-NILM-Net state head. |
| (b) | Buddhahai et al., Sustainable Cities and Society 39 (2018), pp. 623-624, equations (2)-(3), section 2.3 and Table 2. DOI: 10.1016/j.scs.2018.02.002. Tanoni et al., IEEE TSG 14(1), 440-452. DOI: 10.1109/TSG.2022.3191908. | Per-label and label-set methods represent alternatives, not a single shared architecture. Buddhahai estimates power using mean ON/OFF power; Tanoni is a deep weak-label example and does not use that lookup by implication. Temporal granularity is intentionally unspecified. |
| (c) | Faustine et al., UNet-NILM, sections 2.1-2.3 and equations (6)-(8), DOI: 10.1145/3427771.3427859. Xiong et al., MATNilm, section III-B. https://arxiv.org/abs/2307.14778 | Paired supervision over multiple appliances. The appliance grouping shows the task structure; UNet-NILM's actual network has shared features and vector output layers, not MATNilm-style private decoders. |
| (d) | MATNilm, section III-B, description of n appliance branches followed by regression/classification branches. | Only its branching pattern is drawn. The complete MATNilm already contains temporal and appliance-wise attention; the figure must not imply otherwise. |
| (e) | Liu et al., SAMNet, section III-A, p. 2414, explicitly one target appliance; section III-B and Fig. 3, p. 2415, shared experts and task-specific gates/attention; section III-C for the final power gate. DOI: 10.1109/TSG.2021.3139395. General precedent: Ma et al., KDD 2018, DOI: 10.1145/3219819.3220007. | Three experts illustrate the routing principle, not a fixed expert count. Fusion is task-specific attention in SAMNet, not merely an unweighted sum. Its final output power gate and learned state-label construction are omitted to isolate routing. It is not a state-only or joint multi-appliance implementation. |
| (f) | MART-NILM relation_attention configuration and forward path. General cross-task precedent: Xu et al., PAD-Net, CVPR 2018, multimodal distillation and attention-guided message passing. https://arxiv.org/abs/1805.04409 | Private appliance feature exchange before power/state heads. This is not PAD-Net's full architecture: MART-NILM has no intermediate task predictions or distillation loss. State-conditioned output gating is omitted here and shown in the main architecture figure. |

Additional expert-routing example: Liu and Fan (2026), source preprint (not a
peer-reviewed journal citation), https://arxiv.org/abs/2609.18315. Local PDF,
section III-A and Fig. 1: shared/appliance-specific experts, many-to-one
aggregation then one-to-many refinement, followed by paired heads. Its full
two-stage topology is not panel (e); the note identifies a multi-appliance use
of routing discussed separately in the prose.

General architecture vocabulary: Vandenhende et al., IEEE TPAMI 44(7),
3614-3633 (2022), section II: sharing and encoder/decoder interactions.
https://arxiv.org/abs/2004.13379.

Training distinction: Chen et al., ICML 2018, pp. 794-803, GradNorm balances
gradient magnitudes. MART-NILM balances loss scales; it must not be described as
gradient-norm equalization. https://proceedings.mlr.press/v80/chen18a.html.

The former protocol comparison table has been removed from the manuscript at
the author's request. Its detailed audit remains in
`multi_appliances_NILM/docs/multi_appliance_literature_positioning.md`.

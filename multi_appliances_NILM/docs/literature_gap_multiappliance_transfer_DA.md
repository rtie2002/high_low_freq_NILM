# NILM Literature Gap Report: Multi-Appliance × Transfer / Domain Adaptation

**Scope:** Transfer / DA / Meta PDFs under `Literature Review/NILM/transfer learning/` (incl. Domain Adaption + Meta Learning) plus all PDFs under `Literature Review/NILM/multi-appliances/`.  
**Method:** Text extracted with PyMuPDF (`fitz`); UTF-8 temp extracts in `docs/_lit_extract/`. Claims below are limited to what was extractable. One multi-appliance PDF is image-only (no text layer; OCR unavailable) and is flagged.  
**Duplicates:** `Li_etal_AE_2022_...pdf` and `Transfer learning for multi-objective...pdf` are the same Applied Energy paper. `Deep_Domain_Adaptation_...` appears in root and Domain Adaption; counted once.  
**Student context used in synthesis:** planned focus = multi-appliance + multi-label + transfer; MultiNILM experiments show unsupervised DA often fails to make UK-DALE H2 usable.

---

## 1. Per-paper briefs

### 1.1 Transfer / Domain Adaptation / Meta

#### Adversarial and Hierarchical Distribution Alignment Network for Nonintrusive Load Monitoring  
**Xiong, Tan, Hu, Cai, Hu (Electronics / MDPI, 2026)**  
1. Single-appliance energy disaggregation (KT/MV/DW/WM/FG evaluated separately).  
2. **Unsupervised DA** (adversarial + CORAL + MK-MMD hierarchical alignment); target unlabeled.  
3. REDD, UK-DALE, SynD; cross mappings like U1→R1, U1→U2, SynD→UK-DALE/REDD. House-level splits are reported via `U#→R#` style; not an H1+H5→H2 multi-appliance protocol.  
4. Novelty: hierarchical feature+label-space alignment (CORAL + MK-MMD) with adversarial training under limited source labels / unlabeled target.

#### Deep Domain Adaptation for NILM Based on a Knowledge Transfer Learning Network  
**Lin, Ma, Zhu, Liang (IEEE TSG, 2022)**  
1. Single-appliance Seq2Seq TCN disaggregation.  
2. **Unsupervised DA** (joint disaggregation + domain-adaptation loss; unlabeled target houses).  
3. REDD / UK-DALE / REFIT; e.g. REDD H3, UK-DALE H2, REFIT H2 as unlabeled targets.  
4. Novelty: end-to-end TLN with TCN + distribution discrepancy loss for domain-invariant appliance features under data shortage.

#### Enhancing NILM through Transfer Learning with Transformer Models (TransDisNILM)  
**Rong, Wang, Zhou, He, Wu (Energy & Buildings, 2025)**  
1. Single-appliance (per-load) transformer disaggregation; multi-load scenarios discussed as context, not joint multi-label.  
2. **Fine-tune with target labels** (freeze CNN / fine-tune; also direct transfer; source-task selection e.g. WM→other loads). Not UDA.  
3. REFIT / UK-DALE / REDD / PLAID; tables show train/val on house 1, test house 2 within datasets; REFIT→UK-DALE/REDD CTL.  
4. Novelty: transformer + sinusoidal encoding + systematic source-task selection and freeze/fine-tune recipes.

#### Transfer Learning for Multi-Objective NILM in Smart Building  
**Li, Li, Zeng, Stankovic, Stankovic, Xiao, Shi (Applied Energy, 2023 / online 2022)** — *same paper as the other “multi-objective” PDF*  
1. **Multi-appliance / one-to-many** multi-objective (state + consumption) with adaptive #outputs.  
2. **Fine-tune with target labels** (pretrain REFIT → retrain dense → fine-tune CNN on labeled target). Not unsupervised DA.  
3. REFIT / REDD / UK-DALE / private SC-EDNRR. Case iii claims **UK-DALE H3 & H4** for fine-tune/test with Fridge/DW/WM/Kettle metrics — **not honest vs official UK-DALE metadata** (H3/H4 lack those clean submeters; see repo note `li_ae2022_ukdale_h3h4_non_reproducible.md`).  
4. Novelty: one-to-many transfer vs one-to-one; adaptive output heads; building deployment case.

#### Neural Load Disaggregation: Meta-Analysis, Federated Learning and Beyond  
**Bousbiat, Himeur, Varlamis, Bensaali, Amira (Energies, 2023)** — *review*  
1. Survey across single/multi setups; not a new model.  
2. Transfer type: N/A (reviews FL, some meta/transfer mentions).  
3. Summarizes others’ UK-DALE/REFIT/REDD FL results; repeatedly notes many FL papers **test only on training buildings**.  
4. Novelty: meta-analysis of NILM reviews + federated NILM taxonomy / toolkit gap.

#### NILM Domain Adaptation: When Does It Work?  
**Muaz, Zinnikus, Shahid (ICSCC, 2024)**  
1. Single-appliance classification-style transferability study (DW, washer-dryer etc.).  
2. **Direct transfer / source-data scaling** analysis (not proposing a new UDA algorithm); discusses when fine-tuning is needed.  
3. REFIT as source; targets UK-DALE, REDD, GeLaP. Honest negative notes: **UK-DALE washer-dryer overfits early**; same-country helps; F1≥0.7 often needs ~5–6 houses; cross-region needs fine-tune.  
4. Novelty: large-scale empirical map of *when* DA/transfer works vs overfits (641 models) — rare **honest failure/overfit** framing.

#### RTNILM: Deep Robust Transfer Neural Network for Practical NILM  
**Pan, Ye, Weng, Chen, Yin (IEEE TII, 2025)**  
1. Appliance **recognition** (high-frequency / submetered waveform-style datasets), not low-freq multi-label house disaggregation.  
2. **Meta-learning (MAML) + few-shot fine-tune**; NAS + OGE2E pretrain; new-appliance detection via feature similarity.  
3. PLAID / WHITED / COOLL + self-collected — **not UK-DALE house-split NILM**.  
4. Novelty: joint NAS + robust loss + MAML for domain shift, noise, and unseen appliances.

#### Semi-Supervised Domain Adaptation for Multi-Label Classification on NILM  
**Hur, Lee, Kim, Kang (Sensors, 2022)**  
1. **Multi-label** appliance **usage (ON) classification** (not power regression).  
2. **Semi-supervised DA**: labeled source + unlabeled target; pseudo-labeling, teacher–student, gkMMD, TCN, GRL.  
3. UK-DALE **H1↔H2**, REDD H1↔H3, and UK-DALE↔REDD. Downsampled “real environment”. Reports improved F1 vs baselines (claims success).  
4. Novelty: multi-label SSDA stack for NILM with pseudo-label domain stabilization — **closest prior to multi-label + DA**, but classification-only and generally positive reporting.

#### Transfer Learning for Non-Intrusive Load Monitoring  
**D’Incecco, Squartini, Zhong (IEEE TSG, 2020)**  
1. Single-appliance Seq2Point.  
2. **Fine-tune with target labels** (ATL appliance-to-appliance; CTL cross-domain freeze CNN / train dense). Also zero-shot REFIT→UK-DALE (same country works better).  
3. REFIT / UK-DALE (mostly H1–H2) / REDD. Notes prior seq2point used UK-DALE **H1,3,4,5→H2** and REDD H2–6→H1 as *same-domain* eval. Cross-region: needs fine-tune.  
4. Novelty: ATL + CTL recipes for seq2point; empirical transferability conclusions.

#### Unsupervised Domain Adaptation for NILM via Adversarial and Joint Adaptation Network  
**Liu, Zhong, Qiu, Lu, Wang (IEEE TII, 2022)**  
1. Single-appliance disaggregation (WM/DW/MV/kettle/fridge etc.). Mentions multi-appliance co-activation as future interest, not the method.  
2. **Unsupervised DA** (adversarial + joint probability adaptation); limited source labels + unlabeled target.  
3. REDD, UK-DALE, REFIT, Pecan Street; intradomain e.g. Ui H1→… and interdomain transfers.  
4. Novelty: first-wave claim of using unlabeled target aggregates in NILM UDA; feature+label space alignment.

#### Unsupervised Lightweight Transfer Learning for Edge NILM  
**Lu, Li, Yao, Wang (submitted / IEEE TSG-style preprint in folder)**  
1. Single-appliance Seq2Seq-style disaggregation (WM/DW/MW/fridge).  
2. **Unsupervised** edge fine-tune of compressed subnets on unlabeled target (DNC + LCT).  
3. REDD & UK-DALE; Table I: UK-DALE train **H1,3,4 → test H2** (and similar REDD splits); also cross-dataset.  
4. Novelty: dynamic compression + unsupervised lightweight residual adaptation for edge memory budgets.

#### Privacy-Preserving NILM: Self-Alignment Source-Aware Domain Adaptation  
**Hao, Yan, Wen (IEEE TIM, 2025)**  
1. Single-appliance disaggregation (FR/DW/WM etc.).  
2. **Unsupervised / source-free-ish DA**: adversarial DA then fine-tune with **pseudo-labels without source data** (privacy); self-alignment (SAM).  
3. SynD, REDD, UK-DALE; intradomain + interdomain `S/R/U` transfers.  
4. Novelty: source-free fine-tuning + SAM for stable adversarial DA under privacy constraints.

#### Pre-Trained Models for Non-Intrusive Appliance Load Monitoring  
**Wang, Mao, Wilamowski, Nelms (IEEE TGCN, 2022)** — *Meta Learning folder*  
1. Single-appliance disaggregation.  
2. **Meta-learning (MAML) + ensemble**; **few-shot fine-tune** on target.  
3. Pretrain REFIT → fine-tune/test UK-DALE. Addresses negative transfer vs naive TL.  
4. Novelty: BERT/GPT-inspired pretrain+few-shot recipes for NILM transferability.

---

### 1.2 Multi-appliance folder

#### A NILM System Using Multi-Label Classification Approach  
**Buddhahai, Wongcharee, Rakkwamsuk (Sustainable Cities & Society-era)**  
1. **Multi-label** ON/OFF classification (RAkEL + DT etc.).  
2. Supervised (no transfer/DA); mentions semi-supervised as future.  
3. Private field measurement (1-min electrical features); not UK-DALE house-transfer.  
4. Novelty: end-to-end experimental ML design pipeline for multi-label NILM (features, algorithms, learning curves).

#### Attention-Based DL for Simultaneous State Detection of Multiple Appliances  
**IEEE TIM-style (2023; authors in extract: attention + dilated causal CNN)**  
1. **Multi-label** simultaneous state detection (one model, many appliances).  
2. Supervised; no transfer/DA in extracted claims.  
3. Validated in multiple scenarios (dataset names in body; UK-DALE/REDD-style public NILM setting implied by related work — treat house honesty as not strongly pinned in skim).  
4. Novelty: lightweight attention + dilated causal CNN for multi-appliance state MLC.

#### Conv-NILM-Net: Causal Multi-Appliance Energy Source Separation  
**Alami et al.**  
1. **Multi-appliance** causal fully-conv source separation (all appliances jointly).  
2. Supervised (no DA/transfer).  
3. REDD & UK-DALE; building-level experiments (e.g. building 1 mentioned).  
4. Novelty: Conv-TasNet-inspired causal multi-appliance separator; small model size vs SOTA.

#### MATNilm: Multi-Appliance-Task NILM with Limited Labeled Data  
**Xiong, Hong, Zhao, Zhang**  
1. **Multi-appliance multi-task** (regression + classification per appliance, shared hierarchy + 2D attention).  
2. **Few-label / limited labeled data** + sample augmentation — **same-domain** (not cross-house unsupervised DA). Extreme: ~1-day training.  
3. REDD & UK-DALE; e.g. UK-DALE train/val H1 windows, test H2; REDD S1 houses 2–6→1.  
4. Novelty: sample augmentation + MAT architecture for scarce labels without requiring DA.

#### Multi-Label Learning for Appliance Recognition using Fryze-Current Decomposition + CNN  
**Faustine et al. (Energies-style)**  
1. **Multi-label** appliance recognition from aggregate current (high-freq / VI-image path).  
2. Supervised; no transfer.  
3. **PLAID** (not UK-DALE low-freq house TL).  
4. Novelty: Fryze active/non-active current + EDS image + CNN multi-label recognition.

#### Multi-Target Energy Disaggregation using CNNs  
**Ayub, El-Alfy**  
1. **Multi-target regression** (joint appliances).  
2. Supervised; discusses “knowledge transfer” as generalization rhetoric; not a formal DA protocol in extract.  
3. ENERTALK & REDD.  
4. Novelty: point-to-point multi-target CNN vs single-target; simultaneous operation claims.

#### Multilabel Appliance Classification with Weakly Labeled Data  
**Tanoni, Principi, Squartini (IEEE TSG, 2023)**  
1. **Multi-label** state classification.  
2. **Weakly supervised / MIL** (segment-level weak labels); compared to supervised & semi-supervised — **not cross-domain DA**. Mixes UK-DALE+REFIT in some label-budget experiments.  
3. UK-DALE & REFIT; UK-DALE H2 held out in described split; H3/H4 short windows used in construction notes.  
4. Novelty: first (claimed) weak-label MIL multi-label appliance classification for NILM.

#### On Time Series Representations for Multi-Label NILM  
**Authors/year not extractable**  
1–4. **Extraction failed:** PDF is image-only (16 pages, no text layer; Tesseract not installed). Do not invent content. Filename indicates multi-label representation study.

#### Transfer Learning for Multiappliance-Task NILM  
**Sun, Feng, Yuan, Su, Luan (IEEE TIM, 2025)**  
1. **Multi-appliance multi-task** (disaggregation + state via SGN) with attention shared/specific layers.  
2. **Fine-tune with target labels** (shared layers from REFIT; appliance-specific layers fine-tuned on UK-DALE/REDD). Not unsupervised DA.  
3. REFIT → UK-DALE / REDD; also zero-shot vs fine-tune ablations. House IDs not as carefully stressed as Li’s H3/H4 claim in the skim.  
4. Novelty: **simultaneous transfer of two objectives** in multi-appliance SGN; attention for shared features; fine-tune only specific layers.

#### UNet-NILM: Multi-Task State Detection and Power Estimation  
**Faustine, Pereira, Bousbiat, Kulkarni (NILM Workshop)**  
1. **Multi-appliance multi-task** (multi-label states + multi-target quantile power).  
2. Supervised; inductive multi-task transfer between tasks, **not cross-house DA**.  
3. UK-DALE (resampled 6 s); artificial aggregates mentioned in preprocessing.  
4. Novelty: 1D U-Net for joint multi-appliance state+power with quantile regression.

#### Variational Regression for Multi-Target Energy Disaggregation  
**Virtsionis Gkalinikis, Nalmpantis, Vrakas (Sensors, 2023)**  
1. **Multi-target / multi-appliance** regression (shared variational encoder + heads).  
2. Supervised; no DA.  
3. Compared vs UNet-NILM variant / single-target baselines (datasets in body; UK-DALE/REDD-style public NILM).  
4. Novelty: variational multi-target regressor with KL regularization for compact multi-appliance models.

---

## 2. Synthesis

### A) Combinations already well covered

| Combination | Evidence density | Typical pattern |
|---|---|---|
| **Single-appliance + unsupervised DA** | High | Liu TII’22, Lin TSG’22, Xiong’26, Lu edge, Hao TIM’25 — adversarial / MMD / CORAL / source-free variants |
| **Single-appliance + fine-tune CTL/ATL** | High | D’Incecco TSG’20, Rong EnBuild’25, Li AE’23 (one-to-many still needs target labels) |
| **Single-appliance + meta / few-shot** | Medium | Wang TGCN’22 (MAML), RTNILM’25 (high-freq recognition), Muaz cites ensemble/meta needs |
| **Multi-appliance / multi-label / multi-task, supervised, same domain** | High | UNet-NILM, MATNilm, Conv-NILM-Net, Attention MLC, Buddhahai, Fryze-CNN, Variational multi-target, Tanoni weak labels |
| **Multi-appliance + supervised fine-tune transfer** | Emerging but present | **Sun TIM’25** (multi-task SGN fine-tune); **Li AE’23** (one-to-many fine-tune) |
| **Multi-label + semi-supervised DA (classification)** | Thin but exists | **Hur Sensors’22** (H1↔H2, REDD; pseudo-label SSDA) |
| **Empirical “when does transfer work?”** | Rare but important | **Muaz ICSCC’24** (country/continent, #houses, overfit warnings) |

**Takeaway:** The literature is dense on (i) single-appliance UDA claiming large MAE gains, and (ii) multi-appliance models **without** unsupervised DA. The intersection is thin.

### B) Rare or missing combinations (critical)

| Gap | Status in this corpus | Why it matters for the student |
|---|---|---|
| **Multi-appliance + multi-label + unsupervised DA** | Essentially **missing** as a joint power+state system. Hur is multi-label SSDA but **ON classification only**. Sun/Li are multi-appliance but **need target labels**. | Matches PhD plan; open method space. |
| **Unsupervised DA for multi-appliance regression** | **Missing / unclaimed** in extracted papers. Liu explicitly flags multi-appliance co-activation as interesting future work. | Aligns with MultiNILM finding that UDA often fails H2 — literature over-reports single-appliance UDA wins. |
| **Honest negative / failure results for UDA** | **Rare.** Muaz warns overfit / region limits; most UDA papers report large % improvements. Almost none say “H2 remains unusable under UDA.” | High credibility / contribution if framed carefully (conditions of failure). |
| **Few-label target + multi-appliance** under **cross-house DA** | MATNilm = few-label **same protocol**, not UDA. Meta papers = few-shot **single-appliance**. | Practical deployment story (1 day of plugs, then unsupervised). |
| **Event-sparse appliances under DA** | Under-treated as a *protocol*. Kettle often easy; DW/WM harder; Muaz notes washer-dryer signature uniqueness causes early overfit. Sparse events + multi-label imbalance + DA not systematically studied. | Explains brittle H2 / minority-class collapse. |
| **UK-DALE H1+H5 → H2 (hard target) multi-appliance honesty** | Seq2point-era **H1,3,4,5→H2** appears as *same-domain supervised* baseline (D’Incecco citing Kelly). Lu edge uses **H1,3,4→H2** single-appliance unsupervised. **No extracted paper** runs a careful multi-label multi-appliance UDA study with H2 as hard target and reports failure modes. Li’s **H3/H4** fine-tune case is a **reproducibility red flag**, not a gold split. | Student can own an honest hard-split benchmark. |
| **Multi-label + source-free / privacy DA** | Hao is source-free but single-appliance. | Privacy × multi-label open. |
| **Negative-transfer diagnostics for multi-task heads under DA** | Wang meta discusses negative transfer for single-appliance TL; not for shared multi-appliance encoders + UDA. | MultiNILM-relevant: DA may help one head, hurt another. |

### C) Ranked novelty directions (top 8) for this PhD plan

Given: **multi-appliance + multi-label + transfer**, and evidence that **unsupervised DA often fails to make H2 usable**.

1. **Honest hard-target benchmark for multi-appliance UDA (UK-DALE H1+H5→H2, plus REFIT↔UK-DALE)**  
   Publish *when* UDA helps vs harms vs fine-tune vs source-only, per appliance and per head (state vs power). Fill the Muaz-style honesty gap that single-appliance UDA papers omit. Highest differentiation vs Liu/Lin/Xiong/Lu.

2. **Multi-label / multi-task unsupervised (or source-free) DA**  
   Extend beyond Hur (class-only SSDA) and beyond Sun/Li (label-hungry fine-tune): shared encoder + multi-head alignment that respects label imbalance and co-activation. Directly occupies the empty cell in the matrix.

3. **Failure-aware / selective DA instead of always-on alignment**  
   Detect domain shift severity (or event-sparsity) and gate DA / fall back to source prior or few-label calibration. Motivated by MultiNILM H2 failures and Muaz overfit findings — “DA that knows when not to adapt.”

4. **Few-label target calibration for multi-appliance models (not full fine-tune)**  
   Bridge MATNilm (few labels, little DA) and Sun (full fine-tune): e.g. 1-day strong labels or weak segment labels (Tanoni-style) + unlabeled target for multi-head adaptation. Strong practical story.

5. **Event-sparse / long-tail appliance protocol under transfer**  
   Explicit evaluation design: DW/WM/kettle rarity, F1 vs MAE disagreement, co-occurrence with always-on fridge. Most papers average metrics and bury sparse failures that kill H2 usability.

6. **Negative-transfer audit for shared multi-appliance encoders**  
   Quantify when aligning domains for Appliance A destroys Appliance B; propose appliance-conditional or task-conditional adaptation. Novel relative to single-appliance UDA literature.

7. **Reproducibility + split hygiene contribution**  
   Document non-reproducible H3/H4-style claims (Li Case iii) and propose a locked UK-DALE multi-appliance transfer checklist (houses, channels, appliances present, train/val/test windows). Meta-contribution that reviewers increasingly value; supports Direction 1.

8. **Privacy-preserving / source-free multi-appliance DA**  
   Lift Hao’s source-free idea to multi-label multi-task NILM for edge deployment (near Lu’s edge theme but joint appliances). Slightly more incremental unless tied to Directions 1–3.

**Deprioritize (already crowded unless a sharp twist):** another single-appliance adversarial UDA with bigger claimed MAE%; another same-domain multi-appliance CNN without transfer; high-frequency few-shot recognition (RTNILM space) unless linked to low-freq multi-label house transfer.

---

## 3. Coverage matrix (compact)

|  | No transfer | Fine-tune / few-label target | Unsupervised / semi DA | Meta / few-shot |
|---|---|---|---|---|
| **Single-appliance** | Classic seq2point etc. | D’Incecco, Rong, Wang-meta | Liu, Lin, Xiong, Lu, Hao | Wang-meta, RTNILM* |
| **Multi-appliance / multi-task** | UNet, MATNilm, Conv-NILM, Variational | **Sun**, **Li** (labels required) | **Gap** | **Gap** |
| **Multi-label classification** | Buddhahai, Attention, Fryze, Tanoni(weak) | Weak/few labels: Tanoni, MAT-ish | **Hur (SSDA only)** | **Gap** |

\*RTNILM = high-freq recognition, not low-freq UK-DALE multi-appliance.

---

## 4. Caveats

- Paper *On time series representations for multi-label NILM* could not be read (image PDF).  
- Some multi-appliance papers’ exact house IDs were only partially recovered in skims; where uncertain, this report does not invent splits.  
- “Claimed novelty” is authors’ framing from abstracts/contributions, not an endorsement of correctness.  
- Li et al. UK-DALE H3/H4 Case iii should be treated as **non-reproducible as written**, not as prior art validating that split.

---

## 5. Bottom line

The folder shows a mature **single-appliance UDA** literature and a mature **multi-appliance supervised** literature, with **supervised fine-tune multi-appliance transfer** just emerging (Sun, Li). The PhD-relevant empty region is **multi-appliance / multi-label systems under unsupervised or few-label DA on honest hard targets (esp. UK-DALE H2), including negative results**. That is the sharpest, least crowded, and most consistent direction with the student’s MultiNILM evidence.

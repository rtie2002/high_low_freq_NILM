# Papers used for MATUDA design (local copies)

## Classic domain adaptation (any field)

| File | Role in our design |
|------|--------------------|
| `Deep_CORAL_Sun_Saenko_ECCV2016.pdf` | CORAL on **FC** activations (arxiv 1607.01719) |
| `DAN_Long_ICML2015_MMD.pdf` | Multi-layer MMD on adaptation (FC) layers (arxiv 1502.02791) |
| `DANN_Ganin_JMLR2016.pdf` | Adversarial DA (optional future branch) |
| `CDAN_Long_NeurIPS2018.pdf` | Conditional + entropy-aware DA → **EGC-DA** novelty |

## NILM transfer / UDA

| File | Role |
|------|------|
| `Lin_TSG2022_Deep_Domain_Adaptation_Knowledge_Transfer_NILM.pdf` | TCN + **domain loss on fc6–fc8**; unsupervised target |
| `Unsupervised_Domain_Adaptation_...Adversarial_and_Joint...pdf` (Liu TII) | Feature + joint adaptation |
| `Lu_TSG_Unsupervised_Lightweight_Transfer_Edge_NILM.pdf` | Sinkhorn + CORAL; edge residual (optional) |
| `Adversarial and Hierarchical Distribution Alignment...pdf` | Hierarchical CORAL + MK-MMD |
| `Transfer_Learning_for_Non-Intrusive_Load_Monitoring.pdf` | Freeze CNN, tune **FC** |
| `NILM_Domain_Adaptation_When_Does_It_Work.pdf` | Honest failure / when DA works |
| `Privacy-Preserving_NILM_...pdf` | Source-free / self-alignment ideas |
| `Li_AE2022_Transfer_Learning_MultiObjective_NILM.pdf` | Multi-objective TL (**labeled** target; H3/H4 caveat) |

**Design takeaway we follow:** compute MMD/CORAL on **fully connected embeddings**, multi-layer, with supervised multi-task loss only on source. **Not** on raw TCN maps with ad-hoc mean-pool as the sole DA feature.

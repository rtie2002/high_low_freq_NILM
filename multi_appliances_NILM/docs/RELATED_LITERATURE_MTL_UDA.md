# Related literature for MATUDA (multi-appliance + UDA)

Reading notes for later. Papers are **not all NILM**; they address the same *structure* as our project.

## 0. What our project is (context for every paper below)

| Piece | Our setup |
|-------|-----------|
| Input | Aggregate power window (seq2seq, T=480) |
| Outputs | K=5 appliances × (ON/OFF state + power) |
| Model | Shared stem/TCN (+ optional FC) → K heads |
| Transfer | H1+H5 labeled → H2 unlabeled aggregates → H2 test |
| DA recipe | Lin-style \(L=(1-\lambda)L_{\mathrm{NILM}}+\lambda L_{\mathrm{domain}}\), MMD+CORAL, optional EGC |
| Hard parts | (1) temporal **shape** over many timesteps; (2) heads not independent; (3) global DA can hurt sparse appliances |

```text
Literature map
  Lin / CORAL / MMD     →  make source/target features similar
  Cross-task / MTL      →  heads help each other
  Multi-label graphs    →  ON/OFF co-occurrence / exclusion
  Mixture consistency   →  Σ power̂ ≈ aggregate
  Multi-task UDA        →  do MTL + domain shift together  ← closest combo
```

---

## 1. Domain similarity (CORAL / MMD / Lin)

How models learn “source ≈ target” in feature space.

| Topic | Paper | DOI / link |
|-------|--------|------------|
| Deep CORAL | Sun & Saenko, ECCV Workshops 2016 | https://doi.org/10.1007/978-3-319-49409-8_35 |
| DAN / MMD | Long et al., ICML 2015 | https://doi.org/10.48550/arXiv.1502.02791 |
| DANN (adversarial) | Ganin et al. | https://doi.org/10.5555/2946645.2946703 |
| CDAN (conditional) | Long et al. | https://doi.org/10.48550/arXiv.1705.10667 |
| **Lin NILM UDA** (single-appliance) | Lin et al., TSG 2022 | https://doi.org/10.1109/TSG.2021.3115910 |

**Relation to us:** We already implement Lin-style hybrid MMD+CORAL on pooled FC maps. Lin is **single-appliance**; we are multi-head. Pooling throws away temporal shape — a known limit for our case.

**Implementation note (our code):** CORAL/MMD formulas in `MATUDA_loss.py` are correct. Bug fixed: EGC/conditional weights must be applied **after** L2-norm (√w then no re-normalize). `da_mode: global` never used those weights.

---

## 2. Multi-appliance / multi-label NILM + transfer (field-specific)

| Topic | Paper | DOI |
|-------|--------|-----|
| Semi-supervised multi-label DA + PL | Hur et al., Sensors 2022 | https://doi.org/10.3390/s22155838 |
| Hierarchical CORAL + MK-MMD (AHDA) | Electronics 2026 | https://doi.org/10.3390/electronics15030655 |
| Multi-objective transfer NILM (target labels) | Li et al., Applied Energy 2023 | https://doi.org/10.1016/j.apenergy.2022.120223 |
| Multiappliance-task transfer (TIM) | Sun et al., TIM 2025 | https://doi.org/10.1109/TIM.2025.3541652 |
| MATNilm (multi-app structure, not UDA) | Xiong et al., TII 2024 | https://doi.org/10.1109/TII.2023.3301026 |
| Transfer learning for NILM (freeze CNN, tune FC) | D’Incecco et al., TSG 2020 | https://doi.org/10.1109/TSG.2019.2938068 |
| When DA helps / hurts | Muaz et al., ICSCC 2024 | conference — check IEEE Xplore |

**Relation to us:** Hur / AHDA are the best NILM-side guides for **multi-label + DA**. Li/Sun often use **target labeled fine-tuning**, which is easier than our fully unlabeled H2 protocol.

---

## 3. Cross-task / multi-head (heads help each other) — not NILM

| Topic | Paper | Venue | DOI / link |
|-------|--------|-------|------------|
| Cross-task affinity distillation | **CTAL** | WACV 2025 | https://arxiv.org/abs/2401.11124 · [CVF](https://openaccess.thecvf.com/content/WACV2025/html/Sinodinos_Cross-Task_Affinity_Learning_for_Multitask_Dense_Scene_Predictions_WACV_2025_paper.html) |
| Inter-task attention + noise filter | **KEM** | ACCV 2024 | [PDF](https://openaccess.thecvf.com/content/ACCV2024/papers/Zhang_KEM_SGW-based_Multi-Task_Learning_in_Vision_Tasks_ACCV_2024_paper.pdf) |
| Efficient inter-task attention | Deformable Inter-Task Self-Attention | arXiv 2025 | https://doi.org/10.48550/arXiv.2508.04422 |

**Relation to us:** Fridge / kettle / dishwasher heads are like dense MTL tasks. Independent heads miss this; CTAL/KEM = explicit cross-head message.

---

## 4. Multi-label dependency (co-occur / exclusive / causal)

| Topic | Paper | Venue | DOI / link |
|-------|--------|-------|------------|
| Causal vs spurious label correlations | Causal Label Correlations | NeurIPS 2024 | [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/5c54e016197805946481d786d80a662e-Paper-Conference.pdf) |
| Correlative + discriminative grouping | ML-VPT | CVPR 2025 | [HTML](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_Correlative_and_Discriminative_Label_Grouping_for_Multi-Label_Visual_Prompt_Tuning_CVPR_2025_paper.html) |
| Scene-aware label graph | SALGL | ICCV 2023 | https://doi.org/10.1109/ICCV51070.2023.00142 |

**Relation to us:** Appliance ON states are multi-label. Co-occurrence can help or create house-specific spurious links that break H2 transfer (NeurIPS causal paper).

---

## 5. Multi-source from one mixture (audio analogy)

| Topic | Paper | DOI |
|-------|--------|-----|
| Flow matching + strict mixture consistency | FLOSS | https://doi.org/10.48550/arXiv.2505.16119 |
| Weak mixture-to-mixture supervision | M2M | https://doi.org/10.1109/LSP.2024.3417284 |

**Relation to us:** Aggregate ≈ mixture; appliances ≈ sources. Optional loss: \(\sum_k \hat{P}_k \approx P_{\mathrm{agg}}\) (plus other loads / residual).

---

## 6. Multi-task + domain adaptation (MTL ∩ UDA)

This is the combo closest to “multi-appliance UDA.”

| Topic | Paper | Venue | DOI |
|-------|--------|-------|-----|
| **Unsupervised multi-task DA + cross-task distillation** | **UM-Adapt** | ICCV 2019 | https://doi.org/10.1109/ICCV.2019.00152 |
| UDA as multi-objective + gradient alignment | PGA / MPGA | NeurIPS 2024 | https://doi.org/10.48550/arXiv.2406.09353 |
| Multi-source UDA via prompt alignment | MPA | NeurIPS 2023 | https://doi.org/10.48550/arXiv.2209.15210 |

**Relation to us:**

- **UM-Adapt** — several prediction heads + unlabeled target + cross-task coherency for DA. Read first.
- **PGA** — balance conflicting objectives (NILM vs DA vs heads) without one killing the others.
- **MPA** — multiple sources (H1+H5) → one target (H2).

---

## 7. Surveys (background)

| Topic | Paper | DOI |
|-------|--------|-----|
| Deep MTL survey | Crawshaw 2020 | https://doi.org/10.48550/arXiv.2009.09796 |
| Large MTL survey | 2024 | https://doi.org/10.48550/arXiv.2404.18961 |
| Deep MTL + cross-domain applications | 2025 | https://doi.org/10.1007/s41060-025-00892-y |

---

## 8. Suggested reading order (for this project)

1. **Lin** TSG 2022 — our DA baseline formula  
2. **Hur** Sensors 2022 — multi-label NILM DA  
3. **UM-Adapt** ICCV 2019 — multi-task UDA (closest non-NILM twin)  
4. **CTAL** WACV 2025 — cross-head refinement  
5. **NeurIPS 2024 causal labels** — when co-occurrence hurts transfer  
6. **FLOSS / M2M** — mixture consistency idea  
7. **PGA** — balancing multi-objective DA losses  

---

## 9. Design implications we already discussed

| Idea | Status / direction |
|------|-------------------|
| Match MultiNILM backbone dims + compact FC `128→256→192→128` | Done in `matuda.yaml` / `MATUDA.py` |
| Follow Lin: fixed λ=0.6, **no** mid-train DA freeze | `da_freeze_patience: 0` |
| Fix EGC weight + L2 order | Done in `MATUDA_loss.py` |
| FC as DA-only side branch (`TCN→heads`; `pool→FC→DA`) | Not done yet (optional next) |
| Temporal DA / keep shape (not only full-window mean) | Open research direction |
| Cross-head module / mixture consistency | Inspired by §3–§5; not implemented |

---

*Collected from project chat notes. Revisit DOIs on publisher pages if citing formally.*

# Multi-Appliance NILM: Literature Review & Taxonomy

> **Source Paper**:
> Yao Sun, Jianwei Feng, Liang Yuan, Mei Su, and Wenpeng Luan, *"Transfer Learning for Multiappliance-Task Nonintrusive Load Monitoring,"* **IEEE Transactions on Instrumentation and Measurement (TIM)**, vol. 74, Art no. 2507112, 2025.
> [Paper PDF](file:///C:/Users/Raymond%20Tie/Desktop/PhD/Literature%20Review/NILM/multi-appliances/Transfer_Learning_for_Multiappliance-Task_Nonintrusive_Load_Monitoring.pdf)

---

## 1. Multi-Task Architectures: Soft Sharing vs. Hard Sharing

In multi-appliance NILM, two main architectural structures are utilized for multitask modeling:

```
                          ┌───────────────────────────┐
                          │   Aggregate Input X(t)    │
                          └─────────────┬─────────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
┌─────────────────────────┐                           ┌─────────────────────────┐
│      Soft Sharing       │                           │      Hard Sharing       │
│    (Feature Blending)   │                           │ (Branch Isolation / MTL)│
├─────────────────────────┤                           ├─────────────────────────┤
│ • Features from layers  │                           │ • Shared encoder/trunk  │
│   concatenated/blended  │                           │ • Dedicated appliance   │
│ • Cross-appliance       │                           │   heads without leakage │
│   interaction allowed   │                           │ • Eliminates interference│
│ • Example: SAMNet [7]   │                           │ • Example: MATNilm [8]  │
└─────────────────────────┘                           └─────────────────────────┘
```

### 1.1 Soft-Sharing Architecture (Feature Blending)
* **Mechanism**: After a certain number of layer computations, all inputs/features are concatenated into a sequence. The concatenated result undergoes appropriate blending of their respective features before separation.
* **Core Property**: Shares features among all target appliances, enabling the model to exploit cross-appliance dependencies and concurrent activation patterns.
* **Representative Work**: 
  * **SAMNet** ([7]): Y. Liu, J. Qiu, and J. Ma, *"SAMNet: Toward Latency-Free Non-Intrusive Load Monitoring via Multi-Task Deep Learning,"* *IEEE Transactions on Smart Grid*, vol. 13, no. 3, pp. 2412–2424, May 2022.

### 1.2 Hard-Sharing Architecture (Branch Isolation)
* **Mechanism**: Each appliance operates within an independent, isolated network structure with no data sharing among appliances after the initial representation.
* **Core Property**: Prevents negative transfer or feature interference between appliances with conflicting power profiles.
* **Representative Work**:
  * **MATNilm** ([8]): J. Xiong, T. Hong, D. Zhao, and Y. Zhang, *"MATNilm: Multi-Appliance-Task Non-Intrusive Load Monitoring with Limited Labeled Data,"* *IEEE Transactions on Industrial Informatics*, vol. 20, no. 3, pp. 3177–3187, Mar. 2024.

---

## 2. Disaggregation Approaches: Seq2Point vs. Seq2Seq

In neural network-based NILM, there are two primary learning approaches to disaggregation:

```
(a) Seq2Point: Input Window [t - W/2, t + W/2]  ──►  Predicts single midpoint y(t)
    Requires L evaluations to disaggregate a sequence of length L.

(b) Seq2Seq:   Input Window [t, t + L]          ──►  Predicts entire sequence [y(t), ..., y(t + L)]
    Requires only 1 evaluation to disaggregate length L (significantly reduces inference time).
```

### 2.1 Comparison:
* **Sequence-to-Point (Seq2Point)** ([9], [10]):
  - Maps an aggregate input sequence to the power value of a single midpoint timestamp.
  - **Computational Cost**: Requires $L$ forward passes (evaluations) to disaggregate an entire sequence of length $L$.
  - Representative: Zhang et al. (AAAI 2018) [9], Yang et al. (IEEE TIM 2021) [10].
* **Sequence-to-Sequence (Seq2Seq)** ([8], [11], [12], [13]):
  - Maps an aggregate sequence directly to an entire output appliance sequence of equal or defined length.
  - **Computational Cost**: Needs only a single computation step for the entire window, significantly reducing computational overhead during inference.
  - Representative: Xiong et al. (IEEE TII 2024) [8], Yue et al. (BERT4NILM 2020) [11], Liu et al. (IEEE TNSE 2022) [12].

### 2.2 Attention Mechanisms in Seq2Seq NILM:
* **Feature Extraction & Parameter Efficiency**:
  - Incorporating the **attention mechanism** allows models to achieve high disaggregation accuracy while maintaining a relatively small parameter footprint [12].
  - As noted by Xiong et al. [8] and Yue et al. [11], attention-based models demonstrate performance comparable to, or exceeding, other mainstream deep learning models.
* **Dynamic Focus**:
  - Attention enhances disaggregation by dynamically focusing on the most relevant parts of the input aggregate sequence (such as step changes and operational transience), effectively capturing complex temporal patterns and inter-sample dependencies across long horizons.

---

## 3. Design of Multi-Appliance Output Heads: Subtask-Gated Networks (SGN)

While the multi-task backbone defines how features are shared **across different appliances** (macro level: soft vs. hard sharing), the **output head design** determines how predictions are generated for **each individual appliance** (micro level).

```
                 Appliance-Specific Feature Stream (Hidden Representation)
                                           │
                            ┌──────────────┴──────────────┐
                            ▼                             ▼
                 [Classification Subnet]         [Regression Subnet]
                    Fully Connected Layer          Convolution / FC
                     Sigmoid Activation          Linear / ReLU Activation
                            │                             │
                            ▼                             ▼
                  ON/OFF State Probability:         Uncalibrated Power:
                     ŝ(t) ∈ [0, 1]                   ŷ(t) (Watts)
                            │                             │
                            └──────────────┬──────────────┘
                                           ▼ Element-wise Gating (Product)
                                  Final Disaggregated Power:
                                     P̂(t) = ŝ(t) ⊙ ŷ(t)
```

### 3.1 Dual-Subtask Formulation (Shin et al., AAAI 2019 [26])
Unlike traditional architectures that treat energy disaggregation purely as a single continuous regression task, the **Subtask-Gated Network (SGN)** decomposes each appliance's prediction into two cooperative subtasks:
1. **Classification Subnetwork (Appliance State Detection)**:
   - Employs a fully connected layer with a **Sigmoid** activation function to output a probabilistic sequence $\hat{s}(t) \in [0, 1]$ indicating whether the appliance is ON or OFF at each timestep.
2. **Regression Subnetwork (Energy Disaggregation)**:
   - Estimates the raw continuous power waveform $\hat{y}(t)$ corresponding to the load's active draw.
3. **Subtask Gating Mechanism**:
   - The final output is formed by taking the Hadamard (element-wise) product of the regression output and the state classification probability:
     $$\hat{P}(t) = \hat{s}(t) \odot \hat{y}(t)$$

### 3.2 Key Advantages in Multi-Appliance & Transfer Learning:
* **Background Leakage Suppression**:
  In unconstrained regression heads, deep models often predict small residual baseline wattages ($5\text{--}30\text{ W}$) during standby or inactive periods. The gating multiplication acts as a soft switch: when the classifier predicts an OFF state ($\hat{s}(t) \to 0$), background artifacts are suppressed to near-zero.
* **Enhanced Transferability**:
  As highlighted by Sun et al. (IEEE TIM 2025), absolute operating power varies substantially across different makes and models of the same appliance type across households, whereas the underlying operational ON/OFF state transitions remain invariant. Decoupling state detection into an explicit subtask enhances generalization and stability when transferring learned representations across unseen homes.

---

## 4. Key References Cited in this Paper

* **[7] SAMNet**: Y. Liu, J. Qiu, and J. Ma, *"SAMNet: Toward Latency-Free Non-Intrusive Load Monitoring via Multi-Task Deep Learning,"* *IEEE Trans. Smart Grid*, vol. 13, no. 3, pp. 2412–2424, May 2022.
* **[8] MATNilm**: J. Xiong, T. Hong, D. Zhao, and Y. Zhang, *"MATNilm: Multi-Appliance-Task Non-Intrusive Load Monitoring with Limited Labeled Data,"* *IEEE Trans. Ind. Informat.*, vol. 20, no. 3, pp. 3177–3187, Mar. 2024.
* **[9] Seq2Point**: C. Zhang, M. Zhong, Z. Wang, N. Goddard, and C. Sutton, *"Sequence-to-Point Learning with Neural Networks for Non-Intrusive Load Monitoring,"* in *Proc. AAAI Conf. Artif. Intell.*, 2018, pp. 2604–2611.
* **[10] TCN Seq2Point**: W. Yang, C. Pang, J. Huang, and X. Zeng, *"Sequence-to-Point Learning Based on Temporal Convolutional Networks for Nonintrusive Load Monitoring,"* *IEEE Trans. Instrum. Meas.*, vol. 70, pp. 1–10, 2021.
* **[11] BERT4NILM**: Z. Yue, C. R. Witzig, D. Jorde, and H.-A. Jacobsen, *"BERT4NILM: A Bidirectional Transformer Model for Non-Intrusive Load Monitoring,"* in *Proc. 5th Int. Workshop Non-Intrusive Load Monitor.*, 2020, pp. 89–93.
* **[12] Single-to-Multi Network**: Y. Liu, J. Qiu, J. Lu, W. Wang, and J. Ma, *"A Single-to-Multi Network for Latency-Free Non-Intrusive Load Monitoring,"* *IEEE Trans. Netw. Sci. Eng.*, vol. 9, no. 2, pp. 755–768, Mar. 2022.
* **[13] NILM Review**: P. A. Schirmer and I. Mporas, *"Non-Intrusive Load Monitoring: A Review,"* *IEEE Trans. Smart Grid*, vol. 14, no. 1, pp. 769–784, Jan. 2023.
* **[26] SGN**: C. Shin, S. Joo, J. Yim, H. Lee, T. Moon, and W. Rhee, *"Subtask Gated Networks for Non-Intrusive Load Monitoring,"* in *Proc. AAAI Conf. Artif. Intell.*, 2019, vol. 33, no. 1, pp. 1150–1157.


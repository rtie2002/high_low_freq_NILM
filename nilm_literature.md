# NILM Methodological Categorization: Deep Learning vs. Multi-Label Classification

> **Source Paper**:
> Suryalok Dash and Nirod Chandra Sahoo, *"Attention-Based Deep Learning Approach for Nonintrusive and Simultaneous State Detection of Multiple Appliances in Smart Buildings,"* **IEEE Journal of Emerging and Selected Topics in Industrial Electronics (JESTIE)**, vol. 5, no. 3, pp. 1248–1258, July 2024.  
> [Paper PDF](file:///C:/Users/Raymond%20Tie/Desktop/PhD/Literature%20Review/NILM/multi-appliances/Attention-Based_Deep_Learning_Approach_for_Nonintrusive_and_Simultaneous_State_Detection_of_Multiple_Appliances_in_Smart_Buildings.pdf)

---

## 1. Original Literature Excerpt

> *"A few state-of-the-art DL models that adopt the above approach include the convolutional neural network (CNN) [9]–[13], the long short term memory (LSTM) network [14]–[16], Autoencoder [14], [15], the temporal convolutional network (TCN) [17]–[20], and the generative adversarial network [21], [22].*
>
> *Another group of approaches formulates the NILM problem as a multilabel classification (MLC) problem to identify the states of various appliances using off-the-shelf MLC tools and hand-engineered features [6], [15], [23]–[25]. For example, the authors in [6] used wavelet features, and the authors in [23] used statistical features extracted from the aggregate energy signal along with a support vector machine (SVM) or decision tree as a classifier to identify the appliance states. Similarly, the authors in [24] suggested graph-based semisupervised learning, and the authors in [25] proposed deep dictionary learning for the NILM task."*

---

## 2. Comprehensive Methodological Breakdown

The authors categorize existing NILM literature into two primary methodological tracks:

```
                                  ┌──────────────────────────────┐
                                  │   Existing NILM Literature   │
                                  └──────────────┬───────────────┘
                                                 │
                  ┌──────────────────────────────┴──────────────────────────────┐
                  ▼                                                             ▼
  ┌────────────────────────────────┐                           ┌────────────────────────────────┐
  │  Track 1: End-to-End DL Models │                           │ Track 2: Multi-Label (MLC) +   │
  │     (Feature Representation)   │                           │     Hand-Engineered Features   │
  ├────────────────────────────────┤                           ├────────────────────────────────┤
  │ • Automated feature extraction │                           │ • Focuses on on/off state      │
  │ • CNN, LSTM, DAE, TCN, GAN     │                           │   detection (multi-label)      │
  │ • Power regression & waveforms │                           │ • Wavelet, statistical metrics │
  │ • Mostly single-appliance focus│                           │ • SVM, Decision Tree, Graph    │
  └────────────────────────────────┘                           └────────────────────────────────┘
```

---

### Track 1: End-to-End Deep Learning (DL) Models

This paradigm bypasses manual feature engineering by learning hierarchical representations directly from raw aggregate power sequences:

1. **Convolutional Neural Networks (CNN)** ([9]–[13]):
   - Captures local temporal dependencies and appliance-specific transient edges/switching plateaus.
   - Foundation for popular sequence-to-point (Seq2Point) and sequence-to-sequence (Seq2Seq) disaggregation schemes.

2. **Long Short-Term Memory (LSTM) Networks** ([14]–[16]):
   - Utilizes recurrent gating mechanisms to maintain memory over long time horizons.
   - Well-suited for capturing continuous multi-state operations (e.g., washing machine cycles).

3. **Autoencoders / Denoising Autoencoders (DAE)** ([14], [15]):
   - Treats the aggregate signal as a corrupted/noisy version of the target appliance power signal.
   - Learns latent manifold projections to reconstruct clean appliance waveforms.

4. **Temporal Convolutional Networks (TCN)** ([17]–[20]):
   - Employs causal and dilated 1-D convolutions to achieve exponentially expanding receptive fields without recursive training bottlenecks.

5. **Generative Adversarial Networks (GAN)** ([21], [22]):
   - Pits a generator against a discriminator to reconstruct sharp, high-frequency power contours and avoid over-smoothed power predictions.

* **Primary Limitation Noted in Literature**: Most conventional DL architectures are deployed as single-appliance models, requiring dedicated networks for each load, resulting in linear parameter growth and an inability to model concurrent appliance co-occurrences.

---

### Track 2: Multi-Label Classification (MLC) with Hand-Engineered Features

Instead of disaggregating continuous wattage values, this paradigm focuses on **state detection**, framing NILM as a simultaneous multi-label classification problem:
$$\mathbf{z}_t = [z_{t,1}, z_{t,2}, \dots, z_{t,M}]^\top \in \{0, 1\}^M$$
where multiple appliances can be concurrently active ($z_{t,i}=1$) at timestamp $t$.

1. **Wavelet Transform Features** ([6]):
   - Decomposes the aggregate signal into multiple frequency sub-bands to capture localized transient energy bursts and high-frequency harmonics during appliance state transitions.

2. **Statistical Features + Traditional Classifiers** ([23]):
   - Extracts handcrafted statistical metrics from aggregate power windows (e.g., mean, variance, root-mean-square, crest factor, skewness, and kurtosis).
   - Feeds these summary vectors into standard shallow classifiers, such as **Support Vector Machines (SVM)** or **Decision Trees**.

3. **Graph-Based Semi-Supervised Learning** ([24]):
   - Represents aggregate samples and switching events as nodes on a graph.
   - Propagates multi-appliance labels from limited labeled nodes across the graph topology based on feature affinity.

4. **Deep Dictionary Learning** ([25]):
   - Learns sparse dictionary bases corresponding to individual appliance operating states to linearly reconstruct the composite aggregate signal.

* **Primary Limitation Noted in Literature**: Hand-crafted features depend heavily on domain expertise, degrade substantially when deployed in unseen homes with high background noise, and lack the adaptive capacity of end-to-end representation learning.

---

## 3. Context & Motivation for Modern NILM (Why Dash & Sahoo Proposed Their Model)

By contrasting these two tracks, the authors established the core motivation for their work:
* **MLC formulations** correctly recognize that household monitoring is inherently a multi-appliance problem where multiple devices toggle simultaneously. However, relying on **hand-engineered features and shallow classifiers** makes them fragile in real-world scenarios.
* **Deep learning models** excel at learning robust, high-capacity feature representations automatically, but have historically been restricted to **single-appliance models**.
* **The Proposed Synthesis**: Combines the simultaneous multi-state detection capability of MLC with modern deep learning (specifically attention mechanisms and causal dilated convolutions), eliminating manual feature engineering while enabling simultaneous multi-appliance monitoring.

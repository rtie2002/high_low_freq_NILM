# 中文版规划：高频特征选择驱动的多域 NILM

本节是前面英文规划的中文技术版，方便后续和导师讨论、写中文笔记、或整理研究思路。整体目标是把 **feature selection** 做成项目的正式研究阶段，而不是简单地把所有高频特征直接丢进模型。

## 1. 研究总目标

本项目希望建立一个结合低频和高频信息的 NILM 框架：

```text
低频 aggregate power
        +
高频 voltage-current signatures
        ↓
多任务模型
        ↓
appliance power disaggregation + ON/OFF state classification
```

但是当前高频特征数量较多，而且不同特征之间可能高度重复。因此，正式建复杂模型之前，应该先做：

```text
高频特征清洗 -> 冗余过滤 -> mRMR 排名 -> Random Forest 验证 -> 稳定性筛选 -> 消融实验
```

最终目标不是证明“特征越多越好”，而是证明：

```text
经过选择的高频特征比盲目使用全部高频特征更稳定、更有效、更容易解释。
```

---

## 2. 为什么 feature selection 是第一步

当前高频特征包括：

* 时间域特征：`V_rms`, `I_rms`, `P_active`, `S_apparent`, `PF`, `Fcv`, `Fci`
* 波形形状统计：`I_skew`, `I_kurt`, `V_skew`, `I_std`, `V_std`
* 谐波特征：`I1`, `I3`, `I5`, ..., `I15`
* 失真特征：`IH`, `VH`, `THDI`, `THDV`
* 频带能量：`I_BP_low`, `I_BP_mid`, `I_BP_high`, `V_BP_low`
* 频谱包络：`I_env_0` 到 `I_env_7`
* 小波时频特征：`DWT_E0` 到 `DWT_E4`

问题是，有些特征可能表达类似信息。例如：

```text
I_rms / I_std / I1 / I_BP_low
```

它们都可能和电流强度或低频电流能量有关。如果全部保留，会导致：

* 模型复杂度增加
* 训练时间增加
* 过拟合风险增加
* 论文解释性变差
* 很难证明到底是哪一类高频特征有贡献

所以 feature selection 的目标是：

```text
选择与 target 高相关、但彼此低冗余的特征集合。
```

---

## 3. 数据输入定义

对于每个 appliance，例如 kettle，融合后的 CSV 大概是：

| readable_time | V_rms | I_rms | P_active | I3 | THDI | DWT_E1 | aggregate | kettle_power | on_off |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2013-07-22 01:00:00 | 240.4 | 0.58 | 107.3 | 0.33 | 0.74 | 0.057 | 105.9 | 1.0 | 0 |
| 2013-07-22 01:00:06 | 240.5 | 0.58 | 107.1 | 0.33 | 0.74 | 0.058 | 105.8 | 1.0 | 0 |

特征选择时定义：

```text
X_hf = 所有高频特征列
y_reg = kettle_power
y_cls = on_off
```

不要把这些列放入高频特征选择：

```text
readable_time
aggregate
kettle_power
on_off
```

原因：

* `readable_time` 是时间索引，不是电气特征。
* `aggregate` 是低频输入，不属于高频特征。
* `kettle_power` 是 regression target。
* `on_off` 是 classification target。

---

## 4. Stage 0：特征清洗

在正式排名之前，先删除明显不可靠的特征。

### 4.1 删除常数或近似常数特征

如果一个特征几乎不变化，例如：

| window | feature_A |
| ---: | ---: |
| 1 | 0.001 |
| 2 | 0.001 |
| 3 | 0.001 |
| 4 | 0.001 |

那么：

```text
Var(feature_A) ≈ 0
```

这个特征无法区分 appliance 状态，应删除。

默认阈值：

```text
near_constant_variance_threshold = 1e-8
```

### 4.2 删除无效值过多的特征

如果某个特征有太多：

```text
NaN / Inf / invalid values
```

则认为不可靠。

默认规则：

```text
invalid_ratio > 0.05 -> drop
```

即超过 5% 的数据无效就删除。

输出文件：

```text
feature_cleaning_report.csv
```

示例：

| feature | action | reason |
| :--- | :--- | :--- |
| V_rms | keep | valid |
| I_env_7 | drop | invalid_ratio > 0.05 |
| V_skew | drop | near_constant |

---

## 5. Stage 1：相关性与共线性过滤

这一步的目标是删除重复信息。

Pearson correlation 定义：

```text
r(x, y) = cov(x, y) / (std(x) * std(y))
```

如果：

```text
abs(r) > 0.95
```

说明两个特征高度相似。

简单例子：

| sample | I_rms | I_std |
| ---: | ---: | ---: |
| 1 | 0.50 | 0.50 |
| 2 | 0.60 | 0.60 |
| 3 | 0.70 | 0.70 |
| 4 | 0.80 | 0.80 |

此时：

```text
corr(I_rms, I_std) = 1.0
```

所以只需要保留一个。

保留规则：

```text
1. 保留与 target 更相关的特征。
2. 如果相关性差不多，保留物理意义更清楚的特征。
```

例如：

```text
I_rms 和 I_std 高度相关。
I_rms 的电气意义更清楚。
保留 I_rms，删除 I_std。
```

输出文件：

```text
correlation_drop_report.csv
```

示例：

| dropped_feature | kept_feature | pearson | spearman | reason |
| :--- | :--- | ---: | ---: | :--- |
| I_std | I_rms | 0.998 | 0.997 | duplicated current magnitude |
| V_std | V_rms | 0.999 | 0.999 | duplicated voltage magnitude |

---

## 6. Stage 2：mRMR 特征排名

mRMR 的全称是：

```text
minimum Redundancy Maximum Relevance
```

中文意思：

```text
最小冗余，最大相关性
```

也就是说，mRMR 选择的特征应该：

```text
1. 对 target 有用
2. 和已经选出的特征不重复
```

### 6.1 Mutual Information

Mutual information 衡量知道一个变量后，对另一个变量的不确定性减少了多少。

公式：

```text
I(X;Y) = sum_x sum_y p(x,y) log( p(x,y) / (p(x)p(y)) )
```

在本项目里：

```text
I(I3; on_off)
```

表示 3rd harmonic 对 ON/OFF 状态有多少信息量。

```text
I(P_active; kettle_power)
```

表示 active power 对 kettle power prediction 有多少信息量。

### 6.2 mRMR score

对候选特征 `f`：

```text
score(f) = relevance(f, target) - redundancy(f, selected_features)
```

展开为：

```text
score(f) = I(f; y) - (1 / |S|) * sum I(f; s)
```

其中：

```text
f = 候选特征
y = target，可以是 on_off 或 appliance_power
S = 已选特征集合
I(f; y) = 特征与 target 的 mutual information
I(f; s) = 候选特征与已选特征之间的 mutual information
```

### 6.3 Toy Example

假设目标是选择对 `on_off` 有用的特征：

| feature | MI with on_off | redundancy with selected | mRMR score |
| :--- | ---: | ---: | ---: |
| P_active | 0.80 | 0.00 | 0.80 |
| I_rms | 0.78 | 0.75 | 0.03 |
| THDI | 0.45 | 0.10 | 0.35 |
| DWT_E1 | 0.40 | 0.05 | 0.35 |

虽然 `I_rms` 和 `on_off` 的相关性很高，但它和 `P_active` 太重复，所以 mRMR score 很低。

因此 mRMR 可能更倾向于选择：

```text
THDI 或 DWT_E1
```

因为它们提供了新的信息。

### 6.4 为什么要做两份 mRMR ranking

本项目是 multi-task：

```text
y_cls = on_off
y_reg = appliance_power
```

所以应该输出：

```text
rank_mrmr_cls.csv
rank_mrmr_reg.csv
```

原因是：

```text
某些特征适合判断开关状态；
某些特征适合预测连续功率。
```

例如：

```text
DWT_E1 可能更适合检测 switching transient，因此对 on_off 有帮助。
P_active 可能更适合预测连续功率，因此对 appliance_power 有帮助。
```

---

## 7. Stage 3：Random Forest 特征重要性

mRMR 是 filter method，主要看统计依赖关系。Random Forest 是 model-based method，可以验证特征在预测模型里是否真的有用。

使用：

```text
RandomForestClassifier -> on_off
RandomForestRegressor  -> appliance_power
```

### 7.1 为什么用 Random Forest

Random Forest 适合这个阶段，因为：

* 能处理非线性关系
* 适合 tabular features
* 可以输出 feature importance
* 比深度模型更容易解释

### 7.2 Permutation Importance

Permutation importance 的逻辑：

```text
1. 正常训练模型。
2. 记录 validation performance。
3. 打乱一个 feature column。
4. 再看 performance 掉多少。
5. 掉得越多，说明 feature 越重要。
```

例子：

| feature | F1 before shuffle | F1 after shuffle | importance |
| :--- | ---: | ---: | ---: |
| P_active | 0.90 | 0.65 | 0.25 |
| THDI | 0.90 | 0.82 | 0.08 |
| V_rms | 0.90 | 0.89 | 0.01 |

说明：

```text
P_active 被打乱后 F1 掉很多，因此很重要。
V_rms 被打乱后几乎没影响，因此可能不重要。
```

输出文件：

```text
rank_rf_cls.csv
rank_rf_reg.csv
```

---

## 8. Stage 4：多任务特征集合并

因为项目同时做：

```text
ON/OFF classification
appliance power regression
```

所以不能只根据一个 target 选特征。

定义：

```text
F_cls = 对 ON/OFF 分类有用的特征
F_reg = 对功率回归有用的特征
F_final = F_cls ∪ F_reg
```

默认：

```text
每个任务 top_k = 15
最终特征数控制在 20 到 30 个左右
```

例子：

```text
F_cls = [DWT_E1, THDI, I3, I_kurt, Fci]
F_reg = [P_active, I_rms, I1, PF, S_apparent]
```

最终：

```text
F_final = [DWT_E1, THDI, I3, I_kurt, Fci, P_active, I_rms, I1, PF, S_apparent]
```

这样最终特征集合可以同时支持：

```text
状态检测 + 功率预测
```

---

## 9. Stage 5：稳定性选择

一个特征不能只因为在某一次 split 里表现好就被保留。它应该在不同时间段都稳定有效。

NILM 数据不能随便 random split，因为相邻 6 秒窗口非常相似。random split 可能导致 train/test leakage。

应该使用 time-based folds：

```text
Fold 1 = 时间段 1
Fold 2 = 时间段 2
Fold 3 = 时间段 3
Fold 4 = 时间段 4
Fold 5 = 时间段 5
```

稳定性定义：

```text
stability_frequency(feature) = selected_fold_count / total_fold_count
```

例子：

| feature | selected folds | stability |
| :--- | ---: | ---: |
| P_active | 5/5 | 1.00 |
| THDI | 4/5 | 0.80 |
| DWT_E3 | 3/5 | 0.60 |
| V_skew | 1/5 | 0.20 |

默认规则：

```text
stability >= 0.60 -> keep
```

因此保留：

```text
P_active, THDI, DWT_E3
```

删除：

```text
V_skew
```

除非后续 ablation 证明它对某个 appliance 特别重要。

---

## 10. Stage 6：消融实验验证

Feature ranking 只是候选证据，最终必须靠实验验证。

需要比较：

```text
LF only
LF + all HF features
LF + correlation-filtered HF
LF + mRMR-selected HF
LF + mRMR + RF selected HF
LF + selected time-domain HF only
LF + selected harmonics only
LF + selected spectral envelope only
LF + selected wavelet only
```

理想结果：

```text
LF + selected HF > LF only
LF + selected HF >= LF + all HF
```

如果成立，就可以说明：

```text
经过选择的高频特征确实提升了 NILM，而且比盲目使用所有高频特征更稳。
```

示例结果表：

| Method | Feature Count | MAE lower better | F1 higher better |
| :--- | ---: | ---: | ---: |
| LF only | 1 | 42.0 | 0.71 |
| LF + all HF | 52 | 36.5 | 0.78 |
| LF + mRMR HF | 20 | 34.0 | 0.81 |
| LF + mRMR + RF HF | 24 | 32.8 | 0.84 |

这个结果可以支持 thesis claim。

---

## 11. 论文中可以这样表达

可以写成：

> 高频特征提取器产生了丰富但冗余的多域电气表征。为了降低过拟合风险并提升模型可解释性，本文提出一种混合特征选择策略。首先删除无效和高度共线的特征；其次使用 mRMR 选择与目标高度相关且彼此低冗余的特征；随后通过 Random Forest permutation importance 验证特征在非线性预测模型中的实际贡献；最后通过基于时间划分的稳定性选择和消融实验，确定 appliance-specific 和 global 高频特征子集，用于后续低频-高频融合 NILM 模型。

核心贡献可以概括为：

```text
multi-task, literature-backed, stability-aware HF feature selection for LF-HF NILM fusion
```

中文表达：

```text
面向低频-高频融合 NILM 的多任务、文献支撑、稳定性感知高频特征选择方法。
```

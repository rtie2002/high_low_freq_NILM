# Microwave 时间对齐审计

日期：2026-10-09  
数据集：`mixed_ukdale_refit_5w_house_split`

## 1. 问题

需要区分两个问题：

1. aggregate 与 microwave 的时间不一致是否只存在于 REFIT house 20 test；
2. 当前 1--2 sample augmentation 是否只是针对 test block 的偶然修复。

审计脚本为 `scripts/audit_microwave_alignment.py`。它在每个真实 microwave power
由低于 200 W 上升到不低于 200 W 的位置，搜索前后 3 个 sample 的最大 aggregate
正边缘。该方法会受到同时运行负载影响，因此用于比较分布，不被解释为精确的 sensor
timestamp recovery。

## 2. REFIT 各 split 的结果

下表只统计 REFIT。每一列表示最大 aggregate 正边缘相对于 microwave 起点的位置。

| Split | Events | -3 | -2 | -1 | 0 | +1 | +2 | +3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train: houses 3, 5, 9, 11 | 697 | 3.3% | 2.6% | 13.2% | **50.5%** | 18.7% | 6.2% | 5.6% |
| Validation: house 2 | 298 | 13.8% | 7.4% | 5.4% | **47.0%** | 11.4% | 10.4% | 4.7% |
| Test: house 20 selected block | 145 | 3.4% | 2.8% | 1.4% | 19.3% | 17.2% | **50.3%** | 5.5% |

结论：

- training 与 validation 也存在 temporal mismatch，并非完全同步；
- 但二者仍以 lag 0 为主；
- REFIT house 20 test 明显不同，以 lag +2 为主；
- `aggregate < microwave` 的物理不一致在 microwave 起点的比例约为：training
  REFIT 14%、validation REFIT 23%、test REFIT 59%。

UK-DALE 1、2、5 均以 lag 0 为主，没有表现出 REFIT 20 同等级的 +2 shift。

## 3. 当前 test block 是否是偶然裁剪

完整 REFIT house 20 共有 1197 个可审计 microwave starts：

- lag 0：491；
- lag +1：232；
- lag +2：255。

但该分布随日期明显变化：

- 2014-03 至 2014-09：多数月份以 lag +2 为主；
- 2014-10 至 2015-06：多数月份改为 lag 0。

当前固定 test block 从 2014-05-29 开始，正好位于早期 lag +2 regime。它不是由我们
的 8 s resampling 人工制造的“幸运片段”，也不是一个普通的同步片段；相反，它是
House 20 中对齐问题特别严重的时期。如果换成 2015 年的 block，microwave pointwise
F1 很可能更容易，但这样在看过结果后换 test period 会形成 evaluation leakage，不能做。

## 4. 当前 augmentation 应如何解释

当前实现只在 synthetic half 中，以 0.5 probability 将 microwave contribution
向后移动 1 或 2 samples；真实窗口、validation 和 test 均不修改。

它应称为 **alignment-jitter robustness augmentation**，不能称为对真实 sensor
delay 的精确校正。它有合理依据，因为 training 与 validation 中约 22--25% 的
microwave starts 也表现为 lag +1/+2；但 test house 20 的 +1/+2 比例达到 67.5%，
明显超出训练分布。

因此，REFIT test F1 从约 0.420 提高到 0.521 不能只解释成“模型学会了 microwave
pattern”；其中包含模型对未见过的严重 alignment regime 变得更稳健的贡献。

## 5. 科学上正确的处理方式

1. **保留当前固定 test block。** 不因为后期 House 20 更容易而更换测试日期。
2. **所有 augmentation 概率只从 training 估计。** Test 只用于最终报告，不能用来
   选择 shift 方向、最大 shift 或 probability。
3. **Validation 增加 alignment stress view。** 从 training houses 中按时间保留一个
   对齐较差的 period，只用于选择 robustness setting；REFIT 20 仍保持 report-only。
4. **同时报告两类指标。** 保留严格 pointwise F1，并增加容许 ±16 s 的 event F1。
   后者反映是否找到了正确事件，不能替代严格 F1。
5. **不要用 test labels 把整段序列手工 shift。** Cleaned REFIT 不提供每个 sensor 的
   独立 timestamp，固定平移无法恢复 event-dependent delay，而且会造成标签泄漏。
6. **若继续改训练，优先使用简单的 transition uncertainty。** 在 microwave 开关
   边缘附近允许小范围 label/edge uncertainty，而稳定 ON 区域继续使用原监督；不要
   再增加新的 expert 或 backbone。

## 6. 下一项受控实验

在不查看 REFIT 20 选择结果的前提下：

1. 用 training houses 的 lag 分布生成 categorical alignment jitter；
2. 使用 validation house 2 与 training-held-out stress period 选择唯一设置；
3. 比较现有 one-sided augmentation 与 training-derived jitter；
4. 主要判断 validation AP、strict F1、±16 s event F1；
5. 设置冻结后，才在 REFIT 20 与 UK-DALE 2 上做一次最终报告。

这能判断当前 improvement 是真正的 alignment robustness，还是对 House 20 早期
regime 的偶然适配。

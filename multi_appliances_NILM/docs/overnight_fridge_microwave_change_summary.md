# MultiNILM 整晚改动总结与问题解决判定

日期：2026-10-09  
范围：从恢复 early-relation 基线，到最终保留 meter-lag 版本  
最终配置：`config/models/multinilm_k4.yaml`  
最终实验：`multinilm_meter_lag_house_split`

## 1. 先说结论

这轮工作**没有完全解决所有问题**，因此不能声称 noisy-background fridge 已经被解决。

- **Microwave：明显改善，但仍未完美。** REFIT house 20 的 F1 从约 `0.420` 提高到
  `0.521`，主要改善来自针对真实计量不同步的训练增强，而不是增加复杂网络。
- **Fridge：尚未解决。** REFIT house 20 的 fridge F1 为 `0.723`，在高 residual
  背景下仍有严重误触发；波形仍会把未知的 50--100 W 平台误认为 fridge。
- **模型没有继续变复杂。** long-context、重标标签、transition loss、周期特征、
  guarded background 等失败方案均已撤销。最终恢复 1.38M 参数的 early-relation
  architecture，只保留有受控证据支持的改动。

因此，准确的表述应是：

> 本轮定位并显著缓解了 REFIT microwave 的时间错位问题，同时排除了多种可能的
> fridge 原因；但只使用 8 s active aggregate power 时，noisy-background fridge
> 仍存在可辨识性限制，尚不能视为已解决。

## 2. 最初的问题与判断方法

起始版本在跨房屋测试中有两个不同的失效机制：

1. **Microwave：** 预测事件碎片化、漏检，REFIT 比 UK-DALE 严重；
2. **Fridge：** residual 背景升高后大量误触发，输出不符合真实 fridge 周期。

本轮没有用 training loss 判断成功，而是同时检查：

- validation AP、macro-F1 和 MAE；
- REFIT 20 与 UK-DALE 2 的跨房屋结果；
- fridge 在不同 residual 功率区间的 FPR；
- 真实事件数、预测事件数和持续时间；
- noisy-background waveform 是否连续且符合物理逻辑。

## 3. 从第一版到最后一版改了什么

### 3.1 恢复可解释的起点

首先撤回之前不断增加的 dual expert、local contrast、mixture-consistent output、
background head 和 hard-negative 等复杂方案，恢复 early-relation MultiNILM：

- 13 个输入通道：raw、signed delta、absolute delta、rolling mean/std 和 4 个 GL
  fractional channels；
- multiscale stem、IBN、5-block dilated TCN；
- task attention 与 relation attention；
- 五个 appliance 联合预测；
- 约 1.38M 参数。

恢复基线结果：

| Split | MAE (W) | Macro-F1 | AP |
|---|---:|---:|---:|
| Validation | 14.761 | 0.766 | 0.787 |
| REFIT house 20 | 10.615 | 0.729 | 0.708 |
| UK-DALE house 2 | 7.321 | 0.867 | 0.928 |

这样做的目的不是宣称旧模型最好，而是建立一个结构稳定、可以逐项归因的起点。

### 3.2 改善 full-mix 的采样覆盖

在原有 50:50 real/full-random mix 中加入 focal-event 与 background-bin sampling：

```text
50% real windows
        +
50% synthetic windows
  ├─ 强制包含一个真实 ON appliance event
  └─ residual background 从不同功率区间均衡抽样
```

该改变没有增加模型参数。它使 validation AP 从 `0.787` 提高到 `0.795`，validation
fridge FPR 从 `0.209` 降到 `0.171`，但没有彻底解决 microwave 和 fridge。
`prob=0.25` 的版本虽然提高了某个 test F1，却没有得到 validation 支持，因此拒绝，
最终保留 `prob=1.0`。

### 3.3 检查并恢复 evaluation hard gate

我们怀疑 classification gate 会截断本来正确的 regression waveform，因此用同一个
checkpoint 关闭 gate 做诊断。结果如下：

| Split | 保留 hard gate MAE | 关闭 hard gate MAE |
|---|---:|---:|
| Validation | 14.424 | 23.709 |
| REFIT house 20 | 10.254 | 17.195 |
| UK-DALE house 2 | 7.535 | 9.405 |

关闭后 OFF-state leakage 大幅增加。因此该方案被撤销，`apply_to_power: true` 恢复。
这说明 hard gate 不是分类失败的根因，但在当前模型中仍是必要的输出约束。

### 3.4 Microwave duration decoding

validation 显示真实 microwave 事件中位持续时间约 80 s，而预测只有约 40 s。于是只用
validation 选择：

- minimum ON duration：`32 s`；
- maximum merge gap：`24 s`。

该后处理把 validation microwave F1 从 `0.427` 提高到 `0.439`，REFIT 从 `0.420`
提高到 `0.447`。它只清理短碎片，因此是小幅改善，不是根本解决。

### 3.5 Microwave meter-lag augmentation

数据审计发现，REFIT 的 aggregate 与 appliance channel 分别在 8 s 区间求均值，
两只电表并非总在同一时刻采样。真实 microwave target 已经 ON 时，aggregate 中的
完整跳变有时会晚 1--2 个 sample 出现。模型因此会收到互相矛盾的输入和标签。

训练时只对 synthetic microwave contribution 加入随机延迟：

```text
Target/state:        ────[ microwave ON ]────
Aggregate input:     ──────[ microwave ON ]────
                           1--2 samples lag
```

具体约束：

- 只作用于 training synthetic windows；
- 真实训练窗口不变；
- validation 和 test 完全不变；
- power/state target 不移动；
- 其他四个 appliances 不变；
- 50% 的 synthetic microwave 使用 1--2 sample 延迟。

这是本轮唯一得到清晰跨房屋支持的学习改动：

| Split | Duration-only microwave F1 | + Meter lag |
|---|---:|---:|
| Validation | 0.439 | 0.449 |
| REFIT house 20 | 0.447 | **0.521** |
| UK-DALE house 2 | 0.734 | 0.714 |

REFIT overall macro-F1 同时提高到 `0.746`，UK-DALE overall macro-F1 为 `0.872`。
UK-DALE microwave 有轻微下降，因此这不是无代价改进，但它明显缓解了目标域最严重的
microwave failure，而且没有改变 evaluation 数据。

### 3.6 Fridge 输出物理上限与评估 bug 修正

训练集 fridge ON power 的 99.9% 分位数约为 512 W，因此将 fridge prediction 的
上限设为 `600 W`，阻止明显不可能的 700 W 以上尖峰。

同时修正 evaluation：minimum-power threshold 和 maximum-power cap 只能处理
prediction，不能同步修改 ground truth。该修正提高了评估的科学有效性。

需要强调：这个 cap 只修复**幅值不合理**，不会改变 state probability，也不会降低
fridge FPR，所以它不是 noisy-background fridge 的解决方案。

## 4. 尝试后撤销的方案

| 实验 | 原假设 | 观察结果 | 最终决定 |
|---|---|---|---|
| 67 min long context | 完整周期可区分 fridge 与背景 | REFIT FPR 下降，但 fridge F1 `0.723→0.706` | 撤销 |
| REFIT-11 fridge 重标 | 低功率 ON 标签造成错误监督 | fridge F1 局部上升，但高背景 FPR 恶化到 `0.86--0.91` | 撤销 |
| 排除 REFIT 11 | 该房屋域偏移过大 | REFIT overall F1 降到 `0.711` | 撤销 |
| Fridge transition loss | 强调开关边缘可改善连续性 | fridge F1 降到 `0.707` | 删除代码并撤销 |
| Single-fridge model | 共享梯度是主要问题 | REFIT fridge F1 只有 `0.718` | 否定主假设 |
| 4--48 min periodic features | 固定滞后可捕获压缩机周期 | REFIT overall F1 降到 `0.685` | 删除代码并撤销 |
| Hysteresis / Viterbi | 后处理可消除背景误报 | 最好 F1 约 `0.74`，FPR 仍约 `0.5` | 不保留 |
| Guarded synthetic residual | meter leakage 污染 background mix | fridge MAE 略降，但 classification 与 overall F1 退化 | 删除代码并撤销 |

这些负结果非常重要：它们说明问题不是简单的“网络不够大”、TCN context 不够长、
共享多任务梯度冲突，或后处理参数不够好。

## 5. 最终仍然生效的结构

```text
8 s aggregate active power
          │
          ▼
13-channel physical/time-series features
          │
          ▼
Multiscale stem + IBN + 5-block dilated TCN
          │
          ▼
Task attention + cross-appliance relation attention
          │
          ├──────────────┐
          ▼              ▼
     State head      Power head
          │              │
          └── soft gate ─┘
                 │
                 ▼
 validation-calibrated state + hard power mask
                 │
                 ▼
 microwave duration cleanup + prediction-only physical cap
```

最终仍生效的新增内容只有：

1. focal-event/background-stratified full mix；
2. microwave training-only meter-lag augmentation；
3. microwave 32 s minimum-ON 与 24 s merge-gap；
4. fridge prediction-only 600 W physical cap；
5. ground truth 不再被 prediction postprocessing 错误修改。

没有保留新的 expert、background head、transition loss、periodic features 或 guarded
residual branch。

## 6. 最终结果

| Split | Kettle F1 | Fridge F1 | Dishwasher F1 | Washing machine F1 | Microwave F1 | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.870 | 0.808 | 0.799 | 0.929 | 0.449 | 0.771 |
| REFIT house 20 | 0.796 | 0.723 | 0.810 | 0.880 | **0.521** | **0.746** |
| UK-DALE house 2 | 0.967 | 0.891 | 0.864 | 0.924 | 0.714 | **0.872** |

对应 overall MAE：

- Validation：`15.019 W`；
- REFIT house 20：`10.547 W`；
- UK-DALE house 2：`8.478 W`。

这些数字说明总体性能没有崩溃，且 REFIT microwave 得到明显改善；但 validation
microwave F1 仍低，REFIT fridge 的高背景波形仍不可靠。

## 7. 为什么不能说 fridge 已经解决

REFIT house 20 中，随着真实 residual 增大，fridge recall 上升的同时 FPR 也失控。
典型结果为：

| Residual range | Fridge FPR |
|---|---:|
| 200--400 W | 约 0.795 |
| 400--800 W | 约 0.872 |

这意味着模型并不是稳定识别 fridge，而是在高背景时更倾向输出 ON。波形中的
50--100 W 假平台与真实 fridge 在当前输入中可能拥有相似的边缘、幅值和持续时间。

更关键的证据是：

- single-fridge model 没有解决，说明并非主要由 multi-task 梯度竞争造成；
- 更长 context 降低部分 FPR，却同时损失 recall；
- local/periodic features、hysteresis 和 Viterbi 均不能稳定分离两类信号；
- oracle state 可把 REFIT MAE 从约 `10.7 W` 降到约 `5.6 W`，说明主要瓶颈是
  state detection，而不是 ON-state power regression。

因此当前 fridge 问题更接近**观测可辨识性不足**：只给模型一条 8 s active-power
aggregate，未知负载可能产生与 fridge 相同的低功率平台。增加同源特征或更大的网络
不会凭空恢复输入中不存在的信息。

## 8. 为什么认为 microwave 的方向是正确的

不是因为 training loss 更低，而是因为以下证据同时成立：

1. 数据管线中确认存在符合机制的 aggregate/submeter 时间错位；
2. augmentation 只改变 training input，不污染 validation/test；
3. validation microwave F1 有小幅提升；
4. 最严重的 REFIT microwave F1 从约 `0.420` 提高到 `0.521`；
5. REFIT overall 与 UK-DALE overall 没有随之崩溃；
6. 该方法没有增加模型参数，也没有引入新的 inference module。

所以可以说“microwave failure 被明显缓解”，但不能说所有 microwave 场景已解决。

## 9. 下一步真正值得做什么

不建议继续在同一 active-power 输入上堆叠 expert、attention、FFT 或 loss term。
若目标是解决 noisy-background fridge，优先级应为：

1. 加入 reactive power、current、voltage 或高频 transient；
2. 改善 aggregate 与 submeter 的时间同步；
3. 给目标房屋少量 fridge calibration data，进行轻量适配；
4. 标注或单独建模最常与 fridge 混淆的 unknown loads；
5. 若论文必须限定为 8 s active power，则把贡献定位为真实 aggregate 下的
   multi-appliance robustness、meter-asynchrony augmentation，以及
   background-stratified failure analysis，而不是宣称 fridge 已完全解决。

## 10. 可复现位置

- 最终配置：`multi_appliances_NILM/config/models/multinilm_k4.yaml`
- 数据配置：`multi_appliances_NILM/config/experiment_mixed_ukdale_refit_5w_house_split.yaml`
- 详细诊断：`multi_appliances_NILM/docs/fridge_microwave_failure_literature_diagnosis.md`
- 实验日志：`multi_appliances_NILM/docs/multinilm_experiment_log.md`
- 最终 checkpoint 来源：
  `runs/multinilm_microwave_meter_lag_house_split_v2/multinilm_fractional/best.pt`
- 最终统一评估输出：
  `runs/multinilm_meter_lag_house_split/multinilm_fractional/`

对应 Git 历史从 `01bc7b3` 到 `6773688`。最终提交 `6773688` 已撤销最后一个失败的
guarded-residual 实验，并恢复本报告所述的最佳 meter-lag 配置。

# Fridge 与 Microwave 失效：证据、文献与修复路线

更新日期：2026-10-09

## 1. 结论先行

当前问题不是单一的“网络容量不足”，而是两个不同问题：

1. **Microwave：稀有短事件、异步采样与状态碎片化。** REFIT 的 aggregate
   与 appliance meter 在同一个 8 s 时间格内分别求均值，但两者并非同时采样。
   REFIT house 20 的 microwave 完整 aggregate 边缘经常比 appliance state 晚
   约 1--2 个采样点出现。模型因此会收到“标签已经 ON，但输入边缘仍不完整”
   的样本。
2. **Fridge：弱目标与未知背景不可辨识。** Fridge 通常只有约 80--130 W；
   未监测负载也会形成相近的平台和边缘。REFIT house 20 中，fridge 的 FPR
   随 residual background 明显上升。这首先是状态检测问题，不是 ON 状态下的
   功率幅值问题。

因此，不应再给二者共同叠加一个大模块。Microwave 需要处理 meter timing 与
短事件连续性；fridge 需要更长的周期证据和更可靠的背景拒绝。

## 2. 当前模型的实证诊断

### 2.1 Oracle state 证明首要瓶颈是 classification

将同一个模型的回归输出用真实 state 遮罩，不重新训练：

| Split | 当前 hard gate MAE | Oracle-state gate MAE |
|---|---:|---:|
| Validation | 约 14.5 W | 约 9.3 W |
| REFIT house 20 | 约 10.7 W | 约 5.6 W |
| UK-DALE house 2 | 约 8.5 W | 约 6.9 W |

这说明 regression 在真实 ON 区域已经包含相当多有效信息。继续优先修改 power
head 或增加 energy loss，不能直接解决当前主要错误。

### 2.2 删除 evaluation hard gate 是失败方案

保持 checkpoint 不变，仅关闭最终 binary power mask：

| Split | 原 MAE | 关闭 mask 后 MAE |
|---|---:|---:|
| Validation | 14.424 W | 23.709 W |
| REFIT house 20 | 10.254 W | 17.195 W |
| UK-DALE house 2 | 7.535 W | 9.405 W |

soft-gated regression 在 OFF 区仍有大量 leakage，因此 hard gate 必须暂时保留。

### 2.3 Fridge 的错误与背景强度相关

在 REFIT house 20，fridge 的状态误报并非均匀出现。以真实 residual 分桶后：

| Residual range | Recall | FPR |
|---|---:|---:|
| 0--100 W | 0.450 | 0.210 |
| 100--200 W | 0.541 | 0.021 |
| 200--400 W | 0.874 | 0.685 |
| 400--800 W | 0.965 | 0.787 |
| >800 W | 0.903 | 0.697 |

高背景时 recall 很高但 FPR 同时失控，说明模型倾向于把未知负载分配给 fridge。

### 2.4 Microwave 是短事件碎片化，不只是阈值错误

| Split | 真实事件数 | 预测事件数 | 真实中位持续时间 | 预测中位持续时间 |
|---|---:|---:|---:|---:|
| Validation | 146 | 182 | 80 s | 40 s |
| REFIT house 20 | 101 | 429 | 136 s | 32 s |

Validation 仍选择很高的 microwave threshold（0.96），但改变 threshold 本身不能
同时恢复 recall 与 precision。问题在 state ranking 和时间连续性。

## 3. 文献为什么看起来更容易

### UNet-NILM

UNet-NILM 同时进行 multi-label state detection 与 multi-target quantile
regression，但 UK-DALE 实验使用由少数已知 appliance channels 构成的
**artificial aggregate**。这会显著减少 unknown residual；其高 F1 不能直接说明
真实 aggregate 上的 fridge 背景混淆已经解决。

### Conv-NILM-Net

该工作将 NILM 写成 time-domain source separation：encoder、TCN separator、
per-source masks 和 decoder。论文特别放松了 speech separation 中 masks sum to
one 的假设，因为 NILM 含有未知负载与测量误差。这个观点支持我们的诊断：
不能强迫五个目标电器解释全部 aggregate。

### MATNilm

MATNilm 使用 shared-to-appliance-specific regression/classification branches，并将
最终功率写成 regression 与 state probability 的乘积。其 sample augmentation 会
替换和缩放 appliance profile，包括 horizontal scaling。这说明对 appliance
duration、timing 和幅值做训练期扰动，比单纯增加 backbone 更有针对性。

### SAMNet

SAMNet 使用 shared experts、task-specific gates/towers 和 state-power coupling。
但其训练会主动保持可能 ON 与 OFF 的样本接近 1:1，并使用不同的 window step。
因此论文结果同时受 sampling protocol 影响，不能只归因于 architecture。

## 4. 已保留的小改进：Microwave duration decoding

只在 validation 上搜索 temporal postprocessing，选择：

- minimum ON duration：8 s -> 32 s；
- maximum merge gap：16 s -> 24 s。

正式评估结果：

| Split | 原 microwave F1 | 新 microwave F1 |
|---|---:|---:|
| Validation | 0.427 | 0.439 |
| REFIT house 20 | 0.420 | 0.447 |

REFIT microwave precision 从 0.351 提高到 0.396。该改动只清理明显短碎片，
不改变 raw AP，也不被视为完整解决方案。Fridge 的 hysteresis/duration 搜索没有
稳定迁移，因此没有保留。

## 5. 当前受控实验：Microwave meter-lag augmentation

只修改 synthetic full-mix 输入：

```text
submeter target/state:  ----[ microwave ON ]----
aggregate contribution: ------[ microwave ON ]----
                              1--2 samples lag
```

- 真实训练窗口不变；
- validation/test 不变；
- 其他 appliance 不变；
- 50% 的 synthetic microwave contribution 延迟 1--2 个 8 s sample；
- power/state target 保持原 submeter 时间轴。

该实验直接检验 REFIT asynchronous meter timing 假设。成功标准不是 train loss，
而是 validation microwave AP/F1 提高，并在固定参数下改善 REFIT F1，同时不能
损害 UK-DALE 的 aligned microwave event。

## 6. 下一步：Fridge 的最小架构实验

当前 TCN receptive field 为 249 samples，即约 33.2 min。Validation fridge 的
真实 ON duration 中位数约 32.7 min，而完整 ON/OFF cycle 往往更长。因此模型
常常看不到足够的周期上下文来区分 fridge cycle 与未知背景平台。

若 meter-lag 实验确认 microwave 得到改善，下一项只增加一个 dilation=32 的 TCN
block，将 receptive field 从 249 扩大至 505 samples（约 67.3 min）。其余输入、
loss、sampling 和 head 均保持不变。该实验只在以下条件下保留：

1. validation fridge AP/F1 提高；
2. validation 与 REFIT high-background FPR 降低；
3. representative noisy-background waveform 形成完整、连续的 fridge cycle；
4. microwave、kettle 等短事件不被过度平滑。

## 6.1 Meter-lag 实验结果

Meter-lag augmentation 达到预先定义的保留条件：

| Split | 指标 | Duration-only baseline | Meter-lag |
|---|---|---:|---:|
| Validation | microwave F1 | 0.439 | 0.449 |
| REFIT house 20 | microwave F1 | 0.447 | 0.521 |
| REFIT house 20 | overall macro-F1 | 0.719 | 0.746 |
| UK-DALE house 2 | overall macro-F1 | 0.869 | 0.872 |
| UK-DALE house 2 | microwave F1 | 0.734 | 0.714 |

结论：异步采样是 REFIT microwave 失败的重要组成部分，因此保留 augmentation。
它不是 fridge 方案：validation fridge F1 从 0.832 降至 0.808，REFIT 的
200--400 W 与 400--800 W residual 区间 fridge FPR 仍分别高达 0.795 和
0.872。下一实验只扩展 TCN context，其他设置不变。

## 7. 暂时不做的事情

- 不加入 DWT/FFT：8 s active power 的 microwave/fridge 问题首先是时序监督和
  未知背景，不是缺少频谱分辨率；FFT 不能恢复没有被传感器同步观测的信息。
- 不再增加第三个 expert、background head 或 hard-negative module：现有实验没有
  给出稳定增益，且会使归因更加困难。
- 不删除 hard gate：已被同 checkpoint 诊断明确否定。
- 不用 test house 选择超参数：所有 threshold、duration 与 checkpoint 必须由
  validation 决定，test 只用于最终报告。

## 8. 最终可发表方案应满足

最终模型不需要在每个点都完美，但必须同时满足：

- state AP/F1 在跨房屋 validation 上提高；
- fridge high-background FPR 明显下降；
- microwave event recall 与 precision 同时可接受；
- 波形具备物理连续性，而不是大量短脉冲或无依据平台；
- 每个保留组件都有单独消融，且相对 UNet-NILM/MATNilm 等 baseline 使用同一
  真实 aggregate、同一 split 和同一 metric pipeline。

## 9. Long-context 结果与监督一致性修复

67 min context 对背景拒绝有作用，但未单独解决 fridge：

| REFIT house 20 | Meter-lag | + Long context |
|---|---:|---:|
| Fridge FPR | 0.468 | 0.411 |
| Fridge F1 | 0.723 | 0.706 |
| 200--400 W residual FPR | 0.795 | 0.716 |
| 400--800 W residual FPR | 0.872 | 0.798 |

它降低了误报，但也降低 recall，因此只证明周期上下文方向合理。

随后发现训练监督存在更根本的不一致。统一任务定义使用 fridge ON threshold
50 W，但训练集 REFIT house 11 有 98,601 个 `fridge_on=1` 样本低于 50 W，
其中位 power 只有 35 W；该异常占该房屋全部训练时间点的 26.16%。Validation
与 test houses 几乎没有这种偏差。模型因而被训练成将非常弱的背景平台解释为
fridge。

新数据版本只修正 training REFIT house 11：

1. 按统一的 50 W threshold、12 s maximum OFF gap、60 s minimum ON duration
   重新生成 fridge state；
2. 将新 state 为 OFF 的 supervised fridge target 置零，使 state gate 与 power
   target 不再互相矛盾；
3. aggregate 完全不变，被移除的低功率通道自然成为 unknown residual；
4. validation/test 数据与标签完全不变；
5. normalization 只在修正后的 training split 上重新拟合。

该修改不是用 test 调参，而是执行实验开始时已经声明的统一 label definition。
下一实验保留已经有效的 microwave meter-lag 和 long context，只改变这项训练监督。

## 10. Fridge 输出范围修正

背景波形还揭示了独立的评估错误：`max_on_power_watts` 将 fridge 与 kettle 一样
设为 4000 W，因此假 fridge 波形可达到 700 W 以上。训练集 fridge ON power 的
99.9 分位数为 512 W；新的 prediction-only cap 设为 600 W，以保留启动瞬态并
阻止明显不可能的输出。

同时修正 `apply_power_postprocess_pair`：minimum-power threshold 与 maximum-power
cap 只可作用于 prediction，ground truth 只做非负保护，不能为了让 prediction
看起来更准确而被同步裁剪。该修正不改变 state probability 或 F1，但会提高功率
指标的科学有效性并消除最不合理的 fridge 尖峰。

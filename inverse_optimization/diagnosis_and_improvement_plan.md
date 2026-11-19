# 多电器逆优化的诊断与逐步改进计划

日期：2026-11-19。研究对象为非侵入式负荷监测（Non-Intrusive Load Monitoring，NILM）。

目标是从一条真实总表恢复五台指定电器的功率和开关状态，并检验测试时逆优化带来的实际收益。本文记录已经完成的诊断，以及接下来逐项实施的实验。计划中的方法尚未验证有效。

## 1. 当前两阶段在做什么

固定设备顺序为 kettle、fridge、dishwasher、washingmachine、microwave。每窗长度为 512，步长为 256，数据间隔为 8 秒。

Stage 1 使用长短期记忆网络（Long Short-Term Memory，LSTM）。输入为五路真实分表，输出为一路实测总表。

\[
\mathbf P\in\mathbb R^{T\times5},\quad
\widehat{\mathbf y}=F_\theta(\mathbf P),\quad
\min_\theta\frac1T\|F_\theta(\mathbf P)-\mathbf y\|_2^2.
\]

代码实际在训练集拟合的 z-score 坐标中计算均方误差（Mean Squared Error，MSE）。验证集和测试集复用训练统计量。

Stage 2 固定网络权重，对每个窗口的五路候选功率做 Adam 优化。目前每窗有 32 个重启、500 次更新，学习率为 0.01。

\[
\min_{\mathbf P}\frac1T\|\widetilde F_{\hat\theta}(\mathbf P)-\widetilde{\mathbf y}\|_2^2
+\lambda_{\rm bnd}R_{\rm bnd}(\mathbf P).
\]

波浪号表示使用 checkpoint 统计量的标准化输出。边界项是超出训练功率极值时的软惩罚。范围内的候选波形没有设备状态、持续时间或完整事件约束。

## 2. 已经观察到的证据

### 2.1 原始运行与核查范围

训练机仓库版本为 `953b8a2`。本节数值来自上一次只读审查和小规模诊断，本次编写计划没有重新训练。

```text
D:\Raymond\high_low_freq_NILM\inverse_optimization\results\forward_lstm_baseline_20261117_225146_155830
D:\Raymond\high_low_freq_NILM\inverse_optimization\results\neural_adjoint_baseline_20261118_230709_819051
```

读取的 checkpoint 选择 epoch 2，验证 MSE 为 0.8731494565，包含 21,921 个可训练参数。实际保存的结构是 hidden_size=32、num_layers=3、dropout=0。当前模型 YAML 的值可能已改变，复现实验应使用 checkpoint 中的结构。

保存预测覆盖 2,446 个验证窗口。按当前数据顺序重新计算逐窗平均绝对误差（Mean Absolute Error，MAE），与原 CSV 的最大差约为 0.0011 W，支持预测和评分数据的窗口对应关系。

### 2.2 完整验证集的结果

下表前三类总表误差在相同窗口上计算。逐电器指标先在各连续序列内平均重叠预测，再对每个被覆盖的时间点计分一次。覆盖范围以外的时间点未评分。

| 指标 | REFIT 2 | UK-DALE 5 |
|---|---:|---:|
| 验证窗口数 | 2,000 | 446 |
| 覆盖的唯一时间点数 | 546,304 | 188,416 |
| 真实五路输入 LSTM 的总表 MAE | 416.38 W | 237.41 W |
| 真实五路相加的总表 MAE | 290.87 W | 421.02 W |
| 真实五路相加，再加训练背景中位数 241 W 的总表 MAE | 259.36 W | 192.86 W |
| 估计五路输入 LSTM 的总表 MAE | 139.38 W | 3.47 W |
| 估计五路直接相加的总表 MAE | 760.43 W | 465.39 W |
| 原始估计的逐电器 Macro-MAE | 139.28 W | 47.79 W |
| 将估计负值截零后的逐电器 Macro-MAE，仅作后处理敏感性诊断 | 46.92 W | 39.98 W |
| 全 OFF 的逐电器 Macro-MAE | 28.95 W | 27.72 W |
| 估计的逐电器 Macro-F1，使用当前 Stage 2 门槛 | 0.224 | 0.126 |

全 OFF 的低 MAE 反映 OFF 样本占比高，应和真实 ON 时段误差、F1、召回率共同判断。直接相加遗漏背景，需要结合真实五路相加基线解释。UK-DALE 5 的 3.47 W 表明模型输出可以很好地匹配总表，同时五路分解仍然错误。

### 2.3 两个定位问题的对照

**真实分表起步。** 每个验证住户取 16 个均匀分布的窗口，用真实分表初始化，并沿用原 LSTM 与损失做 50 步优化。

| 住户 | 总表重模拟 MAE，初始 → 第 50 步 | 逐电器 MAE，初始 → 第 50 步 |
|---|---:|---:|
| REFIT 2 | 330.85 → 162.14 W | 0 → 54.71 W |
| UK-DALE 5 | 233.64 → 31.67 W | 0 → 28.97 W |

真实分表是诊断用信息，不能作为实际推断初始化。该结果显示现有目标会主动修改正确答案以改善总表拟合。输入梯度的方向有限差分检查也通过，支持自动微分链路正确。

**精确加法与干净混合。** 取六个至少三类电器活跃的验证窗口，用五路真值之和构造观测，替换 LSTM 为精确加法，沿用原优化器和边界项。

| 指标 | 初始 | 优化后 |
|---|---:|---:|
| 按目标选择的总表 MAE | 269.95 W | 5.19 W |
| 逐电器 Macro-MAE | 106.09 W | 121.68 W |

窗口索引为 `[1002,1371,863,38,352,1785]`，seed=2026，32 个重启、500 步。最终各设备 F1 为 `[0.272,0.700,0.489,0.534,0.000]`，部分状态检测改善，但仍有 26.50% 的通道采样点为负功率。这是六窗机制诊断，尚不能代替完整验证结论。

### 2.4 数据与评分问题

- REFIT 2 的原始窗口预测中，各设备低于 −0.1 W 的比例约为 84%–88%。现有绘图把负值截到零，原始保存值和误差保留负值。
- Stage 2 的门槛为 `[200,50,50,20,200]` W，原五设备基线配置为 `[40,50,30,30,100]` W。CSV 的真值状态还包含持续时间等规则。完整评分应复用既有基线规则，并明确原始预测与规则处理后预测。
- 逐窗平均的能量相对误差在真值能耗为零时被 `1e-6` 分母放大。逐窗 F1、平均精度（Average Precision，AP）也不能直接等价为整段时间轴的指标。当前 AP 的并列分数排序处理需要核对。
- REFIT 2 覆盖时间点有 16.38% 带未观测标记，UK-DALE 5 为 1.17%。当前 loader 没有将这些标记用于损失或评分。
- 观测残差最低为 REFIT 2 的 −3716 W、UK-DALE 5 的 −2640.67 W。严重负值也出现在全部通道均有观测标记的时间点。训练冰箱极值为 2846 W，需要追溯原始记录。
- 目前证据不能确认这些异常来自哪一种对齐、传感器或重采样机制。当前标准 UK-DALE 预处理读取 active W 字段。不能直接把负观测残差解释为物理背景。

## 3. 为什么总表误差下降仍会分错

令 \(\mathbf p_i\) 为第 i 台电器，\(\mathbf b\) 为未监测负荷，\(\boldsymbol\epsilon\) 为有符号测量与对齐误差。

\[
\mathbf y=\sum_{i=1}^5\mathbf p_i+\mathbf b+\boldsymbol\epsilon.
\]

只给五路分表训练实测总表，在平方损失与足够模型能力的理想条件下，预测趋向训练分布的条件均值。

\[
F_\theta(\mathbf P)\approx\sum_i\mathbf p_i+
\mathbb E_{\rm train}[\mathbf y-\sum_i\mathbf p_i\mid\mathbf P].
\]

换住户后，这个条件背景可能改变。逆优化也会进入训练样本之外的五路组合。更低的 LSTM 输出误差因此不能保证更低的分表误差。

即使背景已知且正向加法完全正确，512 个观测对应 2560 个逐点未知功率。精确加法矩阵的零空间有 2048 维。把同一段功率从微波炉搬到洗衣机，总和可以保持不变。非负和功率上限会缩小候选集合，但通常仍存在许多解释。

加入自由背景后，对任意五路候选都可以取

\[
\mathbf b=\mathbf y-\sum_i\mathbf p_i.
\]

若只要求非负，对非负总表也允许五台全 OFF、背景等于总表。因此背景需要限制，电器也需要能区分设备的波形与状态信息。

### 3.1 跨 UK-DALE 与 REFIT 的候选功率范围

以下范围用于后续 **Stage 2 候选功率的硬投影**，不是 ON/OFF 标签阈值，也不是对所有真实家电的额定功率声明。下界为 0 W。上界刻意选择为可同时涵盖两数据集的宽松工程上限：UK-DALE 有官方 appliance metadata；REFIT 没有一套官方、统一的五设备上限，且不同房屋的设备型号不同。

| 通道顺序 | Appliance | 最小功率 | UK-DALE metadata max | REFIT 可核对的公开信息 | 建议的跨数据集候选上限 |
|---:|---|---:|---:|---|---:|
| 1 | kettle | 0 W | 3100 W | REFIT cleaned data 会移除大于 4000 W 的 IAM 尖峰；公开 Seq2Point 配置使用 3998 W | **4000 W** |
| 2 | fridge | 0 W | 300 W | REFIT 文献把 fridge-freezer 的上限设为 1700 W | **1000 W** |
| 3 | dishwasher | 0 W | 2500 W | 多个公开 NILM 预处理方案使用 2500 W | **2500 W** |
| 4 | washingmachine | 0 W | 2500 W | 多个公开 NILM 预处理方案使用 2500 W | **2500 W** |
| 5 | microwave | 0 W | 3000 W | UK-DALE metadata 为 3000 W；REFIT 公开 Seq2Point 配置中的 3969 W 更像清洗上界而非设备额定值 | **3000 W** |

若需要在 YAML 中写成固定顺序：

```yaml
appliance_min_w: [0, 0, 0, 0, 0]
appliance_max_w: [4000, 1000, 2500, 2500, 3000]
```

UK-DALE 的五项 metadata 参数为 kettle 3100 W、fridge 300 W、dishwasher 2500 W、washing machine 2500 W、microwave 3000 W。[官方 UK-DALE metadata conversion](https://github.com/JackKelly/UK-DALE_metadata/blob/master/convert_uk-dale_to_NILM_Metadata.py) REFIT 原始数据论文说明 cleaned data 移除了超过 4000 W 的 individual appliance monitor（IAM）尖峰，但这不代表每类设备的真实物理额定上限。[REFIT dataset paper](https://pureportal.strath.ac.uk/files/63396371/Murray_etal_SD_2016_An_electrical_load_measurements_dataset_of_United_Kingdom.pdf) ELECTRIcity 对 REFIT 使用 1700 W 的 fridge-freezer 与 2500 W 的 washer 上限，说明 UK-DALE 的 300 W fridge 上限不能直接施加到 REFIT fridge/freezer。[ELECTRIcity preprocessing table](https://pmc.ncbi.nlm.nih.gov/articles/PMC9028578/)

该表应称为 `cross_dataset_candidate_bounds`，而不是 ground-truth physical limits。fridge 的 1000 W 是在 UK-DALE 300 W 与 REFIT fridge-freezer 1700 W 之间选择的保守实验上限；它必须作为可替换的配置记录。第一个实验仍应只投影到非负区域；第二个独立实验才使用此完整 box bound。两组实验均不重训 Stage 1，也不改动 validation/test 的标签或数据。

## 4. 按顺序实施的实验

每一步建立独立运行记录，保存配置、数据索引、代码版本和结果。主实验始终是五设备联合输出，单设备拟合只用于检查表示能力。以下判断标准用于决定研究方向，不预先承诺改善幅度。

| 顺序 | 具体工作 | 固定的比较条件 | 产物与判断 |
|---|---|---|---|
| 1 | 从现有预测重建每户时间轴，统一指标和基线 | 原 checkpoint、原保存预测 | 每户指标表、原始/截零波形、覆盖率。建立可复算的比较起点 |
| 2 | 检查观测标记、严重负残差、训练功率异常值 | 保留现有 CSV，另做观测有效行敏感性统计 | 事件级数据审计表，确认哪些测量允许使用强物理约束 |
| 3 | 给现有 LSTM 反演加入硬非负投影和全过程记录 | 原权重、相同种子、初始化、步数、重启 | 初始/最终/目标最优历史候选比较。判断负功率通道造成多少误差 |
| 4 | 建立精确加法加设备状态/事件模板的小型正对照 | 干净五路混合、相同验证窗与预算 | 对比加法加边界、模板初始化 0 步、模板加反演。检验设备结构是否帮助归属 |
| 5 | 训练并筛选具有设备区分能力的深度波形模型 | 训练住户拟合，验证住户评分 | ON/OFF 重建、跨设备拟合矩阵、干净混合五路联合反演 |
| 6 | 逐级加入未知背景 | 固定已筛选的设备模型 | 无背景、常数背景、受限可变背景对照，从合成干扰到真实总表 |
| 7 | 完整验证后冻结方案，跨住户正式评估 | 相同数据、指标与计算预算 | 与初始化、经典反演和直接预测基线比较，多种子与运行时间 |

### 第 1 步先完成可信评分

使用 `(dataset, house, sequence_id, unix_time)` 对齐，平均同一时间点的重叠预测。记录未覆盖时段，不补成模型预测。每户报告逐电器 MAE、ON/OFF MAE、F1、Precision、Recall、AP、能量误差，以及各项支持数。能量误差和 AP 的具体定义对齐原项目。住户间先分别报告，再明确宏平均或时间点加权方式。

从已有 `estimated_appliances.npy` 和 `window_indices.npy` 复算，不需要重新优化。加入全 OFF 和训练均值功率基线。功率指标保留原始结果，同时单列截零敏感性结果。波形至少有每设备 ON 事件、OFF 较多的长时段、总表重模拟和五路物理相加。

### 第 2 步确定物理约束适用范围

按训练住户统计功率极值和分位数。逐一查看最严重的负残差事件及其原始观测标记，追踪生成 CSV 的聚合方式和源通道。对全部已覆盖行、仅真实观测行分别评分。若需要修订数据，保留原版并建立有版本的新数据，所有比较方法统一使用同一版。真实测试分表只可用于最终评分和明确标记的诊断。

### 第 3 步实施最小求解器修正

首先只加入硬非负投影，观察该因素的独立影响。之后再测试经过训练数据审计的设备上界。

\[
\mathbf P^{(k+1)}=
\Pi_{\mathcal U}\bigl[\operatorname{AdamStep}(\mathbf P^{(k)},\nabla_{\mathbf P}L)\bigr],
\qquad\mathcal U=\{\mathbf P:\mathbf P\ge0\}.
\]

候选采用 z-score 时，先转换到瓦特空间投影，再转换回标准化坐标。非负瓦特对应的标准化下界是 \(-\mu_i/\sigma_i\)。

每个窗口记录第 0 步、最终步，以及根据完整逆目标保留的历史最佳候选。使用同一条选中轨迹解释收敛图。每个迭代预算下都评价分表指标。历史最佳目标仍可能选错设备，需要通过验证结果检验。首先比较非负修正，再独立比较历史候选保留，避免混合两个改动。

### 第 4 步用简单设备结构验证可恢复性

训练分表提供每种设备的 OFF、典型功率水平和事件形状。构造明确包含零输出的有限状态或事件模板集合。在干净混合上联合拟合五路模板的幅值、位置或激活。模板是研究控制基线，后续深模型需提供额外可测收益。

例子中微波炉应产生持续一段时间的脉冲。模板约束把整段事件一起调整，减少逐点自由分配的空间。先检验单个事件、两设备重叠，再到五设备混合。另保留代表 OFF 与稀有事件比例的完整验证样本。若只有少数挑选事件有效，应记录其适用范围。

### 第 5 步建立深度设备先验

候选参数化为

\[
\mathbf p_i=G_{\theta,i}(\mathbf z_i),\qquad
F_\theta(\mathbf z_1,\ldots,\mathbf z_5,\mathbf b)
=\sum_iG_{\theta,i}(\mathbf z_i)+\mathbf b.
\]

Stage 1 用训练分表拟合波形模型，可从有瓶颈的编码器/解码器开始。分别评价各设备的 OFF、短事件和长周期。Stage 2 固定解码器，优化五组代码及受限背景。设备可以共享主干并有不同输出头，单独模型也可作为消融。

先完成两项检查再用于总表反演。

1. 正确设备模型能否表达验证住户的真实 ON 波形，重建误差和 ON 误差多大？
2. 给同一个真实设备事件，分别优化五种设备模型，正确身份是否比错误身份有稳定优势？同时看绝对误差，避免把五个都拟合不好当成区分成功。

旧项目已发现独立解码器、简单事件平衡和短步 inverse-aware loss 仍会混淆。应参考 `../Neural-adjoint inverse optimization/experiment_log.md` 的 E1–E4 和 F1 记录。F1 的真实验证 Macro-F1 从 0.296 降到 0.093。新的深模型应先通过设备区分与初始化后反演检查，再考虑复杂外层训练。

### 第 6 步加入受限背景与测量误差

建议研究目标为

\[
\min_{\mathbf z_1,\ldots,\mathbf z_5,\,\mathbf b\ge0}
\frac1T\sum_t\rho_\delta\!\left(
\frac{\sum_iG_{\theta,i}(\mathbf z_i)_t+b_t-y_t}{s_y}
\right)
+\lambda_{\rm app}\sum_iR_i(\mathbf z_i)
+\lambda_{\rm bg}R_{\rm bg}(\mathbf b).
\]

其中 \(s_y\) 是训练功率尺度，\(\rho_\delta\) 是 Huber 损失，\(R_i\) 限制设备代码或事件状态，\(R_{\rm bg}\) 限制背景。各项按窗口长度归一化，尺度与权重写入配置。

先测试每窗一个非负常数背景。随后单独测试少量分段参数或带变化惩罚的背景。背景幅值也需限制，因为只惩罚总变分时任意常数背景均无代价。非目标电器可能有突发事件，平滑背景只作为基线。过于自由的背景可作为负对照，展示其吸收目标事件的风险。

按无背景、常数干扰、训练来源的时变背景、真实总表逐级测试。真值分表参与构造的合成总表必须标记为控制实验。推断时只输入构造后的总表，原始来源分量只用于评分。

### 第 7 步定义成功与失败

成功证据需要在相同数据口径下，展示逆优化相较同一模型的 0 步初始化改善逐电器识别。同时报告设备 MAE、ON MAE、F1、误报、能量误差及运行时间，查看稀有设备是否被牺牲。与直接预测基线比较时，统一数据划分、标签、阈值和后处理。

若能找到好候选但总表目标排名选错，优先修改候选排序与先验。若所有候选都差，检查表示能力和搜索。若正确及错误设备模型都能生成同一波形，需要更强设备信息或承认该观测下的不确定性。单通道测量存在无法唯一分解的组合，实验结论应写明适用条件。

## 5. 数据使用与实验记录

| 分区 | 住户 | 用途 |
|---|---|---|
| Train | UK-DALE 1；REFIT 3、5、9、11 | 模型、归一化、模板、背景统计与候选初始化资源 |
| Validation | UK-DALE 5；REFIT 2 | 逐步消融、迭代预算、正则权重和判定设置 |
| Test | UK-DALE 2；REFIT 20 | 方案冻结后的跨住户评分 |

同一套划分贯穿两阶段。每个新窗口的候选优化使用该窗口总表，权重保持固定。任何以分表真值选择初始化、背景或停止点的运行均记录为 oracle 诊断。历史测试曲线已被查看，应如实记录开发历史，并在可用时增加未用于方法选择的独立住户确认结果。

每次运行记录代码版本、数据版本、住户及时间索引、checkpoint、训练统计量、模型结构、种子、设备、参数量、配置、耗时、覆盖率、指标和波形。远程实现与实验遵循根目录 `training_device.md`。

## 6. 文献与本项目的对应关系

以下来源在上一轮审查中已打开核对。方法迁移到本项目属于待检验方案。

| 来源 | 对本项目的作用 |
|---|---|
| [Ren et al., NeurIPS 2020, Benchmarking Deep Inverse Models over Time, and the Neural-Adjoint Method](https://proceedings.neurips.cc/paper_files/paper/2020/file/007ff380ee5ac49ffc34442f5c2a2b86-Paper.pdf) | 冻结正向代理后优化输入，边界损失约束搜索区域。其方法没有建立 NILM 逐设备可辨识性保证 |
| [Kolter & Jaakkola, AISTATS 2012](https://proceedings.mlr.press/v22/zico12/zico12.pdf) | 加法设备状态模型与受限制的未知负荷联合推断，Fig. 5 提供未知成分消融参考 |
| [Kolter, Batra & Ng, NeurIPS 2010](https://proceedings.neurips.cc/paper/2010/file/7810ccd41bf26faaa2c4e1f20db70a71-Paper.pdf) | 学习设备基，再从总表优化设备表示，是经典 NILM 优化基线 |
| [Narayanaswamy et al., Interspeech 2020, Unsupervised Audio Source Separation Using Generative Priors](https://www.isca-archive.org/interspeech_2020/narayanaswamy20_interspeech.pdf) | 冻结源生成器，在精确加法下联合优化各源代码。最接近拟议的深度反演流程；音频谱损失不能直接视为已验证的功率损失 |
| [Lunz et al., SIAM Journal on Imaging Sciences 2021](https://doi.org/10.1137/20M1338460) | 学习正向算子时需关注逆问题中的梯度与优化轨迹，支持扰动诊断；论文理论条件不能直接套用为当前非线性 LSTM 的保证 |
| [Jayaram & Thickstun, ICML 2020](https://proceedings.mlr.press/v119/jayaram20a/jayaram20a.pdf) | 线性混合配生成先验，展示多种合理解释。NILM 仍需比较真实分表功率和状态 |
| [Frank & Ilse, 2020, Problems Using Deep Generative Models for Probabilistic Audio Source Separation](https://proceedings.mlr.press/v137/frank20a/frank20a.pdf) | 深先验可能缺少可用的平滑性或源区分能力，支持先验的独立质量检查 |
| [Brynjarsdóttir & O’Hagan, Inverse Problems 2014](https://mail.tonyohagan.co.uk/academic/abs/simmach.html) | 模型差异会偏置逆推，过于自由的差异项又与目标变量混淆，支持受限背景设计 |

## 7. 当前执行状态

| 工作 | 状态 |
|---|---|
| 现有代码静态审查、训练机保存结果复算、有限差分和小规模控制 | 已完成，见第 2 节 |
| 可复现的诊断脚本与输出归档 | 待整理；此前部分检查通过内存脚本运行，本文记录数值，原运行目录未新增这些诊断产物 |
| 第 1 步完整评估实现 | 下一项实施工作 |
| 第 2–7 步 | 待按顺序实施和记录 |

下一项从现有保存预测建立可信的分户指标和波形报告，然后完成数据审计，再独立测试硬非负反演。更大 LSTM、更多更新步数或更高重启次数应等到目标与设备约束验证后再评估。

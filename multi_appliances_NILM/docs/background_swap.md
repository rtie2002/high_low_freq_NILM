# Background swap：第一项消融实验

## 改了什么

模型结构、损失函数、验证集和测试集都不变。只改变 50% 训练样本的构造方式。

真实窗口可写成：

\[
x_a(t)=\sum_{i=1}^{5} y_{a,i}(t)+b_a(t)
\]

其中 `y` 是五个目标电器，`b` 是没有子表标签的残差背景：

\[
b_a(t)=\max\left(x_a(t)-\sum_i y_{a,i}(t),0\right)
\]

## 之前：full random mix

五个电器和背景分别来自不同窗口：

\[
x_{full}(t)=y_{1,kettle}(t)+y_{2,fridge}(t)+\cdots+y_{5,microwave}(t)+b_6(t)
\]

这能制造更多组合，但一次同时改变了电器与背景，无法单独检验背景依赖。

## 现在：background swap

保留窗口 `a` 的全部电器功率和 ON/OFF 标签，只换成窗口 `j` 的背景：

\[
x_{swap}(t)=\sum_{i=1}^{5}y_{a,i}(t)+b_j(t)
\]

因此：

\[
y_{swap}=y_a,\qquad z_{swap}=z_a
\]

如果冰箱预测因为背景从 100 W 变成 300 W 就发生明显变化，说明模型学到了背景捷径，而不是稳定的冰箱特征。

## 当前配置

新实验使用 `config/models/multinilm_k4.yaml`；原来的 `multinilm_fractional_relational.yaml` 保留 `mode: full`，用于复现旧基线。

```yaml
training:
  random_mix:
    enabled: true
    mode: background_swap
    prob: 0.5
```

- 50%：原始真实窗口。
- 50%：同一组电器标签 + 另一个合法窗口的残差背景。
- 只用于训练；验证和测试始终使用真实数据。
- 当前没有增加 consistency loss，保证这次只比较采样方法。

## 新增诊断输出

普通 `metrics.csv` 新增：

- `false_positive_rate`：真实 OFF 样本中被错误判为 ON 的比例。
- `false_negative_rate`：真实 ON 样本中被错误判为 OFF 的比例。
- `false_positive_energy_wh`：误报部分累计预测的电量。
- `false_event_count`：完全不与真实 ON 重叠的预测 ON 事件数。

另外保存 `background_fpr.csv`，分别统计残差背景 `[0,100)`、`[100,200)`、`[200,400)`、`[400,800)`、`[800,∞)` W 下的 FPR。

第一项判断标准：比较旧的 `x` 与新实验时，优先看冰箱在 200–400 W 和 400–800 W 背景下的 FPR 是否下降，同时确认微波炉 AP、recall 和 FNR 没有恶化。

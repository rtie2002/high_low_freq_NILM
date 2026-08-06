# Appliance head: default vs SGN

Source: `model/MultiNILM.py` → `ApplianceHead`  
Yaml: `architecture.head_style: default | sgn`

---

## `head_style: sgn`（当前 fractional 试用）

对齐 Shin **Subtask Gated Network** 精神（双塔 + 相乘门控），接在共享 TCN 特征 `h` 上：

```text
共享 TCN → h (B, C, T)
              │
    ┌─────────┴─────────┐
    ▼                   ▼
 power_tower         state_tower
 6×Conv (k=10,8,6,5,5,5
   ch=30,30,40,50,50,50)
 + Conv1d 50→1024 + ReLU
 + Conv1d 1024→1
    │                   │
 power_raw          logits → p
    └──── gate(p) · ────┘
              │
   power = gate·power_raw + (1−gate)·off_norm
```

与论文附录的差异（为适配我们的 **seq2seq 全长 T**）：
- 不用把窗口展平后的 FC→32；最后两层用 **Conv1d k=1** 当“FC”，输出仍是 `(B,1,T)`。
- 输入是共享 TCN 的 `h`，不是从头训练整条从原始功率开始的双塔。

每电器仍是 **独立** 的一对 towers（5 套）。

---

## `head_style: default`（旧）

```text
h → local_decoder → F → power_head / state_head (1×1) → gate
```

---

## 注意

- SGN 头参数量大（5 电器 × 2 塔）；跨屋可能更容易过拟合。
- fractional yaml 里暂关 `cross_appliance`（双塔已够重）。

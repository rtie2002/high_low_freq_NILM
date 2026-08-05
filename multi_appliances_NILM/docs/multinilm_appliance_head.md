# MultiNILM appliance head（分塔）

Source: `model/MultiNILM.py` → `ApplianceHead`  
Fractional yaml: `split_task_bodies: true`, `detach_gate: true`

---

## 当前设计（SAMNet-lite 分塔）

```text
共享 TCN 输出 h  (B, C, T)
        │
        ├──────────────────────────────┐
        ▼                              ▼
  state_body(h)                   power_body(h)
  (+res, dropout)                 (+res, dropout)
        │                              │
       Fs                             Fp
        │                              │
        │                    [可选 CrossApplianceDistill 只混 Fp]
        │                              │
   state_head                     power_head
   Conv1d C→1                     Conv1d C→1
        │                              │
   logits → p=σ                   power_raw
        │                              │
        └──── gate(p) ─── × ───────────┘
                         │
              power = gate·power_raw + (1−gate)·off_norm
```

- **共享停在 TCN**：分类 / 回归各有一小栈（深度仍由 `head_local_layers` 控制）。
- **门控保留**；**不做** class↔power concat / 双向互喂。
- **`detach_gate: true`**：gate 用 `p.detach()`，power MSE **不**回传进 state 塔（只靠 BCE 训分类）。

### Cross-appliance

PAD-lite 只混合 **power 路径 `Fp`**；`state_body` 始终从干净的 `h` 读，避免串扰污染分类。

---

## 对照：旧版（`split_task_bodies: false`）

```text
h → local_decoder → F
         ├─ power_head(F)
         └─ state_head(F)
power = gate(p)·power_raw + …
```

同一 `F` + soft gate → power 梯度会改写分类（你见过的 val BCE↑）。

---

## YAML

| Key | 作用 |
|-----|------|
| `split_task_bodies` | `true` = 分塔 |
| `detach_gate` | `true` = gate 截断对 state 的梯度 |
| `head_local_layers` | 每塔深度（默认 2） |
| `gate_mode` | `soft` / `hard` / `soft_train_hard_eval` |

---

## 参数量

分塔约 **×2** 每电器 head body（5 电器 × 两栈）。主干 TCN 不变。

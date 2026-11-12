# Neural-adjoint NILM

项目只有两个工作流程：训练电器波形 CNN，以及用总表功率做逆优化。

```text
models/cnn.py          网络结构、encode、decode、forward
stage1_forward/        前向模型训练、训练数据、Stage-1 losses
stage2_inverse/        冻结 decoder 后的 latent inverse optimisation
configs/               model.yaml、stage1.yaml、stage2.yaml
utils/                 共用的 CSV 读取、指标和绘图
docs/                  数学、架构、研究路线和方法说明
references/            仅供阅读的第三方参考源码，不参与运行
runs/                  模型权重与实验结果
```

## 修改模型和参数

编辑 `models/cnn.py` 定义网络；在 `configs/model.yaml` 选择其结构。训练和反演参数分别保存在 `configs/stage1.yaml` 与 `configs/stage2.yaml`。

| 修改内容 | 配置位置 |
| --- | --- |
| 共享 CNN / 各电器独立 CNN、网络宽度、latent 通道 | `model.yaml` 的 `model` |
| 窗口长度、步长、共同功率尺度 | `data` |
| MSE / ON-OFF 平衡 / inverse-aware 损失 | `loss.name` |
| 学习率、weight decay | `optimizer` |
| batch size、epochs、early stopping、梯度裁剪 | `training` |
| 可微逆优化训练的步数、步长、权重 | `inverse_aware` |
| Stage 2 步数、学习率、初始化、prior | `stage2.yaml` 的 `inference` |
| 波形数量和分辨率 | `evaluation` |
| CSV 路径、电器顺序、归一化统计和评估规则 | `dataset` |

数据路径相对于 `configs/stage1.yaml` 所在目录解析。先配置 `dataset.data_root` 和 `dataset.normalization_file`。训练只使用训练集；验证集普通 MSE 决定最佳 checkpoint。

## 运行

从项目目录运行：

```powershell
python -m models.inspect_model
python -m stage1_forward.train --config configs/stage1.yaml --run-dir runs/forward_cnn
python -m stage2_inverse.inverse --config configs/stage2.yaml --checkpoint runs/forward_cnn/best.pt --run-dir runs/inverse
```

需要先快速检查真实数据时：

```powershell
python -m stage1_forward.train --smoke --run-dir runs/smoke
python -m stage2_inverse.inverse --checkpoint runs/smoke/best.pt --run-dir runs/inverse_smoke --max-windows 2 --steps 2
```

根目录的 `train.py`、`inverse.py`、`cnn.py` 和 `data.py` 是旧命令兼容入口；新研究代码应使用上面的模块命令。

## 阅读顺序

先看 `models/cnn.py` 的网络，再看 `stage1_forward/train.py` 的 `run_epoch()`。它按“forward → loss → zero_grad → backward → step”排列。接着看 `stage2_inverse/inverse.py` 的 `solve_windows()`，其中 Adam 只接收 latent codes。

`stage2_inverse/inverse.py` 的 `solve_windows()` 会冻结 decoder，初始化 latent codes，用 Adam 根据总表误差更新 codes，并为每个窗口保留最优候选。背景功率是非负常数；held-out 电器标签只进入末尾的评估和绘图。

训练数据为共同尺度下的 `[B,5,T]`，latent 为 `[B,5,Z,ceil(T/8)]`。训练按 CSV sequence ID 切窗；逆优化沿用 loader 的时间连续性规则。

## 输出

训练保存 `best.pt`、`history.csv`、`loss_curve.png` 和 `run_manifest.json`。完整配置记录在 manifest 的 `configuration` 中。逆优化保存预测、指标、背景诊断和波形。

每个实验使用独立输出目录。已有输出目录默认会报错，避免意外覆盖；只有明确加入 `--overwrite` 才会替换其中的文件。历史实验记录位于 `forward_training/experiment_log.md`，数学说明位于 `docs/mathematics.md`。第三方参考源码位于 `references/`，不会被训练或推断代码导入。

本机 Anaconda 的 OpenMP 库冲突时，本次本地验证在进程内使用了 `MKL_THREADING_LAYER=SEQUENTIAL`，没有修改项目算法或系统设置。远程运行前阅读 `training_device.md`。

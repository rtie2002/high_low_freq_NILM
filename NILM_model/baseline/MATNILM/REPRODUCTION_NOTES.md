# MATNILM Reproduction Notes

This note summarizes the difference between the MATNILM paper setting and the released code in this folder. It also records the main problems observed when trying to reproduce the REDD results.

## Short Conclusion

The released MATNILM repository is useful as a reference implementation, but it is not a complete clean reproduction package for the paper results.

The main reason is that the paper setting and the released default code are not fully aligned. In particular, the paper describes REDD experiments with `inputLength=864` and `outputLength=64`, but the released code defaults to `outputLength=864`. The original code also produced a shape mismatch when running with `outputLength=64`, so the paper setting could not be run directly without code changes.

## Paper Setting vs Released Code

| Important point | Paper setting | Released code | Why it matters |
|---|---|---|---|
| Output window | REDD uses `outputLength=64` | Default is `outputLength=864` | The default command does not match the paper |
| S2 data | One day training, one day validation, house 1 testing | `train_small.pkl`, `val_small.pkl`, `test_small.pkl` are provided | S2 data is available |
| S3 data | S2 plus sample augmentation | `--dataAug` loads `REDD_pool.pkl`, `poolx.pkl`, and `offduration.pkl` | S3 data is available, but training can be unstable |
| Random seed | Not clearly specified | No fixed seed in code | Results can change between runs |
| Environment | Paper used an older PyTorch setup | Current runs may use newer PyTorch/CUDA | Results may differ from the paper |

## Data Files

The processed REDD data is stored in two forms.

| File | Type | Meaning | Used in S2 | Used in S3 |
|---|---|---|---:|---:|
| `train_small.pkl` | timestamped DataFrame list | Real training sequence with `main` and appliance labels | Yes | Yes |
| `val_small.pkl` | timestamped DataFrame list | Real validation sequence with `main` and appliance labels | Yes | Yes |
| `test_small.pkl` | timestamped DataFrame list | Real test sequence with `main` and appliance labels | Yes | Yes |
| `REDD_pool.pkl` | nested Python list | Target appliance activation snippets | No | Yes |
| `poolx.pkl` | nested Python list | Extra/background load snippets | No | Yes |
| `offduration.pkl` | nested Python list | OFF-gap duration statistics | No | Yes |
| `onList.pkl` | nested Python list | ON interval metadata | No | No direct use |
| `offList.pkl` | nested Python list | OFF interval metadata | No | No direct use |
| `onduration.pkl` | nested Python list | ON-duration metadata | No | No direct use |
| `out.pkl` | nested Python list | Extracted signal snippets/profile data | No | No direct use |

## Model Pipeline

### S2 Pipeline

```text
train_small.pkl
    -> sliding windows
    -> MATNILM model
    -> regression loss + classification loss
    -> validation on val_small.pkl
    -> test on test_small.pkl
```

S2 does not use the augmentation files.

### S3 Pipeline

```text
train_small.pkl
    -> sliding windows
    -> randomly inject snippets from REDD_pool.pkl
    -> randomly add extra/background loads from poolx.pkl
    -> use offduration.pkl for synthetic OFF gaps
    -> MATNILM model
    -> regression loss + classification loss
    -> validation on val_small.pkl
    -> test on test_small.pkl
```

S3 uses augmentation only during training. The validation and test sets remain real data.

## Observed Training Problem

During training with the paper-style command:

```powershell
python main.py --dataAug --batch 32 --lr 0.001 --inputLength 864 --outputLength 64 --subName redd_s3_paper
```

the validation metrics did not converge properly. The log showed that epoch 0 had the best validation result, then the F1 score collapsed to `0.0` for most appliances.

This suggests the classification branch is predicting mostly OFF.

## Log-Based Result Summary

The S2 run with the paper-style window setting:

```powershell
python main.py --batch 32 --lr 0.001 --inputLength 864 --outputLength 64 --subName redd_s2_paper
```

does not show normal convergence.

| Epoch range | Observed behavior | Interpretation |
|---|---|---|
| Epoch 0 | Some useful result appears. Fridge and washer dryer have non-zero F1. | The model is not completely random at the start. |
| Epoch 1 onward | F1 becomes `0.0` for all appliances. | The classifier likely predicts OFF for nearly all samples. |
| Later epochs | MAE fluctuates but does not clearly improve. | Training does not recover after the collapse. |
| Early stopping | Best checkpoint is effectively from the first epoch. | Later training makes validation behavior worse. |

Example validation trend:

| Appliance | Epoch 0 MAE | Later typical MAE | Epoch 0 F1 | Later F1 |
|---|---:|---:|---:|---:|
| dish washer | about `29` | about `38-51` | `0.0` | `0.0` |
| fridge | about `30` | about `60-62` | about `0.79` | `0.0` |
| microwave | about `26` | about `23-36` | `0.0` | `0.0` |
| washer dryer | about `253` | about `345-380` | about `0.74` | `0.0` |

This means the model is not totally unable to learn, because epoch 0 has some signal. However, the learning is not stable. After the first epoch, the useful on/off detection disappears.

For NILM, this is suspicious because the model should normally learn at least some repeated appliance patterns from the training data. Instead, the classifier collapses and the model behaves like it is predicting mostly OFF states.

The most likely cause is the combination of:

| Suspected cause | Why it matters |
|---|---|
| `outputLength=64` creates a sparse target region | Many center-64 target windows contain no ON event. |
| Unweighted `BCELoss` | OFF samples dominate, so predicting OFF becomes an easy solution. |
| Regression is multiplied by classification output | Once classification predicts OFF, power prediction is also suppressed. |
| Dataset target slicing is inconsistent | The released dataset code computes the center target range but originally returns the full 864-sample target. |
| Validation/train code is not clean | Loss logging and eval mode handling make the trend harder to trust. |

Therefore, the suspected failure mode is:

```text
paper-style 64-output training
    -> many target labels are OFF
    -> classifier learns the all-OFF solution
    -> F1 drops to zero
    -> regression is gated down
    -> MAE/SAE stop improving
```

This suggests a problem in the released training pipeline, not just a normal model training curve.

## Suspected Cause of Collapse

The collapse is suspected to be caused by the interaction between the `outputLength=64` setting, class imbalance, and the model's on/off gate.

With `inputLength=864` and `outputLength=64`, the model receives a long input window but is trained/evaluated only on the center 64 samples. Since NILM appliance activations are sparse, many center-64 target windows contain no appliance activation. This makes the classification target highly imbalanced toward OFF.

The code uses a normal unweighted binary cross-entropy loss:

```python
criterion_c = nn.BCELoss()
```

Because most target points are OFF, the classification branch can reduce its loss by predicting OFF for most samples. Once this happens, the F1 score becomes very poor because F1 depends on detecting ON events. If the model predicts all OFF, recall for ON events becomes zero, so F1 also becomes zero.

This problem is more serious because the regression output is multiplied by the classification probability. Therefore, when the classifier predicts OFF, the predicted power is also suppressed.

In short:

```text
short 64-sample target
    -> many target windows are mostly OFF
    -> classifier learns to predict OFF
    -> F1 becomes 0 because ON events are missed
    -> regression output is gated down
    -> MAE/SAE stop improving
```

This does not mean that `outputLength=64` is impossible. The paper uses this setting. The issue is that the released code appears unstable under this setting because the loss and gating design make OFF-collapse easy.

## Why F1 Collapse Breaks The Model

In `modules.py`, the model predicts both appliance power and appliance on/off probability. The final regression output is multiplied by the classification probability:

```python
dc = torch.sigmoid(self.fc_dc(d_cc))
dr = self.fc_dr(d_rr) * dc
```

This means the classification output acts as a gate.

| Classification output | Effect on regression |
|---|---|
| close to `1` | predicted appliance power can pass through |
| close to `0` | predicted appliance power is forced close to zero |

Therefore, if the classifier learns to predict OFF for most samples, the regression output also collapses. This matches the observed behavior: F1 becomes `0.0`, and MAE/SAE do not improve.

## Code Issues Found

| Issue | Code location | Effect |
|---|---|---|
| Default `outputLength=864` | `main.py` argument parser | Does not match paper REDD setting of `64` |
| Original `outputLength=64` shape mismatch | training/evaluation path | Paper setting could not run directly |
| Classification BCE is unweighted | `main.py`, `nn.BCELoss()` | Easy for classifier to predict OFF because NILM data is mostly OFF |
| Regression is gated by classification | `modules.py` | Classification collapse also suppresses power prediction |
| Training loss is not reset per epoch | `main.py`, `iter_loss` list | Logged train loss is misleading |
| No fixed random seed | global training setup | Repeated runs may differ |
| Validation during training does not explicitly call `model.eval()` | training/evaluation flow | Dropout can make validation noisy |
| README lacks exact reproduction details | `README.md` | Does not define exact S1/S2/S3 commands or hyperparameters |

## Logical Issues Found

The following issues are likely to affect reproduction or make the training logs misleading.

| Logical issue | What the code does | Why it matters |
|---|---|---|
| Dataset ignores `outputLength` target | `SubSet.__getitem__` computes `out_begin` and `out_end`, but returns the full input-length target | Suggests the released code was mainly written for `outputLength=864`, not the paper's `outputLength=64` |
| Classifier can learn all-OFF | Uses unweighted `BCELoss` for on/off labels | NILM data is mostly OFF, so predicting OFF can become an easy solution |
| Regression depends on classifier gate | Power output is multiplied by on/off probability | If F1 collapses, regression prediction is also suppressed |
| Validation may run in train mode | Validation does not clearly call `model.eval()` before prediction | Dropout can stay active and make validation metrics noisy |
| Train loss is averaged incorrectly | `iter_loss` is not reset at the start of each epoch | Logged train loss is not the true per-epoch loss |
| Training function uses global model object | `train()` receives `t_net`, but validation calls `evaluateResult(net, ...)` | Code depends on a global variable and is easier to break |

The most important target-slicing issue is:

```python
out_begin = in_begin + int((self.inLen-self.outLen)/2)
out_end = out_begin + self.outLen

X = self.mains[in_begin:in_end, :]
Y = self.outputs[in_begin:in_end, :]
```

The code computes the intended output range but does not use it. For a center-output setting, the target would logically be:

```python
Y = self.outputs[out_begin:out_end, :]
```

This is one reason the original code failed when running with the paper's `outputLength=64` setting.

## Recommended Interpretation

The public MATNILM code should be treated as a partial/reference implementation, not as a fully reliable reproduction package.

A fair statement is:

> The released MATNILM code provides the model architecture, processed REDD data, and S3 augmentation files. However, it does not directly reproduce the paper setting without modification, because the default window configuration differs from the paper and the original code fails under the paper's `outputLength=64` setting. In addition, the training pipeline can suffer from classification gate collapse, which causes unstable F1 and regression metrics.

## Final Observed MATNILM Result

A later MATNILM run stopped at epoch 38 after the early-stopping counter reached `30/30`. During the late training epochs, validation F1 collapsed to `0.0` for all four appliances:

| Late epoch behavior | Observation |
|---|---|
| Epoch 35-38 validation F1 | `0.0` for dishwasher, fridge, microwave, and washer dryer |
| Late train loss | about `0.406-0.417` |
| Late validation loss | about `326k-329k` |
| Early stopping | triggered after 30 non-improving epochs |

After early stopping, the code loaded the best saved model and reported the following final metrics.

### Final Validation Metrics

| Appliance | MAE | SAE | F1 |
|---|---:|---:|---:|
| dish washer | `27.182` | `27.762` | `0.055` |
| fridge | `38.778` | `19.760` | `0.702` |
| microwave | `17.702` | `16.715` | `0.228` |
| washer dryer | `13.263` | `6.878` | `0.938` |

### Final Test Metrics

| Appliance | MAE | SAE | F1 |
|---|---:|---:|---:|
| dish washer | `22.040` | `21.817` | `0.293` |
| fridge | `38.094` | `28.373` | `0.785` |
| microwave | `21.300` | `17.657` | `0.268` |
| washer dryer | `45.663` | `40.434` | `0.544` |

### Interpretation Of This Result

This is not a successful reproduction of the MATNILM paper result.

The model did not completely fail for every appliance. It learned some useful signal for fridge and washer dryer, especially on validation. However, dishwasher and microwave remained poor, and the final test result was inconsistent across appliances. The late-epoch validation F1 collapse to `0.0` also shows that the training process is unstable.

The result should be described as:

```text
partial learning, but failed reproduction
```

The most important evidence is:

| Evidence | Meaning |
|---|---|
| Dishwasher test F1 `0.293` | poor event detection |
| Microwave test F1 `0.268` | poor event detection |
| Fridge test F1 `0.785` | model learns some repeated pattern |
| Washer dryer validation F1 `0.938`, test F1 `0.544` | validation/test behavior is inconsistent |
| Late training F1 `0.0` for all appliances | classifier/gating collapse during training |

### Data Reproduction Limitation

This run also cannot be claimed as the exact paper data setting.

The repository provides processed REDD pickle files, but the full raw-data preprocessing pipeline is not transparent enough to verify every paper detail, such as exact raw timestamps, raw house selection, preprocessing decisions, window construction, and S2/S3 sampling behavior. Therefore, even though the code can run on the provided processed files, the experiment should not be presented as an exact reproduction of the paper dataset.

Recommended wording:

> The MATNILM run on the released processed REDD data shows partial learning but does not reproduce the reported paper performance. The classifier becomes unstable during training, and final test F1 remains poor for dishwasher and microwave. Because the repository does not fully expose a raw-data preprocessing pipeline that verifies the exact paper splits and processed samples, this result should be treated as a failed or partial reproduction rather than an exact reproduction.

## Suggested Reproduction Checks

Run S2 first, without augmentation:

```powershell
python main.py --batch 32 --lr 0.001 --inputLength 864 --outputLength 64 --subName redd_s2_check
```

Then test S3:

```powershell
python main.py --dataAug --batch 32 --lr 0.001 --inputLength 864 --outputLength 64 --subName redd_s3_check
```

If S2 learns but S3 collapses, the augmentation pipeline is likely the main issue.

If both S2 and S3 collapse, the core model/training setup is likely not sufficient for reproducing the reported paper results.

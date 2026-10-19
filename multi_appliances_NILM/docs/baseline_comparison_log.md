# Baseline comparison experiment log

This log records the controlled Table V/VI baseline runs. All models use the
same five-week training split, validation houses, test houses, target labels,
and evaluation metrics. Model selection uses validation data only.

## Efficiency protocol

- Hardware: the same remote RTX 4090 for every model.
- Inference batch size: 64 for every model.
- Splits: UK-DALE house 2 and REFIT house 20, reported separately.
- Timing includes data loading, host-to-device transfer, forward inference,
  device-to-host transfer, and model-specific output reconstruction.
- Model loading, validation calibration, metrics, and plots are excluded.
- Reported efficiency fields are trainable parameters, checkpoint size,
  training time, end-to-end inference time, throughput, and peak allocated GPU
  memory. Best epoch is intentionally not reported.

## Scheduled run

- MART-NILM: reuse the completed seeds 2026, 2027, and 2028; do not retrain.
- UNet-NILM: train seeds 2026, 2027, and 2028 for the paper's 50 epochs and
  batch size 128, then evaluate both held-out test houses.
- MAT-Conv: run seeds 2026, 2027, and 2028 serially on the same RTX 4090 after
  UNet-NILM finishes.
- TransferNILM: run only after the MART-NILM, UNet-NILM, and MAT-Conv rows are
  complete. Its implementation is prepared as described below.

Results will be appended after the scheduled task finishes.

## TransferNILM implementation audit (2026-10-19)

- Source paper: Sun et al., *Transfer Learning for Multiappliance-Task
  Nonintrusive Load Monitoring*, IEEE Transactions on Instrumentation and
  Measurement, 2025, DOI 10.1109/TIM.2025.3541652.
- Official code: <https://github.com/Franky-Feng/transfer_learning_multi-appliance>,
  commit `57e51e20e6682446e0c5a928191491768871bb2c`.
- Model id: `transfer_multi_appliance`; implementation files are
  `model/TransferNILM.py`, `model/TransferNILM_loss.py`, and
  `config/models/transfer_multi_appliance.yaml`.
- Architecture check: the local five-head model has 7,829,258 parameters,
  exactly matching the official `BERT4NILM + 5 x CNNNetwork` construction.
- Restored the paper's state-conditioned power equation
  `power = raw_power * state_probability`. The previous OFF-normalized blend
  was deliberately removed because it was not part of the published method.
- Restored the official global mean `MSE + BCE` objective with state coefficient
  one, 480-sample windows, 240-sample stride, two shared Transformer blocks,
  one appliance-specific Transformer block, two heads, width 256, kernel 5,
  dropout 0.1, Adam at 1e-4, batch size 32, and 100 epochs.
- No MART-NILM augmentation, threshold calibration, learning-rate scheduler,
  gradient clipping, or mixed-precision training is applied.
- The shared five-week house split and CSV state labels remain unchanged so the
  eventual Table V comparison uses identical ground truth and held-out houses.
  This is therefore a faithful architecture/training reproduction under the
  paper's common benchmark, not a reproduction of the paper's random-window
  REFIT-to-UK-DALE/REDD transfer protocol.
- Verification completed: Python compilation, two model/loss smoke checks, and
  exact official/local parameter-count equality. Training has not yet started.

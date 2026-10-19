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
- MAT-Conv: deferred until the additional training machine is available.

Results will be appended after the scheduled task finishes.

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
- The checkpoint score is explicitly fixed to the released trainer's
  `normalized MAE - macro-F1` with unit coefficients. This avoids inheriting
  the shared runner's default MAE coefficient of seven.
- The shared five-week house split and CSV state labels remain unchanged so the
  eventual Table V comparison uses identical ground truth and held-out houses.
  This is therefore a faithful architecture/training reproduction under the
  paper's common benchmark, not a reproduction of the paper's random-window
  REFIT-to-UK-DALE/REDD transfer protocol.
- Verification completed: Python compilation, two model/loss smoke checks, and
  exact official/local parameter-count equality. Training has not yet started.

## TransferNILM final result (2026-10-21)

- Seeds 2026, 2027, and 2028 completed under the common five-week protocol.
  Checkpoint selection used the validation houses only; UK-DALE house 2 and
  REFIT house 20 were evaluated once for final reporting.
- Three-seed house-macro results were: UK-DALE MAE 26.7 +/- 1.5 W, SAE
  21.8 +/- 1.2 W, ON MAE 579.4 +/- 45.8 W, AP 0.681 +/- 0.075, and macro-F1
  0.49 +/- 0.02; REFIT MAE 29.7 +/- 0.9 W, SAE 21.7 +/- 1.0 W, ON MAE
  585.6 +/- 12.9 W, AP 0.440 +/- 0.000, and macro-F1 0.36 +/- 0.02.
- Two seeds produced no positive microwave prediction. The paper summary uses
  the standard zero-division convention (precision = 0) rather than omitting
  those seeds. This gives microwave F1 values of 0.01 +/- 0.02 on UK-DALE and
  0.00 +/- 0.00 on REFIT.
- Representative waveforms agree with the metrics. Fridge cycles are detected
  but their plateau amplitude is strongly underestimated and false low-power
  activations remain. Microwave events are almost entirely suppressed on both
  held-out houses, explaining the high microwave ON MAE and near-zero F1.
- The reproduced model has 7.829 M parameters, a 59.793 MiB checkpoint, and a
  mean training time of 16,493.8 s on the common RTX 4090. In comparison,
  MART-NILM uses 1.373 M parameters, a 5.335 MiB checkpoint, and 485.2 s of
  training time.
- `table_v_models.csv`, `table_vi_models.csv`, all final metrics, timing JSON
  files, and representative fridge/microwave waveform figures were copied to
  the local run directory. Tables V and VI in the manuscript were updated from
  these frozen summaries; MART-NILM remains the final proposed-model row.

## MART-NILM state-decoding audit (2026-10-23)

- A narrow audit reused the frozen Table-V MART-NILM seed-2026 checkpoint and
  evaluated the same validation and held-out house data with validation
  threshold calibration, microwave temporal cleanup, and evaluator-side hard
  state-to-power gating disabled. It did not retrain or select any checkpoint.
- UK-DALE house 2 changed from MAE/SAE/macro-F1/AP = 7.407 W / 5.287 W /
  0.879 / 0.935 to 8.598 W / 5.076 W / 0.791 / 0.935. REFIT house 20 changed
  from 11.133 W / 6.246 W / 0.697 / 0.742 to 15.883 W / 10.333 W / 0.598 /
  0.742.
- AP is identical because it uses the unthresholded state probabilities. The
  sizeable MAE and F1 changes show that the state-decoding overlay materially
  affects the current Table-V power and F1 rows. This one-seed audit is not a
  replacement for the reported three-seed table; it motivates a uniform final
  evaluation protocol before those rows are treated as architecture-only.

## Transfer multiappliance-task state-decoding audit (2026-10-23)

- The frozen Transfer seed-2026 checkpoint was re-evaluated under the same
  validation-only threshold sweep, microwave temporal cleanup, and hard state
  power mask used by the completed MART-NILM, UNet-NILM, and MAT-Conv runs.
  No model weights or checkpoint selection changed.
- UK-DALE house 2 changed from MAE/SAE/macro-F1/AP = 26.880 W / 21.802 W /
  0.487 / 0.671 to 15.622 W / 14.232 W / 0.604 / 0.671. REFIT house 20
  changed from 30.655 W / 22.843 W / 0.366 / 0.440 to 16.099 W / 11.850 W /
  0.448 / 0.440.
- The unchanged AP confirms that the overlay changes hard decoding and power
  masking, not probability ranking. Its large effect on Transfer confirms that
  it is a benchmark-level protocol choice rather than a MART-specific effect.

# Forward-model experiments

## 2026-11-23 background-cleaning training experiment

Change: apply the supplied background rule to training rows only. Validation
and test data remained raw, matching deployment conditions.

Training rows retained: 1,770,991 / 1,857,045 (95.37%).

Selected checkpoint: epoch 6.

Raw test results:

| Metric | Value |
| --- | ---: |
| MAE | 231.73 W |
| RMSE | 340.47 W |
| R2 | 0.393 |
| Pearson correlation | 0.732 |

The validation MSE remained much larger than the training MSE. The training
target standard deviation was 345.39 W, while the raw validation aggregate
standard deviation was about 878 W. The background rule is therefore retained
as an experimental training setting, not treated as a general solution for the
training-validation distribution gap.

Remote result folder:
`D:\Raymond\high_low_freq_NILM\inverse_optimization\results\forward_lstm_background_cleaning (2)`

## 2026-11-23 background cleaning with normal windows

Change: remove abnormal training rows using the same background rule, then
slide normal 512-sample windows within each original recording. Deleted rows
do not cause additional window rejection. Validation and test data remained raw.

Training windows: 5,977. The aggregate mean and standard deviation entering
training were 480.50 W and 574.23 W.

Selected checkpoint: epoch 14.

| Metric | Value |
| --- | ---: |
| MAE | 180.41 W |
| RMSE | 289.13 W |
| R2 | 0.562 |
| Pearson correlation | 0.799 |

Remote result folder:
`D:\Raymond\high_low_freq_NILM\inverse_optimization\results\forward_lstm_background_normal_windows`

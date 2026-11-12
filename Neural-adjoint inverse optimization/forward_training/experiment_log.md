# Inverse NILM baseline experiments

## 2026-11-11 — Constant-background inversion

- Code: `5f0a769`. RTX 4090, seed 2026, mixed UK-DALE and REFIT house-disjoint protocol.
- Stage 1: 100 epochs on 2,933 training windows. Epoch 92 selected by validation reconstruction MSE of 0.00006013 in scaled power, with validation reconstruction MAE of 10.739 W. Training loop time was 32.6 s.
- Stage 2: 150 Adam updates per window, learning rate 0.05, batch size 8. Nonnegative constant background per window. Inference took 41.5 s before timeline assembly and evaluation.
- Coverage: 945 validation windows covered 558,592 of 1,187,993 loader rows (47.02%). Metrics describe covered rows. The held-out test houses were not run.
- Validation macro metrics: MAE 60.809 W, ON MAE 722.843 W, SAE 42.161 W, F1 0.174, and AP 0.183. AP ranks by raw predicted watts. State decoding uses fixed watt thresholds and the paper's microwave duration rules and power gate.
- Aggregate reconstruction MAE before evaluation gating was 36.345 W. Low aggregate error coexisted with poor appliance attribution, particularly for the microwave (F1 0.045).
- Training-machine outputs under this project's `runs/`: `inverse_baseline_20261111_5f0a769_forward/best.pt` and `inverse_baseline_20261111_5f0a769_validation/`. The validation directory contains the metrics, prediction bundle, and five ON-event plots with paired 10× context plots per appliance.

## 2026-11-11 — Full rerun with separate held-out houses

- Code `8889037`, RTX 4090, seed 2026, and the same house split and inverse settings. Stage 1 stopped at epoch 98 and selected epoch 83 by validation reconstruction MSE (0.0000601734 in scaled power). Training loop time was 28.4 s. The same checkpoint was used for all reconstruction and disaggregation evaluations.
- Raw Stage 1 validation reconstruction highlighted a microwave ON MAE of 274.863 W and refrigerator OFF MAE of 23.581 W. Each split has 55 Stage 1 figures, including the original submeter and reconstructed waveforms, ON-event zooms, 10× context, and OFF-heavy long windows. Each inverse split has 50 waveform figures.

| Stage 2 split | Scored rows / available | Macro MAE (W) | Macro ON MAE (W) | Macro SAE (W) | Macro F1 | Macro AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 558,592 / 1,187,993 | 61.128 | 722.637 | 42.410 | 0.175 | 0.183 |
| UK-DALE House 2 | 593,408 / 604,131 | 36.497 | 968.901 | 25.811 | 0.156 | 0.285 |
| REFIT House 20 | 546,304 / 599,536 | 37.809 | 750.823 | 25.649 | 0.120 | 0.186 |

The inverse solver used only the aggregate as its observation. Test labels were used for evaluation. The results are single-seed estimates on covered rows. Average precision ranks by ungated predicted watts. Aggregate reconstruction MAE was 15.552 W for UK-DALE and 19.370 W for REFIT. The microwave F1 values of 0.140 and 0.051 show that a close meter fit still permits poor source attribution.

The training machine retained the complete run under `runs/rerun_20261111_8889037/`. The scheduled process exited with status 0.

## 2026-11-12 — Zero-background initialisation diagnostic

Used the epoch-83 checkpoint, 48 evenly sampled validation windows, and 150 Adam steps at 0.05. The observation was the sum of the five validation submeters, with background fixed at exactly zero. All rows use raw power and fixed watt thresholds. The training-derived initialisation averages encoded values over 128 training windows and the code time dimension, separately for each appliance and latent channel. The oracle uses the individual validation submeters only to initialise a diagnostic trajectory. Its results are excluded from deployable performance estimates.

| Initialisation | Meter RMSE, initial → selected (W) | Appliance MAE, initial → selected (W) | ON MAE, initial → selected (W) | Macro F1, initial → selected |
| --- | ---: | ---: | ---: | ---: |
| Random | 946.687 → 42.137 | 147.248 → 49.092 | 876.126 → 705.840 | 0.125 → 0.192 |
| Training mean | 515.112 → 47.827 | 38.095 → 40.629 | 860.762 → 688.966 | 0.005 → 0.404 |
| Oracle | 72.705 → 18.538 | 10.739 → 9.144 | 111.691 → 183.849 | 0.774 → 0.592 |

Training-mean initialisation improved final macro F1 over random despite higher meter RMSE. Oracle initialisation improved meter fit and overall appliance MAE during optimisation, while ON MAE increased and F1 fell. Microwave F1 fell from 0.965 to 0.415. In 13 of 48 oracle windows, meter RMSE fell while appliance MAE increased. This isolates a mismatch between the meter-only selection objective and appliance attribution even with no background.

Reproduce with `python diagnose_zero_background.py --checkpoint <epoch-83-best.pt> --run-dir <new-output-dir>`. The script records per-appliance and per-window diagnostics. No held-out test windows were used in this diagnostic.

## Measured-mains diagnostic

Same epoch-83 checkpoint, 48 evenly spaced validation windows, 128 training windows, seed 2026, and 150 Adam steps at 0.05. Use measured validation mains and fit a nonnegative constant background per window. Score raw outputs using the same fixed watt thresholds as in the zero-background diagnostic. The oracle uses validation appliance labels for its initial code and is for diagnosis only.

| Initialisation | Meter RMSE, initial → selected (W) | Appliance MAE, initial → selected (W) | ON MAE, initial → selected (W) | Macro F1, initial → selected |
| --- | ---: | ---: | ---: | ---: |
| Random | 1170.868 → 156.327 | 147.248 → 64.137 | 876.126 → 725.292 | 0.125 → 0.175 |
| Training mean | 897.558 → 134.942 | 38.095 → 57.058 | 860.762 → 675.396 | 0.005 → 0.321 |
| Oracle | 745.292 → 108.556 | 10.739 → 27.742 | 111.691 → 255.322 | 0.774 → 0.463 |

Stage 1 reconstruction had macro MAE 9.67 W on sampled training windows and 10.74 W on sampled validation windows. Macro ON MAE was 110.14 W and 111.69 W, respectively. Kettle and microwave ON MAE exceeded 180 W on both partitions. This sampled comparison does not show a large overall generalisation gap. For the oracle, 24/48 windows had lower meter RMSE and higher appliance MAE. Training-mean initialisation improved selected F1 relative to random, but its overall appliance MAE increased during inversion.

Add `--measured-mains` to the diagnostic command above. Outputs include per-appliance reconstruction and inversion metrics and the two windows with the largest changes in oracle appliance MAE. The held-out test split was not evaluated.

## Frozen-checkpoint window-length ablation

Compare 1024/512 and 512/256 (window/stride) on measured validation mains with the epoch-83 decoder, shared train-mean initialization, constant nonnegative background and 150 Adam steps at 0.05. Sample 24 long anchors per validation house and score both methods on their shared 24,576 unique rows per house. Merge overlapping predictions before scoring raw watts with fixed thresholds. The shorter-window run also samples 24 windows per house wholly outside the full 1024 coverage.

| House | Appliance MAE (W), 1024 → 512 | ON MAE (W), 1024 → 512 | Macro F1, 1024 → 512 |
| --- | ---: | ---: | ---: |
| REFIT 2 | 78.98 → 75.19 | 708.61 → 698.91 | 0.298 → 0.312 |
| UK-DALE 5 | 30.30 → 25.29 | 971.29 → 1019.84 | 0.223 → 0.218 |

CPU solve time was 26.7 s for 48 long windows and 67.5 s for 244 short windows, or approximately 0.56 and 0.28 s per window. The UK-DALE shared sample contains no dishwasher ON rows and only four microwave ON rows. Short windows expand coverage and reduce per-window computation, with mixed ON-event accuracy on matched rows. The held-out test split was not evaluated. Reproduce with `diagnose_window_length.py --checkpoint <epoch-83-best.pt> --run-dir <new-output-dir>`.

## Diagonal latent-prior ablation (dad999f)

RTX 4090 run at `D:/Raymond/high_low_freq_NILM/Neural-adjoint inverse optimization/runs/latent_prior_dad999f`. All trials use the same epoch-83 decoder, the original 1024/512 window/stride, train-mean code initialization, the constant background and 150 Adam steps at learning rate 0.05. Estimate appliance/channel code means and standard deviations from 128 evenly spaced training windows. The objective adds the mean squared standardized code distance times the listed weight, used both for the gradient and candidate selection. Evaluate the same 558,592 covered validation rows with the formal state-gated benchmark.

| Prior weight | MAE (W) | ON MAE (W) | Macro-F1 | AP | SAE (W) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 57.604 | 740.771 | 0.296 | 0.206 | 41.056 |
| 0.001 | 64.114 | 669.972 | 0.156 | 0.212 | 48.359 |
| 0.01 | 53.776 | 674.681 | 0.139 | 0.212 | 40.945 |

Both penalties lowered ON MAE, but substantially worsened Macro-F1. At 0.001, fridge recall fell from 0.595 to 0.061, and dishwasher false-positive rate rose from 0.078 to 0.428. In the matched microwave waveform at evaluation index 507555, the penalty recovered some pulses but missed others. Neither tested weight improved power and state attribution together. A single center per appliance/channel pools different operating modes; an explicit OFF/ON or state-conditioned prior is the next source-model comparison. No new held-out test scores were generated.

## State-conditioned latent-prior ablation (af78772)

RTX 4090 run at `D:/Raymond/high_low_freq_NILM/Neural-adjoint inverse optimization/runs/state_prior_af78772`. This compares a new state-conditioned prior with the same epoch-83 decoder, 1024/512 windows, training-mean initialization, constant nonnegative background, 150 Adam steps, learning rate 0.05, and 558,592 covered validation rows. No test labels or validation labels enter the solver.

Estimate separate OFF and ON latent-code means and standard deviations from 128 evenly spaced training windows. For each candidate decoded appliance block, use its eight-sample mean power to calculate a differentiable ON probability

`q_i,l = sigmoid((mean_block(p_i,l) - threshold_i) / 100 W)`.

The per-block state prior is

`R_state = mean[(1 - q_i,l) * d_off(z_i,l) + q_i,l * d_on(z_i,l)]`,

where each `d` is squared latent-code distance in the matching training-state standard deviations. Thus the solver never receives a test state label. Training blocks available to estimate OFF/ON statistics were kettle 16,174/210, fridge 9,518/6,866, dishwasher 15,327/1,057, washing machine 16,005/379, and microwave 16,254/130.

| State-prior weight | MAE (W) | ON MAE (W) | Macro-F1 | AP | SAE (W) | Aggregate MAE (W) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0, existing training-mean baseline | 57.604 | 740.771 | 0.296 | 0.206 | 41.056 | 35.027 |
| 0.0001 | 64.528 | 694.412 | 0.170 | 0.220 | 45.675 | 40.713 |
| 0.001 | 65.292 | 602.498 | 0.099 | 0.184 | 51.451 | 62.196 |
| 0.01 | 50.604 | 564.312 | 0.073 | 0.162 | 45.591 | 131.686 |

All three weights reduced ON MAE while degrading state attribution. At 0.01, fridge, dishwasher, and washing-machine F1 fell to 0.019, 0.003, and 0.004. The high weight effectively suppresses these appliances to obtain low overall MAE from abundant OFF rows. The lowest state weight also lowers Macro-F1. This simple differentiable state gate therefore does not solve the source-allocation problem. It confirms that a per-block diagonal Gaussian prior and candidate-power-derived state gate are insufficient to capture multi-modal appliance waveforms. No held-out test score was produced.

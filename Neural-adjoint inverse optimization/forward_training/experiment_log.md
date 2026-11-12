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

## 2026-11-12 — Frozen source-attribution diagnostics

Validation only, UK-DALE 5 and REFIT 2, from the mixed house-disjoint split. CPU, epoch-83 checkpoint from `rerun_20261111_8889037/forward/best.pt`, 1024/512 window/stride and 4000 W scaling. The saved zero-background manifest supplies seed 2026 and the original 48 starts. C uses deterministic chronological sampling and two deterministic initial codes. No model or inference interface edits.

From the inverse project directory, the exact local run was:

```powershell
$root="$env:TEMP/rerun_20261111_8889037"
& "$env:USERPROFILE/anaconda3/python.exe" -u diagnose_source_attribution.py $root "$root/source_attribution_verified"
```

**A. Saved Stage 1 reconstruction.** Read the original validation reconstruction NPZ copied from the training machine and reuse its power metrics, covering 558,592 rows. Derive raw threshold F1 and one-to-one onset recall using the benchmark's ±16 s tolerance, without predicted duration filtering or power gating.

| Appliance | MAE (W) | ON MAE (W) | OFF MAE (W) | F1 | Onset recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kettle | 5.378 | 213.350 | 2.676 | 0.911 | 0.872 |
| Fridge | 21.927 | 19.396 | 23.581 | 0.992 | 0.930 |
| Dishwasher | 8.496 | 44.926 | 5.331 | 0.432 | 0.094 |
| Washing machine | 10.415 | 59.899 | 7.489 | 0.591 | 0.149 |
| Microwave | 7.514 | 274.863 | 6.690 | 0.918 | 0.925 |

Fridge true-OFF mean prediction is 24.524 W. True-ON average is 89.614 W measured versus 70.637 W reconstructed. High state F1 coexists with a power bias. The microwave example retains a sustained activation and attenuates two single-sample spikes. See `reconstruction/validation/waveforms/microwave/on_01_t507674.png` and its context10x companion under the archive root. The mechanism of the fridge bias was not isolated.

**B. Oracle trajectory.** Reuse both previous three-initialisation comparisons. Replay only the original zero-background oracle with an Adam post-step observer because prior outputs did not save the intermediate codes. Adam 150 steps at 0.05, batches of eight, seed 2026, no prior. The table tracks current iterations, while the previous report selected each window's best meter-loss iteration.

| Step | Meter RMSE (W) | MAE (W) | ON MAE (W) | Macro-F1 | Mean appliance latent RMS movement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 72.705 | 10.739 | 111.691 | 0.774 | 0 |
| 3 | 59.272 | 7.852 | 103.426 | 0.850 | 0.105 |
| 25 | 39.459 | 6.463 | 116.231 | 0.725 | 0.320 |
| 50 | 28.901 | 7.111 | 133.229 | 0.682 | 0.381 |
| 150 | 18.539 | 9.143 | 183.862 | 0.592 | 0.491 |

Peak F1 and minimum ON MAE occurred at step 3 in this label-assisted replay. This does not establish a deployable stopping rule. At step 150 the microwave had the largest F1 drop (0.965 to 0.414), followed by kettle (0.886 to 0.603). Overall MAE was still lower than at initialization, so the decline is specific to ON accuracy and attribution metrics. The saved real-meter oracle comparison shows the same direction (meter RMSE 745.292 to 108.556 W, ON MAE 111.691 to 255.322 W, F1 0.774 to 0.463). See `B_oracle_convergence.png`, `B_oracle_trajectory.csv`, `B_per_appliance.csv`, and the prior `measured_mains_final/largest_mae_increase.png` waveform.

**C. Cross-appliance fit.** For each source, use 24 nonoverlapping validation windows with a labeled onset and at least two above-threshold ON samples. Pass the same isolated target to each appliance identity of the shared decoder. Optimize each code independently against target-waveform MSE, with no latent constraint or prior, using 150 Adam steps at 0.05 from each of two identical cross-identity starts (zero and the target's shared-encoder code). Select the lowest full-window MSE. The encoded start is only an offline fit diagnostic. Medians of per-window ON RMSE in W:

| True appliance | Kettle ID | Fridge ID | Dishwasher ID | Washer ID | Microwave ID |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kettle | 185.8 | 189.8 | 186.6 | 186.8 | 185.0 |
| Fridge | 9.5 | 3.9 | 7.2 | 5.5 | 7.4 |
| Dishwasher | 43.4 | 43.0 | 41.2 | 40.3 | 41.6 |
| Washing machine | 54.9 | 54.6 | 53.8 | 53.8 | 54.0 |
| Microwave | 259.6 | 260.6 | 257.4 | 257.7 | 256.9 |

For microwave, kettle and correct identities have median full-window RMSE 32.67 and 32.66 W. Kettle ON RMSE is at most 110% of correct-ID ON RMSE in all 24 microwave windows. Their median ON errors normalized by measured ON RMS are 0.241 and 0.240. Fridge shows the clearest identity advantage. Good wrong-ID fits directly establish overlap in the decoder's accessible output space on these samples. Near ties where both fits have substantial absolute error can also reflect shared reconstruction limitations. Two starts and a fixed finite budget do not certify global optima, and isolated-window fitting does not establish physical identifiability or full-cycle overlap. See `C_confusability.png`, `C_waveform_examples.png`, `C_normalized_on_rmse.csv`, `C_relative_to_correct.csv` and `C_window_errors.csv` in the output folder.

**Interpretation.** H1 is supported for specific signatures (e.g. short microwave pulses and fridge amplitude), H2 is directly supported by the late oracle degradation even without background, and H3 is supported by overlapping decoder fits. The diagnostics do not quantify how much each contributes to held-out performance.

## 2026-11-12 — D1–D3 frozen source-identifiability diagnostics

**Scope and reproducibility.** Validation only, using the same UK-DALE 5 and REFIT 2 validation partition, the frozen epoch-83 checkpoint at `rerun_20261111_8889037/forward/best.pt`, 1024-sample windows and 4000 W scale. No decoder training, no new prior, no change to the inverse objective, and no held-out test evaluation occurred. Outputs are under `C:/Users/Raymond Tie/AppData/Local/Temp/rerun_20261111_8889037/identifiability_diagnostics`.

```powershell
$root="$env:TEMP/rerun_20261111_8889037"
& "$env:USERPROFILE/anaconda3/python.exe" -u diagnose_identifiability.py $root "$root/identifiability_diagnostics"
```

The script uses 24 deterministic, non-overlapping validation ON-event windows per appliance, as in C. D2 uses 40 valid codes (eight per appliance) and 16 random normalized latent probes per code/identity. It calculates Jacobian-vector products (JVPs), not dense `1024 x 256` Jacobian matrices. QR bases of the JVP waveforms approximate each decoder identity's local tangent space. The maximum singular value of the two bases is their closest tangent cosine; `sqrt(1-cosine^2)` is the residual of the best two-source cancellation in the sampled subspaces. This is a finite-probe local diagnostic, not a formal rank or identifiability proof.

### D1 — Fixed-code appliance-ID sensitivity

For each valid code `z` encoded from an actual appliance waveform, hold `z` fixed and compare `G(z, origin-ID)` with `G(z, substituted-ID)`. The entries below are the median normalized waveform difference `||p_origin-p_substitute||/(||p_origin||+||p_substitute||)`. Row is the waveform/code origin; column is the substituted decoder identity.

| Origin \ ID | Kettle | Fridge | Dishwasher | Washer | Microwave |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kettle | 0.000 | 0.044 | 0.015 | 0.024 | 0.013 |
| Fridge | 0.784 | 0.000 | 0.559 | 0.436 | 0.639 |
| Dishwasher | 0.011 | 0.023 | 0.000 | 0.005 | 0.004 |
| Washing machine | 0.018 | 0.026 | 0.007 | 0.000 | 0.010 |
| Microwave | 0.027 | 0.102 | 0.013 | 0.032 | 0.000 |

The shared decoder **does use identity strongly for refrigerator codes**: switching an encoded fridge trace to kettle has median 41.95 W RMS difference. For kettle, dishwasher, washing-machine and microwave codes, most substitutions change the output only by roughly 3–30 W RMS and normalized differences remain near zero. Thus the problem is not simply that ID is ignored everywhere. Identity conditioning is highly uneven, and is weak exactly for several sparse/event-like sources. See `D1_identity_sensitivity_normalized.csv`, `D1_identity_sensitivity_rms_w.csv`, and `D1_identity_sensitivity.png`.

### D2 — Jacobian confusability

Median closest tangent cosine is high for several pairs: kettle–microwave 0.943 (best sampled cancellation residual 0.333), kettle–dishwasher 0.940 (0.339), dishwasher–microwave 0.924 (0.382), and washing machine–microwave 0.900 (0.435). Fridge pairs are materially less aligned: 0.709–0.745 with residuals 0.665–0.706. Therefore, under the sampled local codes, there are measurable directions whose two appliance updates can partly cancel in the aggregate while individually changing decoded waveforms. This supports near-null/source-ambiguous directions, particularly among non-fridge sources. It does **not** establish that every actual inverse update follows such a direction, nor that a particular cosine threshold separates identifiable and non-identifiable cases. See `D2_jacobian_pair_samples.csv`, `D2_jacobian_pair_summary.csv`, `D2_jacobian_alignment.png`, and `D2_runtime.json` (18.34 seconds CPU for 3,200 JVPs).

### D3 — latent-space validity along unchanged oracle Adam updates

The oracle replay starts from encoder codes of the true validation submeters solely to diagnose the frozen objective. The deployment row is separate: the existing Stage-2 train-mean initialisation. Training support is represented by 128 evenly sampled training windows per appliance. Nearest-code distance is therefore a **sampled-reference statistic**, not a likelihood or proof that a code is invalid.

| Stage | Median nearest sampled-training-code distance | Median movement from oracle initial code (RMS) | Median decoded change from oracle initial (W RMS) |
| --- | ---: | ---: | ---: |
| Oracle initial, step 0 | 0.000 | 0.000 | 0.000 |
| Oracle step 3 | 1.832 | 0.106 | 4.744 |
| Oracle step 25 | 5.528 | 0.335 | 12.772 |
| Oracle step 150 | 6.456 | 0.401 | 22.278 |
| Deployment train-mean start | 0.811 | 0.069 | 9.167 |

The unchanged meter-only optimizer rapidly leaves the sampled encoder-code neighbourhood while its decoded power changes. This is compatible with, but does not by itself prove, unsupported latent exploration. Association with per-window ON MAE is mixed and underpowered for rare sources. At step 150 the Spearman distance–ON-MAE association is 0.322 for fridge, 0.548 for washing machine, 0.200 for dishwasher, -0.253 for kettle and -0.700 for only five microwave ON windows. We therefore **cannot claim** a robust monotone relationship between distance and source error from this experiment. See `D3_latent_validity_per_window.csv`, `D3_latent_validity_summary.csv`, `D3_on_error_correlations.csv`, and `D3_latent_support_distance.png`.

### D1–D3 conclusion

1. The decoder uses appliance ID strongly for fridge but weakly for several other appliance codes.
2. The accessible decoder spaces overlap in C, and D2 independently finds strongly aligned local response directions for multiple non-fridge pairs.
3. Meter-only latent optimisation moves well beyond sampled encoded training codes, though sampled distance alone is not an invalidity certificate.
4. The evidence now favours **source ambiguity / weak appliance-specific representation** as the first Paper 1 problem, with Jacobian-aware inverse updates as the linked second problem.
5. The next algorithm should first improve or explicitly measure appliance discriminability in the forward representation. A solver-only penalty cannot recover attribution information absent from the representation. Any subsequent identifiability-aware solver should be tested against the unchanged Adam baseline and should not ban genuine simultaneous use.

## 2026-11-12 — E1–E4 forward-representation ablation

**Question.** Can an appliance-specific architecture or ON/OFF-balanced Stage-1 reconstruction loss improve source discrimination and downstream inverse attribution, while preserving the forward-model-to-latent-optimisation paradigm?

**Controlled protocol.** E1 is the unchanged epoch-83 shared conditional CNN. E2 has five separate encoders and five separate decoders with the same `[B,5,T] -> [B,5,Z,T/8] -> [B,5,T]` interface. E3 keeps E1's architecture and trains using a per-appliance balanced reconstruction risk, `0.5 mean(error^2 | ON) + 0.5 mean(error^2 | OFF)`, where ON is derived solely from the fixed training power thresholds. E4 combines E2 and E3. All use the same house-disjoint training/validation data, five-appliance order, 1024/512 windows, 4000 W scale, seed 2026, Adam and validation-MSE checkpoint selection. No test houses were read. The inverse evaluation uses frozen decoders, train-mean code initialization, a nonnegative constant background, 150 Adam steps at 0.05, batch size eight, and the same 945 validation windows (558,592 reconstructed rows).

E1 was re-evaluated without retraining because an older E1 inverse artifact did not explicitly record its initialization. The matched E1 run ensures that all entries below use `--train-mean-init`. All model runs and figures are on the training machine under `D:/Raymond/high_low_freq_NILM/Neural-adjoint inverse optimization/runs/paper1_e{1,2,3,4}_...`.

| Variant | Forward model / training loss | Parameters | Best epoch | Validation MSE |
| --- | --- | ---: | ---: | ---: |
| E1 | Shared conditional CNN / MSE | 9,731 | 83 | 0.0000602 |
| E2 | Separate encoders + decoders / MSE | 45,455 | 7 | 0.0028182 |
| E3 | Shared conditional CNN / balanced ON/OFF MSE | 9,731 | 97 | 0.0000267 |
| E4 | Separate encoders + decoders / balanced ON/OFF MSE | 45,455 | 92 | 0.0000899 |

### Stage-1 reconstruction on validation traces

The following macro values average the five appliance columns in each `source_attribution/A_reconstruction.csv`. F1 is raw fixed-threshold F1 and event recall uses the existing ±16 s onset protocol. These labels are used only for evaluation.

| Variant | MAE (W) | ON MAE (W) | Macro-F1 | Mean event recall |
| --- | ---: | ---: | ---: | ---: |
| E1 | 10.746 | 122.287 | 0.769 | 0.594 |
| E2 | 47.247 | 1,029.351 | 0.030 | 0.019 |
| E3 | 8.458 | 74.478 | 0.863 | 0.661 |
| E4 | 24.781 | 84.346 | 0.509 | 0.471 |

E2 is a clear collapsed reconstruction model. Its MSE-only independent source networks converge to predominantly low-power outputs: kettle ON MAE is 2,348.6 W and microwave ON MAE is 918.5 W. Its decoder JVPs are mostly zero, so a tangent-alignment statistic is undefined rather than evidence of identifiability. This architecture-only ablation must be retained as a failure, not treated as support for independent decoders.

E3 is the only variant that improves isolated Stage-1 reconstruction relative to E1. This confirms that ON events were underrepresented by ordinary MSE. E4 shows that event balancing does not rescue the independent architecture: fridge OFF MAE rises to 51.0 W and washing-machine OFF MAE to 33.0 W.

### Controlled zero-background inversion

For the 48-window zero-background control, compare the selected train-mean initialization. This control removes unmonitored loads; it does not make source attribution unique.

| Variant | Meter RMSE (W) | Appliance MAE (W) | ON MAE (W) | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| E1 | 47.827 | 40.629 | 688.966 | 0.404 |
| E2 | 507.930 | 50.019 | 852.253 | 0.032 |
| E3 | 64.613 | 45.997 | 622.097 | 0.304 |
| E4 | 48.531 | 49.573 | 712.743 | 0.321 |

Neither E2, E3 nor E4 improves controlled source-attribution F1 over E1. E3 improves controlled ON MAE but loses F1; this is another direct instance of different error objectives preferring different source allocations.

### Real aggregate validation inversion

| Variant | Appliance MAE (W) | ON MAE (W) | Macro-F1 | Aggregate MAE (W) | Inference time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 | 57.604 | 740.771 | 0.296 | 35.027 | 23.957 |
| E2 | 36.311 | 866.817 | 0.030 | 349.950 | 50.934 |
| E3 | 55.932 | 699.486 | 0.234 | 78.949 | 26.347 |
| E4 | 73.508 | 764.669 | 0.233 | 22.299 | 48.159 |

E3 reduces ON MAE by 41.3 W relative to E1 but lowers Macro-F1 by 0.062 and more than doubles aggregate MAE. E4 obtains the lowest aggregate MAE, 22.299 W, while lowering Macro-F1 from 0.296 to 0.233. Therefore neither better isolated reconstruction nor better aggregate agreement produced better appliance attribution in this ablation.

The output CSV currently uses the legacy string `appliance_conditioned_cnn_inverse` for all Stage-2 rows. Architecture identity is unambiguous from each checkpoint's `model_config`, saved configuration and run directory. This label should be corrected before any later experiment, but metrics above are grouped by those saved run directories and are not mislabeled analytically.

### Source-space diagnostics

E3 reduces several sampled tangent alignments relative to E1, for example kettle–microwave from 0.943 to 0.832, but cross-source fitting remains overlapping. Its normalized ON-RMSE for a microwave target ranges from 0.154 (fridge decoder) to 0.219 (kettle decoder) versus 0.197 for its own decoder. E4 has strong fixed-code ID sensitivity, often 0.66–0.96 normalized difference, and low sampled tangent alignment for non-dead pairs (kettle–microwave 0.360). Yet independently optimized wrong-source fits still occur: its dishwasher target has normalized ON-RMSE 0.030 with kettle and 0.031 with its own decoder; its fridge target is better fit by the washer decoder (0.045) than its own decoder (0.400). Strong fixed-code identity separation therefore does not imply non-overlapping optimised generator spaces.

### D3 step-0 zero-distance audit

The apparent median E1 step-0 nearest-training-code distance of zero is **not** evidence that all validation codes have perfect training support. With ON/OFF window separation, OFF windows for kettle, dishwasher, microwave and washing machine have distance exactly zero because an all-zero power trace passes through the deterministic encoder to the same code and the 128 sampled training references contain that repeated OFF code. The fridge windows in this 48-window diagnostic are all ON. At step 0, median distances for ON-containing windows are kettle 12.935, fridge 1.089, dishwasher 23.875, washing machine 17.656 and microwave 3.941. D3's all-window median was therefore dominated by repeated OFF codes. The support-distance statistic remains useful only when stratified by ON/OFF status and is not a validity proof.

### Decision after E1–E4

The controlled ablation does **not** justify promoting separate decoder architecture or simple event balancing as the Paper 1 algorithm. E3 is a useful reconstruction baseline and shows that sparse ON events matter. E4 proves that structural identity separation alone does not remove cross-source fit overlap. The next research phase should not tune another diagonal latent prior. It should formulate an inverse-aware forward-learning objective and compare it against E1/E3, then separately test an identifiability-aware inverse update. No such new objective or solver has been implemented in this phase.

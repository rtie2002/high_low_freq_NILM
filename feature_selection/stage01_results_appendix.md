# Stage 01 Results Appendix

Audit tables and greedy logs. Main narrative: [feature_selection.md](feature_selection.md).

## Master feature status (50 x 5)

Legend: K = kept, D = dropped.

| Feature | Domain | kettle | fridge | microwave | dishwasher | washingmachine | n_kept |
|---------|--------|:---:|:---:|:---:|:---:|:---:|:------:|
| `DWT_E1` | wavelet | K | K | K | K | K | 5 |
| `DWT_E2` | wavelet | K | K | K | K | K | 5 |
| `Fci` | time_domain | K | K | K | K | K | 5 |
| `Fcv` | time_domain | K | K | K | K | K | 5 |
| `I11` | harmonics | K | K | K | K | K | 5 |
| `I13` | harmonics | K | K | K | K | K | 5 |
| `I15` | harmonics | K | K | K | K | K | 5 |
| `I5` | harmonics | K | K | K | K | K | 5 |
| `I7` | harmonics | K | K | K | K | K | 5 |
| `I9` | harmonics | K | K | K | K | K | 5 |
| `I_BP_mid` | band_power | K | K | K | K | K | 5 |
| `I_env_1` | spectral_envelope | K | K | K | K | K | 5 |
| `I_env_2` | spectral_envelope | K | K | K | K | K | 5 |
| `I_env_3` | spectral_envelope | K | K | K | K | K | 5 |
| `I_env_4` | spectral_envelope | K | K | K | K | K | 5 |
| `I_env_5` | spectral_envelope | K | K | K | K | K | 5 |
| `I_env_6` | spectral_envelope | K | K | K | K | K | 5 |
| `I_spec_entropy` | spectral_descriptors | K | K | K | K | K | 5 |
| `PF` | time_domain | K | K | K | K | K | 5 |
| `V11` | harmonics | K | K | K | K | K | 5 |
| `V13` | harmonics | K | K | K | K | K | 5 |
| `V15` | harmonics | K | K | K | K | K | 5 |
| `V3` | harmonics | K | K | K | K | K | 5 |
| `V5` | harmonics | K | K | K | K | K | 5 |
| `V7` | harmonics | K | K | K | K | K | 5 |
| `V9` | harmonics | K | K | K | K | K | 5 |
| `VH` | distortion | K | K | K | K | K | 5 |
| `V_BP_low` | band_power | K | K | K | K | K | 5 |
| `V_rms` | time_domain | K | K | K | K | K | 5 |
| `V_skew` | shape_statistics | K | K | K | K | K | 5 |
| `IH` | distortion | K | K | K | K | D | 4 |
| `DWT_E3` | wavelet | D | K | K | D | K | 3 |
| `THDI` | distortion | K | D | D | K | K | 3 |
| `DWT_E4` | wavelet | K | D | D | K | D | 2 |
| `I_env_0` | spectral_envelope | D | K | K | D | D | 2 |
| `I_rms` | time_domain | D | D | K | D | K | 2 |
| `DWT_E0` | wavelet | K | D | D | D | D | 1 |
| `I3` | harmonics | D | D | D | D | K | 1 |
| `I_skew` | shape_statistics | D | K | D | D | D | 1 |
| `P_active` | time_domain | D | D | D | K | D | 1 |
| `I1` | harmonics | D | D | D | D | D | 0 |
| `I_BP_high` | band_power | D | D | D | D | D | 0 |
| `I_BP_low` | band_power | D | D | D | D | D | 0 |
| `I_env_7` | spectral_envelope | D | D | D | D | D | 0 |
| `I_kurt` | shape_statistics | D | D | D | D | D | 0 |
| `I_std` | shape_statistics | D | D | D | D | D | 0 |
| `S_apparent` | time_domain | D | D | D | D | D | 0 |
| `THDV` | distortion | D | D | D | D | D | 0 |
| `V1` | harmonics | D | D | D | D | D | 0 |
| `V_std` | shape_statistics | D | D | D | D | D | 0 |

## Greedy elimination logs (16 steps each)

### kettle

| Step | Dropped | Kept | Pair |r| | Reason |
|------|---------|------|--------|------|
| 1 | `I_std` | `I_rms` | 1.000 | priority |
| 2 | `V_std` | `V_rms` | 1.000 | priority |
| 3 | `S_apparent` | `I_rms` | 1.000 | priority |
| 4 | `I_rms` | `DWT_E0` | 1.000 | target |
| 5 | `I1` | `P_active` | 0.997 | priority |
| 6 | `I_BP_low` | `DWT_E0` | 0.995 | priority |
| 7 | `P_active` | `DWT_E0` | 0.992 | target |
| 8 | `THDV` | `VH` | 0.991 | target |
| 9 | `I3` | `IH` | 0.990 | priority |
| 10 | `V1` | `V_rms` | 0.987 | target |
| 11 | `I_env_7` | `I_env_6` | 0.981 | priority |
| 12 | `I_BP_high` | `DWT_E4` | 0.970 | target |
| 13 | `DWT_E3` | `DWT_E4` | 0.965 | target |
| 14 | `I_kurt` | `THDI` | 0.962 | target |
| 15 | `I_skew` | `DWT_E0` | 0.957 | target |
| 16 | `I_env_0` | `THDI` | 0.951 | target |

### fridge

| Step | Dropped | Kept | Pair |r| | Reason |
|------|---------|------|--------|------|
| 1 | `I_std` | `I_rms` | 1.000 | priority |
| 2 | `V_std` | `V_rms` | 1.000 | priority |
| 3 | `S_apparent` | `I_rms` | 1.000 | priority |
| 4 | `DWT_E0` | `I_rms` | 1.000 | target |
| 5 | `I_rms` | `I1` | 0.998 | target |
| 6 | `I1` | `P_active` | 0.997 | priority |
| 7 | `THDV` | `VH` | 0.991 | priority |
| 8 | `I3` | `IH` | 0.990 | priority |
| 9 | `V1` | `V_rms` | 0.987 | priority |
| 10 | `I_env_7` | `I_env_6` | 0.981 | priority |
| 11 | `I_BP_low` | `P_active` | 0.976 | target |
| 12 | `I_BP_high` | `DWT_E4` | 0.970 | priority |
| 13 | `DWT_E4` | `DWT_E3` | 0.965 | priority |
| 14 | `I_kurt` | `THDI` | 0.962 | target |
| 15 | `P_active` | `I_skew` | 0.960 | target |
| 16 | `THDI` | `I_env_0` | 0.951 | target |

### microwave

| Step | Dropped | Kept | Pair |r| | Reason |
|------|---------|------|--------|------|
| 1 | `I_std` | `I_rms` | 1.000 | priority |
| 2 | `V_std` | `V_rms` | 1.000 | priority |
| 3 | `S_apparent` | `I_rms` | 1.000 | priority |
| 4 | `DWT_E0` | `I_rms` | 1.000 | target |
| 5 | `I1` | `I_rms` | 0.998 | target |
| 6 | `P_active` | `I_rms` | 0.997 | target |
| 7 | `THDV` | `VH` | 0.991 | priority |
| 8 | `I3` | `IH` | 0.990 | target |
| 9 | `V1` | `V_rms` | 0.987 | priority |
| 10 | `I_env_7` | `I_env_6` | 0.981 | priority |
| 11 | `I_BP_low` | `I_rms` | 0.980 | target |
| 12 | `I_BP_high` | `DWT_E4` | 0.970 | priority |
| 13 | `DWT_E4` | `DWT_E3` | 0.965 | priority |
| 14 | `I_kurt` | `THDI` | 0.962 | target |
| 15 | `I_skew` | `I_rms` | 0.956 | target |
| 16 | `THDI` | `I_env_0` | 0.951 | target |

### dishwasher

| Step | Dropped | Kept | Pair |r| | Reason |
|------|---------|------|--------|------|
| 1 | `I_std` | `I_rms` | 1.000 | priority |
| 2 | `V_std` | `V_rms` | 1.000 | priority |
| 3 | `S_apparent` | `I_rms` | 1.000 | priority |
| 4 | `DWT_E0` | `I_rms` | 1.000 | target |
| 5 | `I1` | `I_rms` | 0.998 | priority |
| 6 | `I_rms` | `P_active` | 0.997 | target |
| 7 | `THDV` | `VH` | 0.991 | target |
| 8 | `I3` | `IH` | 0.990 | priority |
| 9 | `V1` | `V_rms` | 0.987 | target |
| 10 | `I_env_7` | `I_env_6` | 0.981 | priority |
| 11 | `I_BP_low` | `P_active` | 0.976 | target |
| 12 | `I_BP_high` | `DWT_E4` | 0.970 | target |
| 13 | `DWT_E3` | `DWT_E4` | 0.965 | target |
| 14 | `I_kurt` | `THDI` | 0.962 | target |
| 15 | `I_skew` | `P_active` | 0.960 | target |
| 16 | `I_env_0` | `THDI` | 0.951 | target |

### washingmachine

| Step | Dropped | Kept | Pair |r| | Reason |
|------|---------|------|--------|------|
| 1 | `I_std` | `I_rms` | 1.000 | priority |
| 2 | `V_std` | `V_rms` | 1.000 | priority |
| 3 | `S_apparent` | `I_rms` | 1.000 | priority |
| 4 | `DWT_E0` | `I_rms` | 1.000 | target |
| 5 | `I1` | `I_rms` | 0.998 | priority |
| 6 | `P_active` | `I_rms` | 0.997 | target |
| 7 | `THDV` | `VH` | 0.991 | target |
| 8 | `IH` | `I3` | 0.990 | target |
| 9 | `V1` | `V_rms` | 0.987 | target |
| 10 | `I_env_7` | `I_env_6` | 0.981 | priority |
| 11 | `I_BP_low` | `I_rms` | 0.980 | target |
| 12 | `I_BP_high` | `DWT_E4` | 0.970 | priority |
| 13 | `DWT_E4` | `DWT_E3` | 0.965 | priority |
| 14 | `I_kurt` | `THDI` | 0.962 | target |
| 15 | `I_skew` | `I_rms` | 0.956 | target |
| 16 | `I_env_0` | `THDI` | 0.951 | target |

## Full Dataset vs ON-Period Scenario Comparison

This section compares the original full-window Stage 01 run with the ON-period run. The ON-period run uses `on_off == 1` plus a two-row buffer before and after each activation event. Since each HF row is approximately 6 seconds, this keeps about 12 seconds before and after each event.

Legend: `K` = kept, `D` = dropped. `full` = all rows. `on` = ON-period plus two-step buffer.

### Dataset Size

| appliance | full | on_only_buffer2 | row_reduction_ratio_on_vs_full |
| --- | --- | --- | --- |
| dishwasher | 100779 | 3218 | 0.0319 |
| fridge | 100779 | 71102 | 0.7055 |
| kettle | 100780 | 734 | 0.0073 |
| microwave | 100778 | 510 | 0.0051 |
| washingmachine | 100778 | 2667 | 0.0265 |

### Target-Correlation Summary

| appliance | mean_pearson_full | mean_pearson_on | median_pearson_full | median_pearson_on | max_pearson_full | max_pearson_on | mean_spearman_full | mean_spearman_on | median_spearman_full | median_spearman_on |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dishwasher | 0.251 | 0.474 | 0.204 | 0.478 | 0.707 | 0.938 | 0.054 | 0.399 | 0.058 | 0.375 |
| fridge | 0.164 | 0.051 | 0.048 | 0.029 | 0.711 | 0.243 | 0.236 | 0.122 | 0.159 | 0.092 |
| kettle | 0.205 | 0.354 | 0.112 | 0.331 | 0.662 | 0.740 | 0.060 | 0.232 | 0.076 | 0.242 |
| microwave | 0.136 | 0.373 | 0.081 | 0.353 | 0.674 | 0.741 | 0.069 | 0.298 | 0.073 | 0.293 |
| washingmachine | 0.129 | 0.400 | 0.119 | 0.477 | 0.325 | 0.789 | 0.052 | 0.288 | 0.040 | 0.346 |

Observation: ON-period target correlations increase strongly for dishwasher, kettle, microwave, and washing machine. Washing machine is the clearest case: max |Pearson| rises from about 0.325 to about 0.789. Fridge decreases because its duty cycle is already high and the ON-only subset removes much of the contrast that helps full-window correlation.

### Top Target-Correlated Features In Each Scenario

| appliance | scenario | rank | feature | domain | |Pearson| | |Spearman| |
| --- | --- | --- | --- | --- | --- | --- |
| kettle | full | 1 | I_BP_low | band_power | 0.662 | 0.093 |
| kettle | full | 2 | DWT_E0 | wavelet | 0.660 | 0.093 |
| kettle | full | 3 | DWT_E4 | wavelet | 0.593 | 0.081 |
| kettle | full | 4 | I_BP_high | band_power | 0.582 | 0.081 |
| kettle | full | 5 | DWT_E3 | wavelet | 0.568 | 0.083 |
| kettle | on | 1 | THDI | distortion | 0.740 | 0.384 |
| kettle | on | 2 | P_active | time_domain | 0.723 | 0.356 |
| kettle | on | 3 | S_apparent | time_domain | 0.713 | 0.341 |
| kettle | on | 4 | I1 | harmonics | 0.708 | 0.262 |
| kettle | on | 5 | I_std | shape_statistics | 0.707 | 0.265 |
| fridge | full | 1 | I_env_0 | spectral_envelope | 0.711 | 0.638 |
| fridge | full | 2 | I_env_1 | spectral_envelope | 0.699 | 0.668 |
| fridge | full | 3 | THDI | distortion | 0.682 | 0.616 |
| fridge | full | 4 | I_env_2 | spectral_envelope | 0.628 | 0.462 |
| fridge | full | 5 | I_spec_entropy | spectral_descriptors | 0.610 | 0.545 |
| fridge | on | 1 | I_env_1 | spectral_envelope | 0.243 | 0.421 |
| fridge | on | 2 | I_env_0 | spectral_envelope | 0.217 | 0.335 |
| fridge | on | 3 | THDI | distortion | 0.188 | 0.319 |
| fridge | on | 4 | I_spec_entropy | spectral_descriptors | 0.177 | 0.224 |
| fridge | on | 5 | PF | time_domain | 0.162 | 0.196 |
| microwave | full | 1 | IH | distortion | 0.674 | 0.114 |
| microwave | full | 2 | I3 | harmonics | 0.662 | 0.115 |
| microwave | full | 3 | I5 | harmonics | 0.567 | 0.124 |
| microwave | full | 4 | DWT_E1 | wavelet | 0.360 | 0.092 |
| microwave | full | 5 | I11 | harmonics | 0.331 | 0.074 |
| microwave | on | 1 | I5 | harmonics | 0.741 | 0.599 |
| microwave | on | 2 | IH | distortion | 0.710 | 0.667 |
| microwave | on | 3 | I3 | harmonics | 0.700 | 0.664 |
| microwave | on | 4 | P_active | time_domain | 0.686 | 0.568 |
| microwave | on | 5 | I11 | harmonics | 0.684 | 0.490 |
| dishwasher | full | 1 | P_active | time_domain | 0.707 | 0.064 |
| dishwasher | full | 2 | I1 | harmonics | 0.697 | 0.066 |
| dishwasher | full | 3 | I_rms | time_domain | 0.696 | 0.065 |
| dishwasher | full | 4 | I_std | shape_statistics | 0.696 | 0.065 |
| dishwasher | full | 5 | S_apparent | time_domain | 0.694 | 0.065 |
| dishwasher | on | 1 | THDI | distortion | 0.938 | 0.720 |
| dishwasher | on | 2 | P_active | time_domain | 0.909 | 0.792 |
| dishwasher | on | 3 | S_apparent | time_domain | 0.907 | 0.791 |
| dishwasher | on | 4 | I1 | harmonics | 0.906 | 0.768 |
| dishwasher | on | 5 | I_std | shape_statistics | 0.905 | 0.768 |
| washingmachine | full | 1 | S_apparent | time_domain | 0.325 | 0.010 |
| washingmachine | full | 2 | I_std | shape_statistics | 0.325 | 0.014 |
| washingmachine | full | 3 | I_rms | time_domain | 0.325 | 0.014 |
| washingmachine | full | 4 | I1 | harmonics | 0.323 | 0.016 |
| washingmachine | full | 5 | P_active | time_domain | 0.315 | 0.029 |
| washingmachine | on | 1 | P_active | time_domain | 0.789 | 0.551 |
| washingmachine | on | 2 | DWT_E0 | wavelet | 0.771 | 0.508 |
| washingmachine | on | 3 | I_rms | time_domain | 0.764 | 0.507 |
| washingmachine | on | 4 | I_std | shape_statistics | 0.764 | 0.507 |
| washingmachine | on | 5 | I1 | harmonics | 0.764 | 0.488 |

### Largest |Pearson| Increases From Full To ON-Period

| appliance | feature | domain | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full | target_spearman_abs_full | target_spearman_abs_on | spearman_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kettle | THDI | distortion | 0.174 | 0.740 | 0.566 | 0.092 | 0.384 | 0.291 |
| kettle | I_skew | shape_statistics | 0.111 | 0.661 | 0.550 | 0.091 | 0.269 | 0.177 |
| kettle | Fci | time_domain | 0.116 | 0.608 | 0.492 | 0.088 | 0.314 | 0.226 |
| kettle | I_spec_entropy | spectral_descriptors | 0.155 | 0.617 | 0.462 | 0.089 | 0.478 | 0.389 |
| kettle | I_env_1 | spectral_envelope | 0.152 | 0.612 | 0.461 | 0.088 | 0.424 | 0.336 |
| fridge | V7 | harmonics | 0.041 | 0.082 | 0.041 | 0.030 | 0.145 | 0.115 |
| fridge | V15 | harmonics | 0.032 | 0.066 | 0.035 | 0.097 | 0.153 | 0.056 |
| fridge | I11 | harmonics | 0.049 | 0.083 | 0.034 | 0.119 | 0.203 | 0.084 |
| fridge | V_skew | shape_statistics | 0.028 | 0.058 | 0.030 | 0.083 | 0.118 | 0.035 |
| fridge | V_std | shape_statistics | 0.005 | 0.030 | 0.025 | 0.074 | 0.069 | -0.005 |
| microwave | I_env_2 | spectral_envelope | 0.061 | 0.646 | 0.585 | 0.030 | 0.515 | 0.485 |
| microwave | I_env_0 | spectral_envelope | 0.127 | 0.633 | 0.506 | 0.053 | 0.616 | 0.562 |
| microwave | P_active | time_domain | 0.182 | 0.686 | 0.504 | 0.116 | 0.568 | 0.452 |
| microwave | Fci | time_domain | 0.059 | 0.535 | 0.476 | 0.104 | 0.296 | 0.192 |
| microwave | I_std | shape_statistics | 0.202 | 0.665 | 0.463 | 0.115 | 0.516 | 0.400 |
| dishwasher | I_skew | shape_statistics | 0.203 | 0.897 | 0.695 | 0.069 | 0.780 | 0.711 |
| dishwasher | I_env_1 | spectral_envelope | 0.237 | 0.888 | 0.651 | 0.146 | 0.708 | 0.562 |
| dishwasher | I_kurt | shape_statistics | 0.019 | 0.647 | 0.629 | 0.102 | 0.732 | 0.630 |
| dishwasher | THDI | distortion | 0.311 | 0.938 | 0.627 | 0.105 | 0.720 | 0.615 |
| dishwasher | Fci | time_domain | 0.207 | 0.834 | 0.626 | 0.108 | 0.754 | 0.646 |
| washingmachine | THDI | distortion | 0.146 | 0.747 | 0.602 | 0.030 | 0.554 | 0.523 |
| washingmachine | DWT_E0 | wavelet | 0.266 | 0.771 | 0.505 | 0.014 | 0.508 | 0.494 |
| washingmachine | I_BP_low | band_power | 0.249 | 0.744 | 0.495 | 0.012 | 0.479 | 0.468 |
| washingmachine | I_env_0 | spectral_envelope | 0.063 | 0.554 | 0.490 | 0.016 | 0.466 | 0.449 |
| washingmachine | PF | time_domain | 0.029 | 0.514 | 0.485 | 0.102 | 0.494 | 0.393 |

### Dropped-Feature Set Summary

| appliance | dropped_full | dropped_on | changed_features | jaccard_dropped_sets |
| --- | --- | --- | --- | --- |
| dishwasher | 16 | 17 | 11 | 0.500 |
| fridge | 16 | 14 | 6 | 0.667 |
| kettle | 16 | 14 | 12 | 0.429 |
| microwave | 16 | 15 | 13 | 0.409 |
| washingmachine | 16 | 15 | 11 | 0.476 |

Observation: the dropped-feature set changes meaningfully. Jaccard similarity is around 0.41-0.50 for kettle, microwave, washing machine, and dishwasher, meaning only about half of the union of dropped features overlaps between the two scenarios. Fridge is more stable at about 0.67.

### Master Feature Status: Full vs ON-Period

| feature | domain | kettle_full | kettle_on | fridge_full | fridge_on | microwave_full | microwave_on | dishwasher_full | dishwasher_on | washingmachine_full | washingmachine_on | n_changed | n_dropped_full | n_dropped_on |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I_env_5 | spectral_envelope | K | D | K | K | K | D | K | D | K | D | 4 | 0 | 4 |
| I_rms | time_domain | D | D | D | K | K | D | D | K | K | D | 4 | 3 | 3 |
| I3 | harmonics | D | K | D | K | D | D | D | K | K | D | 4 | 4 | 2 |
| P_active | time_domain | D | K | D | D | D | K | K | D | D | K | 4 | 4 | 2 |
| I_kurt | shape_statistics | D | K | D | K | D | K | D | D | D | K | 4 | 5 | 1 |
| DWT_E4 | wavelet | K | D | D | D | D | D | K | D | D | K | 3 | 3 | 4 |
| I_env_1 | spectral_envelope | K | K | K | K | K | D | K | D | K | D | 3 | 0 | 3 |
| IH | distortion | K | K | K | D | K | D | K | K | D | K | 3 | 1 | 2 |
| I_BP_low | band_power | D | K | D | D | D | K | D | K | D | D | 3 | 5 | 2 |
| I_skew | shape_statistics | D | K | K | K | D | K | D | D | D | K | 3 | 4 | 1 |
| I_env_0 | spectral_envelope | D | K | K | K | K | K | D | K | D | K | 3 | 3 | 0 |
| THDV | distortion | D | D | D | D | D | K | D | K | D | D | 2 | 5 | 3 |
| DWT_E3 | wavelet | D | K | K | K | K | K | D | D | K | D | 2 | 2 | 2 |
| I_BP_mid | band_power | K | K | K | D | K | K | K | D | K | K | 2 | 0 | 2 |
| VH | distortion | K | K | K | K | K | D | K | D | K | K | 2 | 0 | 2 |
| THDI | distortion | K | K | D | K | D | K | K | K | K | K | 2 | 2 | 0 |
| DWT_E0 | wavelet | K | D | D | D | D | D | D | D | D | D | 1 | 4 | 5 |
| I_BP_high | band_power | D | D | D | D | D | K | D | D | D | D | 1 | 5 | 4 |
| DWT_E2 | wavelet | K | D | K | K | K | K | K | K | K | K | 1 | 0 | 1 |
| I_env_4 | spectral_envelope | K | D | K | K | K | K | K | K | K | K | 1 | 0 | 1 |
| I_env_6 | spectral_envelope | K | K | K | K | K | D | K | K | K | K | 1 | 0 | 1 |
| I1 | harmonics | D | D | D | D | D | D | D | D | D | D | 0 | 5 | 5 |
| I_env_7 | spectral_envelope | D | D | D | D | D | D | D | D | D | D | 0 | 5 | 5 |
| I_std | shape_statistics | D | D | D | D | D | D | D | D | D | D | 0 | 5 | 5 |
| S_apparent | time_domain | D | D | D | D | D | D | D | D | D | D | 0 | 5 | 5 |
| V1 | harmonics | D | D | D | D | D | D | D | D | D | D | 0 | 5 | 5 |
| V_std | shape_statistics | D | D | D | D | D | D | D | D | D | D | 0 | 5 | 5 |
| DWT_E1 | wavelet | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| Fci | time_domain | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| Fcv | time_domain | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| I11 | harmonics | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| I13 | harmonics | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| I15 | harmonics | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| I5 | harmonics | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| I7 | harmonics | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| I9 | harmonics | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| I_env_2 | spectral_envelope | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| I_env_3 | spectral_envelope | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| I_spec_entropy | spectral_descriptors | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| PF | time_domain | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| V11 | harmonics | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| V13 | harmonics | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| V15 | harmonics | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| V3 | harmonics | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| V5 | harmonics | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| V7 | harmonics | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| V9 | harmonics | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| V_BP_low | band_power | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| V_rms | time_domain | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |
| V_skew | shape_statistics | K | K | K | K | K | K | K | K | K | K | 0 | 0 | 0 |

### Features Whose Final Status Changed

| appliance | feature | domain | final_status_full | dropped_at_stage_full | final_status_on | dropped_at_stage_on | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dishwasher | DWT_E4 | wavelet | kept | passed | dropped | correlation | 0.501 | 0.492 | -0.009 |
| dishwasher | I3 | harmonics | dropped | correlation | kept | passed | 0.058 | 0.422 | 0.364 |
| dishwasher | I_BP_low | band_power | dropped | correlation | kept | passed | 0.640 | 0.672 | 0.032 |
| dishwasher | I_BP_mid | band_power | kept | passed | dropped | correlation | 0.218 | 0.314 | 0.096 |
| dishwasher | I_env_0 | spectral_envelope | dropped | correlation | kept | passed | 0.233 | 0.855 | 0.621 |
| dishwasher | I_env_1 | spectral_envelope | kept | passed | dropped | correlation | 0.237 | 0.888 | 0.651 |
| dishwasher | I_env_5 | spectral_envelope | kept | passed | dropped | correlation | 0.267 | 0.432 | 0.165 |
| dishwasher | I_rms | time_domain | dropped | correlation | kept | passed | 0.696 | 0.905 | 0.210 |
| dishwasher | P_active | time_domain | kept | passed | dropped | correlation | 0.707 | 0.909 | 0.202 |
| dishwasher | THDV | distortion | dropped | correlation | kept | passed | 0.162 | 0.506 | 0.344 |
| dishwasher | VH | distortion | kept | passed | dropped | correlation | 0.195 | 0.491 | 0.296 |
| fridge | I3 | harmonics | dropped | correlation | kept | passed | 0.038 | 0.050 | 0.012 |
| fridge | IH | distortion | kept | passed | dropped | correlation | 0.047 | 0.035 | -0.012 |
| fridge | I_BP_mid | band_power | kept | passed | dropped | correlation | 0.030 | 0.001 | -0.029 |
| fridge | I_kurt | shape_statistics | dropped | correlation | kept | passed | 0.053 | 0.043 | -0.009 |
| fridge | I_rms | time_domain | dropped | correlation | kept | passed | 0.111 | 0.010 | -0.101 |
| fridge | THDI | distortion | dropped | correlation | kept | passed | 0.682 | 0.188 | -0.494 |
| kettle | DWT_E0 | wavelet | kept | passed | dropped | correlation | 0.660 | 0.523 | -0.137 |
| kettle | DWT_E2 | wavelet | kept | passed | dropped | correlation | 0.333 | 0.361 | 0.027 |
| kettle | DWT_E3 | wavelet | dropped | correlation | kept | passed | 0.568 | 0.378 | -0.189 |
| kettle | DWT_E4 | wavelet | kept | passed | dropped | correlation | 0.593 | 0.377 | -0.216 |
| kettle | I3 | harmonics | dropped | correlation | kept | passed | 0.015 | 0.144 | 0.129 |
| kettle | I_BP_low | band_power | dropped | correlation | kept | passed | 0.662 | 0.512 | -0.150 |
| kettle | I_env_0 | spectral_envelope | dropped | correlation | kept | passed | 0.125 | 0.557 | 0.431 |
| kettle | I_env_4 | spectral_envelope | kept | passed | dropped | correlation | 0.001 | 0.242 | 0.241 |
| kettle | I_env_5 | spectral_envelope | kept | passed | dropped | correlation | 0.334 | 0.380 | 0.045 |
| kettle | I_kurt | shape_statistics | dropped | correlation | kept | passed | 0.010 | 0.315 | 0.305 |
| kettle | I_skew | shape_statistics | dropped | correlation | kept | passed | 0.111 | 0.661 | 0.550 |
| kettle | P_active | time_domain | dropped | correlation | kept | passed | 0.541 | 0.723 | 0.182 |
| microwave | IH | distortion | kept | passed | dropped | correlation | 0.674 | 0.710 | 0.036 |
| microwave | I_BP_high | band_power | dropped | correlation | kept | passed | 0.148 | 0.341 | 0.194 |
| microwave | I_BP_low | band_power | dropped | correlation | kept | passed | 0.131 | 0.466 | 0.335 |
| microwave | I_env_1 | spectral_envelope | kept | passed | dropped | correlation | 0.167 | 0.579 | 0.412 |
| microwave | I_env_5 | spectral_envelope | kept | passed | dropped | correlation | 0.040 | 0.219 | 0.179 |
| microwave | I_env_6 | spectral_envelope | kept | passed | dropped | correlation | 0.101 | 0.291 | 0.189 |
| microwave | I_kurt | shape_statistics | dropped | correlation | kept | passed | 0.006 | 0.284 | 0.278 |
| microwave | I_rms | time_domain | kept | passed | dropped | correlation | 0.202 | 0.665 | 0.463 |
| microwave | I_skew | shape_statistics | dropped | correlation | kept | passed | 0.063 | 0.137 | 0.074 |
| microwave | P_active | time_domain | dropped | correlation | kept | passed | 0.182 | 0.686 | 0.504 |
| microwave | THDI | distortion | dropped | correlation | kept | passed | 0.033 | 0.166 | 0.134 |
| microwave | THDV | distortion | dropped | correlation | kept | passed | 0.010 | 0.065 | 0.055 |
| microwave | VH | distortion | kept | passed | dropped | correlation | 0.002 | 0.006 | 0.003 |
| washingmachine | DWT_E3 | wavelet | kept | passed | dropped | correlation | 0.239 | 0.574 | 0.335 |
| washingmachine | DWT_E4 | wavelet | dropped | correlation | kept | passed | 0.249 | 0.585 | 0.336 |
| washingmachine | I3 | harmonics | kept | passed | dropped | correlation | 0.192 | 0.109 | -0.082 |
| washingmachine | IH | distortion | dropped | correlation | kept | passed | 0.179 | 0.125 | -0.054 |
| washingmachine | I_env_0 | spectral_envelope | dropped | correlation | kept | passed | 0.063 | 0.554 | 0.490 |
| washingmachine | I_env_1 | spectral_envelope | kept | passed | dropped | correlation | 0.035 | 0.504 | 0.470 |
| washingmachine | I_env_5 | spectral_envelope | kept | passed | dropped | correlation | 0.125 | 0.550 | 0.424 |
| washingmachine | I_kurt | shape_statistics | dropped | correlation | kept | passed | 0.009 | 0.284 | 0.275 |
| washingmachine | I_rms | time_domain | kept | passed | dropped | correlation | 0.325 | 0.764 | 0.439 |
| washingmachine | I_skew | shape_statistics | dropped | correlation | kept | passed | 0.121 | 0.213 | 0.092 |
| washingmachine | P_active | time_domain | dropped | correlation | kept | passed | 0.315 | 0.789 | 0.474 |

### Interpretation For Thesis Use

The ON-period scenario is more appropriate when the aim is to measure feature relevance during actual appliance operation. Full-window correlations answer a different question: whether a feature tracks the appliance target across all time, including long OFF periods. For sparse appliances, full-window correlation can be dominated by OFF windows and therefore understate appliance-specific relationships.

The final Stage 01 feature set changes because target correlation is used to choose which feature survives inside a redundant pair. Therefore, when the target-correlation ranking changes under ON-period filtering, some greedy drop decisions also change.

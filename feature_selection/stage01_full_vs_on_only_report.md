# Stage 01 Full Dataset vs ON-Period Dataset Comparison

This report compares saved Stage 01 feature-selection results from two folders:

- Full dataset: `C:\Users\Raymond Tie\Desktop\PhD\Code\multi-domain NILM\high_low_freq_NILM\feature_selection_outputs`
- ON-period dataset: `C:\Users\Raymond Tie\Desktop\PhD\Code\multi-domain NILM\high_low_freq_NILM\feature_selection_outputs(on only)`

The ON-period run is interpreted as `on_off == 1` plus the two-row event buffer generated earlier. Each HF row is approximately 6 seconds, so the buffer is approximately 12 seconds before and after an ON event.

## Method Decision And Honest Conclusion

Stage 01 can be used, but it should be described carefully as a **first-stage redundancy filter**, not as the final proof of feature usefulness. The method is valid for removing highly duplicated HF descriptors, because many extracted features measure closely related physical quantities. Examples include `I_rms`, `I_std`, `S_apparent`, `P_active`, `I1`, `I_BP_low`, and `DWT_E0`, which can move together when current magnitude increases. The filter is therefore useful to reduce collinearity before later mRMR, model-based importance, stability selection, or ablation experiments.

The comparison shows that the **ON-period scenario is more appropriate for target-correlation interpretation** when the thesis question is appliance-active behavior. Full-window correlation is heavily affected by OFF periods, especially for sparse appliances. In the full dataset, the target is often near zero while the aggregate VI waveform still contains other loads and background behavior. This can understate the relationship between HF features and the appliance power. The clearest example is washing machine, where the maximum target |Pearson| increases from `0.325` in the full dataset to `0.789` in the ON-period dataset.

However, the ON-period result should not be presented as a complete NILM feature-selection solution by itself. It removes most OFF examples, so it is better for understanding active-state feature relevance, but not sufficient to prove ON/OFF detection performance. The final NILM model still needs OFF samples and should be validated downstream. Therefore, the recommended thesis position is:

```text
Use ON-period Stage 01 as the main redundancy-filter result for appliance-active HF feature relevance.
Use full-dataset Stage 01 as a sensitivity comparison.
Do not claim Stage 01 alone is the final optimal feature set.
```

## Major Differences Between The Two Scenarios

The largest difference is not only the number of rows, but the meaning of the correlation being estimated.

Full dataset:

```text
corr(HF feature, appliance_power) over all day/week windows
```

This measures whether a feature tracks appliance power while including long OFF periods. It is useful for broad all-window association, but can be diluted by sparse appliance activations.

ON-period dataset:

```text
corr(HF feature, appliance_power) only near appliance active windows
```

This measures whether a feature tracks appliance behavior while the appliance is running. This is closer to the physical question of which waveform descriptors are relevant during actual appliance operation.

The dropped-feature sets changed substantially:

| appliance | dropped in full dataset | dropped in ON-period dataset | features whose status changed | dropped-set overlap (Jaccard) |
| --- | ---: | ---: | ---: | ---: |
| dishwasher | 16 | 17 | 11 | 0.500 |
| fridge | 16 | 14 | 6 | 0.667 |
| kettle | 16 | 14 | 12 | 0.429 |
| microwave | 16 | 15 | 13 | 0.409 |
| washingmachine | 16 | 15 | 11 | 0.476 |

The low Jaccard values for kettle, microwave, dishwasher, and washing machine mean that the selected feature set is sensitive to whether the target correlation is computed over all windows or active windows only. This is expected because Stage 01 first finds redundant feature pairs, then uses target correlation to decide which feature survives inside each pair.

## Physical Meaning Behind The Change

The HF features are computed from 6-second aggregate voltage-current waveform windows. They can be grouped physically as follows:

| feature family | examples | physical meaning | why the scenario changes the result |
| --- | --- | --- | --- |
| current/power magnitude | `I_rms`, `I_std`, `S_apparent`, `P_active`, `I1`, `I_BP_low`, `DWT_E0` | Load size and low-frequency current energy | In ON-period data, these features better follow appliance active power; in full data they are diluted by OFF windows and other aggregate loads. |
| waveform shape | `PF`, `Fci`, `I_skew`, `I_kurt` | Phase relation, peakiness, asymmetry, impulsiveness | These features can become more meaningful during active appliance states, especially for motors, heaters, and switching loads. |
| harmonic/distortion | `I3`, `I5`, `IH`, `THDI` | Non-sinusoidal current and harmonic content | During appliance operation, harmonic signatures are less masked by unrelated OFF-period background behavior. |
| spectral band/envelope | `I_BP_mid`, `I_BP_high`, `I_env_0` ... `I_env_7` | Frequency distribution shape rather than direct power | ON-period filtering can reveal appliance-specific spectral shape, but these features may also be sensitive to transient events and aggregate contamination. |
| wavelet energy | `DWT_E0` ... `DWT_E4` | Time-frequency energy, including transients and switching content | ON-period windows include start/stop and active waveform behavior; full data includes many unrelated background windows. |

This explains why features such as `P_active`, `I_rms`, `I_env_0`, `I_skew`, and wavelet energies can change status between runs. They are not necessarily becoming mathematically unstable; rather, the dataset has changed the physical question being asked.

## Concerns And Limitations

Several concerns should be stated honestly:

1. The ON-period dataset for rare appliances is much smaller. Kettle has only `734` rows and microwave only `510` rows after buffering. Correlation estimates from these small samples may be less stable than full-dataset estimates.
2. The ON-period mask is derived from `on_off`, which is itself generated from appliance power labels. This is acceptable for supervised feature analysis, but it means the method depends on label quality.
3. The HF features are extracted from aggregate VI waveforms. Even during an appliance ON period, other appliances may also be active, so the feature vector is not a pure isolated appliance signature.
4. Correlation filtering captures linear or monotonic association and feature redundancy. It does not prove that a feature improves final NILM prediction accuracy.
5. Stage 01 uses a fixed redundancy threshold (`0.95`). Some selected/dropped differences may be threshold-sensitive and should be checked later by downstream model validation.
6. ON-period feature selection is better for active-state relevance, but not enough for final deployment because NILM also requires distinguishing OFF from ON.

Final judgement: **Stage 01 is usable as a defensible preliminary redundancy filter**, especially when reported with the ON-period result as the main active-state analysis and the full-dataset result as a sensitivity check. It should not be overclaimed as the final feature-selection method. Later stages must validate predictive usefulness and stability.

## Executive Observations

- `dishwasher`: max |Pearson| changed from 0.707 on full data to 0.938 on ON-period data; dropped-feature set Jaccard similarity = 0.50; changed final feature statuses = 11.
- `fridge`: max |Pearson| changed from 0.711 on full data to 0.243 on ON-period data; dropped-feature set Jaccard similarity = 0.67; changed final feature statuses = 6.
- `kettle`: max |Pearson| changed from 0.662 on full data to 0.740 on ON-period data; dropped-feature set Jaccard similarity = 0.43; changed final feature statuses = 12.
- `microwave`: max |Pearson| changed from 0.674 on full data to 0.741 on ON-period data; dropped-feature set Jaccard similarity = 0.41; changed final feature statuses = 13.
- `washingmachine`: max |Pearson| changed from 0.325 on full data to 0.789 on ON-period data; dropped-feature set Jaccard similarity = 0.48; changed final feature statuses = 11.

Overall, the ON-period dataset changes the target-correlation interpretation substantially for several appliances. This supports the thesis argument that full-dataset correlations can be diluted by many OFF windows. The dropped-feature set can still remain partly stable because Stage 01 first removes feature-to-feature redundancy; target correlation is mainly used to decide which feature survives inside each redundant pair.

## Dataset Size Check

| appliance | full | on_only_buffer2 | row_reduction_ratio_on_vs_full |
| --- | --- | --- | --- |
| dishwasher | 100779 | 3218 | 0.0319 |
| fridge | 100779 | 71102 | 0.7055 |
| kettle | 100780 | 734 | 0.0073 |
| microwave | 100778 | 510 | 0.0051 |
| washingmachine | 100778 | 2667 | 0.0265 |

The ON-period run uses far fewer rows for kettle, microwave, dishwasher, and washing machine. Fridge remains large because its `on_off` duty cycle is high.

## Correlation Summary By Appliance

| appliance | mean_pearson_full | mean_pearson_on | median_pearson_full | median_pearson_on | max_pearson_full | max_pearson_on | mean_spearman_full | mean_spearman_on | median_spearman_full | median_spearman_on |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dishwasher | 0.251066 | 0.473662 | 0.204330 | 0.478043 | 0.706843 | 0.937832 | 0.054169 | 0.398648 | 0.057681 | 0.374629 |
| fridge | 0.163980 | 0.050860 | 0.048411 | 0.028624 | 0.711483 | 0.243228 | 0.236084 | 0.121916 | 0.158510 | 0.092011 |
| kettle | 0.205482 | 0.353683 | 0.111798 | 0.331283 | 0.661840 | 0.739824 | 0.059711 | 0.231720 | 0.075899 | 0.242350 |
| microwave | 0.135736 | 0.373157 | 0.081259 | 0.352976 | 0.674096 | 0.740819 | 0.068970 | 0.298020 | 0.072765 | 0.293205 |
| washingmachine | 0.128821 | 0.400396 | 0.119218 | 0.476908 | 0.325452 | 0.788725 | 0.051649 | 0.287648 | 0.040057 | 0.346327 |

Interpretation: if `mean_pearson_on` or `max_pearson_on` is much larger than the full-data value, the full dataset was masking appliance-specific relationships because OFF windows dominated the calculation.

## Top Target-Correlated Features: Full Dataset

### kettle
| feature | domain | target_pearson_abs | target_spearman_abs |
| --- | --- | --- | --- |
| I_BP_low | band_power | 0.661840 | 0.092584 |
| DWT_E0 | wavelet | 0.660356 | 0.093494 |
| DWT_E4 | wavelet | 0.592774 | 0.081267 |
| I_BP_high | band_power | 0.581922 | 0.081446 |
| DWT_E3 | wavelet | 0.567735 | 0.083371 |
| I_env_6 | spectral_envelope | 0.551700 | 0.073495 |
| I_env_7 | spectral_envelope | 0.547962 | 0.084984 |
| P_active | time_domain | 0.540881 | 0.093714 |
| I_rms | time_domain | 0.535385 | 0.093566 |
| I_std | shape_statistics | 0.535385 | 0.093566 |

### fridge
| feature | domain | target_pearson_abs | target_spearman_abs |
| --- | --- | --- | --- |
| I_env_0 | spectral_envelope | 0.711483 | 0.638043 |
| I_env_1 | spectral_envelope | 0.699363 | 0.668271 |
| THDI | distortion | 0.681921 | 0.615595 |
| I_env_2 | spectral_envelope | 0.628174 | 0.462215 |
| I_spec_entropy | spectral_descriptors | 0.609807 | 0.544828 |
| PF | time_domain | 0.599541 | 0.551740 |
| I7 | harmonics | 0.513223 | 0.442852 |
| I_skew | shape_statistics | 0.457811 | 0.446881 |
| I_env_3 | spectral_envelope | 0.367298 | 0.301767 |
| I5 | harmonics | 0.332073 | 0.340033 |

### microwave
| feature | domain | target_pearson_abs | target_spearman_abs |
| --- | --- | --- | --- |
| IH | distortion | 0.674096 | 0.114299 |
| I3 | harmonics | 0.661995 | 0.115216 |
| I5 | harmonics | 0.567264 | 0.123946 |
| DWT_E1 | wavelet | 0.359972 | 0.091507 |
| I11 | harmonics | 0.331071 | 0.073739 |
| I13 | harmonics | 0.277788 | 0.092410 |
| I9 | harmonics | 0.263323 | 0.085358 |
| I_BP_mid | band_power | 0.232919 | 0.089608 |
| S_apparent | time_domain | 0.204094 | 0.116026 |
| I_std | shape_statistics | 0.202445 | 0.115484 |

### dishwasher
| feature | domain | target_pearson_abs | target_spearman_abs |
| --- | --- | --- | --- |
| P_active | time_domain | 0.706843 | 0.064339 |
| I1 | harmonics | 0.697227 | 0.065665 |
| I_rms | time_domain | 0.695518 | 0.064987 |
| I_std | shape_statistics | 0.695518 | 0.064987 |
| S_apparent | time_domain | 0.693725 | 0.065493 |
| DWT_E0 | wavelet | 0.647015 | 0.064983 |
| I_BP_low | band_power | 0.640257 | 0.065484 |
| DWT_E4 | wavelet | 0.500967 | 0.055223 |
| I_env_6 | spectral_envelope | 0.488769 | 0.063181 |
| I_BP_high | band_power | 0.479700 | 0.056372 |

### washingmachine
| feature | domain | target_pearson_abs | target_spearman_abs |
| --- | --- | --- | --- |
| S_apparent | time_domain | 0.325452 | 0.010325 |
| I_std | shape_statistics | 0.325201 | 0.013843 |
| I_rms | time_domain | 0.325201 | 0.013843 |
| I1 | harmonics | 0.323113 | 0.015761 |
| P_active | time_domain | 0.315044 | 0.029345 |
| DWT_E0 | wavelet | 0.265536 | 0.014185 |
| DWT_E4 | wavelet | 0.248850 | 0.015524 |
| I_BP_low | band_power | 0.248614 | 0.011739 |
| I_BP_high | band_power | 0.241413 | 0.040469 |
| I_env_6 | spectral_envelope | 0.241224 | 0.005578 |

## Top Target-Correlated Features: ON-Period Dataset

### kettle
| feature | domain | target_pearson_abs | target_spearman_abs |
| --- | --- | --- | --- |
| THDI | distortion | 0.739824 | 0.383523 |
| P_active | time_domain | 0.722558 | 0.356325 |
| S_apparent | time_domain | 0.713103 | 0.340601 |
| I1 | harmonics | 0.707851 | 0.261529 |
| I_std | shape_statistics | 0.706998 | 0.265086 |
| I_rms | time_domain | 0.706998 | 0.265085 |
| I_skew | shape_statistics | 0.660907 | 0.268700 |
| I_spec_entropy | spectral_descriptors | 0.617282 | 0.478338 |
| I_env_1 | spectral_envelope | 0.612225 | 0.424105 |
| Fci | time_domain | 0.608263 | 0.314041 |

### fridge
| feature | domain | target_pearson_abs | target_spearman_abs |
| --- | --- | --- | --- |
| I_env_1 | spectral_envelope | 0.243228 | 0.420635 |
| I_env_0 | spectral_envelope | 0.217225 | 0.334834 |
| THDI | distortion | 0.187979 | 0.318532 |
| I_spec_entropy | spectral_descriptors | 0.177485 | 0.224446 |
| PF | time_domain | 0.162373 | 0.195750 |
| I_skew | shape_statistics | 0.122411 | 0.233557 |
| I_env_2 | spectral_envelope | 0.099850 | 0.089882 |
| Fci | time_domain | 0.096090 | 0.207380 |
| I11 | harmonics | 0.083406 | 0.203161 |
| V7 | harmonics | 0.082308 | 0.144916 |

### microwave
| feature | domain | target_pearson_abs | target_spearman_abs |
| --- | --- | --- | --- |
| I5 | harmonics | 0.740819 | 0.598729 |
| IH | distortion | 0.709847 | 0.667342 |
| I3 | harmonics | 0.700142 | 0.664460 |
| P_active | time_domain | 0.686368 | 0.568028 |
| I11 | harmonics | 0.684235 | 0.489778 |
| S_apparent | time_domain | 0.666108 | 0.523356 |
| I_std | shape_statistics | 0.665318 | 0.515818 |
| I_rms | time_domain | 0.665318 | 0.515818 |
| I1 | harmonics | 0.646067 | 0.480998 |
| I_env_2 | spectral_envelope | 0.645713 | 0.515254 |

### dishwasher
| feature | domain | target_pearson_abs | target_spearman_abs |
| --- | --- | --- | --- |
| THDI | distortion | 0.937832 | 0.719748 |
| P_active | time_domain | 0.908865 | 0.791665 |
| S_apparent | time_domain | 0.907159 | 0.791291 |
| I1 | harmonics | 0.905739 | 0.767695 |
| I_std | shape_statistics | 0.905115 | 0.768313 |
| I_rms | time_domain | 0.905115 | 0.768313 |
| I_skew | shape_statistics | 0.897208 | 0.779902 |
| I_env_1 | spectral_envelope | 0.888406 | 0.707603 |
| I_env_0 | spectral_envelope | 0.854660 | 0.724881 |
| Fci | time_domain | 0.833560 | 0.753773 |

### washingmachine
| feature | domain | target_pearson_abs | target_spearman_abs |
| --- | --- | --- | --- |
| P_active | time_domain | 0.788725 | 0.550725 |
| DWT_E0 | wavelet | 0.771022 | 0.507866 |
| I_rms | time_domain | 0.764040 | 0.507433 |
| I_std | shape_statistics | 0.764040 | 0.507432 |
| I1 | harmonics | 0.763559 | 0.488360 |
| S_apparent | time_domain | 0.762432 | 0.507054 |
| THDI | distortion | 0.747422 | 0.553538 |
| I_BP_low | band_power | 0.744090 | 0.479500 |
| DWT_E4 | wavelet | 0.584517 | 0.381835 |
| V_std | shape_statistics | 0.580205 | 0.378655 |

## Largest Pearson Correlation Increases When Using ON-Period Data

### kettle
| feature | domain | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full | target_spearman_abs_full | target_spearman_abs_on | spearman_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| THDI | distortion | 0.173817 | 0.739824 | 0.566007 | 0.092386 | 0.383523 | 0.291137 |
| I_skew | shape_statistics | 0.110951 | 0.660907 | 0.549955 | 0.091450 | 0.268700 | 0.177250 |
| Fci | time_domain | 0.115998 | 0.608263 | 0.492265 | 0.088433 | 0.314041 | 0.225607 |
| I_spec_entropy | spectral_descriptors | 0.155100 | 0.617282 | 0.462182 | 0.088842 | 0.478338 | 0.389497 |
| I_env_1 | spectral_envelope | 0.151686 | 0.612225 | 0.460538 | 0.088410 | 0.424105 | 0.335695 |
| I_env_0 | spectral_envelope | 0.125394 | 0.556760 | 0.431366 | 0.085979 | 0.492736 | 0.406756 |
| PF | time_domain | 0.103710 | 0.517130 | 0.413420 | 0.086380 | 0.283382 | 0.197003 |
| I_env_2 | spectral_envelope | 0.098216 | 0.499862 | 0.401646 | 0.078302 | 0.346703 | 0.268400 |
| I15 | harmonics | 0.170983 | 0.564417 | 0.393433 | 0.085006 | 0.335673 | 0.250667 |
| I_env_3 | spectral_envelope | 0.099843 | 0.459734 | 0.359891 | 0.078395 | 0.384205 | 0.305809 |
| I_kurt | shape_statistics | 0.009774 | 0.314655 | 0.304881 | 0.087578 | 0.374262 | 0.286684 |
| I_env_4 | spectral_envelope | 0.000729 | 0.241832 | 0.241103 | 0.008906 | 0.059686 | 0.050779 |

### fridge
| feature | domain | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full | target_spearman_abs_full | target_spearman_abs_on | spearman_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V7 | harmonics | 0.040870 | 0.082308 | 0.041437 | 0.029820 | 0.144916 | 0.115096 |
| V15 | harmonics | 0.031897 | 0.066498 | 0.034602 | 0.097323 | 0.152897 | 0.055574 |
| I11 | harmonics | 0.049379 | 0.083406 | 0.034026 | 0.118948 | 0.203161 | 0.084213 |
| V_skew | shape_statistics | 0.028346 | 0.057939 | 0.029593 | 0.082847 | 0.117903 | 0.035056 |
| V_std | shape_statistics | 0.004813 | 0.030117 | 0.025304 | 0.073996 | 0.069091 | -0.004905 |
| V_rms | time_domain | 0.004813 | 0.030116 | 0.025303 | 0.073996 | 0.069090 | -0.004906 |
| I_env_7 | spectral_envelope | 0.008478 | 0.033328 | 0.024849 | 0.083764 | 0.019675 | -0.064089 |
| I_env_6 | spectral_envelope | 0.013581 | 0.035279 | 0.021698 | 0.119256 | 0.018047 | -0.101209 |
| V1 | harmonics | 0.007769 | 0.026192 | 0.018423 | 0.074050 | 0.069788 | -0.004262 |
| V9 | harmonics | 0.045070 | 0.057167 | 0.012096 | 0.122729 | 0.186433 | 0.063704 |
| I3 | harmonics | 0.037709 | 0.049515 | 0.011806 | 0.214856 | 0.128640 | -0.086216 |
| V_BP_low | band_power | 0.006960 | 0.017276 | 0.010316 | 0.005176 | 0.004463 | -0.000714 |

### microwave
| feature | domain | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full | target_spearman_abs_full | target_spearman_abs_on | spearman_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| I_env_2 | spectral_envelope | 0.060970 | 0.645713 | 0.584742 | 0.030435 | 0.515254 | 0.484820 |
| I_env_0 | spectral_envelope | 0.127102 | 0.632778 | 0.505676 | 0.053495 | 0.615890 | 0.562395 |
| P_active | time_domain | 0.182106 | 0.686368 | 0.504262 | 0.115615 | 0.568028 | 0.452413 |
| Fci | time_domain | 0.059267 | 0.534992 | 0.475726 | 0.104175 | 0.295908 | 0.191733 |
| I_std | shape_statistics | 0.202445 | 0.665318 | 0.462873 | 0.115484 | 0.515818 | 0.400334 |
| I_rms | time_domain | 0.202445 | 0.665318 | 0.462872 | 0.115484 | 0.515818 | 0.400334 |
| S_apparent | time_domain | 0.204094 | 0.666108 | 0.462014 | 0.116026 | 0.523356 | 0.407330 |
| I1 | harmonics | 0.191965 | 0.646067 | 0.454103 | 0.114864 | 0.480998 | 0.366134 |
| PF | time_domain | 0.004381 | 0.452454 | 0.448074 | 0.009746 | 0.144455 | 0.134709 |
| I_env_1 | spectral_envelope | 0.167216 | 0.579370 | 0.412154 | 0.066208 | 0.555561 | 0.489353 |
| I15 | harmonics | 0.186647 | 0.563842 | 0.377195 | 0.054633 | 0.357307 | 0.302674 |
| I_BP_mid | band_power | 0.232919 | 0.591695 | 0.358776 | 0.089608 | 0.440274 | 0.350665 |

### dishwasher
| feature | domain | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full | target_spearman_abs_full | target_spearman_abs_on | spearman_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| I_skew | shape_statistics | 0.202627 | 0.897208 | 0.694580 | 0.068890 | 0.779902 | 0.711012 |
| I_env_1 | spectral_envelope | 0.237252 | 0.888406 | 0.651153 | 0.145569 | 0.707603 | 0.562034 |
| I_kurt | shape_statistics | 0.018748 | 0.647410 | 0.628662 | 0.102165 | 0.731882 | 0.629717 |
| THDI | distortion | 0.310601 | 0.937832 | 0.627232 | 0.105171 | 0.719748 | 0.614577 |
| Fci | time_domain | 0.207361 | 0.833560 | 0.626199 | 0.107854 | 0.753773 | 0.645919 |
| I_env_0 | spectral_envelope | 0.233472 | 0.854660 | 0.621187 | 0.132711 | 0.724881 | 0.592169 |
| PF | time_domain | 0.206033 | 0.705373 | 0.499340 | 0.087941 | 0.609203 | 0.521262 |
| I_spec_entropy | spectral_descriptors | 0.292255 | 0.731460 | 0.439205 | 0.090962 | 0.705925 | 0.614963 |
| I5 | harmonics | 0.028080 | 0.430329 | 0.402249 | 0.017712 | 0.435642 | 0.417931 |
| V3 | harmonics | 0.090643 | 0.479369 | 0.388726 | 0.075674 | 0.203217 | 0.127543 |
| I3 | harmonics | 0.058251 | 0.421838 | 0.363587 | 0.000134 | 0.345079 | 0.344945 |
| I_env_3 | spectral_envelope | 0.201659 | 0.557758 | 0.356099 | 0.111601 | 0.488151 | 0.376550 |

### washingmachine
| feature | domain | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full | target_spearman_abs_full | target_spearman_abs_on | spearman_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| THDI | distortion | 0.145619 | 0.747422 | 0.601803 | 0.030179 | 0.553538 | 0.523359 |
| DWT_E0 | wavelet | 0.265536 | 0.771022 | 0.505486 | 0.014185 | 0.507866 | 0.493681 |
| I_BP_low | band_power | 0.248614 | 0.744090 | 0.495475 | 0.011739 | 0.479500 | 0.467761 |
| I_env_0 | spectral_envelope | 0.063293 | 0.553599 | 0.490307 | 0.016279 | 0.465775 | 0.449496 |
| PF | time_domain | 0.028846 | 0.513836 | 0.484990 | 0.101671 | 0.494440 | 0.392769 |
| V1 | harmonics | 0.098146 | 0.576748 | 0.478602 | 0.093775 | 0.373892 | 0.280116 |
| VH | distortion | 0.088534 | 0.564094 | 0.475560 | 0.124299 | 0.440025 | 0.315726 |
| P_active | time_domain | 0.315044 | 0.788725 | 0.473681 | 0.029345 | 0.550725 | 0.521380 |
| I_env_1 | spectral_envelope | 0.034632 | 0.504472 | 0.469841 | 0.040125 | 0.405278 | 0.365153 |
| V_std | shape_statistics | 0.122801 | 0.580205 | 0.457404 | 0.096668 | 0.378655 | 0.281987 |
| V_rms | time_domain | 0.122801 | 0.580205 | 0.457404 | 0.096667 | 0.378650 | 0.281983 |
| THDV | distortion | 0.075988 | 0.528131 | 0.452143 | 0.126236 | 0.435530 | 0.309294 |

## Largest Pearson Correlation Decreases When Using ON-Period Data

### kettle
| feature | domain | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full | target_spearman_abs_full | target_spearman_abs_on | spearman_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DWT_E4 | wavelet | 0.592774 | 0.377149 | -0.215625 | 0.081267 | 0.174963 | 0.093696 |
| I_BP_high | band_power | 0.581922 | 0.378633 | -0.203289 | 0.081446 | 0.164829 | 0.083383 |
| DWT_E3 | wavelet | 0.567735 | 0.378285 | -0.189450 | 0.083371 | 0.152869 | 0.069498 |
| I_BP_low | band_power | 0.661840 | 0.511984 | -0.149856 | 0.092584 | 0.401247 | 0.308663 |
| I_env_7 | spectral_envelope | 0.547962 | 0.404182 | -0.143780 | 0.084984 | 0.174065 | 0.089081 |
| I_env_6 | spectral_envelope | 0.551700 | 0.410608 | -0.141092 | 0.073495 | 0.164431 | 0.090935 |
| DWT_E0 | wavelet | 0.660356 | 0.523368 | -0.136989 | 0.093494 | 0.265386 | 0.171892 |
| IH | distortion | 0.022459 | 0.019331 | -0.003128 | 0.030958 | 0.081682 | 0.050724 |

### fridge
| feature | domain | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full | target_spearman_abs_full | target_spearman_abs_on | spearman_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| I_env_2 | spectral_envelope | 0.628174 | 0.099850 | -0.528323 | 0.462215 | 0.089882 | -0.372333 |
| I_env_0 | spectral_envelope | 0.711483 | 0.217225 | -0.494258 | 0.638043 | 0.334834 | -0.303209 |
| THDI | distortion | 0.681921 | 0.187979 | -0.493942 | 0.615595 | 0.318532 | -0.297063 |
| I7 | harmonics | 0.513223 | 0.024785 | -0.488439 | 0.442852 | 0.005208 | -0.437644 |
| I_env_1 | spectral_envelope | 0.699363 | 0.243228 | -0.456134 | 0.668271 | 0.420635 | -0.247636 |
| PF | time_domain | 0.599541 | 0.162373 | -0.437169 | 0.551740 | 0.195750 | -0.355990 |
| I_spec_entropy | spectral_descriptors | 0.609807 | 0.177485 | -0.432322 | 0.544828 | 0.224446 | -0.320383 |
| I_skew | shape_statistics | 0.457811 | 0.122411 | -0.335400 | 0.446881 | 0.233557 | -0.213325 |

### microwave
| feature | domain | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full | target_spearman_abs_full | target_spearman_abs_on | spearman_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fcv | time_domain | 0.011661 | 0.003551 | -0.008110 | 0.041472 | 0.024207 | -0.017265 |
| VH | distortion | 0.002346 | 0.005584 | 0.003237 | 0.010544 | 0.041041 | 0.030496 |
| V_BP_low | band_power | 0.004197 | 0.008786 | 0.004589 | 0.049439 | 0.031914 | -0.017526 |
| V3 | harmonics | 0.022191 | 0.038905 | 0.016714 | 0.045699 | 0.019766 | -0.025934 |
| IH | distortion | 0.674096 | 0.709847 | 0.035751 | 0.114299 | 0.667342 | 0.553043 |
| V11 | harmonics | 0.006990 | 0.043949 | 0.036959 | 0.012416 | 0.064402 | 0.051986 |
| I3 | harmonics | 0.661995 | 0.700142 | 0.038147 | 0.115216 | 0.664460 | 0.549244 |
| V5 | harmonics | 0.024048 | 0.070899 | 0.046851 | 0.038132 | 0.015506 | -0.022626 |

### dishwasher
| feature | domain | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full | target_spearman_abs_full | target_spearman_abs_on | spearman_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| I_env_4 | spectral_envelope | 0.049630 | 0.009261 | -0.040369 | 0.033752 | 0.071489 | 0.037738 |
| V7 | harmonics | 0.128618 | 0.099614 | -0.029004 | 0.003211 | 0.079468 | 0.076257 |
| DWT_E4 | wavelet | 0.500967 | 0.492029 | -0.008938 | 0.055223 | 0.510255 | 0.455033 |
| I_BP_high | band_power | 0.479700 | 0.476716 | -0.002984 | 0.056372 | 0.503763 | 0.447391 |
| DWT_E3 | wavelet | 0.478400 | 0.481129 | 0.002729 | 0.057550 | 0.515940 | 0.458390 |
| I_env_6 | spectral_envelope | 0.488769 | 0.514435 | 0.025667 | 0.063181 | 0.448165 | 0.384984 |
| I11 | harmonics | 0.032743 | 0.060871 | 0.028128 | 0.003195 | 0.155550 | 0.152355 |
| I13 | harmonics | 0.063340 | 0.091655 | 0.028315 | 0.003192 | 0.153568 | 0.150377 |

### washingmachine
| feature | domain | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full | target_spearman_abs_full | target_spearman_abs_on | spearman_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| I7 | harmonics | 0.117794 | 0.016911 | -0.100882 | 0.057465 | 0.016867 | -0.040598 |
| DWT_E1 | wavelet | 0.139555 | 0.055414 | -0.084140 | 0.039990 | 0.072774 | 0.032784 |
| I11 | harmonics | 0.092820 | 0.010308 | -0.082512 | 0.158912 | 0.004612 | -0.154300 |
| I3 | harmonics | 0.191764 | 0.109479 | -0.082285 | 0.066573 | 0.077245 | 0.010671 |
| IH | distortion | 0.179221 | 0.124742 | -0.054479 | 0.044323 | 0.096000 | 0.051677 |
| I9 | harmonics | 0.128407 | 0.089295 | -0.039112 | 0.035733 | 0.033463 | -0.002271 |
| V13 | harmonics | 0.003378 | 0.000899 | -0.002479 | 0.046570 | 0.003187 | -0.043383 |
| V9 | harmonics | 0.016042 | 0.025097 | 0.009055 | 0.083460 | 0.018708 | -0.064753 |

## Dropped-Feature Set Comparison

| appliance | dropped_full | dropped_on | changed_features | jaccard_dropped_sets |
| --- | --- | --- | --- | --- |
| dishwasher | 16 | 17 | 11 | 0.500000 |
| fridge | 16 | 14 | 6 | 0.666700 |
| kettle | 16 | 14 | 12 | 0.428600 |
| microwave | 16 | 15 | 13 | 0.409100 |
| washingmachine | 16 | 15 | 11 | 0.476200 |

`jaccard_dropped_sets` measures overlap between dropped-feature sets: 1.0 means identical dropped features, 0.0 means no overlap. A lower value means the dataset choice changes the Stage 01 final selection more strongly.

## Dropped/Kept Transition Counts

| appliance | drop_change | n_features |
| --- | --- | --- |
| dishwasher | dropped_both | 11 |
| dishwasher | dropped_full_only | 5 |
| dishwasher | dropped_on_only | 6 |
| dishwasher | kept_both | 28 |
| fridge | dropped_both | 12 |
| fridge | dropped_full_only | 4 |
| fridge | dropped_on_only | 2 |
| fridge | kept_both | 32 |
| kettle | dropped_both | 9 |
| kettle | dropped_full_only | 7 |
| kettle | dropped_on_only | 5 |
| kettle | kept_both | 29 |
| microwave | dropped_both | 9 |
| microwave | dropped_full_only | 7 |
| microwave | dropped_on_only | 6 |
| microwave | kept_both | 28 |
| washingmachine | dropped_both | 10 |
| washingmachine | dropped_full_only | 6 |
| washingmachine | dropped_on_only | 5 |
| washingmachine | kept_both | 29 |

Definitions: `dropped_both` = removed in both runs; `dropped_full_only` = removed only using full data; `dropped_on_only` = removed only using ON-period data; `kept_both` = retained in both runs.

## Features Whose Final Status Changed

### kettle
| feature | domain | final_status_full | dropped_at_stage_full | final_status_on | dropped_at_stage_on | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DWT_E0 | wavelet | kept | passed | dropped | correlation | 0.660356 | 0.523368 | -0.136989 |
| DWT_E2 | wavelet | kept | passed | dropped | correlation | 0.333373 | 0.360872 | 0.027499 |
| DWT_E3 | wavelet | dropped | correlation | kept | passed | 0.567735 | 0.378285 | -0.189450 |
| DWT_E4 | wavelet | kept | passed | dropped | correlation | 0.592774 | 0.377149 | -0.215625 |
| I3 | harmonics | dropped | correlation | kept | passed | 0.014516 | 0.143926 | 0.129410 |
| I_BP_low | band_power | dropped | correlation | kept | passed | 0.661840 | 0.511984 | -0.149856 |
| I_env_0 | spectral_envelope | dropped | correlation | kept | passed | 0.125394 | 0.556760 | 0.431366 |
| I_env_4 | spectral_envelope | kept | passed | dropped | correlation | 0.000729 | 0.241832 | 0.241103 |
| I_env_5 | spectral_envelope | kept | passed | dropped | correlation | 0.334235 | 0.379580 | 0.045346 |
| I_kurt | shape_statistics | dropped | correlation | kept | passed | 0.009774 | 0.314655 | 0.304881 |
| I_skew | shape_statistics | dropped | correlation | kept | passed | 0.110951 | 0.660907 | 0.549955 |
| P_active | time_domain | dropped | correlation | kept | passed | 0.540881 | 0.722558 | 0.181677 |

### fridge
| feature | domain | final_status_full | dropped_at_stage_full | final_status_on | dropped_at_stage_on | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I3 | harmonics | dropped | correlation | kept | passed | 0.037709 | 0.049515 | 0.011806 |
| IH | distortion | kept | passed | dropped | correlation | 0.047281 | 0.035464 | -0.011817 |
| I_BP_mid | band_power | kept | passed | dropped | correlation | 0.029913 | 0.000676 | -0.029237 |
| I_kurt | shape_statistics | dropped | correlation | kept | passed | 0.052797 | 0.043302 | -0.009496 |
| I_rms | time_domain | dropped | correlation | kept | passed | 0.110876 | 0.010313 | -0.100563 |
| THDI | distortion | dropped | correlation | kept | passed | 0.681921 | 0.187979 | -0.493942 |

### microwave
| feature | domain | final_status_full | dropped_at_stage_full | final_status_on | dropped_at_stage_on | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IH | distortion | kept | passed | dropped | correlation | 0.674096 | 0.709847 | 0.035751 |
| I_BP_high | band_power | dropped | correlation | kept | passed | 0.147732 | 0.341238 | 0.193507 |
| I_BP_low | band_power | dropped | correlation | kept | passed | 0.131141 | 0.465937 | 0.334796 |
| I_env_1 | spectral_envelope | kept | passed | dropped | correlation | 0.167216 | 0.579370 | 0.412154 |
| I_env_5 | spectral_envelope | kept | passed | dropped | correlation | 0.040275 | 0.219289 | 0.179014 |
| I_env_6 | spectral_envelope | kept | passed | dropped | correlation | 0.101349 | 0.290545 | 0.189196 |
| I_kurt | shape_statistics | dropped | correlation | kept | passed | 0.006305 | 0.284179 | 0.277874 |
| I_rms | time_domain | kept | passed | dropped | correlation | 0.202445 | 0.665318 | 0.462872 |
| I_skew | shape_statistics | dropped | correlation | kept | passed | 0.062571 | 0.136902 | 0.074332 |
| P_active | time_domain | dropped | correlation | kept | passed | 0.182106 | 0.686368 | 0.504262 |
| THDI | distortion | dropped | correlation | kept | passed | 0.032820 | 0.166433 | 0.133613 |
| THDV | distortion | dropped | correlation | kept | passed | 0.009942 | 0.064709 | 0.054767 |
| VH | distortion | kept | passed | dropped | correlation | 0.002346 | 0.005584 | 0.003237 |

### dishwasher
| feature | domain | final_status_full | dropped_at_stage_full | final_status_on | dropped_at_stage_on | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DWT_E4 | wavelet | kept | passed | dropped | correlation | 0.500967 | 0.492029 | -0.008938 |
| I3 | harmonics | dropped | correlation | kept | passed | 0.058251 | 0.421838 | 0.363587 |
| I_BP_low | band_power | dropped | correlation | kept | passed | 0.640257 | 0.672080 | 0.031823 |
| I_BP_mid | band_power | kept | passed | dropped | correlation | 0.217570 | 0.313735 | 0.096165 |
| I_env_0 | spectral_envelope | dropped | correlation | kept | passed | 0.233472 | 0.854660 | 0.621187 |
| I_env_1 | spectral_envelope | kept | passed | dropped | correlation | 0.237252 | 0.888406 | 0.651153 |
| I_env_5 | spectral_envelope | kept | passed | dropped | correlation | 0.266554 | 0.431583 | 0.165029 |
| I_rms | time_domain | dropped | correlation | kept | passed | 0.695518 | 0.905115 | 0.209597 |
| P_active | time_domain | kept | passed | dropped | correlation | 0.706843 | 0.908865 | 0.202022 |
| THDV | distortion | dropped | correlation | kept | passed | 0.162140 | 0.506294 | 0.344155 |
| VH | distortion | kept | passed | dropped | correlation | 0.195398 | 0.491100 | 0.295702 |

### washingmachine
| feature | domain | final_status_full | dropped_at_stage_full | final_status_on | dropped_at_stage_on | target_pearson_abs_full | target_pearson_abs_on | pearson_delta_on_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DWT_E3 | wavelet | kept | passed | dropped | correlation | 0.239118 | 0.574323 | 0.335206 |
| DWT_E4 | wavelet | dropped | correlation | kept | passed | 0.248850 | 0.584517 | 0.335667 |
| I3 | harmonics | kept | passed | dropped | correlation | 0.191764 | 0.109479 | -0.082285 |
| IH | distortion | dropped | correlation | kept | passed | 0.179221 | 0.124742 | -0.054479 |
| I_env_0 | spectral_envelope | dropped | correlation | kept | passed | 0.063293 | 0.553599 | 0.490307 |
| I_env_1 | spectral_envelope | kept | passed | dropped | correlation | 0.034632 | 0.504472 | 0.469841 |
| I_env_5 | spectral_envelope | kept | passed | dropped | correlation | 0.125445 | 0.549579 | 0.424134 |
| I_kurt | shape_statistics | dropped | correlation | kept | passed | 0.009086 | 0.284430 | 0.275344 |
| I_rms | time_domain | kept | passed | dropped | correlation | 0.325201 | 0.764040 | 0.438839 |
| I_skew | shape_statistics | dropped | correlation | kept | passed | 0.120643 | 0.213136 | 0.092493 |
| P_active | time_domain | dropped | correlation | kept | passed | 0.315044 | 0.788725 | 0.473681 |

## Methodological Interpretation

Full-dataset target correlation answers: which HF features track appliance power across all windows, including many OFF periods. This is useful for broad ON/OFF separability, but it can underestimate feature relevance for appliances that are rarely active or have multi-state cycles.

ON-period target correlation answers: which HF features track appliance power while the appliance is active, including a small transition buffer. This is closer to the thesis question of appliance-state feature relevance and should be more informative for washing machine, dishwasher, microwave, and kettle.

Stage 01 redundancy filtering is not purely a target-correlation ranking. It first identifies highly redundant feature pairs using feature-to-feature Pearson/Spearman correlation. The target correlation is then used as a tie-breaker to decide which feature in a redundant pair is retained. Therefore, large target-correlation changes do not always imply equally large dropped-feature changes.

## Files Generated For Audit

- `feature_selection\stage01_full_vs_on_only_tables\correlation_change_all_features.csv`
- `feature_selection\stage01_full_vs_on_only_tables\correlation_summary.csv`
- `feature_selection\stage01_full_vs_on_only_tables\dropped_feature_status_changes.csv`
- `feature_selection\stage01_full_vs_on_only_tables\dropped_feature_status_comparison.csv`
- `feature_selection\stage01_full_vs_on_only_tables\dropped_status_counts.csv`
- `feature_selection\stage01_full_vs_on_only_tables\dropped_status_summary.csv`
- `feature_selection\stage01_full_vs_on_only_tables\rows_compared.csv`
- `feature_selection\stage01_full_vs_on_only_tables\top10_target_corr_full.csv`
- `feature_selection\stage01_full_vs_on_only_tables\top10_target_corr_on_only.csv`
- `feature_selection\stage01_full_vs_on_only_tables\top_pearson_drops_on_minus_full.csv`
- `feature_selection\stage01_full_vs_on_only_tables\top_pearson_gains_on_minus_full.csv`

These CSV files contain the full comparison tables used to build this report.

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

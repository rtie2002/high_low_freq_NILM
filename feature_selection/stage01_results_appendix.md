# Stage 01 Results Appendix

Audit tables and greedy logs. Main narrative: [feature_selection.md](feature_selection.md).

## Master feature status (50 x 5)

Legend: K = kept, D = dropped.

| Feature | Domain | kettle | fridge | microwave | dishwasher | washingmachine | n_kept |
|---------|--------|:---:|:---:|:---:|:---:|:---:|:------:|
| `V_rms` | time_domain | K | K | K | K | K | 5 |
| `I_rms` | time_domain | D | K | K | K | D | 3 |
| `P_active` | time_domain | K | D | D | D | K | 2 |
| `S_apparent` | time_domain | D | D | D | D | D | 0 |
| `PF` | time_domain | K | K | K | K | K | 5 |
| `Fcv` | time_domain | K | K | K | K | K | 5 |
| `Fci` | time_domain | K | K | K | K | K | 5 |
| `I_skew` | shape_statistics | K | K | K | D | K | 4 |
| `I_kurt` | shape_statistics | K | K | K | D | K | 4 |
| `V_skew` | shape_statistics | K | K | K | K | K | 5 |
| `I_std` | shape_statistics | D | D | D | D | D | 0 |
| `V_std` | shape_statistics | D | D | D | D | D | 0 |
| `I1` | harmonics | D | D | D | D | D | 0 |
| `V1` | harmonics | D | D | D | D | D | 0 |
| `I3` | harmonics | K | K | D | K | D | 3 |
| `V3` | harmonics | K | K | K | K | K | 5 |
| `I5` | harmonics | K | K | D | K | K | 4 |
| `V5` | harmonics | K | K | K | K | K | 5 |
| `I7` | harmonics | K | K | K | K | K | 5 |
| `V7` | harmonics | K | K | K | K | K | 5 |
| `I9` | harmonics | K | K | K | K | K | 5 |
| `V9` | harmonics | K | K | K | K | K | 5 |
| `I11` | harmonics | K | K | K | K | K | 5 |
| `V11` | harmonics | D | K | K | K | K | 4 |
| `I13` | harmonics | K | K | K | K | K | 5 |
| `V13` | harmonics | K | K | K | K | K | 5 |
| `I15` | harmonics | K | K | K | K | K | 5 |
| `V15` | harmonics | K | K | K | K | K | 5 |
| `IH` | distortion | D | D | K | K | K | 3 |
| `VH` | distortion | K | K | K | K | K | 5 |
| `THDI` | distortion | K | K | K | K | K | 5 |
| `THDV` | distortion | D | D | D | D | D | 0 |
| `I_BP_low` | band_power | K | D | K | K | D | 3 |
| `I_BP_mid` | band_power | K | K | K | K | K | 5 |
| `I_BP_high` | band_power | D | D | D | D | D | 0 |
| `V_BP_low` | band_power | K | K | K | K | K | 5 |
| `I_spec_entropy` | spectral_descriptors | K | K | K | K | K | 5 |
| `I_env_0` | spectral_envelope | K | K | K | K | K | 5 |
| `I_env_1` | spectral_envelope | K | K | D | D | D | 2 |
| `I_env_2` | spectral_envelope | K | K | K | K | K | 5 |
| `I_env_3` | spectral_envelope | K | K | K | K | K | 5 |
| `I_env_4` | spectral_envelope | K | K | K | K | K | 5 |
| `I_env_5` | spectral_envelope | D | K | D | D | D | 1 |
| `I_env_6` | spectral_envelope | K | K | D | K | K | 4 |
| `I_env_7` | spectral_envelope | D | D | D | D | D | 0 |
| `DWT_E0` | wavelet | D | D | D | D | D | 0 |
| `DWT_E1` | wavelet | D | K | K | K | K | 4 |
| `DWT_E2` | wavelet | D | D | K | K | K | 3 |
| `DWT_E3` | wavelet | K | K | K | D | D | 3 |
| `DWT_E4` | wavelet | D | D | D | D | K | 1 |

## Greedy elimination logs (16 steps each)

### kettle

| Step | Dropped | Kept | Pair |r| | Reason |
|------|---------|------|--------|------|
| 1 | `V_std` | `V_rms` | 1.000 | priority |
| 2 | `I_std` | `I_rms` | 1.000 | priority |
| 3 | `DWT_E0` | `I_rms` | 1.000 | target |
| 4 | `S_apparent` | `I_rms` | 1.000 | priority |
| 5 | `I_env_7` | `I_env_6` | 0.998 | priority |
| 6 | `I_rms` | `P_active` | 0.995 | target |
| 7 | `I1` | `P_active` | 0.994 | target |
| 8 | `THDV` | `VH` | 0.991 | target |
| 9 | `I_env_5` | `I_BP_high` | 0.983 | target |
| 10 | `I_BP_high` | `I_env_6` | 0.981 | target |
| 11 | `IH` | `I3` | 0.981 | target |
| 12 | `V1` | `V_rms` | 0.974 | target |
| 13 | `DWT_E4` | `DWT_E3` | 0.969 | priority |
| 14 | `DWT_E2` | `DWT_E3` | 0.966 | target |
| 15 | `DWT_E1` | `I_BP_mid` | 0.966 | target |
| 16 | `V11` | `V13` | 0.956 | target |

### fridge

| Step | Dropped | Kept | Pair |r| | Reason |
|------|---------|------|--------|------|
| 1 | `I_std` | `I_rms` | 1.000 | priority |
| 2 | `V_std` | `V_rms` | 1.000 | priority |
| 3 | `S_apparent` | `I_rms` | 1.000 | priority |
| 4 | `DWT_E0` | `I_rms` | 1.000 | priority |
| 5 | `I1` | `I_rms` | 0.999 | priority |
| 6 | `P_active` | `I_rms` | 0.997 | priority |
| 7 | `IH` | `I3` | 0.992 | target |
| 8 | `THDV` | `VH` | 0.990 | priority |
| 9 | `V1` | `V_rms` | 0.987 | priority |
| 10 | `I_env_7` | `I_env_6` | 0.985 | priority |
| 11 | `I_BP_low` | `I_rms` | 0.977 | priority |
| 12 | `I_BP_high` | `DWT_E4` | 0.972 | priority |
| 13 | `DWT_E2` | `I_BP_mid` | 0.962 | target |
| 14 | `DWT_E4` | `DWT_E3` | 0.961 | priority |

### microwave

| Step | Dropped | Kept | Pair |r| | Reason |
|------|---------|------|--------|------|
| 1 | `V_std` | `V_rms` | 1.000 | priority |
| 2 | `I_std` | `I_rms` | 1.000 | priority |
| 3 | `DWT_E0` | `I_rms` | 1.000 | target |
| 4 | `S_apparent` | `I_rms` | 1.000 | priority |
| 5 | `I3` | `IH` | 0.999 | priority |
| 6 | `V1` | `V_rms` | 0.998 | priority |
| 7 | `I_env_7` | `I_env_6` | 0.997 | priority |
| 8 | `I1` | `I_rms` | 0.997 | target |
| 9 | `P_active` | `I_rms` | 0.994 | priority |
| 10 | `I_env_6` | `I_BP_high` | 0.984 | target |
| 11 | `I_env_1` | `I_env_0` | 0.981 | target |
| 12 | `THDV` | `VH` | 0.981 | priority |
| 13 | `DWT_E4` | `DWT_E3` | 0.972 | priority |
| 14 | `I_env_5` | `I_BP_high` | 0.968 | target |
| 15 | `I5` | `IH` | 0.963 | priority |
| 16 | `I_BP_high` | `DWT_E3` | 0.956 | priority |

### dishwasher

| Step | Dropped | Kept | Pair |r| | Reason |
|------|---------|------|--------|------|
| 1 | `I_std` | `I_rms` | 1.000 | priority |
| 2 | `V_std` | `V_rms` | 1.000 | priority |
| 3 | `DWT_E0` | `I_rms` | 1.000 | target |
| 4 | `S_apparent` | `I_rms` | 1.000 | priority |
| 5 | `P_active` | `I_rms` | 1.000 | priority |
| 6 | `I1` | `I_rms` | 0.999 | priority |
| 7 | `I_env_7` | `I_env_6` | 0.998 | priority |
| 8 | `THDV` | `VH` | 0.993 | priority |
| 9 | `V1` | `V_rms` | 0.986 | target |
| 10 | `I_BP_high` | `I_env_6` | 0.983 | target |
| 11 | `DWT_E3` | `DWT_E4` | 0.982 | target |
| 12 | `I_env_5` | `I_env_6` | 0.975 | target |
| 13 | `I_kurt` | `THDI` | 0.969 | target |
| 14 | `I_env_1` | `THDI` | 0.961 | target |
| 15 | `DWT_E4` | `I_env_6` | 0.959 | target |
| 16 | `I_skew` | `THDI` | 0.955 | target |

### washingmachine

| Step | Dropped | Kept | Pair |r| | Reason |
|------|---------|------|--------|------|
| 1 | `V_std` | `V_rms` | 1.000 | priority |
| 2 | `I_std` | `I_rms` | 1.000 | priority |
| 3 | `DWT_E0` | `I_rms` | 1.000 | target |
| 4 | `S_apparent` | `I_rms` | 1.000 | priority |
| 5 | `I3` | `IH` | 0.997 | target |
| 6 | `V1` | `V_rms` | 0.997 | priority |
| 7 | `I1` | `I_rms` | 0.994 | priority |
| 8 | `I_env_7` | `I_env_6` | 0.993 | priority |
| 9 | `I_rms` | `P_active` | 0.990 | target |
| 10 | `THDV` | `VH` | 0.985 | target |
| 11 | `DWT_E3` | `DWT_E4` | 0.983 | target |
| 12 | `I_env_5` | `I_env_6` | 0.978 | target |
| 13 | `I_BP_high` | `I_env_6` | 0.975 | target |
| 14 | `I_env_1` | `I_env_0` | 0.964 | target |
| 15 | `I_BP_low` | `P_active` | 0.960 | target |

# REFIT house summary (channels, appliance coverage, usable weeks)

Status note for local copy under `dataset_preprocess/REFIT/` (2026-08-12).

Notes:

1. **Source files** in this repo: `dataset_preprocess/REFIT/House_*.csv` (20 houses present; **House 14 missing**).
2. **Columns per file:** `Time`, `Unix`, `Aggregate`, `Appliance1` … `Appliance9`.
   - **1 aggregate channel** + **9 IAM appliance channels** per house (**10** power channels total).
3. **Sampling:** official REFIT cleaned data is **8 seconds** (these CSVs match that layout).
4. **Weeks:** computed from the `Time` column. `covered_weeks` subtracts gaps where consecutive samples are more than **30s** apart.
5. Our multi-appliance trainer expects **5 appliances:** `kettle`, `fridge`, `dishwasher`, `washingmachine`, `microwave` (`config/experiment_refit.yaml`).

Raw summary CSV (machine-readable):  
`dataset_preprocess/created_data/REFIT/_refit_house_summary.csv`

---

## Selected 6 houses (full 5-appliance set)

Same style as the REDD overview table. These are the only houses we use for the 5-app protocol (`2, 3, 5, 9, 11, 20`).

| REFIT house | meter channels | Calendar weeks |
| ---: | ---: | ---: |
| 2 | 10 | 88.20 |
| 3 | 10 | 87.81 |
| 5 | 10 | 92.62 |
| 9 | 10 | 81.15 |
| 11 | 10 | 56.02 |
| 20 | 10 | 65.76 |

Notes for this table:

- **meter channels = 10** for every REFIT house: `Aggregate` + `Appliance1`…`Appliance9` (unlike REDD, channel count does not vary by house).
- Calendar weeks are from the raw `House_*.csv` time span (native ~8 s). After 6 s preprocess + gap trim, usable length is a bit shorter (see covered weeks below).

| REFIT house | covered weeks (gaps >30 s removed) |
| ---: | ---: |
| 2 | 65.95 |
| 3 | 77.41 |
| 5 | 82.98 |
| 9 | 68.09 |
| 11 | 49.35 |
| 20 | 61.37 |

---

## 1) How many houses and channels?

| Item | Value |
|------|------:|
| Houses in local folder | **20** (`House_1` … `House_21`, no `House_14`) |
| Aggregate channels | 1 per house |
| IAM appliance channels | **9** per house (`Appliance1`–`Appliance9`) |
| Total measurement columns | **10** (aggregate + 9 IAMs), plus `Time` / `Unix` |

This is much richer than REDD (~2–6 weeks, 4 appliances, no kettle).

---

## 2) How much data (weeks) is inside each house?

Computed from `Time` in each `House_*.csv`.

| REFIT house | rows | calendar_weeks | covered_weeks |
|---:|---:|---:|---:|
| 1 | 6,960,008 | 91.28 | 79.33 |
| 2 | 5,733,526 | 88.20 | 65.95 |
| 3 | 6,994,594 | 87.81 | 77.41 |
| 4 | 6,760,511 | 90.57 | 77.72 |
| 5 | 7,430,755 | 92.62 | 82.98 |
| 6 | 6,241,971 | 82.49 | 69.29 |
| 7 | 6,756,034 | 87.60 | 75.99 |
| 8 | 6,118,469 | 79.29 | 70.54 |
| 9 | 6,169,525 | 81.15 | 68.09 |
| 10 | 6,739,284 | 83.85 | 73.50 |
| 11 | 4,431,541 | 56.02 | 49.35 |
| 12 | 5,859,544 | 69.66 | 64.55 |
| 13 | 4,737,371 | 71.22 | 53.75 |
| 15 | 6,225,696 | 81.05 | 69.42 |
| 16 | 5,722,544 | 77.66 | 63.73 |
| 17 | 5,431,577 | 67.13 | 62.43 |
| 18 | 5,007,721 | 63.29 | 59.25 |
| 19 | 5,622,610 | 67.21 | 62.24 |
| 20 | 5,168,605 | 65.76 | 61.37 |
| 21 | 5,383,993 | 69.97 | 62.95 |

Typical good houses: **~80–93 calendar weeks** (~1.5–1.8 years), **~66–83 covered weeks** after gaps.

Shorter houses: **11** (~56 wk), **13** (~71 wk), **18–21** (~63–70 wk).

---

## 3) Which of our 5 appliances are covered?

REFIT IAM columns are generic (`Appliance1`–`Appliance9`). Appliance identity is **house-specific**.

This repo’s baseline mapping (from `NILM_model/baseline/transfer_learning_multi-appliance/dataset_management/refit/create_dataset.py`) defines **6 houses** that carry all **5 target appliances**:

**Houses with full 5-appliance mapping:** `2`, `3`, `5`, `9`, `11`, `20`

| REFIT house | kettle (IAM) | microwave (IAM) | fridge (IAM) | dishwasher (IAM) | washingmachine (IAM) |
|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 5 | 1 | 3 | 2 |
| 3 | 9 | 8 | 2 | 5 | 6 |
| 5 | 8 | 7 | 1 | 4 | 3 |
| 9 | 7 | 6 | 1 | 4 | 3 |
| 11 | 7 | 6 | 1 | 4 | 3 |
| 20 | 9 | 8 | 1 | 5 | 4 |

For other houses (`1`, `4`, `6`–`10`, `12`, `13`, `15`–`19`, `21`), you need `MetaData_Tables.xlsx` / `CLEAN_READ_ME_081116.txt` to map IAM → appliance name before using them in a 5-app protocol.

---

## 4) ON activity check (baseline 6 houses)

Thresholds from baseline `Arguments.refit_params_appliance` ON cutoffs used for a quick sanity check (`power > threshold`):

| REFIT house | kettle ON% | microwave ON% | fridge ON% | dishwasher ON% | washingmachine ON% |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.99% | 0.28% | 39.78% | 7.76% | 3.73% |
| 3 | 1.24% | 0.19% | 51.77% | 3.99% | 4.50% |
| 5 | 0.93% | 1.46% | 50.34% | 9.04% | 6.84% |
| 9 | 1.11% | 0.15% | 50.27% | 6.63% | 2.19% |
| 11 | 1.16% | 0.14% | 18.32% | 0.68% | 1.13% |
| 20 | 0.54% | 0.40% | 41.32% | 1.30% | 2.14% |

All six baseline houses show **real washing-machine activity** (unlike REDD H2/H5). Max power on WM IAM is **~2.4–3.6 kW** on these houses.

House **11** has weaker dishwasher / WM ON rates but still non-zero.

---

## 5) Comparison vs REDD (why REFIT matters for WM)

| | REDD (H1+H3) | REFIT (baseline 6 houses) |
|--|--|--|
| Duration | ~2–2.6 covered weeks | ~50–83 covered weeks |
| Appliances | 4 (no kettle) | **5** (includes kettle) |
| Houses with all targets | 2 (H1, H3) | **6** (2,3,5,9,11,20) |
| Washing machine | H1/H3 OK; H2 dead | **Active on all 6 baseline houses** |

REFIT is the practical next domain for **more WM examples** and longer training windows.

---

## 6) Recommended starting protocol

Preprocess is implemented:

```text
python dataset_preprocess/refit_processing_multi_appliance.py --split_houses 2,3,5,9,11,20 --full_range
```

Config: `config/preprocess/refit.yaml` (`sample_seconds: 6`, houses `2,3,5,9,11,20`).

Exported CSVs:

```text
multi_appliances_NILM/datasets/refit/refit_house{2,3,5,9,11,20}_lf_6s.csv
```

Suggested first split (same spirit as UK-DALE / REDD cross-house):

- **Source (labeled):** houses `2, 3, 5, 9, 11`
- **Target (eval):** house `20`

Then copy/split into:

```text
multi_appliances_NILM/datasets/refit/training/multi_appliance_training.csv
multi_appliances_NILM/datasets/refit/validating/multi_appliance_validating.csv
multi_appliances_NILM/datasets/refit/testing/multi_appliance_testing.csv
```

---

## 7) Local file location

```text
dataset_preprocess/REFIT/House_1.csv
...
dataset_preprocess/REFIT/House_21.csv   (no House_14)
```

When you build preprocess, point `data_dir` at this folder. Files are named `House_N.csv` (not `CLEAN_HouseN.csv`), but the column layout matches the cleaned REFIT release.

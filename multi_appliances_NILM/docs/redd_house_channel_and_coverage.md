# REDD house summary (channels, appliance coverage, usable weeks)

Notes:

1. **Channels** = number of meter "tables" found under `/buildingX/elec/meter*/table` in `dataset_preprocess/REDD/redd.h5`.
2. **Weeks** = span of `meter1` timestamps. `covered_weeks` subtracts long recording gaps where consecutive `meter1` samples are more than **30s** apart.
3. Your current 4-appliance REDD preprocess config only selects **houses 1–3**:
   - `config/preprocess/redd.yaml` uses 4 appliances: `fridge`, `dishwasher`, `microwave`, `washingmachine` (no `kettle` in REDD).

---

## 1) How many channels (meter tables) are inside each REDD house?

| REDD house | # meter channels |
|---:|---:|
| 1 | 43 |
| 2 | 13 |
| 3 | 24 |
| 4 | 20 |
| 5 | 28 |
| 6 | 19 |

## 2) How much data (weeks) is inside each house?

Computed from raw `meter1` timestamps in `redd.h5`.

| REDD house | calendar_weeks | covered_weeks (gaps>30s subtracted) |
|---:|---:|---:|
| 1 | 5.18 | 2.61 |
| 2 | 5.00 | 1.99 |
| 3 | 6.40 | 2.43 |
| 4 | 6.85 | 2.78 |
| 5 | 6.26 | **0.52** |
| 6 | 3.34 | 2.64 |

## 3) What appliances exist in each house? (from NILMTK official metadata)

Source: NILMTK `dataset_converters/redd/metadata/buildingX.yaml` on GitHub.

| REDD house | fridge | dishwasher | microwave | washer dryer | kettle | other notable |
|---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | m5 | m6 | m11 | **m10+m20** (dual-phase) | **no** | oven(m3), stove(m4), light, sockets |
| **2** | m9 | m10 | m6 | m7 | **no** | stove(m5), light(m4), sockets |
| **3** | m7 | m9 | m16 | m13(+m14) | **no** | furnace(m10), disposal(m8), electronics |
| **4** | **no** | m15 | **no** | m7 | **no** | furnace(m4), stove(m8), A/C(m9+m10+m20) |
| **5** | m18 | m20 | m3 | m8+m9 | **no** | furnace(m6), electric heater(m12+m13), subpanel |
| **6** | m8 | m9 | **no** | m4 | **no** | stove(m5), space heater(m12), A/C(m15+m16+m17) |

## 4) Why we only use houses 1–3 (not 4–6)

Our protocol needs **all 4 appliances** (`fridge`, `dishwasher`, `microwave`, `washingmachine`) in **every** house so cross-house transfer is fair. Looking at the table above:

| House | fridge | dishwasher | microwave | washer dryer | All 4 present? | Why excluded? |
|---:|:---:|:---:|:---:|:---:|:---|:---|
| 1 | yes | yes | yes | yes | **yes** | — |
| 2 | yes | yes | yes | yes (but 0 ON) | yes (but WM unusable) | — |
| 3 | yes | yes | yes | yes | **yes** | — |
| **4** | **no** | yes | **no** | yes | **no** | **missing fridge and microwave** |
| **5** | yes | yes | yes | yes (but dead: max ~48 W) | technically yes | **WM dead meter + only ~0.5 weeks coverage** |
| **6** | yes | yes | **no** | yes (weak: max ~370 W, 92 samples >20 W) | **no** | **missing microwave, WM near-dead** |

Summary:

- **House 4**: no fridge meter, no microwave meter → cannot form the same 4-appliance set.
- **House 5**: has all 4 appliance labels, but only **~0.5 weeks covered** (massive gaps in recording). Washer dryer meters 8+9 max ~48 W → essentially dead / standby only.
- **House 6**: no microwave meter. Washer dryer meter 4 max ~370 W with only 92 raw samples >20 W → near-dead WM. Only ~3.3 calendar weeks.

So **houses 1–3 are the only three** where you get a consistent 4-appliance set with usable data coverage (~2–2.6 weeks each).

## 5) Why we still include house 2 (despite WM being empty)

House 2 has **usable fridge, dishwasher, microwave** data (~2 weeks covered). It is only the **washingmachine** channel that is dead (no real wash activity during the recording).

We use H2 as the **evaluation/target house** for the H1+H3 → H2 cross-house split. The protocol is:

- **Train/val** on H1 + H3 (all 4 appliances, labeled)
- **Test** on H2:
  - Fridge, dishwasher, microwave → **valid** evaluation
  - Washingmachine → **skip / report N/A** (0 ground truth ON, so F1/MAE is meaningless)

This is the standard approach in REDD multi-appliance papers. H2 is useful for 3 of 4 appliances. For WM-only transfer evaluation, use a **H1 ↔ H3 time hold-out** instead.

## 6) No kettle in any REDD house

**None** of the 6 REDD houses have a kettle meter. This is a US dataset; electric kettles are uncommon. Our UK-DALE 5-appliance set includes kettle, but REDD is limited to 4 appliances.

## 7) Exported REDD (6s) ON rates (houses 1–3)

Using `dataset_preprocess/created_data/REDD/redd_house{H}_lf_6s.csv`:

| REDD house | rows | fridge ON% | dishwasher ON% | microwave ON% | washingmachine ON% |
|---:|---:|---:|---:|---:|---:|
| 1 | 262,723 | 24.92% | 4.53% | 1.33% | 2.25% |
| 2 | 200,229 | 44.52% | 1.47% | 0.40% | **0.00%** |
| 3 | 240,821 | 37.42% | 1.27% | 0.44% | 2.94% |

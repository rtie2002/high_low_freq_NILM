# Reviewer controls, 2026-11-09

## Frozen protocol

Reference: final no-LRE model; seeds 2026, 2027, 2028; fixed five-week whole-house split.
No coefficient search, threshold change, label change or architecture selection.
Validation alone selects checkpoints and calibration. Test houses are report-only.
The default model YAML still enables LRE, so the launcher explicitly overrides its
coefficient to zero for every new run. Do not reuse historical LRE-enabled ablations.

## Controlled experiments

- Random recomposition: disable focal sampling and background stratification together;
  retain 50% recomposition and bounded balancing.
- Global balancing: use `task_balance=equal`, retain focal mixing.
- Unbounded appliance-wise balancing: use `per_appliance_equal`, retain focal mixing.
- No relation attention: disable only cross-appliance attention for a fair no-LRE audit.

Each variant runs all three seeds. These are evidential ablations, not a search to
replace the frozen reference; degraded results remain part of the report.
Saved metric CSVs and waveforms must be inspected before manuscript claims are added.

Concurrency audit: group frozen test samples by true active-target count 0, 1, 2, 3,
4-5; report sample counts, exact match, macro F1, MAE, ON MAE, and class support.
Undefined per-appliance F1/ON MAE is excluded and counted explicitly. Zero-active
samples are separate because their abundance can inflate exact match.
Verify paired prediction alignment before comparing models; no thresholds are fitted.

## Literature

Dash and Sahoo, TSG 15(3), 3337-3340 (2024), DOI 10.1109/TSG.2024.3373258:
local full text inspected; shared convolution/bi-LSTM encoder, state module,
attention/recurrent power decoder. Joint multi-appliance power-state prediction is prior art.

Zhang et al., TCE 72(2), 2852-2865 (2026), DOI 10.1109/TCE.2026.3677408:
IEEE-deposited Crossref publication metadata checked. Weighting equations unavailable;
do not claim equivalence, a distinct mechanism, or superiority from the title/abstract.

## Status

Prepared for execution. No new numerical results asserted yet.
Outputs: `runs/paper_5w/reviewer_controls`; launcher logs under `runs/_logs`.
Appendix tables and main-text result sentences are added only after completion and audit.

# Journal revision plan (maps reviewer critique → actions)

**Verdict accepted:** current evidence is workshop-level. Do not claim journal-ready until items below are green.

## P0 — Credibility blockers (do first)

| Reviewer point | Action | Status |
|----------------|--------|--------|
| H2 adapt overlaps H2 test | Chronological split: first 70% H2 = unlabeled adapt; last 30% = test only | **code done** (`target_adapt_frac=0.7`) |
| One seed | Report mean±std over seeds `{2024,2025,2026}` | **runner ready** (`scripts/run_journal_p0.py`) |
| SAE worse but abstract soft-pedals | Rewrite abstract/discussion: state F1↑ and SAE↑ honestly | **paper done** |
| Under-specified method | Caps, thresholds, gate on/off, ON-mass, norm source | **paper done** |
| Informal tone / conference 4pp | Switch to `IEEEtran` journal; polish prose | **paper started** (still short until results land) |

## P1 — Experiments required for journal

| Experiment | Why | Status |
|------------|-----|--------|
| Multi-seed chronological H2 (3 methods × 3 seeds) | Statistical reliability + no leak | **launch next** |
| Ablations: −entropy, −conditional, MMD-only, CORAL-only | Prove EGC-DA components | configs ready; run after main |
| Multi-house UK-DALE | One split ≠ cross-building | planned |
| Stronger baselines + supervised FT upper bound | Fair comparison | planned |
| Cross-dataset UK↔REDD/REFIT | Reviewer required | planned |
| Per-app MAE/SAE/P/R + DW event analysis | Dishwasher failure | planned |

## Claim discipline

Only claim what tables support. Until multi-house + ablations + seeds exist, claim:
> “On UK-DALE H1+H5→H2 with a held-out H2 test segment, EGC-DA improves multi-label F1 vs Source-Only and Global FC-UDA; energy SAE remains a limitation.”

Pilot tables in `paper/main.tex` are explicitly marked **preliminary / overlapping protocol** and must be replaced after P0 completes.

Paper figures (from `multi_appliances_NILM/evaluation`-style plots) are in `paper/figures/`:
- `train_compare_h2_f1_mae.png`, `train_curves_*.png`
- `h2_power_grid_source_only.png`, `h2_power_grid_matuda.png`
- ON-period waveforms under `waveforms_*`

Regenerate: `python scripts/plot_paper_figures.py`

**Honesty note:** figures make the Source-Only collapse and DW↔WM confusion visible; they do **not** replace multi-house / cross-dataset / ablation requirements for Applied Energy.

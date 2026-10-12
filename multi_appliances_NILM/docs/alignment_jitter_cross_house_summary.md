# Data Alignment and Jitter: Cross-House Summary

## Purpose

This experiment tested whether REFIT event alignment and microwave training
jitter improve the current raw-input MultiNILM model. Four controlled variants
used the same architecture, loss, training houses, validation houses, seed,
training schedule, checkpoint rule, calibration, and postprocessing. Only the
dataset treatment and alignment jitter were changed.

The four variants were:

1. original data without jitter;
2. high-confidence-aligned data without jitter;
3. original data with microwave jitter;
4. high-confidence-aligned data with microwave jitter.

## Evaluation protocol

- Training: UK-DALE 1 and REFIT 3, 5, 9, and 11.
- Validation: UK-DALE 5 and REFIT 2.
- Cross-house development: UK-DALE 2 and REFIT 20.
- Checkpoints and thresholds were selected using validation only.
- Every saved checkpoint was subsequently audited on both cross-house
  development scenarios using metrics and fridge/microwave waveforms.

Because UK-DALE 2 and REFIT 20 now influence method retention, they must not be
described as untouched final test houses. A separate unseen house split is
required for the final paper evaluation.

## Overall results

| Data / augmentation | Validation MAE/F1/AP | REFIT20 MAE/F1/AP | UK-DALE2 MAE/F1/AP |
|---|---|---|---|
| **original / no jitter** | 16.915 / **0.760** / **0.798** | 11.275 / **0.703** / 0.737 | **8.373** / 0.861 / **0.938** |
| aligned / no jitter | **16.436** / **0.760** / 0.797 | 11.385 / 0.684 / 0.740 | 8.485 / **0.868** / **0.938** |
| original / jitter | 16.721 / 0.751 / 0.794 | 11.388 / 0.693 / 0.733 | 8.379 / 0.862 / 0.924 |
| aligned / jitter | 16.875 / 0.750 / 0.790 | **11.121** / 0.699 / **0.746** | 8.522 / 0.858 / 0.926 |

The simple mean across the two cross-house development scenarios is:

| Data / augmentation | Mean MAE (W) | Mean F1 | Mean AP |
|---|---:|---:|---:|
| **original / no jitter** | 9.824 | **0.782** | 0.838 |
| aligned / no jitter | 9.935 | 0.776 | **0.839** |
| original / jitter | 9.884 | 0.778 | 0.829 |
| aligned / jitter | **9.822** | 0.779 | 0.836 |

The 0.002 W mean-MAE difference between aligned/jitter and original/no-jitter
is negligible. Original/no-jitter gives the strongest mean F1 without either
additional processing step.

## Fridge and microwave findings

None of the four methods solves the fridge problem. All retain long false
fridge plateaus under high residual background. The apparent validation gains
from alignment do not consistently transfer to REFIT 20.

For microwave, aligned/jitter improves REFIT20 F1/AP to 0.550/0.554, but the
same model reduces UK-DALE2 microwave F1/AP to 0.731/0.734. In comparison,
original/no-jitter obtains 0.798/0.803 on UK-DALE2. Alignment and jitter can
improve the timing of selected REFIT events, but they also retain large false
microwave activations in confounding background segments. The improvement is
therefore domain-specific rather than robust.

## Decision

Retain **original data without alignment jitter** as the working protocol.

Reasons:

- highest mean cross-house F1;
- lowest UK-DALE2 MAE;
- highest REFIT20 overall F1;
- strongest UK-DALE2 microwave F1/AP;
- no deterministic event-realignment dependency;
- no stochastic alignment-jitter hyperparameters;
- simpler and easier to defend in the paper.

The working configuration is:

```yaml
alignment_jitter:
  enabled: false
```

## Rule for future improvements

A new component is retained only when:

1. its validation checkpoint is not materially worse;
2. it remains stable on both cross-house development scenarios;
3. fridge and microwave waveforms do not become less physical;
4. the improvement is not restricted to one house at the expense of the
   other;
5. its contribution can be explained clearly and justifies its complexity.

This protocol prevents another small validation-only improvement from being
mistaken for robust cross-house progress.

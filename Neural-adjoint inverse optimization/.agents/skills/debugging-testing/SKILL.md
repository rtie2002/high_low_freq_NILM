---
name: debugging-testing
description: "Diagnose and fix Python/PyTorch research code failures, incorrect output shapes, numerical instability, exploding gradients, NaN losses, data leakage and silent model regressions. Use for traceback analysis, test writing, reproducible bug isolation, gradient checks and regression verification. Do not trigger for pure style-only edits."
---

# Debugging and Testing

## Goal
Find the smallest reproducible cause of a failure and verify a minimal fix without masking research errors.

## Workflow
1. Reproduce the failure with exact command, input size, relevant environment and error output when available.
2. Localize the failing operation: assertions, isolated tensor shape inspections, controlled examples or targeted tests.
3. Distinguish code bug from invalid assumption: data shape, dtype/device, target alignment, loss axis, gradient break, NaN/Inf, solver convergence, import path.
4. Apply the smallest corrective change and keep error handling explicit. Don't suppress exceptions to make tests green.
5. Add a focused regression test or short reproducibility check when possible.
6. Run tests and report actual commands, pass/fail status and any untested cases.

## Checks for research ML
- Verify train/eval mode, seed behavior, optimizer steps, masked labels, scaler/inverse scaler and prediction alignment.
- Test model forward and backward with representative shapes and small synthetic inputs.
- For numerical optimization, validate residuals, step sizes, convergence tolerances, gradient magnitude and constraints.
- Treat surprising 'improvements' after refactoring as possible evaluation leakage or altered behavior until confirmed.

## Deliverable
Root cause, minimal fix, exact checks executed, remaining risk.

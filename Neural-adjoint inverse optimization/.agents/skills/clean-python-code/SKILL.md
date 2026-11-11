---
name: clean-python-code
description: "Write or simplify ordinary Python code to be easy for a human research engineer to read. Use when generating Python functions, improving names, reducing nested control flow, removing cryptic one-liners, or making a research script easier to maintain. Not for repo restructuring, mathematical derivation, or behavior-changing algorithm redesign."
---

# Clean Python Code

## Goal
Write ordinary Python that a researcher can understand quickly, debug easily, and modify months later. Focus on *how the code reads*, not on architectural novelty.

## Workflow
1. Locate the actual entry point and nearby conventions; inspect only the relevant modules.
2. Trace the real flow from input to output and identify unclear naming, tangled conditions, repeated work, and unnecessary indirection.
3. Rewrite in a direct top-down sequence with meaningful variable names and helpful intermediate values.
4. Keep abstraction proportional: a function should name a meaningful operation, not wrap one trivial line. Avoid utility files for one-off logic.
5. Preserve numerical operations, vectorization, evaluation order where meaningful, and error behavior unless asked to change them.
6. Run available narrow checks (or explain what was not run); summarize what became easier to read.

## Style
- Simple loops/conditions are fine. Use vectorized NumPy/PyTorch where performance or autograd requires it.
- Favor ordinary functions/classes over clever inheritance, decorators or registries.
- Add short docstrings where input/output meaning is not obvious; comments explain why, invariants and non-obvious choices.
- Keep logging and exception handling proportionate; do not add layers of configuration for a simple script.

## Example
Avoid `pred = net(inp.to(dev)).detach().cpu().numpy()` when the steps are important to explain. Prefer explicit `input_tensor`, `model_output`, `prediction_array` intermediate values, while leaving familiar idioms concise when already clear.

## Deliverable
A readable implementation with minimal churn and a brief account of significant naming/control-flow changes.

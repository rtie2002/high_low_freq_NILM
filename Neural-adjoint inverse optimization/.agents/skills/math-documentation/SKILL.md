---
name: math-documentation
description: "Add precise mathematical comments and docstrings that connect Python/PyTorch code to equations, tensor dimensions, units, assumptions, and reduction axes. Use when users ask to explain models, algorithms, losses, adjoints, gradients, normalization, signal processing, or to make math-heavy code readable. Documentation must reflect existing code, never invented derivations."
---

# Mathematical Documentation

## Goal
Mathematical comments should make the implementation auditably match the formula—not decorate every line.

## Workflow
1. Read the actual expressions, paper/specification if provided, and any shape conventions; verify the formula that the code computes.
2. At each mathematically important block, state the operation in an accurate ASCII or LaTeX-friendly equation, define symbols, axes, shape and units as needed.
3. Describe why the operation exists, any assumptions, and numerical implications only where they help comprehension.
4. Cross-check code vs comment for `sum` vs `mean`, dimension order, epsilon, scale, signs, padding/interpolation, target alignment and boundary conditions.
5. Avoid duplicate verbose prose. Skip comments for obvious assignments, imports, standard `relu` calls, etc.
6. When code and claimed equation disagree, report this clearly; never quietly change either when documentation-only was requested.

## Example with correct semantics
```python
# Mean squared error: L = (1 / (B*T)) * sum_{b=1..B,t=1..T}(y_hat[b,t] - y[b,t])^2
# y_hat, y: [B, T] in watts. Mean is taken across both axes.
loss = torch.mean((prediction - target) ** 2)
```

## Adjoint-specific caution
The form of an adjoint depends on the state equation and sign convention. Define primal residual, adjoint variable, transposed Jacobian and boundary or terminal conditions from the *implemented system*, and trace each into code; do not assert a universal sign or gradient expression.

## Tensor shape conventions
Use a consistent legend, e.g. B=batch size, C=channels, T=time steps, F=features, and specify where sequences are downsampled. Every axis used in reshape/transpose/reduction should be explicit when non-obvious.

## Deliverable
Concise math comments that a researcher can verify against the implementation, with any uncertainties flagged.

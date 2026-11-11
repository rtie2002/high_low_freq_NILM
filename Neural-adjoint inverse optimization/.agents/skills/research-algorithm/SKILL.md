---
name: research-algorithm
description: "Implement, explain, or review a scientific algorithm, signal-processing method, NILM workflow, inverse optimization procedure, adjoint method, mathematical loss or numerical solver in Python. Use when correctness and step-by-step mathematical logic matter more than software abstraction. Pair with math-documentation when equations need explicit code mapping."
---

# Research Algorithm

## Goal
Readable scientific implementations with steps that follow the underlying mathematics and preserve experimental validity.

## Workflow
1. Identify input/output definitions, objective, assumptions, symbol conventions, units and edge cases from the task, paper or existing implementation. Never assume a formula not supplied or implied by code.
2. Write a short algorithm outline in execution order. Map each mathematical quantity to a descriptive variable.
3. Implement contiguous blocks in that order: preprocessing -> computations/iterations -> constraints/loss -> outputs. Use helper functions only for genuine named operations.
4. Add equations adjacent to non-obvious computations, define symbols and explain reductions/normalization. Explain approximations, tolerances, stopping criteria, boundary/initial conditions and stability checks as applicable.
5. Validate with small analytical/synthetic sanity checks, shape/unit checks, and regression comparisons when feasible.
6. If asked for refactoring only, preserve mathematical equivalence, update schedule, random generator behavior and gradient paths.

## Mathematical correctness checks
- Distinguish sums from means, batch from time axes, and power from energy units.
- For objective functions, check signs, scaling, reduction and regularizer weights.
- For an adjoint or implicit gradient, verify transpose/conjugation, the exact state equation, boundary conditions and gradient signs against the *actual formulation*; do not paste a generic adjoint formula blindly.
- For NILM, verify time alignment, window target offsets, appliance aggregate consistency and split leakage.
- For numerical operations, watch dtype, conditioning, convergence and in-place autograd mutations.

## Example math comment
```python
# Objective for each batch: L = (1 / (B*T)) * sum_{b,t} (y_hat[b,t] - y[b,t])**2
# prediction, target: [B, T], in watts; reduction averages over batch and time.
loss = torch.mean((prediction - target) ** 2)
```

## Deliverable
A step-by-step readable algorithm, mapping between variables and math, and checks supporting correctness.

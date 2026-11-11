# Project instructions

<!-- BEGIN: Clean Research Skills Auto Router -->
## Research-code rules and automatic skill routing

Repository style: simple, human-readable Python research code with a clear file layout, logical algorithms, readable PyTorch architecture, and precise mathematical comments. Prefer the repository's established conventions. This file is guidance, not a mandate to rewrite unrelated code.

### Always-on defaults
- Make the smallest sensible change for the task. Do not invent frameworks, layers, helper files, wrapper classes, or generic interfaces without a demonstrated need.
- Keep a clear top-down flow (input -> processing/model -> loss -> evaluation/output). Use explicit intermediate variables when they aid understanding.
- Use descriptive names and focused functions. Do not split a coherent algorithm into many tiny helpers or files just to appear modular.
- Put new code in the most natural existing module; introduce or move files only when the current structure demonstrably hinders navigation.
- Explain *why* non-obvious algorithms work. On math-heavy code, document equations, assumptions, symbol-to-variable correspondence, reduction axes, units and tensor shapes. Never invent a formula not supported by the implementation.
- Preserve math, forward semantics, gradients, data splits, configurations and result comparability unless the user explicitly requests a behavior change. Verify when possible; never claim tests passed unless run.
- Keep comments informative rather than narrating obvious syntax. Prefer simple Python with efficient NumPy/PyTorch operations; do not replace vectorized computations with slow loops merely for readability.
- For substantial refactoring, first identify affected entry points and dependencies, then work in small reviewable steps. Explain any risky migration.

### Automatic skill selection
Use the matching project-local skill(s) below *when the task calls for them*; use a maximum of the fewest relevant skills, typically one primary plus one or two supporting skills. Do **not** load all eight for routine work. The skill metadata in `.agents/skills/*/SKILL.md` is designed for Codex automatic discovery.

| User intent / task | Primary skill | Optional supporting skill |
| --- | --- | --- |
| Write or tidy Python / improve naming and control flow | `clean-python-code` | `math-documentation` for math-heavy code |
| Reorganize files, imports, entry points, modules | `project-structure` | `code-refactoring` |
| Implement a method, optimization objective, derivation, signal-processing procedure | `research-algorithm` | `math-documentation` |
| Build or modify PyTorch network, layers, forward, loss-connected shapes | `pytorch-architecture` | `math-documentation` |
| Add/repair equations, docstrings, tensor-shape explanations | `math-documentation` | Task-specific skill if needed |
| Simplify existing implementation *without changing behavior* | `code-refactoring` | `project-structure` only if files need moving |
| Diagnose a crash, NaNs, gradients, mismatch, regression, tests | `debugging-testing` | `pytorch-architecture` or `research-algorithm` if implicated |
| Manage seeds, splits, checkpoints, configs, experiments | `research-reproducibility` | `debugging-testing` |

When a task spans multiple areas, follow the primary workflow and consult the relevant supporting skills only for affected files. If a legacy `clean-research-code` skill is also installed, prefer these focused skills for their respective tasks; do not delete the legacy skill automatically.

### Communication
- For non-trivial code changes, report changed files, main logic, mathematical assumptions, and actual validation results concisely in Mandarin unless asked otherwise.
- If the request is only to review or explain, do not edit files. If proposed structural changes could affect imports or experiment reproducibility, call out the risk.
<!-- END: Clean Research Skills Auto Router -->


---
name: code-refactoring
description: "Refactor existing Python research code for clarity while preserving behavior: remove unnecessary helper functions, wrappers, excess classes, repeated logic and difficult-to-read control flow. Use when asked to clean up, simplify or humanize AI-generated code without changing algorithms, trained model semantics or research results. For moving many modules, pair with project-structure."
---

# Behavior-Preserving Code Refactoring

## Goal
Make the existing implementation feel human-written, but treat research outputs as a correctness contract.

## Workflow
1. Determine change scope and establish baseline: Git status, entry points, tests, sample outputs and stable experiment settings. Don't overwrite uncommitted user work.
2. Describe concrete readability problems first: unnecessary indirection, duplicate branches, obscure names, over-splitting, complex conditionals.
3. Refactor one coherent module at a time. Prefer deleting useless abstractions over adding new architectural layers.
4. Preserve mathematical operations, initialization, randomness, data ordering, augmentations, gradient tracking, tensor shape and checkpoint interfaces. Flag compatibility risks from renames/moves.
5. Compare old vs new outputs under controlled settings when practical. Check import/CLI entry points and relevant tests.
6. Report any behavior you could not establish as equivalent. Do not claim bitwise identity without evidence.

## Special research guardrails
- Avoid changing precision, device placement, loss aggregation, optimizer schedule, forward operator or solver iteration order as cosmetic cleanup.
- No silent new dependencies, new configuration frameworks or full project rewrites.
- If too broad, choose a small high-impact path and document remaining work rather than touching unrelated modules.

## Deliverable
Readable code, minimal diffs, explicit verification and a summary of what changed vs what deliberately did not.

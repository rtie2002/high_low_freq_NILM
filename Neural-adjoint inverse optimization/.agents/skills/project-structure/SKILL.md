---
name: project-structure
description: "Make a Python research repository file structure clear, navigable and appropriately small. Use when asked to organize folders, reduce scattered code, simplify imports, decide what belongs in train.py/model.py/data.py/evaluate.py, or consolidate excessive helper modules. Do not trigger merely because a user edits one function."
---

# Project Structure

## Goal
A researcher should find the model, data, training, evaluation, experiments and plots without searching through dozens of tiny files. A five-file layout is a *possible example*, never a mandatory template.

## Workflow
1. Inspect current tree, runnable entry points, imports, tests, configs and artifact paths before proposing changes.
2. Identify real navigation problems: circular imports, overlapping modules, duplicate entry points, unintuitive placement, huge mixed-responsibility files.
3. Propose the smallest new layout; explain old-to-new mapping and why each move helps. Prefer preserving reasonable existing modules.
4. If user requested changes, move in small steps; update all imports, CLI paths, notebooks/scripts, relative resource paths and documentation.
5. Avoid hidden filesystem side effects: don't move datasets, experiment results, checkpoints or trained weights unless expressly requested.
6. Check module imports, available smoke tests and original entry points. Report any breakage or unverified assumptions.

## Practical boundaries
- Co-locate tightly coupled model operations; don't spread one forward pass across many modules.
- Prefer one coherent `model.py` or model package only when scope warrants it.
- Extract shared utilities only after actual repeated use; do not turn every helper into a file.
- Avoid introducing package boilerplate or build tooling merely for appearances.
- User-facing runnable scripts should remain easy to discover.

## Deliverable
Before/after tree when material, responsibility of each changed file, import changes, and exact verification performed.

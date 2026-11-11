---
name: research-reproducibility
description: "Make Python/NILM/deep-learning experiments reproducible and easy to compare: random seeds, dataset splits, configuration, experiment naming, training/evaluation protocols, checkpoints, logging and data leakage checks. Use when implementing experimental workflows or investigating inconsistent results across runs. Do not trigger for small non-experiment code formatting."
---

# Research Reproducibility

## Goal
A research collaborator should be able to discover how an experiment was run, what data split it used and which checkpoint/results belong to it, without adopting unnecessary infrastructure.

## Workflow
1. Identify actual experiment entry points, split logic, preprocessing fit scope, random number generators, model init, checkpoints and metrics.
2. Keep parameters discoverable in existing configs/CLI. Don't introduce a full configuration framework for a few values.
3. Make randomness and split definitions explicit; document when device/kernel operations remain nondeterministic.
4. Keep train/validation/test preprocessing isolated and transformations reproducible; guard against overlapping NILM windows leaking between splits.
5. Record only useful metadata: seed, dataset/split, appliance/task, window length, model/config, metrics, checkpoint path and code version when available.
6. Verify a short smoke run or compare a few outputs; avoid expensive full retraining unless requested and feasible.

## NILM-specific cautions
- TSTR and TRTS must explicitly state whether train data are real or synthetic and what population real test data represent.
- Keep normalization fitted on the appropriate training subset; avoid household/temporal leakage.
- Verify timestamps, resampling, window stride and appliance labels stay aligned.

## Deliverable
Traceable experiment setup and explicit notes on what is / is not deterministically reproducible.

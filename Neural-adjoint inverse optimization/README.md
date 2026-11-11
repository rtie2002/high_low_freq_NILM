# Neural Adjoint NILM: minimal research pipeline

This is a standalone inverse-NILM research project.  It owns its Python data,
metrics, plotting, output, and evaluation code under `utils/`.  It reads the
same preprocessed CSV dataset assets used for the direct-model comparison, but
does not import or execute code from `multi_appliances_NILM` at runtime.

## Current implementation

Phase A implements and tests the differentiable bounded power projection in
`power_projection.py`. Phase B implements an independent supervised initial
estimator in `initial_estimator.py`. The Phase B model is a direct aggregate-to-
appliance baseline; it does not yet contain inverse refinement or projection.

All additive quantities in Phase B are restored to watts and divided by one
fixed `power_scale_watts`. Independently z-scored appliance channels are never
summed. The model predicts five nonnegative appliance traces, five differentiable
state probabilities, and one nonnegative residual-background proxy.

Run the cheap verification before any full experiment:

```powershell
conda run -n base python -m pytest tests/test_power_projection.py tests/test_initial_estimator.py -q
conda run -n base python train_initial.py --config configs/initial.yaml --smoke
```

Evaluate a checkpoint without training:

```powershell
conda run -n base python evaluate_initial.py `
  --config configs/initial.yaml `
  --checkpoint runs/phase_b_smoke/best.pt `
  --split validation --max-batches 2
```

Smoke metrics are integration diagnostics, not full-dataset research results.

## First experiment: NA-0 physical inverse baseline

For a test aggregate window `y` (watts), estimate five target appliance powers
`p_1, ..., p_5` and an unmonitored-background residual `r` by optimising the
candidate source trajectories directly:

```text
candidate sources u = [p_1, ..., p_5, r]
        -> F(u) = sum_i p_i + r
        -> meter mismatch + constraints
        -> gradient with respect to u
        -> updated candidate sources
```

The physical forward model is exactly known:

`F(u) = sum_i p_i + r`.

It is intentionally **not** a neural network in NA-0.  A CNN trained on
labelled appliance powers to predict their aggregate could learn this addition
without learning appliance identity or waveform logic.  NA-0 is therefore a
negative-control baseline: it demonstrates precisely what power conservation,
non-negativity, and wattage bounds can and cannot identify.

No test appliance labels enter the optimisation.  They are read only after
inference to construct the same `PredictionBundle`, metrics table, and
waveform figures as the established supervised project.

## Reuse boundary

All reusable components are under `utils/`:

- `utils/data_loader.py`: CSV split, normalisation, sequence boundaries, and windows;
- `utils/metrics.py`: MAE, SAE, F1, event F1, false events, and background FPR;
- `utils/waveform_capture.py`: the same aggregate and ON-period figures;
- `utils/state_postprocess.py`: state calibration and duration cleanup;
- `utils/output_format.py`: standard CSV/JSON result formatting;
- `utils/protocol.py`: one entry point which writes the complete evaluation bundle.

The readable implementations are directly in the files above.  Start with
`utils/README.md` for the data/evaluation map and metric equations.  New code
should call `utils.protocol.load_protocol(...)` and
`utils.protocol.save_inverse_evaluation(...)` rather than recreate metrics or
waveform code.

## Run

From this directory:

```powershell
python na0_physical_inverse.py --config config_na0_physical.yaml --split validation
python na0_physical_inverse.py --config config_na0_physical.yaml --split refit_house20
```

The result folder contains `predictions.npz`, `metrics.csv`,
`background_fpr.csv`, residual diagnostics, and the same ON-period waveform
figures used by the existing project.

## Research sequence

1. **NA-0, now:** exact physical forward model plus explicit constraints.
   It should expose non-identifiability; it is not expected to outperform a
   supervised NILM model.
2. **NA-1:** add auditable hand-designed priors such as appliance-specific
   feasible power ranges and duration rules.  Test each term separately.
3. **NA-2:** train one differentiable temporal appliance-prior model per
   appliance using *training labels only*.  Replace hand-designed waveform
   penalties with learned prior penalties during inverse optimisation.
4. **NA-3:** use score-based/generative priors and multiple inference chains
   to represent ambiguity rather than hiding it in one point estimate.

Only NA-2/NA-3 require a genuine learned model-training stage.  NA-0 is still
valuable because it provides the necessary failure baseline and reuses the
published data/evaluation protocol unchanged.

## Scientific warning

The residual is an unobserved combined quantity, not an identified appliance.
It may include non-target devices, submeter alignment differences, and meter
error.  A low aggregate mismatch does not prove correct appliance attribution.
Always report appliance-label metrics, aggregate reconstruction, and waveform
figures together.

#!/usr/bin/env python
"""Run controlled MultiNILM simplification candidates on validation only.

The script deliberately starts from ``config/models/multinilm_k4.yaml`` and
changes one experiment family at a time.  Each run stores its fully merged
configuration in the normal run directory, so temporary candidate YAML files
are unnecessary.

The feature family tests whether the 13-channel fixed feature bank is needed:

``feature_1``
    Raw aggregate only.
``feature_2``
    Raw aggregate and its signed first difference.
``feature_3``
    Add one 45-sample causal rolling mean.
``feature_4``
    Add the matching 45-sample causal rolling standard deviation.

Training and model selection use the configured validation houses.  This
script never evaluates a test split.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_experiment, load_model_config, merge_configs
from main import get_adapter
from runner import evaluate_model, train_model


DEFAULT_EXPERIMENT = ROOT / "config" / "experiment_mixed_ukdale_refit_5w_house_split.yaml"
DEFAULT_MODEL_CONFIG = ROOT / "config" / "models" / "multinilm_k4.yaml"

FEATURE_VARIANTS = {
    "feature_1": {
        "experiment_id": "multinilm_simplify_feature_1_raw",
        "include_delta": False,
        "include_rolling_mean": False,
        "include_rolling_std": False,
    },
    "feature_2": {
        "experiment_id": "multinilm_simplify_feature_2_raw_delta",
        "include_delta": True,
        "include_rolling_mean": False,
        "include_rolling_std": False,
    },
    "feature_3": {
        "experiment_id": "multinilm_simplify_feature_3_raw_delta_mean",
        "include_delta": True,
        "include_rolling_mean": True,
        "include_rolling_std": False,
    },
    "feature_4": {
        "experiment_id": "multinilm_simplify_feature_4_raw_delta_mean_std",
        "include_delta": True,
        "include_rolling_mean": True,
        "include_rolling_std": True,
    },
    # Replacement candidate used only if signed delta proves unhelpful. It
    # keeps one directly interpretable slow-background channel beside raw.
    "feature_2_mean": {
        "experiment_id": "multinilm_simplify_feature_2_raw_mean",
        "include_delta": False,
        "include_rolling_mean": True,
        "include_rolling_std": False,
    },
}

LOSS_VARIANTS = {
    # A single interpretable task-weight check on the raw-input model. This
    # tests whether the compact input can recover state ranking without adding
    # a component or changing the bounded per-appliance balancing rule.
    "lambda_state_1": {
        "experiment_id": "multinilm_simplify_seeded_raw_lambda_state_1",
        "lambda_state": 1.0,
    },
    # Remove the two aggregate reconstruction terms that are most directly
    # duplicated by all-sample MSE. Keep edge shape and false-positive control.
    "loss_compact": {
        "experiment_id": "multinilm_simplify_seeded_loss_compact_raw",
        "power_off_weight": 0.0,
        "power_energy_relative_weight": 0.0,
    },
    # Smallest objective that preserves the two reasons for having two heads:
    # ON-weighted power reconstruction and imbalanced ON/OFF classification.
    # The bounded per-appliance scale matching is retained at this stage so
    # the experiment removes auxiliary losses without changing task coupling.
    "loss_core": {
        # The earlier folder used this loss before model initialization was
        # seeded.  Keep the corrected controlled run under an unambiguous ID.
        "experiment_id": "multinilm_simplify_seeded_loss_core_raw",
        "power_off_weight": 0.0,
        "power_delta_weight": 0.0,
        "power_energy_relative_weight": 0.0,
        "state_fp_weight": 0.0,
    },
    # Same training objective, but select the checkpoint by the standard,
    # threshold-independent validation AP instead of a bespoke composite.
    "loss_core_ap_monitor": {
        "experiment_id": "multinilm_simplify_loss_core_ap_monitor_raw",
        "power_off_weight": 0.0,
        "power_delta_weight": 0.0,
        "power_energy_relative_weight": 0.0,
        "state_fp_weight": 0.0,
        "checkpoint_monitor": "val_ap",
    },
}

ARCHITECTURE_VARIANTS = {
    # Preserve the soft probabilistic power gate in the forward pass, but stop
    # the regression loss from updating the state classifier through it.
    "detached_train_power_gate": {
        "experiment_id": "multinilm_simplify_seeded_raw_detached_train_power_gate",
        "gate_mode": "soft_detached",
    },
    # Decouple the regression value from the state probability during
    # training. Evaluation still applies the calibrated binary state mask, so
    # this removes a duplicated gate without changing the reported ON rule.
    "no_train_power_gate": {
        "experiment_id": "multinilm_simplify_seeded_raw_no_train_power_gate",
        "gate_mode": "none",
    },
    # Shorter TCN with dilations 1, 2, 4 instead of 1, 2, 4, 8, 16.
    "three_tcn_blocks": {
        "experiment_id": "multinilm_simplify_seeded_raw_three_tcn_blocks",
        "num_blocks": 3,
        "max_dilation": 4,
    },
    # Replace the mixed instance/batch normalization in the stem with ordinary
    # BatchNorm; all convolutional paths and later normalization stay fixed.
    "batch_stem_norm": {
        "experiment_id": "multinilm_simplify_seeded_raw_batch_stem_norm",
        "stem_norm_type": "batch",
    },
    # First remove the per-head channel attention. Appliance-specific heads
    # and explicit cross-appliance relation attention remain unchanged.
    "no_task_attention": {
        "experiment_id": "multinilm_simplify_seeded_raw_no_task_attention",
        "task_attention": {"enabled": False},
    },
    # Keep both attention mechanisms but reduce each appliance decoder from
    # two residual local blocks to one.
    "one_head_block": {
        "experiment_id": "multinilm_simplify_seeded_raw_one_head_block",
        "head_local_layers": 1,
    },
    # Bottom-up relational TCN: retain the shared temporal encoder and the
    # explicit multi-appliance interaction, remove the nested refinements
    # around them. The full retained loss is used so this tests architecture.
    "plain_relation": {
        "experiment_id": "multinilm_simplify_seeded_plain_relation_raw",
        "use_multiscale_stem": False,
        "stem_norm_type": "batch",
        "head_local_layers": 1,
        "task_attention": {"enabled": False},
    },
}

REPRO_VARIANTS = {
    "seeded_baseline": {
        "experiment_id": "multinilm_simplify_seeded_baseline",
        "feature_base": None,
    },
    "seeded_raw": {
        "experiment_id": "multinilm_simplify_seeded_raw",
        "feature_base": "feature_1",
    },
    "seeded_raw_ap_monitor": {
        "experiment_id": "multinilm_simplify_seeded_raw_ap_monitor",
        "feature_base": "feature_1",
        "checkpoint_monitor": "val_ap",
    },
}

HYPERPARAMETER_VARIANTS = {
    # Relation attention was useful in the earlier no-relation ablation. Test
    # one stronger residual value without changing its parameters or topology.
    "relation_scale_05": {
        "experiment_id": "multinilm_simplify_seeded_raw_relation_scale_05",
        "relation_residual_scale": 0.5,
    },
    # The seeded histories show a widening train/validation gap after the
    # early epochs.  This is a single regularisation check, not a grid search:
    # all temporal, head, and relation dropout values still share one number.
    "dropout_035": {
        "experiment_id": "multinilm_simplify_seeded_raw_dropout_035",
        "dropout": 0.35,
    },
}


def _feature_candidate(base: dict, name: str) -> dict:
    """Return one feature-only ablation while preserving every other setting."""
    spec = FEATURE_VARIANTS[name]
    candidate = copy.deepcopy(base)
    candidate["experiment_id"] = spec["experiment_id"]

    features = copy.deepcopy(candidate.get("fractional", {}))
    features.pop("k", None)
    features.update(
        {
            # An explicit empty list disables all GL channels without changing
            # the model implementation or introducing a special-case network.
            "alphas": [],
            "include_raw": True,
            "include_delta": spec["include_delta"],
            "include_abs_delta": False,
            "include_local_contrast": False,
            "include_rolling_mean": spec["include_rolling_mean"],
            "include_rolling_std": spec["include_rolling_std"],
            "rolling_windows": [45]
            if (spec["include_rolling_mean"] or spec["include_rolling_std"])
            else [],
            "channel_normalize": "none",
        }
    )
    candidate["fractional"] = features
    return candidate


def _loss_candidate(base: dict, name: str, feature_base: str) -> dict:
    """Return one loss-only ablation on a previously tested feature base."""
    if feature_base not in {"feature_1", "feature_2_mean"}:
        raise ValueError("loss candidates require feature_1 or feature_2_mean")
    candidate = _feature_candidate(base, feature_base)
    spec = LOSS_VARIANTS[name]
    candidate["experiment_id"] = spec["experiment_id"].replace(
        "_raw", "_raw_mean" if feature_base == "feature_2_mean" else "_raw"
    )
    loss = copy.deepcopy(candidate.get("loss", {}))
    loss.update(
        {
            key: value
            for key, value in spec.items()
            if key not in {"experiment_id", "checkpoint_monitor"}
        }
    )
    candidate["loss"] = loss
    if "checkpoint_monitor" in spec:
        training = copy.deepcopy(candidate.get("training", {}))
        training["checkpoint_monitor"] = spec["checkpoint_monitor"]
        candidate["training"] = training
    return candidate


def _architecture_candidate(base: dict, name: str) -> dict:
    """Return a bottom-up architecture candidate with raw aggregate input."""
    candidate = _feature_candidate(base, "feature_1")
    spec = ARCHITECTURE_VARIANTS[name]
    candidate["experiment_id"] = spec["experiment_id"]
    architecture = copy.deepcopy(candidate.get("architecture", {}))
    architecture.update(
        {key: copy.deepcopy(value) for key, value in spec.items() if key != "experiment_id"}
    )
    candidate["architecture"] = architecture
    return candidate


def _repro_candidate(base: dict, name: str) -> dict:
    """Build a reference whose seed is applied before model construction."""
    spec = REPRO_VARIANTS[name]
    feature_base = spec["feature_base"]
    candidate = (
        copy.deepcopy(base)
        if feature_base is None
        else _feature_candidate(base, feature_base)
    )
    candidate["experiment_id"] = spec["experiment_id"]
    if "checkpoint_monitor" in spec:
        training = copy.deepcopy(candidate.get("training", {}))
        training["checkpoint_monitor"] = spec["checkpoint_monitor"]
        candidate["training"] = training
    return candidate


def _hyperparameter_candidate(base: dict, name: str) -> dict:
    """Return one scalar hyperparameter check on the seeded raw model."""
    candidate = _feature_candidate(base, "feature_1")
    spec = HYPERPARAMETER_VARIANTS[name]
    candidate["experiment_id"] = spec["experiment_id"]
    architecture = copy.deepcopy(candidate.get("architecture", {}))
    if "dropout" in spec:
        architecture["dropout"] = float(spec["dropout"])
    if "relation_residual_scale" in spec:
        cross = copy.deepcopy(architecture.get("cross_appliance", {}))
        cross["residual_scale"] = float(spec["relation_residual_scale"])
        architecture["cross_appliance"] = cross
    candidate["architecture"] = architecture
    return candidate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument(
        "--stage",
        choices=("features", "loss", "architecture", "hyperparameters", "repro"),
        default="features",
        help="Controlled simplification family to run.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=None,
        help="Named candidates to run in order (default: every candidate in the stage).",
    )
    parser.add_argument(
        "--feature-base",
        choices=("feature_1", "feature_2_mean"),
        default="feature_1",
        help="Validated compact input used by loss-stage candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    experiment = load_experiment(args.experiment)
    base_model_cfg = load_model_config(args.model_config)

    variants_by_stage = {
        "features": FEATURE_VARIANTS,
        "loss": LOSS_VARIANTS,
        "architecture": ARCHITECTURE_VARIANTS,
        "hyperparameters": HYPERPARAMETER_VARIANTS,
        "repro": REPRO_VARIANTS,
    }
    variants = variants_by_stage[args.stage]
    candidates = args.candidates or list(variants)
    unknown = sorted(set(candidates) - set(variants))
    if unknown:
        raise ValueError(f"Unknown {args.stage} candidates: {unknown}")

    for name in candidates:
        if args.stage == "features":
            model_cfg = _feature_candidate(base_model_cfg, name)
        elif args.stage == "loss":
            model_cfg = _loss_candidate(base_model_cfg, name, args.feature_base)
        elif args.stage == "architecture":
            model_cfg = _architecture_candidate(base_model_cfg, name)
        elif args.stage == "hyperparameters":
            model_cfg = _hyperparameter_candidate(base_model_cfg, name)
        else:
            model_cfg = _repro_candidate(base_model_cfg, name)
        merged = merge_configs(experiment, model_cfg)
        data_root = Path(merged["data_root"])
        if not data_root.is_absolute():
            data_root = ROOT / data_root

        run_dir = ROOT / "runs" / merged["experiment_id"] / merged["model_name"]
        validation_metrics = run_dir / "validation_metrics.csv"
        checkpoint = run_dir / "best.pt"

        print(f"\n=== {name}: {merged['experiment_id']} ===", flush=True)
        adapter = get_adapter(merged["model_name"], merged, data_root=str(data_root))

        if validation_metrics.exists():
            print(f"Already complete: {validation_metrics}", flush=True)
            continue
        if not checkpoint.exists():
            checkpoint = train_model(adapter, run_dir)
            print(f"Saved checkpoint: {checkpoint}", flush=True)

        prediction_path = evaluate_model(
            adapter,
            checkpoint,
            run_dir,
            split="validation",
            show_cost_summary=False,
        )
        print(f"Saved validation predictions: {prediction_path}", flush=True)


if __name__ == "__main__":
    main()

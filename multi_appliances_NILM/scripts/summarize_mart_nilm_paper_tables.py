#!/usr/bin/env python
"""Summarise the three-seed paper experiments for Tables V--VIII.

Table V uses the two held-out test houses separately. Table VI compares cost
under one inference batch size and hardware, without reporting selected epoch.
Table VII is computed only from the validation houses: metrics are first
calculated per house and then averaged with equal house weight.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.common import PredictionBundle
from data.dataloader import get_state_label_source
from evaluation.metrics import evaluate_bundle, resolve_power_postprocess


SEEDS = (2026, 2027, 2028)
MODEL_LABELS = {
    "unetnilm": "UNet-NILM",
    "mat_nilm": "MAT-Conv",
    "transfer_multi_appliance": "TransferNILM",
    "mart_nilm": "MART-NILM",
}
APPLIANCES = ("kettle", "fridge", "dishwasher", "washingmachine", "microwave")
TEST_SCENARIOS = ("ukdale_house2", "refit_house20")
VARIANTS = (
    "complete",
    "single_scale_cnn",
    "no_feature_encoder",
    "no_shared_temporal_encoder",
    "no_appliance_decoder",
    "no_relation_attention",
)
TRAINING_VARIANTS = (
    "standard_joint_training",
    "focal_event_mixing",
    "appliance_wise_balancing",
    "complete",
)
METRICS = (
    "mae",
    "sae",
    "on_mae",
    "average_precision",
    "precision",
    "recall",
    "f1",
)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def _subset(bundle: PredictionBundle, mask: np.ndarray, split: str) -> PredictionBundle:
    def take(value):
        return None if value is None else np.asarray(value)[mask]

    return replace(
        bundle,
        split=split,
        sample_index=np.arange(int(mask.sum()), dtype=np.int32),
        y_true_watts=take(bundle.y_true_watts),
        y_pred_watts=take(bundle.y_pred_watts),
        y_true_on=take(bundle.y_true_on),
        y_pred_on=take(bundle.y_pred_on),
        y_pred_state_prob=take(bundle.y_pred_state_prob),
        csv_timesteps=take(bundle.csv_timesteps),
        segment_ids=take(bundle.segment_ids),
    )


def _validation_house_rows(dataset_root: Path) -> list[tuple[str, int, int]]:
    summary = pd.read_csv(dataset_root / "selection_summary.csv")
    rows = summary.loc[summary["role"].eq("validation")]
    result: list[tuple[str, int, int]] = []
    start = 0
    for row in rows.itertuples(index=False):
        stop = start + int(row.n_rows)
        result.append((f"{row.dataset}_house{row.house}", start, stop))
        start = stop
    if len(result) != 2:
        raise ValueError(f"Expected two validation houses, found {len(result)}")
    return result


def _equal_house_validation_metrics(run_dir: Path, dataset_root: Path) -> pd.Series:
    bundle = PredictionBundle.load(run_dir / "validation_predictions.npz")
    if bundle.csv_timesteps is None:
        raise ValueError(f"Missing csv_timesteps in {run_dir}")
    with (run_dir / "config_merged.yaml").open("r", encoding="utf-8") as handle:
        saved_cfg = yaml.safe_load(handle)
    experiment_cfg = saved_cfg.get("experiment", saved_cfg)
    model_cfg = saved_cfg.get("model", saved_cfg)
    power_postprocess = resolve_power_postprocess(
        experiment_cfg,
        bundle.appliances,
        model_cfg,
    )
    house_overall = []
    for house, start, stop in _validation_house_rows(dataset_root):
        mask = (bundle.csv_timesteps >= start) & (bundle.csv_timesteps < stop)
        if not np.any(mask):
            raise ValueError(f"No reconstructed validation samples for {house}")
        house_bundle = _subset(bundle, mask, house)
        metrics = evaluate_bundle(
            house_bundle,
            sae_period=int(experiment_cfg["evaluation"].get("sae_period", 1200)),
            state_label_source=get_state_label_source(model_cfg),
            power_postprocess=power_postprocess,
            sample_seconds=float(experiment_cfg["csv"]["sample_seconds"]),
            event_tolerance_seconds=float(
                experiment_cfg["evaluation"].get("event_tolerance_seconds", 0.0)
            ),
        )
        house_overall.append(metrics.loc[metrics["appliance"].eq("overall")].iloc[0])
    return pd.DataFrame(house_overall).mean(numeric_only=True)


def table_v(
    run_root: Path,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
) -> pd.DataFrame:
    records = []
    for model in models:
        for scenario in TEST_SCENARIOS:
            for seed in seeds:
                path = (
                    run_root
                    / "table_v"
                    / model
                    / f"seed_{seed}"
                    / "test"
                    / scenario
                    / "metrics.csv"
                )
                frame = pd.read_csv(path).set_index("appliance")
                for metric in METRICS:
                    for appliance in (*APPLIANCES, "overall"):
                        value = float(frame.loc[appliance, metric])
                        # Precision is undefined when a model predicts no
                        # positive samples.  Use the standard zero-division
                        # convention so every seed contributes to the reported
                        # three-seed mean and standard deviation.
                        if metric == "precision" and not np.isfinite(value):
                            value = 0.0
                        records.append(
                            {
                                "model": MODEL_LABELS[model],
                                "scenario": scenario,
                                "seed": seed,
                                "metric": metric,
                                "appliance": appliance,
                                "value": value,
                            }
                        )
    raw = pd.DataFrame(records)
    return (
        raw.groupby(["model", "scenario", "metric", "appliance"], sort=False)["value"]
        .agg(mean="mean", std=lambda x: x.std(ddof=1))
        .reset_index()
    )


def table_vi(
    run_root: Path,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for model in models:
        for seed in seeds:
            run_dir = run_root / "table_v" / model / f"seed_{seed}"
            with (run_dir / "training_time.json").open("r", encoding="utf-8") as handle:
                timing = json.load(handle)
            with (run_dir / "inference_benchmark.json").open("r", encoding="utf-8") as handle:
                benchmark = json.load(handle)
            split_rows = {row["split"]: row for row in benchmark["splits"]}
            row = {
                "model": MODEL_LABELS[model],
                "seed": seed,
                "parameters_m": float(benchmark["parameters_trainable_millions"]),
                "model_size_mib": float(benchmark["checkpoint_size_mib"]),
                "training_seconds": float(timing["total_seconds"]),
            }
            for scenario in TEST_SCENARIOS:
                result = split_rows[scenario]
                row[f"{scenario}_inference_seconds"] = float(result["seconds"])
                row[f"{scenario}_timesteps_per_second"] = float(
                    result["timesteps_per_second"]
                )
                row[f"{scenario}_peak_gpu_mib"] = float(result["peak_gpu_memory_mib"])
            rows.append(row)
    raw = pd.DataFrame(rows)
    aggregations = {
        "parameters_m": ("parameters_m", "mean"),
        "model_size_mib_mean": ("model_size_mib", "mean"),
        "model_size_mib_std": ("model_size_mib", lambda x: x.std(ddof=1)),
        "training_seconds_mean": ("training_seconds", "mean"),
        "training_seconds_std": ("training_seconds", lambda x: x.std(ddof=1)),
    }
    for scenario in TEST_SCENARIOS:
        for metric in (
            "inference_seconds",
            "timesteps_per_second",
            "peak_gpu_mib",
        ):
            column = f"{scenario}_{metric}"
            aggregations[f"{column}_mean"] = (column, "mean")
            aggregations[f"{column}_std"] = (column, lambda x: x.std(ddof=1))
    return raw.groupby("model", sort=False).agg(**aggregations).reset_index()


def table_vii(run_root: Path, dataset_root: Path, seeds: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for seed in seeds:
            if variant == "complete":
                run_dir = run_root / "table_v" / "mart_nilm" / f"seed_{seed}"
            else:
                run_dir = run_root / "table_vii" / variant / f"seed_{seed}"
            metric = _equal_house_validation_metrics(run_dir, dataset_root)
            with (run_dir / "training_time.json").open("r", encoding="utf-8") as handle:
                timing = json.load(handle)
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "parameters_m": float(timing["parameters_trainable_millions"]),
                    "mae": float(metric["mae"]),
                    "on_mae": float(metric["on_mae"]),
                    "off_mae": float(metric["off_mae"]),
                    "sae": float(metric["sae"]),
                    "macro_f1": float(metric["f1"]),
                }
            )
    raw = pd.DataFrame(rows)
    return (
        raw.groupby("variant", sort=False)
        .agg(
            parameters_m=("parameters_m", "mean"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", lambda x: x.std(ddof=1)),
            on_mae_mean=("on_mae", "mean"),
            on_mae_std=("on_mae", lambda x: x.std(ddof=1)),
            off_mae_mean=("off_mae", "mean"),
            off_mae_std=("off_mae", lambda x: x.std(ddof=1)),
            sae_mean=("sae", "mean"),
            sae_std=("sae", lambda x: x.std(ddof=1)),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", lambda x: x.std(ddof=1)),
        )
        .reset_index()
    )


def table_vii_test(run_root: Path, seeds: tuple[int, ...]) -> pd.DataFrame:
    """Summarise frozen architecture variants on each test house.

    The additional ``house_macro`` rows give the equal-weight mean of UK-DALE
    house 2 and REFIT house 20. They do not pool samples across houses.
    """
    rows = []
    for variant in VARIANTS:
        for seed in seeds:
            run_dir = (
                run_root / "table_v" / "mart_nilm" / f"seed_{seed}"
                if variant == "complete"
                else run_root / "table_vii" / variant / f"seed_{seed}"
            )
            house_values = []
            for scenario in TEST_SCENARIOS:
                metrics = pd.read_csv(run_dir / "test" / scenario / "metrics.csv")
                overall = metrics.loc[metrics["appliance"].eq("overall")].iloc[0]
                values = {
                    "mae": float(overall["mae"]),
                    "on_mae": float(overall["on_mae"]),
                    "off_mae": float(overall["off_mae"]),
                    "sae": float(overall["sae"]),
                    "macro_f1": float(overall["f1"]),
                }
                house_values.append(values)
                rows.append({"variant": variant, "scenario": scenario, "seed": seed, **values})
            rows.append(
                {
                    "variant": variant,
                    "scenario": "house_macro",
                    "seed": seed,
                    **{
                        key: float(np.mean([item[key] for item in house_values]))
                        for key in ("mae", "on_mae", "off_mae", "sae", "macro_f1")
                    },
                }
            )
    raw = pd.DataFrame(rows)
    return (
        raw.groupby(["scenario", "variant"], sort=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", lambda x: x.std(ddof=1)),
            on_mae_mean=("on_mae", "mean"),
            on_mae_std=("on_mae", lambda x: x.std(ddof=1)),
            off_mae_mean=("off_mae", "mean"),
            off_mae_std=("off_mae", lambda x: x.std(ddof=1)),
            sae_mean=("sae", "mean"),
            sae_std=("sae", lambda x: x.std(ddof=1)),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", lambda x: x.std(ddof=1)),
        )
        .reset_index()
    )


def _table_viii_run_dir(run_root: Path, variant: str, seed: int) -> Path:
    if variant == "complete":
        return run_root / "table_v" / "mart_nilm" / f"seed_{seed}"
    return run_root / "table_viii" / variant / f"seed_{seed}"


def table_viii(
    run_root: Path,
    dataset_root: Path,
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return overall and per-appliance validation summaries for Table VIII."""
    overall_rows = []
    appliance_rows = []
    house_ranges = _validation_house_rows(dataset_root)
    for variant in TRAINING_VARIANTS:
        for seed in seeds:
            run_dir = _table_viii_run_dir(run_root, variant, seed)
            overall = _equal_house_validation_metrics(run_dir, dataset_root)
            overall_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "mae": float(overall["mae"]),
                    "on_mae": float(overall["on_mae"]),
                    "sae": float(overall["sae"]),
                    "average_precision": float(overall["average_precision"]),
                    "macro_f1": float(overall["f1"]),
                }
            )

            bundle = PredictionBundle.load(run_dir / "validation_predictions.npz")
            with (run_dir / "config_merged.yaml").open("r", encoding="utf-8") as handle:
                saved_cfg = yaml.safe_load(handle)
            experiment_cfg = saved_cfg.get("experiment", saved_cfg)
            model_cfg = saved_cfg.get("model", saved_cfg)
            postprocess = resolve_power_postprocess(
                experiment_cfg, bundle.appliances, model_cfg
            )
            house_frames = []
            for house, start, stop in house_ranges:
                mask = (bundle.csv_timesteps >= start) & (bundle.csv_timesteps < stop)
                metrics = evaluate_bundle(
                    _subset(bundle, mask, house),
                    sae_period=int(experiment_cfg["evaluation"].get("sae_period", 1200)),
                    state_label_source=get_state_label_source(model_cfg),
                    power_postprocess=postprocess,
                    sample_seconds=float(experiment_cfg["csv"]["sample_seconds"]),
                    event_tolerance_seconds=float(
                        experiment_cfg["evaluation"].get("event_tolerance_seconds", 0.0)
                    ),
                )
                house_frames.append(metrics.set_index("appliance"))
            for appliance in APPLIANCES:
                appliance_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "appliance": appliance,
                        **{
                            metric: float(
                                np.mean([frame.loc[appliance, metric] for frame in house_frames])
                            )
                            for metric in ("mae", "on_mae", "sae", "average_precision", "f1")
                        },
                    }
                )

    overall_raw = pd.DataFrame(overall_rows)
    overall_agg = {}
    for metric in ("mae", "on_mae", "sae", "average_precision", "macro_f1"):
        overall_agg[f"{metric}_mean"] = (metric, "mean")
        overall_agg[f"{metric}_std"] = (metric, lambda x: x.std(ddof=1))
    overall_summary = (
        overall_raw.groupby("variant", sort=False).agg(**overall_agg).reset_index()
    )

    appliance_raw = pd.DataFrame(appliance_rows)
    appliance_agg = {}
    for metric in ("mae", "on_mae", "sae", "average_precision", "f1"):
        appliance_agg[f"{metric}_mean"] = (metric, "mean")
        appliance_agg[f"{metric}_std"] = (metric, lambda x: x.std(ddof=1))
    appliance_summary = (
        appliance_raw.groupby(["variant", "appliance"], sort=False)
        .agg(**appliance_agg)
        .reset_index()
    )
    return overall_summary, appliance_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs" / "paper_5w")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "datasets" / "mixed_ukdale_refit_5w_all_house_split",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "paper_5w" / "summaries")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_LABELS),
        default=["mart_nilm"],
        help="Table V/VI models whose completed runs should be summarised.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=("v", "vi", "vii", "vii_test", "viii"),
        default=("v", "vi", "vii"),
        help="Summaries to create; useful while later experiment families are still running.",
    )
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    models = tuple(args.models)
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for a standard deviation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if "v" in args.tables:
        outputs["table_v_models.csv"] = table_v(args.run_root, seeds, models)
    if "vi" in args.tables:
        outputs["table_vi_models.csv"] = table_vi(args.run_root, seeds, models)
    if "vii" in args.tables:
        outputs["table_vii_mart_nilm.csv"] = table_vii(
            args.run_root,
            args.dataset_root,
            seeds,
        )
    if "vii_test" in args.tables:
        outputs["table_vii_test_mart_nilm.csv"] = table_vii_test(
            args.run_root,
            seeds,
        )
    if "viii" in args.tables:
        overall, appliances = table_viii(args.run_root, args.dataset_root, seeds)
        outputs["table_viii_mart_nilm.csv"] = overall
        outputs["table_viii_per_appliance_mart_nilm.csv"] = appliances
    for name, frame in outputs.items():
        path = args.output_dir / name
        frame.to_csv(path, index=False, float_format="%.6f")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

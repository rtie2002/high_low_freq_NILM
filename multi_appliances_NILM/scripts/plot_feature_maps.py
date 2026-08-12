#!/usr/bin/env python
"""Plot conv feature maps from a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from adapters.config import load_experiment, load_model_config, merge_configs, model_name_from_config, resolve_tensor_dtype
from evaluation.feature_maps import FeatureMapConfig, save_feature_maps
from main import MODELS, get_adapter, _default_model_config, _default_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dynamic conv feature maps for a NILM model.")
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--experiment", type=Path, default=ROOT / "config" / "experiment_ukdale.yaml")
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--appliance", type=str, default=None, help="Plot only this appliance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment(args.experiment)
    model_cfg_path = args.model_config or _default_model_config(args.model)
    model_cfg = load_model_config(model_cfg_path)
    if model_name_from_config(model_cfg) != args.model:
        raise ValueError(f"--model {args.model!r} does not match {model_cfg_path}")

    merged = merge_configs(experiment, model_cfg)
    data_root = args.data_path or merged.get("data_root")
    if data_root is not None:
        data_root = Path(data_root)
        if not data_root.is_absolute():
            data_root = ROOT / data_root

    adapter = get_adapter(args.model, merged, data_root=str(data_root) if data_root else None)
    run_dir = args.run_dir or _default_run_dir(merged["experiment_id"], args.model)
    ckpt = args.checkpoint or (run_dir / "best.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, tensor_dtype = resolve_tensor_dtype(adapter.model_cfg)
    model = adapter.build_model(device)
    if tensor_dtype == torch.float64:
        model = model.double()
    payload = torch.load(ckpt, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    appliances = adapter.cfg["appliances"]
    if args.appliance:
        if args.appliance not in appliances:
            raise ValueError(f"Unknown appliance {args.appliance!r}. Available: {appliances}")
        appliances = [args.appliance]

    cfg = FeatureMapConfig.from_dict(
        adapter.model_cfg.get("training", {}).get("plots", {}).get("feature_maps")
    )
    cfg.enabled = True
    cfg.max_examples = args.max_examples

    out_dir = run_dir / "feature_maps" / args.split
    loader = adapter.build_dataloader(args.split)
    save_feature_maps(
        adapter,
        model,
        loader,
        out_dir,
        split=args.split,
        device=device,
        appliances=appliances,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()

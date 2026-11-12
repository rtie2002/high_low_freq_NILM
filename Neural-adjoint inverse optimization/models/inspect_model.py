"""Print and save a reproducible summary of the configured CNN."""

import argparse
from pathlib import Path

import torch

from models.cnn import build_forward_model
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "configs/stage1.yaml")
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "docs/model_architecture.md")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if "model_config" in config:
        config["model"] = load_yaml(args.config.parent / config["model_config"])["model"]
    model = build_forward_model(config["model"])
    length = int(config["data"]["window_length"])
    total = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    latent = (length + model.DOWNSAMPLE - 1) // model.DOWNSAMPLE
    text = f"""# CNN architecture report

Generated from `{args.config.name}`.

| Item | Value |
| --- | --- |
| Input power | `[B, 5, {length}]` |
| Encoder single-trace input | `[B, 1, {length}]` |
| Latent code | `[B, 5, {model.latent_channels}, {latent}]` |
| Decoder output | `[B, 5, {length}]` |
| Downsampling factor | {model.DOWNSAMPLE} |
| Trainable parameters | {total:,} |

## Encoder

`{model.encoder}`

## Decoder

`{model.decoder}`

The shared conditional decoder concatenates one fixed five-dimensional appliance identity vector to each latent code before decoding. The final Softplus enforces nonnegative power.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

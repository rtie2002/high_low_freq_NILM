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
    encoder_summary = model.encoder if hasattr(model, "encoder") else f"5 independent encoders:\n{model.encoders}"
    decoder_summary = model.decoder if hasattr(model, "decoder") else f"5 independent decoders:\n{model.decoders}"
    text = f"""# CNN architecture report

Generated from `{args.config.name}`. Model: `{config['model']['name']}`.

| Item | Value |
| --- | --- |
| Model type | `{model.__class__.__name__}` |
| Input power | `[B, 5, {length}]` |
| Latent code | `[B, 5, {model.latent_channels}, {latent}]` |
| Decoder output | `[B, 5, {length}]` |
| Downsampling factor | {model.DOWNSAMPLE} |
| Trainable parameters | {total:,} |

## Encoder

`{encoder_summary}`

## Decoder

`{decoder_summary}`
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

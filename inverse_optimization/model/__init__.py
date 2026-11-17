"""Choose a forward model from config.yaml."""

from pathlib import Path

import yaml

from .lstm import LSTM_NILM


MODEL_CLASSES = {
    "lstm": LSTM_NILM,
}


def build_model(model_config):
    """Build the model named by model.name in config.yaml."""
    model_name = model_config["name"]
    config_path = Path(__file__).resolve().parent / "config" / f"{model_name}.yaml"
    with config_path.open(encoding="utf-8") as file:
        settings = yaml.safe_load(file)
    return MODEL_CLASSES[model_name](**settings)

"""Predict a total-meter waveform from appliance powers and known inputs."""

from pathlib import Path

import torch
import yaml
from torch import nn

class LSTM_NILM(nn.Module):
    def __init__(self, input_size, hidden_size, **lstm_options):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, **lstm_options)
        directions = 2 if self.lstm.bidirectional else 1
        self.output_layer = nn.Linear(hidden_size * directions, 1)

    def forward(self, appliance_power):
        # The loader and loss always use batch-first tensors.
        if self.lstm.batch_first:
            hidden_sequence, _ = self.lstm(appliance_power)
        else:
            hidden_sequence, _ = self.lstm(appliance_power.transpose(0, 1))
            hidden_sequence = hidden_sequence.transpose(0, 1)
            
        aggregate = self.output_layer(hidden_sequence)  # [B, T, 1]
        return aggregate.squeeze(-1)  # [B, T]; z-score targets can be negative.


if __name__ == "__main__":
    model_dir = Path(__file__).resolve().parent

    # Load the LSTM settings, such as input_size and hidden_size.
    with (model_dir / "config" / "lstm.yaml").open(encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    # Use the project's window length for the example input.
    with (model_dir.parent / "stage1_forward" / "config.yaml").open(encoding="utf-8") as file:
        window_length = yaml.safe_load(file)["data"]["window_length"]

    model = LSTM_NILM(**settings)
    example = torch.zeros(2, window_length, settings["input_size"])
    print(model)
    with torch.no_grad():
        print("Input:", tuple(example.shape), "Output:", tuple(model(example).shape))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

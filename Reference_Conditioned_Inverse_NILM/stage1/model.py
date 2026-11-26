"""Shared Stage 1 encoder for individual appliance power events."""

from collections.abc import Sequence

import torch
from torch import nn


# Stage 1 class order. The reference CSV folder for washing_machine is washingmachine.
APPLIANCES = ("kettle", "fridge", "dishwasher", "washing_machine", "microwave")


class ApplianceReferenceEncoder(nn.Module):
    """Stage 1 model for appliance power sampled every 8 seconds.

    Input: B normalized power traces, each with shape [L_i].
    The data loader applies normalization before passing traces to the model.
    Output: B reference vectors [B, 128] and five class scores [B, 5].
    B is the number of events. L_i is the number of samples in event i.
    """

    def __init__(self, use_batch_norm=True):
        super().__init__()

        # L = timestamps
        # 1. Feature extractor: [1, 1, L] -> [1, 64, L].
        # The axes are [events, channels, time]. Encode one event at a time.
        # Unit stride and symmetric padding keep all L positions at each layer.
        # When enabled, BatchNorm uses one event's time positions during training.
        # Use Identity to compare the same CNN without these per-event statistics.
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),   # [1, 1, L]  -> [1, 32, L]
            nn.BatchNorm1d(num_features=32) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # [1, 32, L] -> [1, 64, L]
            nn.BatchNorm1d(num_features=64) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  # [1, 64, L] -> [1, 64, L]
            nn.BatchNorm1d(num_features=64) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
        )

        # 2. Embedding layer: 64 mean features + 64 max features -> 128 values.
        self.embedding = nn.Linear(2 * 64, 128)

        # 3. Classification head: [B, 128] -> [B, 5] raw scores.
        self.classifier = nn.Linear(128, len(APPLIANCES))

    def encode(self, power: torch.Tensor) -> torch.Tensor:
        """Encode one normalized event [L] into a reference vector [128]."""
        conv_input = power[None, None, :]                 # [1, 1, L]
        features = self.features(conv_input).squeeze(0)  # [64, L]

        # Pool across time. Mean captures typical responses; max retains peaks.
        mean_features = features.mean(dim=-1)  # [64]
        max_features = features.amax(dim=-1)   # [64]
        pooled = torch.cat((mean_features, max_features), dim=0)  # [128]
        reference_vector = self.embedding(pooled)                # [128]
        return reference_vector

    def forward(self, events: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the whole model on a batch of variable-length events."""
        event_vectors = [self.encode(event) for event in events]  # B vectors of size 128
        embeddings = torch.stack(event_vectors, dim=0)           # [B, 128]
        class_scores = self.classifier(embeddings)                # [B, 5]
        return embeddings, class_scores

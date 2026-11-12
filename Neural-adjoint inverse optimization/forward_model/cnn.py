"""Small source-waveform CNNs used by the Paper 1 forward-model ablation."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ApplianceConditionedCNN(nn.Module):
    """Reconstruct five appliance traces and their target-only aggregate.

    Power tensors use one common scale, e.g. watts / 4000. The input shape
    is [batch, 5, time], in the channel order specified by APPLIANCES.
    Train against the five submeter traces using the first forward output.
    For inverse optimization, freeze the decoder and optimize codes passed
    directly to decode(). The class condition comes from the channel index.
    """

    APPLIANCES = ("kettle", "fridge", "dishwasher", "washingmachine", "microwave")
    DOWNSAMPLE = 8

    def __init__(
        self,
        hidden_channels: int = 32,
        latent_channels: int = 2,
        output_softplus_beta: float = 10.0,
        output_bias: float = -0.3,
    ):
        super().__init__()
        if hidden_channels < 1 or not 1 <= latent_channels < self.DOWNSAMPLE:
            raise ValueError("Use positive hidden channels and 1–7 latent channels.")
        if output_softplus_beta <= 0:
            raise ValueError("output_softplus_beta must be positive.")
        self.latent_channels = latent_channels
        self.output_softplus_beta = float(output_softplus_beta)
        appliance_count = len(self.APPLIANCES)
        self.register_buffer("appliance_ids", torch.eye(appliance_count), persistent=False)

        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, latent_channels, kernel_size=4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                latent_channels + appliance_count, hidden_channels, 4, stride=2, padding=1
            ),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_channels, 1, 4, stride=2, padding=1),
        )
        # Initialize near low power in the shared scaled-watt coordinate.
        nn.init.constant_(self.decoder[-1].bias, float(output_bias))

    def encode(self, appliance_power: Tensor) -> Tensor:
        """Map [B, 5, T] to [B, 5, Z, ceil(T / 8)]."""
        if (
            appliance_power.ndim != 3
            or appliance_power.shape[1] != len(self.APPLIANCES)
            or appliance_power.shape[-1] == 0
        ):
            raise ValueError("appliance_power must have shape [B, 5, T] with T > 0.")
        batch, appliances, time = appliance_power.shape
        pad_right = (-time) % self.DOWNSAMPLE
        # Treat appliance traces as separate examples with shared encoder weights.
        traces = F.pad(appliance_power.reshape(batch * appliances, 1, time), (0, pad_right))
        codes = self.encoder(traces)
        return codes.reshape(batch, appliances, self.latent_channels, codes.shape[-1])

    def decode(self, codes: Tensor, output_length: int) -> Tensor:
        """Map candidate codes [B, 5, Z, ceil(T / 8)] to powers [B, 5, T]."""
        appliances = len(self.APPLIANCES)
        if codes.ndim != 4 or codes.shape[1:3] != (appliances, self.latent_channels):
            raise ValueError("codes must have shape [B, 5, latent_channels, latent_time].")
        batch, _, _, latent_time = codes.shape
        expected_latent_time = (output_length + self.DOWNSAMPLE - 1) // self.DOWNSAMPLE
        if output_length <= 0 or latent_time != expected_latent_time:
            raise ValueError("latent_time must equal ceil(output_length / 8).")

        # Each device has its own code and a fixed one-hot ID. The convolutional
        # decoder weights are shared, and device traces are decoded independently.
        ids = self.appliance_ids.to(dtype=codes.dtype)[None, :, :, None]
        conditioned = torch.cat((codes, ids.expand(batch, -1, -1, latent_time)), dim=2)
        power = self.decoder(conditioned.flatten(0, 1))
        power = F.softplus(power, beta=self.output_softplus_beta)  # Smooth, nonnegative power.
        return power.reshape(batch, appliances, -1)[..., :output_length]

    def forward(self, appliance_power: Tensor) -> tuple[Tensor, Tensor]:
        """Return reconstructed appliances [B, 5, T] and their sum [B, T]."""
        codes = self.encode(appliance_power)
        reconstructed = self.decode(codes, output_length=appliance_power.shape[-1])
        # Known measurement operator for the five monitored sources.
        target_aggregate = reconstructed.sum(dim=1)
        return reconstructed, target_aggregate


class ApplianceSpecificCNN(nn.Module):
    """Five independent waveform autoencoders with the same external interface.

    This is E2's controlled structural ablation.  Each source has its own
    encoder and decoder, so a code from source ``i`` can only be decoded by
    that source's decoder during normal reconstruction and inverse search.
    Inputs and outputs remain ``[B, 5, T]`` and latent codes remain
    ``[B, 5, Z, ceil(T/8)]``.  Consequently the physical Stage-2 summation and
    its optimisation budget are unchanged from :class:`ApplianceConditionedCNN`.
    """

    APPLIANCES = ApplianceConditionedCNN.APPLIANCES
    DOWNSAMPLE = ApplianceConditionedCNN.DOWNSAMPLE

    def __init__(
        self,
        hidden_channels: int = 32,
        latent_channels: int = 2,
        output_softplus_beta: float = 10.0,
        output_bias: float = -0.3,
    ):
        super().__init__()
        if hidden_channels < 1 or not 1 <= latent_channels < self.DOWNSAMPLE:
            raise ValueError("Use positive hidden channels and 1–7 latent channels.")
        if output_softplus_beta <= 0:
            raise ValueError("output_softplus_beta must be positive.")
        self.latent_channels = latent_channels
        self.output_softplus_beta = float(output_softplus_beta)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden_channels, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden_channels, latent_channels, kernel_size=4, stride=2, padding=1),
            )
            for _ in self.APPLIANCES
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(latent_channels, hidden_channels, 4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose1d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose1d(hidden_channels, 1, 4, stride=2, padding=1),
            )
            for _ in self.APPLIANCES
        ])
        for decoder in self.decoders:
            nn.init.constant_(decoder[-1].bias, float(output_bias))

    def encode(self, appliance_power: Tensor) -> Tensor:
        """Encode independent source traces from [B,5,T] to [B,5,Z,L]."""
        if (
            appliance_power.ndim != 3
            or appliance_power.shape[1] != len(self.APPLIANCES)
            or appliance_power.shape[-1] == 0
        ):
            raise ValueError("appliance_power must have shape [B, 5, T] with T > 0.")
        pad_right = (-appliance_power.shape[-1]) % self.DOWNSAMPLE
        codes = [encoder(F.pad(appliance_power[:, i : i + 1], (0, pad_right)))
                 for i, encoder in enumerate(self.encoders)]
        return torch.stack(codes, dim=1)

    def decode(self, codes: Tensor, output_length: int) -> Tensor:
        """Decode source-specific latent codes [B,5,Z,L] to [B,5,T]."""
        expected_latent_time = (output_length + self.DOWNSAMPLE - 1) // self.DOWNSAMPLE
        if (
            codes.ndim != 4
            or codes.shape[1:3] != (len(self.APPLIANCES), self.latent_channels)
            or codes.shape[-1] != expected_latent_time
        ):
            raise ValueError("codes must have shape [B, 5, latent_channels, ceil(T / 8)].")
        traces = [F.softplus(decoder(codes[:, i]), beta=self.output_softplus_beta)[..., :output_length]
                  for i, decoder in enumerate(self.decoders)]
        return torch.cat(traces, dim=1)

    def forward(self, appliance_power: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.encode(appliance_power)
        reconstructed = self.decode(codes, output_length=appliance_power.shape[-1])
        return reconstructed, reconstructed.sum(dim=1)


def build_forward_model(model_config: dict[str, object]) -> nn.Module:
    """Create one ablation model from the checkpoint/YAML model mapping."""
    config = dict(model_config)
    name = config.pop("name", None)
    models = {
        "appliance_conditioned_cnn": ApplianceConditionedCNN,
        "appliance_specific_cnn": ApplianceSpecificCNN,
    }
    if name not in models:
        raise ValueError(f"Unknown forward model {name!r}; choose one of {tuple(models)}.")
    return models[name](**config)

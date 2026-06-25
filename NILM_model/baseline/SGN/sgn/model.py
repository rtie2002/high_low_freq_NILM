import torch
from torch import nn


class ConvSeq2SeqSubNet(nn.Module):
    """CNN Seq2Seq subnetwork used for SGN regression/classification branches."""

    def __init__(
        self,
        input_channels: int,
        input_length: int,
        output_length: int,
        hidden_fc: int = 1024,
        dropout: float = 0.0,
        num_outputs: int = 1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 30, kernel_size=10, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(30, 30, kernel_size=8, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(30, 40, kernel_size=6, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(40, 50, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(50, 50, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(50, 50, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * input_length, hidden_fc),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_fc, output_length * num_outputs),
        )
        self.output_length = output_length
        self.num_outputs = num_outputs
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.features(x))
        return out.view(out.shape[0], self.num_outputs, self.output_length)


class SGN(nn.Module):
    """Subtask Gated Network with multi-appliance power and on/off outputs."""

    def __init__(
        self,
        input_length: int,
        output_length: int,
        input_channels: int = 1,
        hidden_fc: int = 1024,
        dropout: float = 0.0,
        num_appliances: int = 1,
        gate_mode: str = "soft",
        standby_power: bool = False,
    ) -> None:
        super().__init__()
        if gate_mode not in {"soft", "hard"}:
            raise ValueError("gate_mode must be 'soft' or 'hard'")
        self.gate_mode = gate_mode
        self.standby_power = standby_power
        self.num_appliances = num_appliances
        self.regression = ConvSeq2SeqSubNet(
            input_channels, input_length, output_length, hidden_fc, dropout, num_appliances
        )
        self.classification = ConvSeq2SeqSubNet(
            input_channels, input_length, output_length, hidden_fc, dropout, num_appliances
        )
        self.standby = nn.Parameter(torch.zeros(num_appliances, 1)) if standby_power else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        power = self.regression(x)
        on_prob = torch.sigmoid(self.classification(x))
        if self.gate_mode == "hard":
            # Straight-through estimator: binary gate in forward, soft gradient in backward.
            hard = (on_prob >= 0.5).to(on_prob.dtype)
            gate = on_prob + (hard - on_prob).detach()
        else:
            gate = on_prob
        gated_power = power * gate
        if self.standby is not None:
            gated_power = gated_power + (1.0 - gate) * self.standby
        return {
            "power": power,
            "on_prob": on_prob,
            "gate": gate,
            "gated_power": gated_power,
        }

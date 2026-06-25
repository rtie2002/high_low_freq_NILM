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
    """Subtask Gated Network (Shin et al., AAAI 2019).

    Paper notation (one appliance i, one window):
      x~     input aggregate window          -> batch["x"]
      p_hat  regression output (power head)  -> power
      o_hat  classification ON probability   -> on_prob  (sigmoid)
      y_hat  final gated prediction          -> gated_power = p_hat * o_hat   (Eq. 6)

    Two identical Zhang Seq2Seq CNN branches; only the final gating connects them.
    """

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
        # --- [TRANSFER ADD] optional cross-house helpers (all off by default) ---
        gate_floor: float = 0.0,
        transfer_input_norm: bool = False,
        transfer_adapter: bool = False,
    ) -> None:
        super().__init__()
        if gate_mode not in {"soft", "hard"}:
            raise ValueError("gate_mode must be 'soft' or 'hard'")
        if not 0.0 <= gate_floor < 1.0:
            raise ValueError("gate_floor must be in [0, 1)")
        self.gate_mode = gate_mode
        self.standby_power = standby_power
        self.num_appliances = num_appliances

        # Paper SGN: two independent Zhang CNN branches (architecture unchanged).
        self.regression = ConvSeq2SeqSubNet(
            input_channels, input_length, output_length, hidden_fc, dropout, num_appliances
        )
        self.classification = ConvSeq2SeqSubNet(
            input_channels, input_length, output_length, hidden_fc, dropout, num_appliances
        )
        self.standby = nn.Parameter(torch.zeros(num_appliances, 1)) if standby_power else None

        # [TRANSFER ADD] Leaky gate floor applied after paper soft/hard gate.
        # Keeps a minimum fraction of regression alive when classifier is uncertain
        # on unseen houses (e.g. H2); does not change branch weights or depth.
        self.gate_floor = float(gate_floor)

        # [TRANSFER ADD] Per-window input normalization before both paper branches.
        # Reduces sensitivity to house-specific aggregate scale/shape (domain shift).
        self.transfer_input_norm = (
            nn.InstanceNorm1d(input_channels, affine=True) if transfer_input_norm else None
        )

        # [TRANSFER ADD] Residual input adapter shared by both branches.
        # Learns a house-invariant correction on x~ while keeping original subnets intact.
        # x' = x~ + adapter(x~); output still has input_channels for paper subnets.
        self.transfer_adapter = (
            nn.Sequential(
                nn.Conv1d(input_channels, input_channels, kernel_size=10, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv1d(input_channels, input_channels, kernel_size=8, padding="same"),
            )
            if transfer_adapter
            else None
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply optional transfer preprocessing; identity when all transfer flags are off."""
        if self.transfer_input_norm is not None:
            x = self.transfer_input_norm(x)
        if self.transfer_adapter is not None:
            x = x + self.transfer_adapter(x)
        return x

    def _apply_gate_floor(self, gate: torch.Tensor) -> torch.Tensor:
        """[TRANSFER ADD] Map gate in [0,1] to [gate_floor, 1]."""
        if self.gate_floor <= 0.0:
            return gate
        return self.gate_floor + (1.0 - self.gate_floor) * gate

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._prepare_input(x)

        # f_power(x~): regression subnetwork — raw power estimate p_hat (no sigmoid).
        power = self.regression(x)

        # f_on(x~): classification subnetwork — logits passed through sigmoid -> o_hat in [0, 1].
        # Paper Appendix A: sigmoid only on the classification branch.
        on_prob = torch.sigmoid(self.classification(x))

        if self.gate_mode == "hard":
            # Hard SGN variant (Eq. 10): gate = 1 if o_hat >= 0.5 else 0.
            # Straight-through estimator keeps soft o_hat for backward through the gate.
            hard = (on_prob >= 0.5).to(on_prob.dtype)
            gate = on_prob + (hard - on_prob).detach()
        else:
            # Soft SGN (Eq. 6): use o_hat directly as the gate.
            gate = on_prob

        # [TRANSFER ADD] Optional leaky floor on top of paper gate (see gate_floor above).
        gate = self._apply_gate_floor(gate)

        # Eq. (6): y_hat = p_hat * o_hat  (element-wise over output timesteps).
        # Autograd note: d(L_output)/d(p_hat) is scaled by o_hat, so regression learns
        # mainly when the classifier predicts ON; if o_hat -> 0, regression gradient -> 0.
        gated_power = power * gate

        # SGN-sp variant (Eq. 9): add learnable standby b when gate is closed.
        if self.standby is not None:
            gated_power = gated_power + (1.0 - gate) * self.standby

        return {
            "power": power,              # p_hat — used by optional reg_on_loss, plots (orange)
            "on_prob": on_prob,          # o_hat — trained by L_on (BCE)
            "gate": gate,                # soft o_hat or hard STE gate
            "gated_power": gated_power,  # y_hat — trained by L_output (MSE)
        }

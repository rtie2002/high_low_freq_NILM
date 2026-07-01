"""UNet-NILM model architecture only (Faustine et al., BuildSys 2020).

Hyperparameters live in config/models/unet_nilm.yaml and config/experiment.yaml.
The pipeline loads that file and passes values into the constructors below.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MLPLayer(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_arch: list[int],
        output_size: int | None = None,
        activation: nn.Module | None = None,
        batch_norm: bool = True,
    ):
        super().__init__()
        activation = activation or nn.PReLU()
        layer_sizes = [in_size, *hidden_arch]
        layers: list[nn.Module] = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            layers.append(layer)
            if batch_norm and i != 0:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            layers.append(activation)

        if output_size is not None:
            layers.append(nn.Linear(layer_sizes[-1], output_size))
            layers.append(activation)

        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.utils.weight_norm(layer)
                init.xavier_uniform_(layer.weight)

        self.mlp_network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp_network(z)


class Conv1D(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_kernels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        last: bool = False,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        activation = activation or nn.PReLU()
        self.conv = nn.Conv1d(n_channels, n_kernels, kernel_size, stride, padding)
        self.net = self.conv if last else nn.Sequential(self.conv, nn.BatchNorm1d(n_kernels), activation)
        nn.utils.weight_norm(self.conv)
        init.xavier_uniform_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Deconv1D(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_kernels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        last: bool = False,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        activation = activation or nn.PReLU()
        self.deconv = nn.ConvTranspose1d(n_channels, n_kernels, kernel_size, stride, padding)
        self.net = self.deconv if last else nn.Sequential(self.deconv, nn.BatchNorm1d(n_kernels), activation)
        init.xavier_uniform_(self.deconv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_kernels: int,
        n_layers: int,
        seq_size: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.feat_size = (seq_size - 1) // 2**n_layers + 1
        self.feat_dim = self.feat_size * n_kernels
        self.conv_stack = nn.Sequential(
            *(
                [Conv1D(n_channels, n_kernels // 2 ** (n_layers - 1), kernel_size, stride, padding)]
                + [
                    Conv1D(
                        n_kernels // 2 ** (n_layers - l),
                        n_kernels // 2 ** (n_layers - l - 1),
                        kernel_size,
                        stride,
                        padding,
                    )
                    for l in range(1, n_layers - 1)
                ]
                + [Conv1D(n_kernels // 2, n_kernels, kernel_size, stride, padding, last=True)]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.upsample = Deconv1D(in_ch, in_ch // 2, kernel_size, stride, padding)
        self.conv = Conv1D(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.upsample(x1)
        diff = x2.shape[2] - x1.shape[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet1D(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_layers: int,
        features_start: int,
        n_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        layers: list[nn.Module] = [Conv1D(n_channels, features_start, kernel_size, stride, padding)]
        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Conv1D(feats, feats * 2, kernel_size, stride, padding))
            feats *= 2
        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, kernel_size, stride, padding))
            feats //= 2

        head = nn.Conv1d(feats, num_classes, kernel_size=1)
        head = nn.utils.weight_norm(head)
        init.xavier_uniform_(head.weight)
        layers.append(head)
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xi = [self.layers[0](x)]
        for layer in self.layers[1 : self.num_layers]:
            xi.append(layer(xi[-1]))
        for i, layer in enumerate(self.layers[self.num_layers : -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class UNETNiLM(nn.Module):
    """
    UNet-NILM: joint multi-label state detection + multi-target quantile regression.

    Expected input shape:  (B, seq_len, in_size)
    State output shape:    (B, 2, output_size)
    Power output shape:    (B, n_quantiles, output_size)  if n_quantiles > 1
                           (B, output_size)                otherwise
    """

    def __init__(
        self,
        in_size: int,
        output_size: int,
        seq_len: int,
        d_model: int,
        n_layers: int,
        n_quantiles: int,
        features_start: int,
        pool_filter: int,
        encoder_n_layers: int,
        mlp_hidden: list[int],
        dropout: float,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.pool_filter = pool_filter
        self.output_size = output_size
        self.mlp_out_dim = mlp_hidden[-1]

        self.unet = UNet1D(
            num_classes=output_size,
            num_layers=n_layers,
            features_start=features_start,
            n_channels=in_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv_layer = Encoder(
            n_channels=output_size,
            n_kernels=d_model,
            n_layers=encoder_n_layers,
            seq_size=seq_len,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.mlp_layer = MLPLayer(in_size=d_model * pool_filter, hidden_arch=mlp_hidden)
        self.dropout = nn.Dropout(dropout)

        self.fc_out_state = nn.Linear(self.mlp_out_dim, output_size * 2)
        self.fc_out_power = nn.Linear(self.mlp_out_dim, output_size * n_quantiles)
        nn.init.xavier_normal_(self.fc_out_state.weight)
        nn.init.xavier_normal_(self.fc_out_power.weight)
        self.fc_out_state.bias.data.fill_(0)
        self.fc_out_power.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        x = x.permute(0, 2, 1)

        unet_out = self.dropout(self.unet(x))
        conv_out = self.conv_layer(unet_out)
        conv_out = F.adaptive_avg_pool1d(conv_out, self.pool_filter).reshape(b, -1)
        mlp_out = self.dropout(self.mlp_layer(conv_out))

        states_logits = self.fc_out_state(mlp_out).reshape(b, 2, self.output_size)
        power_logits = self.fc_out_power(mlp_out)
        if self.n_quantiles > 1:
            power_logits = power_logits.reshape(b, self.n_quantiles, self.output_size)

        return states_logits, power_logits


__all__ = ["UNETNiLM", "UNet1D", "Encoder", "MLPLayer"]

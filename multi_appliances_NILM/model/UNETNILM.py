"""UNet-NILM: joint multi-appliance state and power prediction.

This is a small, readable PyTorch port of the architecture published in:

    Faustine et al., "UNet-NILM: A Deep Neural Network for Multi-tasks
    Appliances State Detection and Power Estimation in NILM", NILM 2020.
    DOI: 10.1145/3427771.3427859

The public author repository contains a few mechanical inconsistencies
(``UNetBaseline`` is missing, ``num_classes`` is undefined, and ``block.py``
is not present). This port keeps the published/official model structure while
repairing those names and using stride 1 after an upsampling concatenation,
which is required for the block to behave as the U-Net described in the paper.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader

from data.common import BaseNILMAdapter, StepOutput
from model.UNETNILM_loss import PAPER_QUANTILES, UNETNILMLoss


def _xavier_uniform(module: nn.Module) -> nn.Module:
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> PReLU block used by the author code."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 2) -> None:
        super().__init__()
        conv = _xavier_uniform(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        )
        self.net = nn.Sequential(
            weight_norm(conv),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FinalEncoderConv(nn.Module):
    """Final author Encoder convolution: no batch norm or activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        conv = _xavier_uniform(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.conv = weight_norm(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsample, concatenate the matching encoder feature, then fuse."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        deconv = _xavier_uniform(
            nn.ConvTranspose1d(
                in_channels,
                in_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        )
        self.upsample = nn.Sequential(
            deconv,
            nn.BatchNorm1d(in_channels // 2),
            nn.PReLU(),
        )
        self.fuse = ConvBlock(in_channels, out_channels, stride=1)

    def forward(self, decoder: torch.Tensor, encoder: torch.Tensor) -> torch.Tensor:
        decoder = self.upsample(decoder)
        difference = encoder.shape[-1] - decoder.shape[-1]
        if difference >= 0:
            decoder = F.pad(decoder, (difference // 2, difference - difference // 2))
        else:
            crop = -difference
            decoder = decoder[..., crop // 2 : decoder.shape[-1] - (crop - crop // 2)]
        return self.fuse(torch.cat((encoder, decoder), dim=1))


class UNetFeatureExtractor(nn.Module):
    """One-dimensional U-Net feature extractor from Section 2.3."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        num_layers: int,
        features_start: int,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("UNet-NILM requires at least two U-Net levels")

        down_blocks: list[nn.Module] = []
        channels = input_channels
        features = features_start
        for _ in range(num_layers):
            down_blocks.append(ConvBlock(channels, features))
            channels = features
            features *= 2
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks: list[nn.Module] = []
        features = channels
        for _ in range(num_layers - 1):
            up_blocks.append(UpBlock(features, features // 2))
            features //= 2
        self.up_blocks = nn.ModuleList(up_blocks)

        self.output = weight_norm(
            _xavier_uniform(nn.Conv1d(features, output_channels, kernel_size=1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_features = []
        for block in self.down_blocks:
            x = block(x)
            encoder_features.append(x)

        x = encoder_features[-1]
        for index, block in enumerate(self.up_blocks):
            x = block(x, encoder_features[-2 - index])
        return self.output(x)


class UNetNILM(nn.Module):
    """Paper model: shared U-Net features with state and quantile heads.

    Input:
        aggregate window ``(B, L, 1)``

    Outputs:
        state logits ``(B, 2, A)``
        power quantiles ``(B, Q, A)``
    """

    def __init__(
        self,
        *,
        num_appliances: int,
        input_channels: int = 1,
        d_model: int = 128,
        dropout: float = 0.25,
        num_layers: int = 5,
        features_start: int = 32,
        pool_size: int = 32,
        mlp_hidden: int = 1024,
        num_quantiles: int = len(PAPER_QUANTILES),
    ) -> None:
        super().__init__()
        if num_appliances < 1:
            raise ValueError("num_appliances must be positive")

        self.num_appliances = num_appliances
        self.num_quantiles = num_quantiles
        self.unet = UNetFeatureExtractor(
            input_channels=input_channels,
            output_channels=num_appliances,
            num_layers=num_layers,
            features_start=features_start,
        )

        # The official code applies N/2 additional strided convolutions after
        # the U-Net and then pools to a fixed temporal size.
        output_layers = max(num_layers // 2, 1)
        encoder_channels = [num_appliances]
        for layer_index in range(output_layers):
            out_channels = d_model if layer_index == output_layers - 1 else d_model // 2
            encoder_channels.append(out_channels)

        conv_layers: list[nn.Module] = []
        for layer_index in range(output_layers):
            block_type = FinalEncoderConv if layer_index == output_layers - 1 else ConvBlock
            conv_layers.append(
                block_type(
                    encoder_channels[layer_index], encoder_channels[layer_index + 1]
                )
            )
        self.output_encoder = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            _xavier_uniform(nn.Linear(d_model * pool_size, mlp_hidden)),
            nn.PReLU(),
        )

        self.state_head = _xavier_uniform(nn.Linear(mlp_hidden, 2 * num_appliances))
        self.power_head = _xavier_uniform(
            nn.Linear(mlp_hidden, num_quantiles * num_appliances)
        )
        # The author code uses Xavier normal initialization for the two heads.
        nn.init.xavier_normal_(self.state_head.weight)
        nn.init.xavier_normal_(self.power_head.weight)

    def forward(self, aggregate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if aggregate.ndim != 3 or aggregate.shape[-1] != 1:
            raise ValueError("aggregate must have shape (batch, window, 1)")

        batch_size = aggregate.shape[0]
        features = aggregate.transpose(1, 2)
        features = self.dropout(self.unet(features))
        features = self.output_encoder(features)
        features = self.dropout(self.pool(features).flatten(start_dim=1))
        features = self.dropout(self.mlp(features))

        state_logits = self.state_head(features).reshape(
            batch_size, 2, self.num_appliances
        )
        power_quantiles = self.power_head(features).reshape(
            batch_size, self.num_quantiles, self.num_appliances
        )
        return state_logits, power_quantiles


class UNetNILMAdapter(BaseNILMAdapter):
    """Connect the paper model to this repository's shared train/eval runner."""

    name = "unetnilm"

    def build_model(self, device: torch.device) -> torch.nn.Module:
        architecture = self.model_cfg["architecture"]
        quantiles = self.model_cfg.get("loss", {}).get("quantiles", PAPER_QUANTILES)
        return UNetNILM(
            num_appliances=len(self.cfg["appliances"]),
            input_channels=int(architecture.get("input_channels", 1)),
            d_model=int(architecture.get("d_model", 128)),
            dropout=float(architecture.get("dropout", 0.25)),
            num_layers=int(architecture.get("num_layers", 5)),
            features_start=int(architecture.get("features_start", 32)),
            pool_size=int(architecture.get("pool_size", 32)),
            mlp_hidden=int(architecture.get("mlp_hidden", 1024)),
            num_quantiles=len(quantiles),
        ).to(device)

    def build_loss(self) -> UNETNILMLoss:
        quantiles = self.model_cfg.get("loss", {}).get("quantiles", PAPER_QUANTILES)
        return UNETNILMLoss(quantiles)

    def configure_optimizer(self, model: torch.nn.Module):
        training = self.model_cfg["training"]
        betas = training.get("adam_betas", [0.9, 0.98])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(training.get("learning_rate", 0.001)),
            betas=(float(betas[0]), float(betas[1])),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        scheduler_cfg = training.get("lr_scheduler", {})
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler_cfg.get("factor", 0.1)),
            patience=int(scheduler_cfg.get("patience", 5)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )
        return optimizer, scheduler

    def step(self, model, loss_fn: UNETNILMLoss, batch) -> StepOutput:
        aggregate, power_target, state_target = batch
        state_logits, power_quantiles = model(aggregate)
        output = loss_fn(state_logits, power_quantiles, power_target, state_target)

        state_probability = torch.softmax(state_logits, dim=1)[:, 1, :]
        predicted_state = state_logits.argmax(dim=1)
        median_power = power_quantiles[:, loss_fn.median_index, :]

        appliance_logs: dict[str, float] = {}
        for appliance_index, appliance in enumerate(self.cfg["appliances"]):
            state_loss = F.cross_entropy(
                state_logits[:, :, appliance_index],
                state_target[:, appliance_index].long(),
            )
            power_loss = loss_fn.power_loss(
                power_quantiles[:, :, appliance_index : appliance_index + 1],
                power_target[:, appliance_index : appliance_index + 1],
            )
            appliance_logs[f"loss_{appliance}"] = float(
                (state_loss + power_loss).detach()
            )

        return StepOutput(
            loss=output.loss,
            logs={
                "loss": float(output.loss.detach()),
                "loss_power": float(output.loss_power.detach()),
                "loss_state": float(output.loss_state.detach()),
                "mae": float(output.mae.detach()),
                **appliance_logs,
            },
            aux={
                "pred_state": predicted_state.detach().cpu(),
                "state_prob": state_probability.detach().cpu(),
                "true_state": state_target.long().detach().cpu(),
                "pred_power": median_power.detach().cpu(),
                "true_power": power_target.detach().cpu(),
            },
        )

    @torch.no_grad()
    def predict_dataloader(
        self,
        model,
        loader: DataLoader,
        device,
        *,
        max_batches=None,
        split="test",
    ):
        model.eval()
        pred_power = []
        pred_state = []
        true_power = []
        true_state = []
        sample_indices = []
        offset = 0
        median_index = self.build_loss().median_index

        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            aggregate, power_target, state_target = batch
            state_logits, power_quantiles = model(aggregate.to(device))

            power = power_quantiles[:, median_index, :].cpu().numpy()
            state_probability = torch.softmax(state_logits, dim=1)[:, 1, :].cpu().numpy()
            batch_size = len(aggregate)

            pred_power.append(power[:, None, :])
            pred_state.append(state_probability[:, None, :])
            true_power.append(power_target.numpy()[:, None, :])
            true_state.append(state_target.numpy()[:, None, :])
            sample_indices.append(self._sample_index(offset, batch_size))
            offset += batch_size

        return self.finalize_prediction_bundle(
            split=split,
            sample_indices=sample_indices,
            pred_power_batches=pred_power,
            pred_state_batches=pred_state,
            true_power_batches=true_power,
            true_state_batches=true_state,
        )

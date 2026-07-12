"""MultiNILM adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from adapters.common import BaseNILMAdapter, StepOutput
from model.MultiNILM import MultiNILM
from model.MultiNILM_loss import MultiNILMLoss


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """NumPy does not support bfloat16; cast to float32 when AMP is enabled."""
    return t.detach().float().cpu().numpy()


def _resolve_pos_weight(adapter: "MultiNILMAdapter", loss_cfg: dict) -> list[float] | None:
    """Use yaml pos_weight, or auto-balance rare ON events from the train split."""
    configured = loss_cfg.get("pos_weight")
    if configured is not None and str(configured).lower() not in {"auto", "null", "none"}:
        return configured
    return adapter._data_loader().estimate_state_pos_weights("train").tolist()


def _pred_on_from_config(
    adapter: "MultiNILMAdapter",
    power_norm: np.ndarray,
    state_prob: np.ndarray,
) -> np.ndarray:
    """Build binary ON/OFF predictions for metrics and saved bundles."""
    source = str(adapter.model_cfg.get("evaluation", {}).get("pred_on_source", "state_head")).lower()
    if source == "state_head":
        return (state_prob >= 0.5).astype(np.int32)

    loader = adapter._data_loader()
    power_watts = loader.denorm_to_watts(power_norm)
    threshold = loader.state_threshold_watts
    if threshold is None:
        raise ValueError("pred_on_source=power_threshold requires threshold training labels")
    threshold = np.asarray(threshold, dtype=np.float32)
    power_on = (power_watts > threshold).astype(np.int32)
    if source == "power_threshold":
        return power_on
    if source == "combined":
        state_on = (state_prob >= 0.5).astype(np.int32)
        return np.maximum(state_on, power_on).astype(np.int32)
    raise ValueError(
        "evaluation.pred_on_source must be one of: state_head, power_threshold, combined"
    )


class MultiNILMAdapter(BaseNILMAdapter):
    # This name is used by main.py, logs, run folders, and saved predictions.
    name = "multinilm"

    def build_model(self, device: torch.device) -> torch.nn.Module:
        # Read architecture settings from config/models/multinilm.yaml.
        arch = self.model_cfg["architecture"]

        # Create the MultiNILM neural network.
        model = MultiNILM(
            # Number of input channels. For normal aggregate power, this is 1.
            input_channels=int(arch.get("input_channels", arch.get("input_size", 1))),

            # Number of output appliances comes from the experiment CSV config.
            # REDD = 4 appliances, UK-DALE = 5 appliances.
            num_appliances=len(self.cfg["appliances"]),

            # Number of output timesteps predicted by the model.
            # Example: 64 for REDD MATNILM-style center output.
            output_length=int(self.model_cfg["windowing"].get("output_window_length", 1)),

            # CNN hidden feature channels (final width for TCN + heads).
            hidden_channels=int(arch.get("hidden_channels", arch.get("hidden", 64))),

            # Gradual widening before TCN, e.g. [16, 32, 64].
            channel_schedule=arch.get("channel_schedule"),

            # First staged conv kernel (default 7); later stages use stage_kernel_size.
            stem_kernel_size=int(arch.get("stem_kernel_size", 7)),
            stage_kernel_size=int(arch.get("stage_kernel_size", 5)),

            # Number of residual temporal convolution blocks.
            num_blocks=int(arch.get("num_blocks", 5)),

            # Temporal convolution kernel size.
            kernel_size=int(arch.get("kernel_size", 5)),

            # Dropout rate inside temporal blocks.
            dropout=float(arch.get("dropout", 0.1)),
        )

        # Move model to GPU if available, otherwise CPU.
        return model.to(device)

    def build_loss(self) -> MultiNILMLoss:
        # Read loss settings from config/models/multinilm.yaml.
        loss_cfg = self.model_cfg.get("loss", {})

        # Create loss:
        #   total_loss = MSE(power) + lambda_state * BCEWithLogits(state)
        return MultiNILMLoss(
            # Weight of ON/OFF classification loss.
            lambda_state=float(loss_cfg.get("lambda_state", 0.1)),

            # Auto-balance rare ON timesteps when pos_weight is null/auto.
            pos_weight=_resolve_pos_weight(self, loss_cfg),

            # Used only for MAE logging. This comes from dataset normalization
            # stats when available, otherwise from legacy scalar power_scale.
            power_scale=self._data_loader().loss_scale,
        )

    def step(
        self,
        model: torch.nn.Module,
        loss_fn: MultiNILMLoss,
        batch: Any,
    ) -> StepOutput:
        """Run one batch through MultiNILM and return loss/logs.

        This method is used by runner.py for both:

            training   -> runner enables gradients and calls backward()
            validation -> runner disables gradients and only records logs

        This method only handles model-specific forward and loss logic.
        """
        # batch comes from WindowDataset.__getitem__:
        #   x = aggregate input window
        #   y = true appliance power
        #   z = true appliance ON/OFF label
        #
        # Shapes:
        #   x: (B, input_length, 1)
        #   y: (B, output_length, appliances)
        #   z: (B, output_length, appliances)
        x, y, z = batch

        # The dataloader now owns normalization and optional state rebuilding,
        # so adapters receive model-ready tensors and only run model logic.
        z = z.float()

        # Forward pass through MultiNILM.
        #
        # power_pred:
        #   predicted appliance power, shape (B, output_length, appliances)
        #
        # state_logits:
        #   raw ON/OFF logits, shape (B, output_length, appliances)
        #   sigmoid is NOT applied before BCEWithLogitsLoss.
        power_pred, state_logits = model(x)

        # Compute paper-style multitask loss:
        #   L = sum_i ( L_power^i + lambda_state * L_state^i )
        out = loss_fn(power_pred, state_logits, y, z)

        # Convert logits to probabilities only for metric/F1 logging.
        # This is not used in the loss.
        state_prob = torch.sigmoid(state_logits)

        # Binary ON/OFF prediction for metrics (configurable scoring source).
        pred_state = torch.from_numpy(
            _pred_on_from_config(
                self,
                _to_numpy(power_pred),
                _to_numpy(state_prob),
            )
        ).long()

        app_logs = {}
        for app_i, app in enumerate(self.cfg["appliances"]):
            app_logs[f"loss_power_{app}"] = float(out.loss_power_per_appliance[app_i].detach())
            app_logs[f"loss_state_{app}"] = float(out.loss_state_per_appliance[app_i].detach())

        # StepOutput is what runner.py expects.
        # runner.py uses:
        #   step.loss for backward()
        #   step.logs for training/validation logs
        #   step.aux for F1 calculation
        return StepOutput(
            loss=out.loss,
            logs={
                "loss": float(out.loss.detach()),
                "loss_power": float(out.loss_power.detach()),
                "loss_state": float(out.loss_state.detach()),
                "mae": float(out.mae.detach()),
                **app_logs,
            },
            # Move state tensors to CPU because runner.py collects them across batches.
            aux={
                "pred_state": pred_state.detach().cpu(),
                "true_state": z.long().detach().cpu(),
                "pred_power": power_pred.detach().float().cpu(),
                "true_power": y.detach().cpu(),
            },
        )

    @torch.no_grad()
    def predict_dataloader(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        *,
        max_batches: int | None = None,
        split: str = "test",
    ) -> Any:
        # Evaluation/inference mode.
        # This disables dropout behavior and uses running BatchNorm statistics.
        model.eval()

        # Lists collect batch predictions before concatenating into one array.
        pred_power, pred_state = [], []
        true_power, true_state = [], []
        sample_indices = []
        offset = 0

        for batch_idx, batch in enumerate(loader):
            # Optional limit for quick debugging.
            if max_batches is not None and batch_idx >= max_batches:
                break

            # x/y/z: normalized mains input, appliance power targets, ON/OFF labels
            x, y, z = batch

            # Move already-normalized aggregate input to GPU/CPU.
            x = x.to(device)

            # Predict appliance power and state logits.
            power_pred, state_logits = model(x)

            # Convert logits to probabilities for ON/OFF prediction.
            state_prob = torch.sigmoid(state_logits)

            pred_power.append(_to_numpy(power_pred))

            pred_state.append(
                _pred_on_from_config(
                    self,
                    _to_numpy(power_pred),
                    _to_numpy(state_prob),
                )
            )

            # y and z are ground truth from dataloader.
            # They are already on CPU because we did not move them to device.
            true_power.append(y.numpy())
            true_state.append(z.numpy())

            # Track window indices so prediction files can align with waveform plots.
            sample_indices.append(self._sample_index(offset, len(x)))
            offset += len(x)

        return self.finalize_prediction_bundle(
            split=split,
            sample_indices=sample_indices,
            pred_power_batches=pred_power,
            pred_state_batches=pred_state,
            true_power_batches=true_power,
            true_state_batches=true_state,
        )

"""Shared adapter helpers.

Full pipeline:

    CSV rows
    -> NILMDataLoader reads mains / target / state columns
    -> NILMDataLoader applies normalization and windowing
    -> adapter.step() defines model-specific forward/loss logic
    -> adapter.predict_dataloader() collects outputs over a split
    -> finalize_prediction_bundle(...) converts batched outputs into one
       PredictionBundle
    -> runner/evaluation compute metrics and save plots from that bundle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from adapters.dataloader import NILMDataLoader, _resolve_input_length, _split_key


@dataclass
class PredictionBundle:
    """Standard prediction object saved by every model.

    Where it is used:

        adapter.predict_dataloader(...)
            -> builds a PredictionBundle
            -> runner/evaluation saves it as .npz
            -> evaluation/metrics.py reads it to calculate MAE, SAE, F1
            -> evaluation/compare.py reads it to compare models

    Shape convention:

        y_true_watts : (N, A)
        y_pred_watts : (N, A)
        y_true_on    : (N, A)
        y_pred_on    : (N, A)

    where:

        N = flattened output timesteps after window reconstruction
        A = number of appliances
    """

    experiment_id: str
    model_name: str
    split: str
    appliances: list[str]
    sample_index: np.ndarray
    y_true_watts: np.ndarray
    y_pred_watts: np.ndarray
    y_true_on: np.ndarray | None = None
    y_pred_on: np.ndarray | None = None
    csv_timesteps: np.ndarray | None = None

    def save(self, path: Path) -> None:
        """Save predictions to a compressed .npz file.

        We save empty arrays for optional fields when they are missing because
        np.savez does not preserve Python None cleanly.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            experiment_id=self.experiment_id,
            model_name=self.model_name,
            split=self.split,
            appliances=np.array(self.appliances),
            sample_index=self.sample_index,
            y_true_watts=self.y_true_watts,
            y_pred_watts=self.y_pred_watts,
            y_true_on=self.y_true_on if self.y_true_on is not None else np.array([]),
            y_pred_on=self.y_pred_on if self.y_pred_on is not None else np.array([]),
            csv_timesteps=self.csv_timesteps if self.csv_timesteps is not None else np.array([]),
        )

    @classmethod
    def load(cls, path: Path) -> "PredictionBundle":
        """Load a PredictionBundle saved by save(...)."""
        data = np.load(path, allow_pickle=True)
        appliances = data["appliances"].tolist()
        if isinstance(appliances, np.ndarray):
            appliances = appliances.tolist()
        y_true_on = data["y_true_on"]
        y_pred_on = data["y_pred_on"]
        csv_ts = data["csv_timesteps"] if "csv_timesteps" in data else np.array([])
        return cls(
            experiment_id=str(data["experiment_id"]),
            model_name=str(data["model_name"]),
            split=str(data["split"]),
            appliances=list(appliances),
            sample_index=data["sample_index"],
            y_true_watts=data["y_true_watts"],
            y_pred_watts=data["y_pred_watts"],
            y_true_on=None if y_true_on.size == 0 else y_true_on,
            y_pred_on=None if y_pred_on.size == 0 else y_pred_on,
            csv_timesteps=None if csv_ts.size == 0 else csv_ts,
        )


@dataclass
class StepOutput:
    """Standard output from one adapter training/evaluation step.

    Where it is used:

        runner.py
            -> calls adapter.step(...)
            -> receives StepOutput
            -> uses output.loss for backpropagation/checkpoint selection
            -> logs output.logs to console/history.csv/live plots

    Fields:

        loss : tensor used for backward() during training
        logs : simple float values, such as loss_power/loss_state/mae
        aux  : optional extra objects for a specific model
    """

    loss: object
    logs: dict[str, float] = field(default_factory=dict)
    aux: dict[str, object] = field(default_factory=dict)


def center_output_slice(windowing: dict[str, Any]) -> slice:
    """Return the center target region inside a longer input window.

    Example for MATNILM/REDD paper setting:

        input length  = 864
        output length = 64

    The model sees 864 samples but predicts/evaluates only the center 64
    samples. This helper returns the Python slice for that center region.
    """
    seq_len = _resolve_input_length(windowing)
    out_len = int(windowing.get("output_window_length", 1))
    start = (seq_len - out_len) // 2
    return slice(start, start + out_len)


def build_dataloader(
    dataset: Dataset,
    train_cfg: dict[str, Any],
    *,
    shuffle: bool,
) -> DataLoader:
    """Build the PyTorch DataLoader used by train/validation/test loops.

    Where it is used:

        AdapterDataMixin.build_standard_dataloader(split)
            -> runner.py

    The model-specific adapter does not need to know how many workers,
    prefetching, or pin_memory are used. It just asks this helper for a loader.
    """
    num_workers = int(train_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        # Batch size comes from config/models/*.yaml.
        "batch_size": int(train_cfg["batch_size"]),

        # Training windows are shuffled. Validation/test windows are not.
        "shuffle": shuffle,

        # num_workers > 0 uses separate worker processes for data loading.
        "num_workers": num_workers,

        # pin_memory helps GPU transfer when CUDA is available.
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        # Keep workers alive across epochs to reduce startup overhead.
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))

        # Number of batches each worker prepares ahead of time.
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
    return DataLoader(dataset, **kwargs)


class AdapterDataMixin:
    """Shared data-loading methods inherited by model adapters.

    Each adapter class has:

        self.experiment  -> experiment YAML dictionary
        self.model_cfg   -> model YAML dictionary
        self.data_root   -> optional root folder

    This mixin uses those fields to build datasets/loaders without duplicating
    the same code inside every adapter.
    """

    cfg: dict[str, Any]
    experiment: dict[str, Any]
    model_cfg: dict[str, Any]
    data_root: str | None
    _data: NILMDataLoader | None = None

    def _data_loader(self) -> NILMDataLoader:
        """Create NILMDataLoader only once, then reuse it.

        NILMDataLoader is the object that actually reads CSV files and creates
        sliding-window datasets.
        """
        if self._data is None:
            self._data = NILMDataLoader(self.experiment, self.model_cfg, self.data_root)
        return self._data

    def build_dataset(self, split: str) -> Dataset:
        """Build one split dataset: train, validation, or test."""
        return self._data_loader().build_dataset(split)

    def build_standard_dataloader(self, split: str) -> DataLoader:
        """Build one split DataLoader using the shared DataLoader settings."""
        return build_dataloader(
            self.build_dataset(split),
            self.model_cfg["training"],
            shuffle=(split == "train"),
        )


class BaseNILMAdapter(AdapterDataMixin):
    """Base adapter for all NILM models.

    This class contains the boring shared parts that should not be repeated in
    every model adapter:

        - store merged config
        - store experiment/model config shortcuts
        - build train/validation/test DataLoaders
        - create the default Adam optimizer

    Model-specific adapter files should only implement the parts that differ:

        - build_model(...)
        - build_loss(...)
        - step(...)
        - predict_dataloader(...) if the model output format is special

    The selected model is still controlled by hyperparameter/config:

        python main.py --model multinilm --model-config config/models/multinilm.yaml
        python main.py --model mat_nilm --model-config config/models/mat_nilm.yaml
    """

    name = "base"

    def __init__(self, merged_cfg: dict[str, Any], data_root: str | None = None):
        # Full merged config = experiment YAML + model YAML.
        self.cfg = merged_cfg

        # Dataset side: CSV paths, column names, metrics settings.
        self.experiment = merged_cfg["experiment"]

        # Model side: windowing, architecture, loss, training settings.
        self.model_cfg = merged_cfg["model"]

        # Optional root folder for dataset files.
        self.data_root = data_root or merged_cfg.get("data_root")

        # Lazy-loaded NILMDataLoader. It is created only when a split is needed.
        self._data = None

    def build_dataloader(self, split: str) -> DataLoader:
        """Build train/validation/test DataLoader using common settings."""
        return self.build_standard_dataloader(split)

    def configure_optimizer(self, model: torch.nn.Module):
        """Default optimizer controlled by config/models/*.yaml.

        If a future model needs a special optimizer or scheduler, that adapter
        can override this method.
        """
        train_cfg = self.model_cfg["training"]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        return optimizer, None

    def _prediction_context(self, split: str) -> tuple[list[str], str]:
        """Return the appliance order and normalized split name used by evaluation."""
        return self.cfg["appliances"], _split_key(split)

    def _sample_index(self, offset: int, batch_size: int) -> np.ndarray:
        """Create monotonically increasing sample ids for one prediction batch."""
        return np.arange(offset, offset + batch_size)

    def finalize_prediction_bundle(
        self,
        *,
        split: str,
        sample_indices: list[np.ndarray],
        pred_power_batches: list[np.ndarray],
        pred_state_batches: list[np.ndarray],
        true_power_batches: list[np.ndarray],
        true_state_batches: list[np.ndarray],
    ) -> PredictionBundle:
        """Assemble batched inference outputs into the shared prediction format."""
        # Step 1:
        # Recover the appliance order and canonical split name used by evaluation.
        appliances, split_key = self._prediction_context(split)

        # Step 2:
        # Concatenate per-batch arrays into one long (N, A) timeline.
        y_pred = np.concatenate(pred_power_batches, axis=0).reshape(-1, len(appliances))
        y_true = np.concatenate(true_power_batches, axis=0).reshape(-1, len(appliances))
        z_pred = np.concatenate(pred_state_batches, axis=0).reshape(-1, len(appliances))
        z_true = np.concatenate(true_state_batches, axis=0).reshape(-1, len(appliances))

        # Step 3:
        # Convert predictions and targets from normalized space back to watts.
        loader = self._data_loader()
        y_pred = loader.denorm_to_watts(y_pred)
        y_true = loader.denorm_to_watts(y_true)

        # Step 4:
        # Reconstruct which original CSV timestep each flattened output belongs to.
        csv_timesteps = loader.window_output_timesteps(split_key, len(y_true))

        # Step 5:
        # Return the standard prediction object used everywhere else in the repo.
        return build_prediction_bundle(
            experiment_id=self.experiment["experiment_id"],
            model_name=self.name,
            split=split,
            appliances=appliances,
            sample_index=np.concatenate(sample_indices),
            y_true_watts=y_true,
            y_pred_watts=y_pred,
            y_true_on=z_true,
            y_pred_on=z_pred,
            csv_timesteps=csv_timesteps,
        )


def build_prediction_bundle(
    *,
    experiment_id: str,
    model_name: str,
    split: str,
    appliances: list[str],
    sample_index: np.ndarray,
    y_true_watts: np.ndarray,
    y_pred_watts: np.ndarray,
    y_true_on: np.ndarray,
    y_pred_on: np.ndarray,
    csv_timesteps: np.ndarray | None = None,
) -> PredictionBundle:
    """Create a PredictionBundle with consistent dtype handling.

    Adapters call this at the end of inference. Keeping this constructor here
    makes sure every model saves predictions in the same format, so the same
    metrics/plotting code can be reused.
    """
    return PredictionBundle(
        experiment_id=experiment_id,
        model_name=model_name,
        split=split,
        appliances=appliances,
        sample_index=sample_index,
        y_true_watts=y_true_watts,
        y_pred_watts=y_pred_watts,
        y_true_on=y_true_on.astype(np.int32),
        y_pred_on=y_pred_on.astype(np.int32),
        csv_timesteps=csv_timesteps,
    )

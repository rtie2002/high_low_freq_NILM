"""Live PNG updates in the run folder during training."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.plots import (
    plot_loss_components,
    plot_training_history,
    save_appliance_on_waveforms,
)


class LiveTrainingMonitor:
    def __init__(
        self,
        run_dir: Path,
        *,
        model_name: str,
        appliances: list[str],
        plot_cfg: dict[str, Any],
        seed: int = 0,
    ):
        self.run_dir = Path(run_dir)
        self.model_name = model_name
        self.appliances = list(appliances)
        self.plot_cfg = plot_cfg
        self.seed = int(seed)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.waveforms_dir = self.run_dir / "waveforms"

        self.history_path = self.run_dir / "history.csv"
        self.loss_detail_path = self.run_dir / "loss_detail.csv"
        self.live_history_png = self.run_dir / "live_training_loss.png"
        self.live_loss_png = self.run_dir / "live_loss_components.png"

        self._history_file: TextIO | None = None
        self._loss_file: TextIO | None = None
        self._history_writer: csv.DictWriter | None = None
        self._loss_writer: csv.DictWriter | None = None

    def plot_mode(self) -> str:
        return str(self.plot_cfg.get("plot_mode", "live")).lower()

    def plot_interval(self) -> int:
        return max(1, int(self.plot_cfg.get("plot_interval", 1)))

    def plot_on_periods(self) -> int:
        return int(self.plot_cfg.get("plot_on_periods", 5))

    def on_period_samples(self) -> int:
        return int(self.plot_cfg.get("on_period_samples", 400))

    def waveform_dpi(self) -> int:
        return int(self.plot_cfg.get("waveform_dpi", 300))

    def plot_max_batches(self) -> int | None:
        value = self.plot_cfg.get("plot_max_batches")
        return None if value is None else int(value)

    def should_plot(self, epoch_no: int) -> bool:
        mode = self.plot_mode()
        if mode == "end":
            return False
        if mode == "interval":
            return epoch_no % self.plot_interval() == 0
        return True

    def append_epoch(self, *, epoch: int, train_logs: dict[str, float], val_logs: dict[str, float]) -> None:
        history_row = {
            "epoch": epoch,
            "train_loss": train_logs.get("loss", float("nan")),
            "val_loss": val_logs.get("loss", float("nan")),
            "train_mae": train_logs.get("mae", float("nan")),
            "val_mae": val_logs.get("mae", float("nan")),
        }
        loss_row = {
            "epoch": epoch,
            "train_loss": train_logs.get("loss", float("nan")),
            "val_loss": val_logs.get("loss", float("nan")),
            "train_loss_state": train_logs.get("loss_state", float("nan")),
            "val_loss_state": val_logs.get("loss_state", float("nan")),
            "train_loss_power": train_logs.get("loss_power", float("nan")),
            "val_loss_power": val_logs.get("loss_power", float("nan")),
        }
        self._write_csv_row(history_row, self.history_path, "_history_file", "_history_writer")
        self._write_csv_row(loss_row, self.loss_detail_path, "_loss_file", "_loss_writer")

    def _write_csv_row(
        self,
        row: dict[str, Any],
        path: Path,
        file_attr: str,
        writer_attr: str,
    ) -> None:
        writer = getattr(self, writer_attr)
        if writer is None:
            handle = path.open("w", newline="", encoding="utf-8")
            setattr(self, file_attr, handle)
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            setattr(self, writer_attr, writer)
        writer.writerow(row)
        getattr(self, file_attr).flush()

    def save_loss_plots(self, *, epoch: int, best_epoch: int | None = None) -> None:
        if not self.history_path.exists():
            return
        plot_training_history(
            self.history_path,
            self.live_history_png,
            title=f"{self.model_name} training (epoch {epoch})",
            best_epoch=best_epoch,
        )
        if self.loss_detail_path.exists():
            plot_loss_components(
                self.loss_detail_path,
                self.live_loss_png,
                title=f"{self.model_name} loss components",
            )

    @torch.no_grad()
    def save_waveforms(
        self,
        adapter,
        model: torch.nn.Module,
        *,
        val_loader: DataLoader,
        test_loader: DataLoader | None,
        device: torch.device,
        epoch: int,
        include_best: bool = False,
    ) -> list[Path]:
        tag = "best" if include_best else "live"
        saved: list[Path] = []
        saved.extend(
            self._save_split_waveforms(
                adapter, model, val_loader, device, split="validation", epoch=epoch, tag=tag
            )
        )
        if test_loader is not None:
            saved.extend(
                self._save_split_waveforms(
                    adapter, model, test_loader, device, split="test", epoch=epoch, tag=tag
                )
            )
        return saved

    def _save_split_waveforms(
        self,
        adapter,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        *,
        split: str,
        epoch: int,
        tag: str,
    ) -> list[Path]:
        bundle = adapter.predict_dataloader(
            model,
            loader,
            device,
            max_batches=self.plot_max_batches(),
        )
        aggregate = self._split_mains(adapter, split, len(bundle.y_true_watts))
        output_dir = self.waveforms_dir / split / tag / f"epoch_{epoch:03d}"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        split_id = 0 if split == "validation" else 1
        rng = np.random.default_rng(self.seed + epoch * 1009 + split_id)
        prefix = "best" if tag == "best" else f"epoch_{epoch:03d}"
        return save_appliance_on_waveforms(
            output_dir,
            appliances=self.appliances,
            y_true_watts=bundle.y_true_watts,
            y_pred_watts=bundle.y_pred_watts,
            y_true_on=bundle.y_true_on,
            y_pred_on=bundle.y_pred_on,
            aggregate=aggregate,
            n_periods=self.plot_on_periods(),
            period_samples=self.on_period_samples(),
            dpi=self.waveform_dpi(),
            rng=rng,
            file_prefix=prefix,
            title_prefix=f"{self.model_name} {split} epoch {epoch} — ",
        )

    def _split_mains(self, adapter, split: str, n_points: int) -> np.ndarray | None:
        try:
            key = "validation" if split == "validation" else "test"
            data_loader = adapter._data_loader()
            x, _, _ = data_loader.get_splits()[key]
            windowing = data_loader.model_cfg["windowing"]
            seq_len = int(windowing["input_window_length"])
            if windowing.get("force_even_input_length", False) and seq_len % 2 != 0:
                seq_len += 1
            offset = seq_len - 1
            end = min(offset + n_points, len(x))
            if end <= offset:
                return None
            return x[offset:end].astype(np.float32)
        except Exception:
            return None

    def close(self) -> None:
        if self._history_file is not None:
            self._history_file.close()
        if self._loss_file is not None:
            self._loss_file.close()

    def finalize(self, *, best_epoch: int) -> None:
        plot_training_history(
            self.history_path,
            self.run_dir / "training_loss.png",
            title=f"{self.model_name} training",
            best_epoch=best_epoch,
        )
        if self.loss_detail_path.exists():
            plot_loss_components(
                self.loss_detail_path,
                self.run_dir / "loss_components.png",
                title=f"{self.model_name} loss components",
            )

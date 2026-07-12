"""Live PNG updates in the run folder during training."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.feature_maps import FeatureMapConfig, save_feature_maps
from evaluation.plots import (
    FULL_CYCLE_APPLIANCES,
    plot_loss_components,
    plot_matnilm_training_losses,
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
        if "plot_interval" in self.plot_cfg:
            return max(1, int(self.plot_cfg["plot_interval"]))
        return max(1, int(self.plot_cfg.get("every_n_epochs", 1)))

    def should_plot(self, epoch_no: int) -> bool:
        if self.plot_cfg.get("enabled") is False:
            return False
        mode = self.plot_mode()
        if mode == "end":
            return False
        if mode == "interval":
            return epoch_no % self.plot_interval() == 0
        return True

    def plot_on_periods(self) -> int:
        return int(self.plot_cfg.get("plot_on_periods", 5))

    def on_period_samples(self) -> int | None:
        raw = self.plot_cfg.get("on_period_samples", 0)
        if raw is None:
            return None
        val = int(raw)
        return None if val <= 0 else val

    def full_cycle_appliances(self) -> list[str]:
        raw = self.plot_cfg.get("full_cycle_appliances")
        if raw is None:
            return list(FULL_CYCLE_APPLIANCES)
        return list(raw)

    def waveform_dynamic_figsize(self) -> bool:
        return bool(self.plot_cfg.get("waveform_dynamic_figsize", True))

    def on_period_margin_min(self) -> int:
        return int(self.plot_cfg.get("on_period_margin_min", 40))

    def on_period_margin_frac(self) -> float:
        return float(self.plot_cfg.get("on_period_margin_frac", 0.08))

    def waveform_figsize(self) -> float:
        return float(self.plot_cfg.get("waveform_figsize", 5.5))

    def waveform_dpi(self) -> int:
        return int(self.plot_cfg.get("waveform_dpi", 300))

    def plot_max_batches(self) -> int | None:
        value = self.plot_cfg.get("plot_max_batches")
        return None if value is None else int(value)

    def append_epoch(self, *, epoch: int, train_logs: dict[str, float], val_logs: dict[str, float]) -> None:
        train_time_sec = float(train_logs.get("elapsed_sec", float("nan")))
        val_time_sec = float(val_logs.get("elapsed_sec", float("nan")))
        epoch_time_sec = (
            train_time_sec + val_time_sec
            if np.isfinite(train_time_sec) and np.isfinite(val_time_sec)
            else float("nan")
        )
        history_row = {
            "epoch": epoch,
            "train_loss": train_logs.get("loss", float("nan")),
            "val_loss": val_logs.get("loss", float("nan")),
            "train_mae": train_logs.get("mae", float("nan")),
            "val_mae": val_logs.get("mae", float("nan")),
            "val_f1": val_logs.get("val_f1", float("nan")),
            "val_acc": val_logs.get("val_acc", float("nan")),
            "val_mif1": val_logs.get("val_mif1", float("nan")),
            "val_miacc": val_logs.get("val_miacc", float("nan")),
            "train_time_sec": train_time_sec,
            "val_time_sec": val_time_sec,
            "epoch_time_sec": epoch_time_sec,
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
        for key, value in train_logs.items():
            if key.startswith("loss_") and key not in ("loss_state", "loss_power"):
                loss_row[f"train_{key}"] = value
        for key, value in val_logs.items():
            if key.startswith("loss_") and key not in ("loss_state", "loss_power"):
                loss_row[f"val_{key}"] = value
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
            path.parent.mkdir(parents=True, exist_ok=True)
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
        figsize = self.waveform_figsize()
        plot_training_history(
            self.history_path,
            self.live_history_png,
            title=f"{self.model_name} training (epoch {epoch})",
            best_epoch=best_epoch,
            figsize=figsize,
        )
        if self.loss_detail_path.exists():
            plot_loss_components(
                self.loss_detail_path,
                self.live_loss_png,
                title=f"{self.model_name} loss components",
                figsize=figsize,
            )
            if self.model_name == "mat_nilm":
                plot_matnilm_training_losses(
                    self.loss_detail_path,
                    self.run_dir / "live_matnilm_training_losses.png",
                    appliances=self.appliances,
                    figsize=figsize,
                )

    def feature_map_cfg(self) -> FeatureMapConfig:
        return FeatureMapConfig.from_dict(self.plot_cfg.get("feature_maps"))

    def should_plot_feature_maps(self) -> bool:
        return self.feature_map_cfg().enabled

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
        self._prune_legacy_epoch_waveform_dirs()
        saved: list[Path] = []
        if not include_best:
            saved.extend(
                self._save_split_waveforms(
                    adapter, model, val_loader, device, split="validation", epoch=epoch, tag="latest"
                )
            )
            if test_loader is not None:
                saved.extend(
                    self._save_split_waveforms(
                        adapter, model, test_loader, device, split="test", epoch=epoch, tag="latest"
                    )
                )
            if self.should_plot_feature_maps():
                saved.extend(
                    self._save_feature_maps(
                        adapter, model, val_loader, device, split="validation", epoch=epoch, tag="latest"
                    )
                )
                if test_loader is not None:
                    saved.extend(
                        self._save_feature_maps(
                            adapter, model, test_loader, device, split="test", epoch=epoch, tag="latest"
                        )
                    )
        else:
            saved.extend(
                self._save_split_waveforms(
                    adapter, model, val_loader, device, split="validation", epoch=epoch, tag="best"
                )
            )
            if test_loader is not None:
                saved.extend(
                    self._save_split_waveforms(
                        adapter, model, test_loader, device, split="test", epoch=epoch, tag="best"
                    )
                )
            if self.should_plot_feature_maps():
                saved.extend(
                    self._save_feature_maps(
                        adapter, model, val_loader, device, split="validation", epoch=epoch, tag="best"
                    )
                )
                if test_loader is not None:
                    saved.extend(
                        self._save_feature_maps(
                            adapter, model, test_loader, device, split="test", epoch=epoch, tag="best"
                        )
                    )
        return saved

    def _save_feature_maps(
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
        cfg = self.feature_map_cfg()
        if not cfg.enabled:
            return []
        output_dir = self.run_dir / "feature_maps" / split / tag / f"epoch_{epoch:04d}"
        return save_feature_maps(
            adapter,
            model,
            loader,
            output_dir,
            split=split,
            device=device,
            cfg=cfg,
        )

    def _waveform_tag_dir(self, split: str, tag: str) -> Path:
        return self.waveforms_dir / split / tag

    def _prune_legacy_epoch_waveform_dirs(self) -> None:
        """Remove old live/epoch_XXX and best/epoch_XXX folders from earlier runs."""
        if not self.waveforms_dir.exists():
            return
        for split_dir in self.waveforms_dir.iterdir():
            if not split_dir.is_dir():
                continue
            for tag_dir in split_dir.iterdir():
                if not tag_dir.is_dir():
                    continue
                if tag_dir.name == "live":
                    shutil.rmtree(tag_dir, ignore_errors=True)
                    continue
                for child in tag_dir.iterdir():
                    if child.is_dir() and child.name.startswith("epoch_"):
                        shutil.rmtree(child, ignore_errors=True)

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
            split=split,
            max_batches=self.plot_max_batches(),
        )
        aggregate = self._split_mains(adapter, split, bundle)
        output_dir = self._waveform_tag_dir(split, tag)
        if output_dir.exists():
            shutil.rmtree(output_dir)

        split_id = 0 if split == "validation" else 1
        rng = np.random.default_rng(self.seed + epoch * 1009 + split_id)
        # Waveform plots always use dataset CSV *_on labels for true ON periods.
        waveform_true_on = adapter._data_loader().csv_on_labels_at_timesteps(
            split,
            bundle.csv_timesteps[: len(bundle.y_true_watts)],
        ) if bundle.csv_timesteps is not None and len(bundle.csv_timesteps) >= len(bundle.y_true_watts) else adapter._data_loader().window_flattened_csv_states(
            split,
            len(bundle.y_true_watts),
        )
        return save_appliance_on_waveforms(
            output_dir,
            appliances=self.appliances,
            y_true_watts=bundle.y_true_watts,
            y_pred_watts=bundle.y_pred_watts,
            y_true_on=waveform_true_on,
            y_pred_on=bundle.y_pred_on,
            aggregate=aggregate,
            csv_timesteps=bundle.csv_timesteps,
            n_periods=self.plot_on_periods(),
            period_samples=self.on_period_samples(),
            full_cycle_appliances=self.full_cycle_appliances(),
            margin_min=self.on_period_margin_min(),
            margin_frac=self.on_period_margin_frac(),
            figsize=self.waveform_figsize(),
            dynamic_figsize=self.waveform_dynamic_figsize(),
            dpi=self.waveform_dpi(),
            rng=rng,
            file_prefix=tag,
            title_prefix=f"{self.model_name} {split} {tag} epoch {epoch} — ",
        )

    def _split_mains(self, adapter, split: str, bundle) -> np.ndarray | None:
        try:
            n_points = len(bundle.y_true_watts)
            data_loader = adapter._data_loader()
            key = "validation" if split == "validation" else "test"
            if bundle.csv_timesteps is not None and len(bundle.csv_timesteps) >= n_points:
                raw_x, _, _ = data_loader.get_raw_csv_arrays(key)
                return raw_x[bundle.csv_timesteps[:n_points]].astype(np.float32)
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
        figsize = self.waveform_figsize()
        plot_training_history(
            self.history_path,
            self.run_dir / "training_loss.png",
            title=f"{self.model_name} training",
            best_epoch=best_epoch,
            figsize=figsize,
        )
        if self.loss_detail_path.exists():
            plot_loss_components(
                self.loss_detail_path,
                self.run_dir / "loss_components.png",
                title=f"{self.model_name} loss components",
                figsize=figsize,
            )
            if self.model_name == "mat_nilm":
                plot_matnilm_training_losses(
                    self.loss_detail_path,
                    self.run_dir / "matnilm_training_losses.png",
                    appliances=self.appliances,
                    figsize=figsize,
                )

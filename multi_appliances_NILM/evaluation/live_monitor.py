"""Live PNG updates in the run folder during training."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from adapters.dataloader import get_state_label_source, resolve_state_thresholds_watts
from evaluation.feature_maps import FeatureMapConfig, save_feature_maps
from evaluation.metrics import evaluate_bundle
from evaluation.plots import (
    FULL_CYCLE_APPLIANCES,
    bundle_aggregate_watts,
    bundle_csv_appliance_watts,
    dataset_on_labels_for_bundle,
    plot_loss_components,
    plot_matnilm_training_losses,
    plot_training_history,
    plot_validation_metrics,
    save_appliance_on_waveforms,
    save_multi_epoch_metrics_collage,
    save_multi_epoch_waveform_collages,
    save_val_test_comparison_figure,
)
from evaluation.power_postprocess import resolve_power_postprocess


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
        self.live_metrics_png = self.run_dir / "live_validation_metrics.png"

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

    def waveform_context_scale(self) -> float:
        # >1 saves a long ×N context strip beside each focused ON plot; <=1 disables.
        raw = self.plot_cfg.get("waveform_context_scale", 10)
        if raw is None:
            return 0.0
        return float(raw)

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
        val_mae = float(val_logs.get("mae", float("nan")))
        history_row = {
            "epoch": epoch,
            "train_loss": train_logs.get("loss", float("nan")),
            "val_loss": val_logs.get("loss", float("nan")),
            "train_mae": train_logs.get("mae", float("nan")),
            "val_mae": val_mae,
            "val_mae_norm": float(val_logs.get("mae_norm", float("nan"))),
            "val_mae_watts": float(val_logs.get("mae_watts_epoch", val_mae)),
            "val_f1": val_logs.get("val_f1", float("nan")),
            "val_acc": val_logs.get("val_acc", float("nan")),
            "val_mif1": val_logs.get("val_mif1", float("nan")),
            "val_miacc": val_logs.get("val_miacc", float("nan")),
            "train_time_sec": train_time_sec,
            "val_time_sec": val_time_sec,
            "epoch_time_sec": epoch_time_sec,
        }

        # Same-scale NILM objective: power + balanced state. No domain.
        def _nilm_objective(logs: dict[str, float]) -> float:
            p = float(logs.get("loss_power", float("nan")))
            s = float(logs.get("loss_state_term", logs.get("loss_state", float("nan"))))
            if p != p or s != s:
                return float("nan")
            return p + s

        train_nilm = _nilm_objective(train_logs)
        val_nilm = _nilm_objective(val_logs)
        history_row["train_loss_nilm"] = train_nilm
        history_row["val_loss_nilm"] = val_nilm

        loss_row = {
            "epoch": epoch,
            "train_loss": train_logs.get("loss", float("nan")),
            "val_loss": val_logs.get("loss", float("nan")),
            "train_loss_nilm": train_nilm,
            "val_loss_nilm": val_nilm,
            "train_loss_state": train_logs.get("loss_state", float("nan")),
            "val_loss_state": val_logs.get("loss_state", float("nan")),
            "train_loss_power": train_logs.get("loss_power", float("nan")),
            "val_loss_power": val_logs.get("loss_power", float("nan")),
            "train_loss_state_term": train_logs.get("loss_state_term", float("nan")),
            "val_loss_state_term": val_logs.get("loss_state_term", float("nan")),
            "train_loss_domain": train_logs.get("loss_domain", float("nan")),
        }
        for key, value in train_logs.items():
            if key.startswith("loss_") and key not in (
                "loss_state",
                "loss_power",
                "loss_state_term",
                "loss_domain",
            ):
                loss_row[f"train_{key}"] = value
        for key, value in val_logs.items():
            if key.startswith("loss_") and key not in (
                "loss_state",
                "loss_power",
                "loss_state_term",
            ):
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
        plot_validation_metrics(
            self.history_path,
            self.live_metrics_png,
            title=f"{self.model_name} validation metrics (epoch {epoch})",
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
        self._prune_legacy_epoch_feature_map_dirs()
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
                fig_path = self._save_epoch_val_test_comparison_figure(epoch)
                if fig_path is not None:
                    saved.append(fig_path)
                # One-picture dashboards so you don't jump epoch folders.
                saved.extend(self._save_epoch_comparison_dashboards(epoch))
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

    def _feature_map_tag_dir(self, split: str, tag: str) -> Path:
        return self.run_dir / "feature_maps" / split / tag

    def _epoch_tag(self, epoch: int) -> str:
        return f"epoch_{int(epoch):04d}"

    def _prune_legacy_epoch_feature_map_dirs(self) -> None:
        """Remove obsolete ``live/`` feature-map folders only (keep epoch_* history)."""
        feature_root = self.run_dir / "feature_maps"
        if not feature_root.exists():
            return
        for split_dir in feature_root.iterdir():
            if not split_dir.is_dir():
                continue
            live = split_dir / "live"
            if live.is_dir():
                shutil.rmtree(live, ignore_errors=True)

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
        # Keep a durable epoch copy + refresh latest/best pointer tag.
        tags = [self._epoch_tag(epoch), tag] if tag in {"latest", "best"} else [tag]
        saved: list[Path] = []
        for out_tag in tags:
            output_dir = self._feature_map_tag_dir(split, out_tag)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            saved.extend(
                save_feature_maps(
                    adapter,
                    model,
                    loader,
                    output_dir,
                    split=split,
                    device=device,
                    cfg=cfg,
                    max_batches=self.plot_max_batches(),
                )
            )
        return saved

    def _waveform_tag_dir(self, split: str, tag: str) -> Path:
        return self.waveforms_dir / split / tag

    def _metrics_epoch_dir(self, epoch: int) -> Path:
        return self.run_dir / "metrics_by_epoch" / self._epoch_tag(epoch)

    def _prune_legacy_epoch_waveform_dirs(self) -> None:
        """Remove obsolete ``live/`` waveform folders only (keep epoch_* history)."""
        if not self.waveforms_dir.exists():
            return
        for split_dir in self.waveforms_dir.iterdir():
            if not split_dir.is_dir():
                continue
            live = split_dir / "live"
            if live.is_dir():
                shutil.rmtree(live, ignore_errors=True)

    def _append_metrics_history(
        self,
        *,
        epoch: int,
        split: str,
        metrics: pd.DataFrame,
    ) -> None:
        """Append overall (+ per-app) rows into a long-running comparison CSV."""
        history_path = self.run_dir / "metrics_history.csv"
        df = metrics.copy()
        # evaluate_bundle already has a ``split`` column; only add ``epoch``.
        df.insert(0, "epoch", int(epoch))
        df["split"] = str(split)
        if history_path.exists():
            prev = pd.read_csv(history_path)
            out = pd.concat([prev, df], ignore_index=True)
        else:
            out = df
        # Prefer epoch, split, appliance first for easy reading.
        cols = list(out.columns)
        front = [c for c in ("epoch", "split", "appliance") if c in cols]
        rest = [c for c in cols if c not in front]
        out = out[front + rest]
        out.to_csv(history_path, index=False)

    def _save_split_metrics_table(
        self,
        adapter,
        bundle,
        *,
        split: str,
        epoch: int,
    ) -> Path:
        """Write the same MAE/SAE/F1 table as final evaluate, keyed by epoch."""
        power_postprocess = resolve_power_postprocess(
            adapter.experiment,
            bundle.appliances,
            adapter.model_cfg,
        )
        metrics = evaluate_bundle(
            bundle,
            sae_period=int(adapter.experiment["evaluation"].get("sae_period", 1200)),
            on_threshold_watts=(
                resolve_state_thresholds_watts(adapter.experiment, self.appliances)
                if get_state_label_source(adapter.model_cfg) == "threshold"
                else None
            ),
            state_label_source=get_state_label_source(adapter.model_cfg),
            power_postprocess=power_postprocess,
        )
        epoch_dir = self._metrics_epoch_dir(epoch)
        epoch_dir.mkdir(parents=True, exist_ok=True)
        path = epoch_dir / f"{split}_metrics.csv"
        metrics.to_csv(path, index=False)
        # Also keep a rolling "latest" copy of the table for this split.
        latest_dir = self.run_dir / "metrics_by_epoch" / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(latest_dir / f"{split}_metrics.csv", index=False)
        self._append_metrics_history(epoch=epoch, split=split, metrics=metrics)
        return path

    def _save_epoch_val_test_comparison_figure(self, epoch: int) -> Path | None:
        """PNG table of validation vs test metrics for this plot-interval epoch."""
        epoch_dir = self._metrics_epoch_dir(epoch)
        val_path = epoch_dir / "validation_metrics.csv"
        test_path = epoch_dir / "test_metrics.csv"
        if not val_path.exists() or not test_path.exists():
            return None
        out = epoch_dir / "validation_test_comparison.png"
        save_val_test_comparison_figure(
            val_path,
            test_path,
            out,
            epoch=epoch,
            title=f"ep{epoch} val vs test",
            dpi=max(600, int(self.plot_cfg.get("comparison_dpi", 600))),
        )
        latest_dir = self.run_dir / "metrics_by_epoch" / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        latest_png = latest_dir / "validation_test_comparison.png"
        latest_csv = latest_dir / "validation_test_comparison.csv"
        shutil.copy2(out, latest_png)
        csv_src = out.with_suffix(".csv")
        if csv_src.exists():
            shutil.copy2(csv_src, latest_csv)
        return out

    def _save_epoch_comparison_dashboards(self, epoch: int) -> list[Path]:
        """Build single-picture comparisons across epochs (metrics XOR waveforms)."""
        saved: list[Path] = []
        dpi = int(self.plot_cfg.get("comparison_dpi", 600))
        # One collage per ON-period example (same count as plot_on_periods).
        n_periods = max(1, self.plot_on_periods())

        # Metrics only (no waveforms mixed in).
        metrics_all = save_multi_epoch_metrics_collage(
            self.run_dir,
            title="",  # panels already labeled; keep collage compact
            dpi=dpi,
        )
        if metrics_all is not None:
            saved.append(metrics_all)

        # Waveforms: separate PNG for each ON-period case (01..N).
        for period_index in range(1, n_periods + 1):
            saved.extend(
                save_multi_epoch_waveform_collages(
                    self.run_dir,
                    self.appliances,
                    period_index=period_index,
                    prefer_context=False,
                    dpi=dpi,
                    title_prefix=f"{self.model_name} ",
                )
            )
        return saved

    def _write_waveforms_for_bundle(
        self,
        adapter,
        bundle,
        *,
        split: str,
        epoch: int,
        tag: str,
    ) -> list[Path]:
        aggregate = bundle_aggregate_watts(
            adapter._data_loader(),
            split,
            n_points=len(bundle.y_true_watts),
            csv_timesteps=bundle.csv_timesteps,
        )
        y_true_plot = bundle_csv_appliance_watts(
            adapter._data_loader(),
            split,
            n_points=len(bundle.y_true_watts),
            csv_timesteps=bundle.csv_timesteps,
        )
        if y_true_plot is None:
            y_true_plot = bundle.y_true_watts

        waveform_true_on = dataset_on_labels_for_bundle(
            adapter._data_loader(),
            split,
            len(bundle.y_true_watts),
            bundle.csv_timesteps,
        )
        state_src = get_state_label_source(adapter.model_cfg)
        on_thresholds = (
            resolve_state_thresholds_watts(adapter.experiment, self.appliances)
            if state_src == "threshold"
            else None
        )
        # Epoch-stable seed → same ON periods every plot interval (fair collage).
        split_id = 0 if split == "validation" else 1
        rng = np.random.default_rng(self.seed + 17 * split_id)

        # Durable epoch folder + pointer tag (latest / best).
        tags = [self._epoch_tag(epoch)]
        if tag in {"latest", "best"}:
            tags.append(tag)

        saved: list[Path] = []
        for out_tag in tags:
            output_dir = self._waveform_tag_dir(split, out_tag)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            saved.extend(
                save_appliance_on_waveforms(
                    output_dir,
                    appliances=self.appliances,
                    y_true_watts=y_true_plot,
                    y_pred_watts=bundle.y_pred_watts,
                    y_true_on=waveform_true_on,
                    y_pred_on=bundle.y_pred_on,
                    on_thresholds_watts=on_thresholds,
                    state_label_source=state_src,
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
                    context_scale=self.waveform_context_scale(),
                    rng=rng,
                    file_prefix=out_tag,
                    title_prefix=f"{self.model_name} {split} {out_tag} epoch {epoch} — ",
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
        # Waveforms need the full split timeline; partial batches break overlap
        # reconstruction and ON-period selection on later CSV rows.
        bundle = adapter.predict_dataloader(
            model,
            loader,
            device,
            split=split,
            max_batches=None,
        )
        saved = self._write_waveforms_for_bundle(
            adapter, bundle, split=split, epoch=epoch, tag=tag
        )
        # Same inference → full MAE/SAE/F1 table for this epoch (val/test).
        if tag == "latest":
            metrics_path = self._save_split_metrics_table(
                adapter, bundle, split=split, epoch=epoch
            )
            # Touch a small README so the epoch folder is self-describing.
            note = self._metrics_epoch_dir(epoch) / "README.txt"
            note.write_text(
                f"Metrics and waveforms for training epoch {epoch}.\n"
                f"Waveforms: waveforms/{{validation,test}}/epoch_{epoch:04d}/\n"
                f"Table: {metrics_path.name}\n"
                f"All epochs appended to metrics_history.csv\n",
                encoding="utf-8",
            )
        return saved

    def close(self) -> None:
        if self._history_file is not None:
            self._history_file.close()
        if self._loss_file is not None:
            self._loss_file.close()

    def finalize(self, *, best_epoch: int, last_epoch: int | None = None) -> None:
        """Write final loss/metric PNGs (and refresh live_* copies).

        Always runs at training end — including early stop — so curves are not
        left stale when the last epoch falls between ``plot_interval`` ticks.
        """
        if self.plot_cfg.get("enabled") is False:
            return
        if not self.history_path.exists():
            return
        epoch = int(last_epoch if last_epoch is not None else best_epoch or 0)
        # Refresh live_* first (same paths used during interval updates).
        self.save_loss_plots(epoch=epoch, best_epoch=best_epoch or None)
        figsize = self.waveform_figsize()
        plot_training_history(
            self.history_path,
            self.run_dir / "training_loss.png",
            title=f"{self.model_name} training",
            best_epoch=best_epoch,
            figsize=figsize,
        )
        plot_validation_metrics(
            self.history_path,
            self.run_dir / "validation_metrics.png",
            title=f"{self.model_name} validation metrics",
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

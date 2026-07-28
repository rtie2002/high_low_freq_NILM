"""
Independent UK-DALE window dataset for MATUDA (no MultiNILM imports).

CSV columns (multi_appliances_NILM processed UK-DALE):
  readable_time, house, aggregate,
  {app}_power, {app}_on  for each appliance

Locked protocol (already reflected in the split files):
  training / validating  -> Houses 1+5 (labeled source)
  testing               -> House 2 (target; labels used only at eval)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_APPLIANCES = (
    "kettle",
    "fridge",
    "dishwasher",
    "washingmachine",
    "microwave",
)


@dataclass
class NormStats:
    """Z-score stats (same defaults as common UK-DALE multi-appliance baselines)."""

    agg_mean: float = 400.0
    agg_std: float = 500.0
    app_mean: Optional[np.ndarray] = None  # (K,)
    app_std: Optional[np.ndarray] = None  # (K,)

    @classmethod
    def ukdale_defaults(cls, appliances: Sequence[str]) -> "NormStats":
        table = {
            "kettle": (100.0, 500.0),
            "fridge": (50.0, 50.0),
            "dishwasher": (700.0, 1000.0),
            "washingmachine": (400.0, 700.0),
            "microwave": (60.0, 300.0),
        }
        means = np.asarray([table[a][0] for a in appliances], dtype=np.float32)
        stds = np.asarray([table[a][1] for a in appliances], dtype=np.float32)
        return cls(agg_mean=400.0, agg_std=500.0, app_mean=means, app_std=stds)

    def norm_agg(self, x: np.ndarray) -> np.ndarray:
        return (x - self.agg_mean) / (self.agg_std + 1e-8)

    def norm_power(self, y: np.ndarray) -> np.ndarray:
        assert self.app_mean is not None and self.app_std is not None
        return (y - self.app_mean) / (self.app_std + 1e-8)

    def denorm_power(self, y: np.ndarray) -> np.ndarray:
        assert self.app_mean is not None and self.app_std is not None
        return np.maximum(y * self.app_std + self.app_mean, 0.0)


def load_csv_arrays(
    csv_path: Path | str,
    appliances: Sequence[str] = DEFAULT_APPLIANCES,
    houses: Optional[Iterable[int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      aggregate: (T,) float32
      powers:    (T, K) float32
      states:    (T, K) float32 in {0,1}
    """
    path = Path(csv_path)
    usecols = ["house", "aggregate"]
    for a in appliances:
        usecols.append(f"{a}_power")
        usecols.append(f"{a}_on")
    df = pd.read_csv(path, usecols=usecols)
    if houses is not None:
        house_set = set(int(h) for h in houses)
        df = df[df["house"].isin(house_set)]
    agg = df["aggregate"].to_numpy(dtype=np.float32)
    powers = np.stack(
        [df[f"{a}_power"].to_numpy(dtype=np.float32) for a in appliances], axis=1
    )
    states = np.stack(
        [df[f"{a}_on"].to_numpy(dtype=np.float32) for a in appliances], axis=1
    )
    return agg, powers, states


class AggregateWindowDataset(Dataset):
    """
    Sliding windows over aggregate (+ optional source labels).

    Each item:
      x: (1, T) normalized aggregate
      y: (K,)   normalized power at window center (zeros if unlabeled)
      z: (K,)   ON/OFF at window center (zeros if unlabeled)
    """

    def __init__(
        self,
        aggregate: np.ndarray,
        powers: Optional[np.ndarray],
        states: Optional[np.ndarray],
        *,
        seq_len: int = 600,
        stride: int = 30,
        norm: Optional[NormStats] = None,
        labeled: bool = True,
        appliances: Sequence[str] = DEFAULT_APPLIANCES,
    ):
        self.seq_len = int(seq_len)
        self.stride = max(1, int(stride))
        self.labeled = labeled
        self.norm = norm or NormStats.ukdale_defaults(appliances)
        self.k = len(appliances)

        agg = np.asarray(aggregate, dtype=np.float32).reshape(-1)
        if len(agg) < self.seq_len:
            raise ValueError(f"Series length {len(agg)} < seq_len {self.seq_len}")

        self.agg = self.norm.norm_agg(agg)
        if labeled:
            if powers is None or states is None:
                raise ValueError("Labeled dataset requires powers and states")
            self.powers = self.norm.norm_power(np.asarray(powers, dtype=np.float32))
            self.states = np.asarray(states, dtype=np.float32)
        else:
            # Target UDA path: aggregates only; placeholders unused by domain loss.
            self.powers = np.zeros((len(agg), self.k), dtype=np.float32)
            self.states = np.zeros((len(agg), self.k), dtype=np.float32)

        self.starts = np.arange(0, len(agg) - self.seq_len + 1, self.stride, dtype=np.int64)
        self.center = self.seq_len // 2

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = int(self.starts[idx])
        e = s + self.seq_len
        c = s + self.center
        x = torch.from_numpy(self.agg[s:e]).unsqueeze(0)  # (1, T)
        y = torch.from_numpy(self.powers[c])
        z = torch.from_numpy(self.states[c])
        return {"x": x, "y": y, "z": z}


def estimate_pos_weight(states: np.ndarray, cap: float = 50.0) -> torch.Tensor:
    """
    BCE pos_weight per appliance from source ON rates.
    pos_weight_k = n_neg / n_pos, capped at `cap` (default 50).
    """
    z = np.asarray(states, dtype=np.float64)
    n_pos = z.sum(axis=0)
    n_neg = z.shape[0] - n_pos
    w = n_neg / np.maximum(n_pos, 1.0)
    w = np.minimum(w, cap)
    return torch.tensor(w, dtype=torch.float32)


def _chrono_split(
    agg: np.ndarray,
    powers: np.ndarray,
    states: np.ndarray,
    adapt_frac: float,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Chronological split of a single-house series to avoid adapt/test leakage.
    First `adapt_frac` of timesteps -> unlabeled adaptation; remainder -> test.
    """
    n = len(agg)
    cut = int(n * float(adapt_frac))
    cut = max(cut, 1)
    cut = min(cut, n - 1)
    return (
        (agg[:cut], powers[:cut], states[:cut]),
        (agg[cut:], powers[cut:], states[cut:]),
    )


def make_loaders(
    data_root: Path | str,
    *,
    appliances: Sequence[str] = DEFAULT_APPLIANCES,
    seq_len: int = 599,
    stride_train: int = 30,
    stride_eval: int = 60,
    batch_size: int = 64,
    num_workers: int = 0,
    target_adapt_frac: float | None = 0.7,
) -> dict:
    """
    Source train/val: H1+H5 labeled.
    Target House 2 is split chronologically when target_adapt_frac in (0, 1):
      - first target_adapt_frac  -> unlabeled adaptation (aggregates only)
      - remaining                -> held-out test (labels for metrics only)
    If target_adapt_frac is None or <=0: use full House-2 as test and as
    unlabeled adapt (legacy / figure regeneration for older checkpoints).

    Normalization uses fixed literature constants (NILM multi-appliance UK-DALE
    baseline Arguments.py), not statistics computed on the target/test set.
    """
    root = Path(data_root)
    train_csv = root / "training" / "multi_appliance_training.csv"
    val_csv = root / "validating" / "multi_appliance_validating.csv"
    test_csv = root / "testing" / "multi_appliance_testing.csv"
    norm = NormStats.ukdale_defaults(appliances)

    agg_tr, p_tr, z_tr = load_csv_arrays(train_csv, appliances)
    agg_va, p_va, z_va = load_csv_arrays(val_csv, appliances)
    agg_te, p_te, z_te = load_csv_arrays(test_csv, appliances)
    pos_weight = estimate_pos_weight(z_tr, cap=50.0)

    if target_adapt_frac is None or float(target_adapt_frac) <= 0.0:
        agg_ad, p_ev, z_ev = agg_te, p_te, z_te
        agg_ev = agg_te
        used_frac = 0.0
    else:
        (agg_ad, _, _), (agg_ev, p_ev, z_ev) = _chrono_split(
            agg_te, p_te, z_te, float(target_adapt_frac)
        )
        used_frac = float(target_adapt_frac)

    ds_src = AggregateWindowDataset(
        agg_tr, p_tr, z_tr, seq_len=seq_len, stride=stride_train, norm=norm, labeled=True,
        appliances=appliances,
    )
    ds_val = AggregateWindowDataset(
        agg_va, p_va, z_va, seq_len=seq_len, stride=stride_eval, norm=norm, labeled=True,
        appliances=appliances,
    )
    ds_tgt = AggregateWindowDataset(
        agg_ad, None, None, seq_len=seq_len, stride=stride_train, norm=norm, labeled=False,
        appliances=appliances,
    )
    ds_test = AggregateWindowDataset(
        agg_ev, p_ev, z_ev, seq_len=seq_len, stride=stride_eval, norm=norm, labeled=True,
        appliances=appliances,
    )

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return {
        "source": DataLoader(ds_src, shuffle=True, drop_last=True, **common),
        "target": DataLoader(ds_tgt, shuffle=True, drop_last=True, **common),
        "val": DataLoader(ds_val, shuffle=False, drop_last=False, **common),
        "test": DataLoader(ds_test, shuffle=False, drop_last=False, **common),
        "norm": norm,
        "appliances": list(appliances),
        "pos_weight": pos_weight,
        "on_rates": {a: float(z_tr[:, i].mean()) for i, a in enumerate(appliances)},
        "target_split": {
            "adapt_frac": used_frac,
            "adapt_timesteps": int(len(agg_ad)),
            "test_timesteps": int(len(agg_ev)),
            "adapt_windows": len(ds_tgt),
            "test_windows": len(ds_test),
        },
        "label_thresholds_watts": {
            # Thresholds used when building UK-DALE multi-appliance CSV on/off columns
            # (see multi_appliances_NILM experiment_ukdale.yaml).
            "kettle": 40,
            "fridge": 50,
            "dishwasher": 30,
            "washingmachine": 30,
            "microwave": 100,
        },
    }

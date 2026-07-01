"""UNet-NILM paper preprocessing (Faustine et al., BuildSys 2020).

Ported from NILM_model/baseline/UNETNILM/src/data/load_data.py:
  - median (p50) sliding-window filter per channel
  - per-appliance z-score on filtered appliance power
  - binary states from filtered power + on_power_threshold
  - mains noise or denoise path with z-score
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _effective_window(window: int) -> int:
    return window - 1 if window % 2 == 0 else window


def _sliding_windows(sequence_length: int, data: np.ndarray) -> np.ndarray:
    seq_len = _effective_window(sequence_length)
    pad = seq_len // 2
    padded = np.pad(data, (pad, pad), mode="constant", constant_values=0.0)
    return np.stack([padded[i : i + seq_len] for i in range(len(padded) - seq_len + 1)], axis=0)


def quantile_filter(window: int, data: np.ndarray, p: float = 50.0) -> np.ndarray:
    """Median filter implemented as sliding-window percentile (author repo)."""
    windows = _sliding_windows(window, np.asarray(data, dtype=np.float64))
    return np.percentile(windows, p, axis=1, method="nearest").astype(np.float32)


def binarize(power: np.ndarray, threshold: float) -> np.ndarray:
    return (power >= threshold).astype(np.int64)


def _mains_noise_path(
    mains: np.ndarray,
    filter_window: int,
    *,
    sub_mains: np.ndarray | None = None,
) -> np.ndarray:
    """noise_inputs.npy path (author load_data.pre_process_uk_dale)."""
    sub = np.asarray(sub_mains if sub_mains is not None else mains, dtype=np.float64)
    mains_denoise = quantile_filter(filter_window, sub, p=50.0)
    floored = mains - float(np.percentile(mains, 1))
    floored = np.where(floored < mains_denoise, mains_denoise, floored)
    return quantile_filter(filter_window, floored, p=50.0)


def _mains_denoise_path(mains: np.ndarray, filter_window: int) -> np.ndarray:
    """denoise_inputs.npy path — uses aggregate as stand-in for sub_mains."""
    return quantile_filter(filter_window, mains, p=50.0)


def preprocess_mains(
    mains_watts: np.ndarray,
    seq2quantile: dict[str, Any],
    *,
    use_denoised: bool = False,
    sub_mains_watts: np.ndarray | None = None,
) -> np.ndarray:
    cfg = seq2quantile.get("mains", {})
    filter_window = int(
        seq2quantile.get("mains_denoise_window", 10) if use_denoised
        else seq2quantile.get("mains_noise_window", 10)
    )
    if use_denoised:
        sub = np.asarray(sub_mains_watts if sub_mains_watts is not None else mains_watts, dtype=np.float64)
        filtered = _mains_denoise_path(sub, filter_window)
        norm = cfg.get("denoise", {"mean": 123.0, "std": 369.0})
    else:
        filtered = _mains_noise_path(
            mains_watts,
            filter_window,
            sub_mains=sub_mains_watts,
        )
        norm = cfg.get("noise", {"mean": 389.0, "std": 445.0})
    mean = float(norm["mean"])
    std = float(norm["std"])
    return ((filtered - mean) / std).astype(np.float32)


def preprocess_appliances(
    power_watts: np.ndarray,
    appliances: list[str],
    seq2quantile: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Filter, z-score power targets, and rebuild ON/OFF states (author pipeline)."""
    app_cfg = seq2quantile["appliances"]
    p = float(seq2quantile.get("quantile_percent", 50.0))
    n = len(power_watts)
    n_apps = len(appliances)
    y = np.empty((n, n_apps), dtype=np.float32)
    z = np.empty((n, n_apps), dtype=np.int64)

    for i, app in enumerate(appliances):
        stats = app_cfg[app]
        filtered = quantile_filter(int(stats["window"]), power_watts[:, i], p=p)
        y[:, i] = (filtered - float(stats["mean"])) / float(stats["std"])
        z[:, i] = binarize(filtered, float(stats["on_power_threshold"]))

    return y, z


def denorm_appliance_power(
    power_norm: np.ndarray,
    appliances: list[str],
    seq2quantile: dict[str, Any],
    *,
    style: str = "author",
) -> np.ndarray:
    """Inverse scaling for eval watts (author repo uses norm * std + std)."""
    out = np.array(power_norm, dtype=np.float64, copy=True)
    app_cfg = seq2quantile["appliances"]
    for i, app in enumerate(appliances):
        stats = app_cfg[app]
        std = float(stats["std"])
        mean = float(stats["mean"])
        if style == "author":
            out[..., i] = out[..., i] * std + std
        else:
            out[..., i] = out[..., i] * std + mean
        out[..., i] = np.maximum(out[..., i], 0.0)
    return out.astype(np.float32)


def preprocess_unet_arrays(
    mains_watts: np.ndarray,
    power_watts: np.ndarray,
    appliances: list[str],
    model_cfg: dict[str, Any],
    *,
    sub_mains_watts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq2quantile = model_cfg["seq2quantile"]
    data_cfg = model_cfg.get("data", {})
    use_denoised = bool(data_cfg.get("use_denoised_mains", False))
    x = preprocess_mains(
        mains_watts,
        seq2quantile,
        use_denoised=use_denoised,
        sub_mains_watts=sub_mains_watts,
    )
    y, z = preprocess_appliances(power_watts, appliances, seq2quantile)
    return x, y, z

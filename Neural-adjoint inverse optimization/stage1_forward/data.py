"""Stage-1 sliding-window dataset over raw appliance power arrays.

The forward model receives [B, 5, T] windows of scaled power (watts / S).
This dataset extracts those windows from the pre-loaded SplitArrays
produced by NILMDataLoader, respecting CSV sequence boundaries so windows
never straddle two discontinuous recordings.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class RawPowerWindows(Dataset):
    """Sliding-window dataset of five-appliance power, scaled to [0, 1].

    Each item is a [5, T] float32 tensor of p_watts / power_scale_watts.
    The DataLoader shuffles these for Stage-1 training; Stage-1 uses only
    the appliance traces (not the aggregate mains) as input and target.

    Parameters
    ----------
    windows : np.ndarray
        Pre-extracted windows of shape [N, 5, T], already scaled.
    """

    def __init__(self, windows: np.ndarray) -> None:
        # [N, 5, T] float32, pre-scaled power.
        self._windows = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> Tensor:
        # Return [5, T]: five appliance traces in scaled watts.
        return self._windows[index]

    @classmethod
    def from_split(
        cls,
        *,
        powers_watts: np.ndarray,
        segment_ids: np.ndarray,
        length: int,
        stride: int,
        power_scale_watts: float,
    ) -> "RawPowerWindows":
        """Build the dataset from a full split's power array.

        Parameters
        ----------
        powers_watts : np.ndarray
            Shape [T_total, 5] — raw appliance watts from NILMDataLoader.
        segment_ids : np.ndarray
            Shape [T_total] — monotonic segment IDs; windows that cross a
            segment boundary are dropped.
        length : int
            Number of samples per window (e.g. 1024).
        stride : int
            Step between consecutive window starts (e.g. 512).
        power_scale_watts : float
            Common physical scale; all five traces are divided by this value.
        """
        # Collect per-segment window start indices, restarting stride at each
        # segment boundary (matching the NILMDataLoader convention).
        boundaries = np.flatnonzero(
            np.r_[True, segment_ids[1:] != segment_ids[:-1], True]
        )
        starts_list = [
            np.arange(start, end - length + 1, stride, dtype=np.int64)
            for start, end in zip(boundaries[:-1], boundaries[1:])
            if end - start >= length
        ]
        if not starts_list:
            # Return an empty dataset so smoke runs don't crash.
            empty = np.zeros((0, 5, length), dtype=np.float32)
            return cls(empty)

        starts = np.concatenate(starts_list)

        # Extract all windows at once: [N, length, 5] → transpose → [N, 5, length].
        offsets = np.arange(length, dtype=np.int64)                 # [length]
        rows    = starts[:, None] + offsets[None, :]                # [N, length]
        windows = powers_watts[rows]                                 # [N, length, 5]
        windows = windows.transpose(0, 2, 1) / power_scale_watts    # [N, 5, length]
        return cls(windows)


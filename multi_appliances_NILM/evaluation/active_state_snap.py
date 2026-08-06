"""Active-state power snap (Schirmer / MSDC-style discrete levels).

Fit power-level centers on **source** appliance watts (active samples only),
then quantize predictions that are ON to the nearest center. No manual labels:
centers come from KMeans / MeanShift on existing submeter power Y.

Modes:
  - pointwise: if p > eps, replace with nearest center (Schirmer).
  - segment: each contiguous pred-ON run -> median snapped to one center
    and filled (better for mid-ON dropouts).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _as_2d(power: np.ndarray) -> np.ndarray:
    arr = np.asarray(power, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _fill_short_gaps(on: np.ndarray, max_gap: int) -> np.ndarray:
    """Close OFF holes of length <= max_gap inside ON runs (1D bool)."""
    if max_gap <= 0 or not np.any(on):
        return on
    out = np.asarray(on, dtype=bool).copy()
    n = int(out.size)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        left_on = i > 0 and bool(out[i - 1])
        right_on = j < n and bool(out[j])
        if left_on and right_on and (j - i) <= max_gap:
            out[i:j] = True
        i = j
    return out


def fit_active_centers_1d(
    power_watts: np.ndarray,
    *,
    on_threshold_watts: float,
    n_clusters: int,
    method: str = "kmeans",
    max_samples: int = 200_000,
    seed: int = 0,
) -> np.ndarray:
    """Cluster active (p > threshold) watts -> sorted center levels (W)."""
    x = np.asarray(power_watts, dtype=np.float64).reshape(-1)
    active = x[x > float(on_threshold_watts)]
    if active.size == 0:
        return np.asarray([], dtype=np.float64)

    k = max(1, int(n_clusters))
    if active.size > max_samples:
        rng = np.random.default_rng(seed)
        active = rng.choice(active, size=max_samples, replace=False)

    method = str(method).lower()
    if method in {"meanshift", "mean_shift"}:
        from sklearn.cluster import MeanShift, estimate_bandwidth

        bw = estimate_bandwidth(
            active.reshape(-1, 1), quantile=0.1, n_samples=min(1000, int(active.size))
        )
        if not np.isfinite(bw) or bw <= 0:
            bw = max(float(np.std(active)) * 0.2, 1.0)
        ms = MeanShift(bandwidth=bw, bin_seeding=True)
        ms.fit(active.reshape(-1, 1))
        return np.sort(ms.cluster_centers_.reshape(-1)).astype(np.float64)

    if active.size < k:
        return np.unique(np.round(active, 1)).astype(np.float64)

    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        km.fit(active.reshape(-1, 1))
        return np.sort(km.cluster_centers_.reshape(-1)).astype(np.float64)
    except ImportError:
        qs = np.linspace(0.5 / k, 1.0 - 0.5 / k, k)
        return np.unique(np.quantile(active, qs)).astype(np.float64)


def label_states_1d(
    power_watts: np.ndarray,
    centers: np.ndarray,
    *,
    on_threshold_watts: float,
) -> np.ndarray:
    """Map watts -> state id: 0 = OFF, 1..K = nearest active center."""
    x = np.asarray(power_watts, dtype=np.float64).reshape(-1)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    out = np.zeros(x.shape[0], dtype=np.int32)
    if centers.size == 0:
        return out
    active = x > float(on_threshold_watts)
    if not np.any(active):
        return out
    d = np.abs(x[active, None] - centers[None, :])
    out[active] = d.argmin(axis=1).astype(np.int32) + 1
    return out


def snap_pointwise(
    power_watts: np.ndarray,
    centers: np.ndarray,
    *,
    eps_watts: float,
) -> np.ndarray:
    """Schirmer: leave p <= eps; else nearest active center."""
    x = np.asarray(power_watts, dtype=np.float64).copy().reshape(-1)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    if centers.size == 0:
        return x
    active = x > float(eps_watts)
    if np.any(active):
        d = np.abs(x[active, None] - centers[None, :])
        x[active] = centers[d.argmin(axis=1)]
    return x


def snap_segments(
    power_watts: np.ndarray,
    centers: np.ndarray,
    on_mask: np.ndarray,
    *,
    eps_watts: float,
) -> np.ndarray:
    """Fill each contiguous ON run with one snapped level (median -> center)."""
    x = np.asarray(power_watts, dtype=np.float64).copy().reshape(-1)
    on = np.asarray(on_mask, dtype=bool).copy().reshape(-1)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    if centers.size == 0:
        return x
    if int(x.size) != int(on.size):
        raise ValueError(f"power/on length mismatch: {x.size} vs {on.size}")

    n = int(x.size)
    eps = float(eps_watts)
    i = 0
    while i < n:
        if not bool(on[i]):
            i += 1
            continue
        j = i + 1
        while j < n and bool(on[j]):
            j += 1
        seg = x[i:j]
        strong = seg[seg > eps]
        level = float(np.median(strong)) if strong.size else float(np.median(seg))
        if level <= eps:
            level = float(centers[len(centers) // 2])
        center = float(centers[int(np.argmin(np.abs(centers - level)))])
        x[i:j] = center
        i = j

    off = np.logical_not(on)
    x[off] = np.where(x[off] > eps, 0.0, x[off])
    return x


@dataclass
class ActiveStateSnapConfig:
    enabled: bool
    method: str
    mode: str  # pointwise | segment
    on_source: str  # state_head | power_eps
    fill_gaps_max: int
    fit_split: str
    centers_path: Path | None
    n_clusters: dict[str, int]
    eps_watts: dict[str, float]
    seed: int = 0


def resolve_active_state_snap(
    experiment_cfg: dict[str, Any],
    appliances: list[str],
    model_cfg: dict[str, Any] | None = None,
) -> ActiveStateSnapConfig | None:
    """Read evaluation.active_state_snap from model (preferred) or experiment."""
    model_eval = (model_cfg or {}).get("evaluation", {})
    exp_eval = experiment_cfg.get("evaluation", {})
    raw = model_eval.get("active_state_snap")
    if raw is None:
        raw = exp_eval.get("active_state_snap", {})
    if raw is False or raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    if not bool(raw.get("enabled", False)):
        return None

    thr_map = exp_eval.get("on_thresholds_watts", {})
    default_k = int(raw.get("n_clusters_default", 2))
    n_map_raw = raw.get("n_clusters", {})
    if isinstance(n_map_raw, int):
        n_clusters = {app: int(n_map_raw) for app in appliances}
    else:
        n_clusters = {
            app: int(n_map_raw.get(app, default_k)) if isinstance(n_map_raw, dict) else default_k
            for app in appliances
        }

    eps_raw = raw.get("eps_watts")
    eps: dict[str, float] = {}
    for app in appliances:
        if isinstance(eps_raw, dict) and app in eps_raw:
            eps[app] = float(eps_raw[app])
        elif eps_raw is not None and not isinstance(eps_raw, dict):
            eps[app] = float(eps_raw)
        else:
            eps[app] = float(thr_map.get(app, 5.0))

    path = raw.get("centers_path")
    return ActiveStateSnapConfig(
        enabled=True,
        method=str(raw.get("method", "kmeans")).lower(),
        mode=str(raw.get("mode", "segment")).lower(),
        on_source=str(raw.get("on_source", "state_head")).lower(),
        fill_gaps_max=int(raw.get("fill_gaps_max", 20)),
        fit_split=str(raw.get("fit_split", "train")),
        centers_path=None if not path else Path(str(path)),
        n_clusters=n_clusters,
        eps_watts=eps,
        seed=int(raw.get("seed", experiment_cfg.get("seed", 0))),
    )


def save_centers(path: Path, centers: dict[str, np.ndarray], meta: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "centers_watts": {k: np.asarray(v, dtype=float).tolist() for k, v in centers.items()},
        "meta": meta or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_centers(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("centers_watts", data)
    return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}


def fit_centers_from_power_matrix(
    y_watts: np.ndarray,
    appliances: list[str],
    cfg: ActiveStateSnapConfig,
    on_thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    y = _as_2d(y_watts)
    if y.shape[1] != len(appliances):
        raise ValueError(f"power cols {y.shape[1]} != appliances {len(appliances)}")
    thr = np.asarray(on_thresholds, dtype=np.float64).reshape(-1)
    out: dict[str, np.ndarray] = {}
    for i, app in enumerate(appliances):
        out[app] = fit_active_centers_1d(
            y[:, i],
            on_threshold_watts=float(thr[i]) if i < len(thr) else cfg.eps_watts[app],
            n_clusters=cfg.n_clusters[app],
            method=cfg.method,
            seed=cfg.seed + i,
        )
    return out


def apply_active_state_snap(
    y_pred_watts: np.ndarray,
    centers: dict[str, np.ndarray],
    appliances: list[str],
    cfg: ActiveStateSnapConfig,
    *,
    y_pred_on: np.ndarray | None = None,
) -> np.ndarray:
    """Return snapped prediction watts (N, A)."""
    y = _as_2d(y_pred_watts).copy()
    on_all = None if y_pred_on is None else _as_2d(y_pred_on).astype(bool)

    for i, app in enumerate(appliances):
        c = centers.get(app)
        if c is None or len(c) == 0:
            continue
        eps = float(cfg.eps_watts[app])
        col = y[:, i]

        if cfg.mode == "pointwise":
            y[:, i] = snap_pointwise(col, c, eps_watts=eps)
            continue

        if cfg.on_source == "state_head" and on_all is not None:
            on = on_all[:, i].astype(bool)
        else:
            on = col > eps
        on = _fill_short_gaps(on, cfg.fill_gaps_max)
        y[:, i] = snap_segments(col, c, on, eps_watts=eps)

    return y

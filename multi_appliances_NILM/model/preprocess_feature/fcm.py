"""
Active-state Fuzzy C-Means post-processing (Schirmer Sec. III-C / Eq. 9–10).

References
----------
1. Schirmer & Mporas, IEEE OAJPE 2022 — Device and Time Invariant Features…
   Only **ON** estimates are snapped to active-state centers; near-OFF left alone.
2. Ji et al., IEEE TSG 2019 — AFAMAP based on Iterative Fuzzy c-Means (IFCM)
   Adaptive number of states via FCM + centroid-deviation threshold
   (``m_i ∈ [2, 8]`` in their Table I / Fig. 1).

This module is a **standalone block** (fit on source ON watts → snap predictions).
Wire into evaluation later; training loss is unchanged when used eval-only.

Watt space
----------
Fit and apply in **watts** (denormalized). Do not mix z-scored tensors here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

ArrayLike = np.ndarray | Sequence[float]


# ---------------------------------------------------------------------------
# 1. Core Fuzzy C-Means (1-D power samples)
# ---------------------------------------------------------------------------

def fuzzy_cmeans(
    samples: ArrayLike,
    n_clusters: int,
    *,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-5,
    rng: np.random.Generator | int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard Bezdek FCM on 1-D active-power samples.

    Args:
        samples: (N,) ON-power values in watts (OFF already filtered out).
        n_clusters: number of centers ``c``.
        m: fuzzifier (``m > 1``); paper-typical value 2.
        max_iter / tol: convergence.
        rng: seed or Generator for reproducible init.

    Returns:
        centers: (c,) sorted ascending (watts).
        membership: (c, N) fuzzy memberships.
    """
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("fuzzy_cmeans: empty samples")
    c = int(n_clusters)
    if c < 1:
        raise ValueError(f"n_clusters must be >= 1, got {c}")
    if x.size < c:
        # Too few points: unique values / mean as degenerate centers.
        uniq = np.unique(x)
        if uniq.size >= c:
            centers = np.sort(uniq[:c].astype(np.float64))
        else:
            pad = np.full(c - uniq.size, float(uniq[-1]) if uniq.size else 0.0)
            centers = np.sort(np.concatenate([uniq.astype(np.float64), pad]))
        # Hard membership to nearest center.
        d = np.abs(x[None, :] - centers[:, None])
        u = np.zeros((c, x.size), dtype=np.float64)
        u[np.argmin(d, axis=0), np.arange(x.size)] = 1.0
        return centers, u

    gen = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(rng)
    )
    # Init membership randomly, columns sum to 1.
    u = gen.random((c, x.size))
    u /= np.clip(u.sum(axis=0, keepdims=True), 1e-12, None)

    centers = np.zeros(c, dtype=np.float64)
    for _ in range(int(max_iter)):
        um = u**float(m)
        denom = np.clip(um.sum(axis=1), 1e-12, None)
        centers_new = (um @ x) / denom

        # Distances; avoid exact-zero for numerical stability.
        dist = np.abs(x[None, :] - centers_new[:, None])
        dist = np.maximum(dist, 1e-12)

        # u_ij ∝ 1 / (d_ij ^{2/(m-1)}), columns normalized.
        expo = 2.0 / (float(m) - 1.0)
        inv = dist ** (-expo)
        u_new = inv / np.clip(inv.sum(axis=0, keepdims=True), 1e-12, None)

        shift = float(np.linalg.norm(centers_new - centers))
        centers = centers_new
        u = u_new
        if shift < float(tol):
            break

    order = np.argsort(centers)
    return centers[order], u[order]


def _merge_close_centers(centers: np.ndarray, mu_th: float) -> np.ndarray:
    """Merge centers closer than ``mu_th`` watts (Ji-style centroid deviation)."""
    if centers.size == 0:
        return centers
    sorted_c = np.sort(np.asarray(centers, dtype=np.float64).reshape(-1))
    merged = [float(sorted_c[0])]
    for v in sorted_c[1:]:
        if abs(float(v) - merged[-1]) < float(mu_th):
            merged[-1] = 0.5 * (merged[-1] + float(v))
        else:
            merged.append(float(v))
    return np.asarray(merged, dtype=np.float64)


def iterative_fuzzy_cmeans(
    samples: ArrayLike,
    *,
    c_min: int = 2,
    c_max: int = 8,
    mu_th: float = 50.0,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-5,
    rng: np.random.Generator | int | None = 0,
) -> np.ndarray:
    """
    Iterative FCM (Ji et al.): try ``c ∈ [c_min, c_max]``, merge centers
    closer than ``mu_th``, keep the richest set that remains separated.

    Returns:
        centers (N,) sorted watts, ``2 ≤ N ≤ c_max`` when data allow.
    """
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)

    c_lo = max(1, int(c_min))
    c_hi = max(c_lo, int(c_max))
    # Cap by unique magnitudes so IFCM cannot invent fake states.
    n_unique = int(np.unique(np.round(x, decimals=1)).size)
    c_hi = min(c_hi, max(1, n_unique), x.size)

    best = np.array([float(np.mean(x))], dtype=np.float64)
    for c in range(c_lo, c_hi + 1):
        centers, _ = fuzzy_cmeans(
            x, c, m=m, max_iter=max_iter, tol=tol, rng=rng
        )
        merged = _merge_close_centers(centers, mu_th)
        if merged.size >= best.size:
            best = merged
        # If merge collapsed below requested c, further c will over-cluster.
        if merged.size < c:
            break
    return best


# ---------------------------------------------------------------------------
# 2. Schirmer Eq. (9)–(10) snap
# ---------------------------------------------------------------------------

def snap_active_power(
    p_hat: ArrayLike,
    centers: ArrayLike,
    *,
    eps: float,
) -> np.ndarray:
    """
    Schirmer Eq. (10) for one appliance series.

    ``p̂_m = p̂'`` if ``p̂' ≤ ε``; else nearest center ``s^{n_min}`` (Eq. 9).

    Note: this does **not** force OFF noise to exact 0 when ``p̂' ≤ ε`` —
    it leaves near-OFF alone (usage / state-probability invariance).
    """
    p = np.asarray(p_hat, dtype=np.float64)
    s = np.asarray(centers, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return p.copy()

    out = p.copy()
    active = out > float(eps)
    if not np.any(active):
        return out
    vals = out[active]
    # Eq. (9): n_min = argmin_n |p' - s_n|
    idx = np.argmin(np.abs(vals[:, None] - s[None, :]), axis=1)
    out[active] = s[idx]
    return out


# ---------------------------------------------------------------------------
# 3. Multi-appliance post-process block
# ---------------------------------------------------------------------------

@dataclass
class ActiveStateFCMConfig:
    """Per-appliance / global knobs for fit + apply."""

    # ON sample selection when fitting from ground-truth power (watts).
    on_threshold_watts: float = 5.0
    # Eq. (10) margin ε (near-OFF kept). Often ≈ on_threshold.
    eps_watts: float = 5.0
    # IFCM (Ji): search c in [c_min, c_max], merge if |s_i-s_j| < mu_th.
    use_ifcm: bool = True
    n_clusters: int | None = None  # fixed c if use_ifcm=False
    c_min: int = 2
    c_max: int = 8
    mu_th_watts: float = 50.0
    fuzzifier_m: float = 2.0
    max_iter: int = 200
    seed: int = 0
    # Optional per-appliance overrides: {name: {eps_watts, on_threshold_watts, ...}}
    per_appliance: Mapping[str, Mapping[str, float | int | bool]] | None = None


@dataclass
class ActiveStateFCMPostProcess:
    """
    Fit active centers on source ON watts; snap predictions (Schirmer C).

    Example
    -------
    >>> pp = ActiveStateFCMPostProcess(appliances, config)
    >>> pp.fit_from_power(y_train_watts)           # (T, K) source labels
    >>> y_hat_pp = pp.apply(y_pred_watts)          # (T, K) or (N, K)
    """

    appliances: list[str]
    config: ActiveStateFCMConfig
    centers: dict[str, np.ndarray]

    def __init__(
        self,
        appliances: Sequence[str],
        config: ActiveStateFCMConfig | None = None,
    ) -> None:
        self.appliances = [str(a) for a in appliances]
        self.config = config or ActiveStateFCMConfig()
        self.centers = {a: np.zeros(0, dtype=np.float64) for a in self.appliances}

    def _app_cfg(self, name: str) -> ActiveStateFCMConfig:
        base = self.config
        raw = (base.per_appliance or {}).get(name)
        if not raw:
            return base
        # Shallow override of known fields.
        kw = {
            "on_threshold_watts": float(raw.get("on_threshold_watts", base.on_threshold_watts)),
            "eps_watts": float(raw.get("eps_watts", base.eps_watts)),
            "use_ifcm": bool(raw.get("use_ifcm", base.use_ifcm)),
            "n_clusters": (
                None
                if raw.get("n_clusters", base.n_clusters) is None
                else int(raw.get("n_clusters", base.n_clusters))  # type: ignore[arg-type]
            ),
            "c_min": int(raw.get("c_min", base.c_min)),
            "c_max": int(raw.get("c_max", base.c_max)),
            "mu_th_watts": float(raw.get("mu_th_watts", base.mu_th_watts)),
            "fuzzifier_m": float(raw.get("fuzzifier_m", base.fuzzifier_m)),
            "max_iter": int(raw.get("max_iter", base.max_iter)),
            "seed": int(raw.get("seed", base.seed)),
            "per_appliance": base.per_appliance,
        }
        return ActiveStateFCMConfig(**kw)

    def fit_appliance(self, name: str, power_watts: ArrayLike) -> np.ndarray:
        """Fit centers for one appliance from 1-D power (watts)."""
        cfg = self._app_cfg(name)
        p = np.asarray(power_watts, dtype=np.float64).reshape(-1)
        active = p[p > float(cfg.on_threshold_watts)]
        if active.size == 0:
            self.centers[name] = np.zeros(0, dtype=np.float64)
            return self.centers[name]

        if cfg.use_ifcm or cfg.n_clusters is None:
            centers = iterative_fuzzy_cmeans(
                active,
                c_min=cfg.c_min,
                c_max=cfg.c_max,
                mu_th=cfg.mu_th_watts,
                m=cfg.fuzzifier_m,
                max_iter=cfg.max_iter,
                rng=cfg.seed,
            )
        else:
            centers, _ = fuzzy_cmeans(
                active,
                int(cfg.n_clusters),
                m=cfg.fuzzifier_m,
                max_iter=cfg.max_iter,
                rng=cfg.seed,
            )
        self.centers[name] = np.asarray(centers, dtype=np.float64)
        return self.centers[name]

    def fit_from_power(self, power_watts: np.ndarray) -> "ActiveStateFCMPostProcess":
        """
        Fit all appliances.

        Args:
            power_watts: ``(T, K)`` or ``(N, K)`` source ground-truth watts,
                column order = ``self.appliances``.
        """
        arr = np.asarray(power_watts, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] != len(self.appliances):
            raise ValueError(
                f"Expected {len(self.appliances)} columns, got {arr.shape[1]}"
            )
        for i, name in enumerate(self.appliances):
            self.fit_appliance(name, arr[:, i])
        return self

    def apply_appliance(self, name: str, p_hat_watts: ArrayLike) -> np.ndarray:
        cfg = self._app_cfg(name)
        return snap_active_power(
            p_hat_watts, self.centers.get(name, np.zeros(0)), eps=cfg.eps_watts
        )

    def apply(self, power_watts: np.ndarray) -> np.ndarray:
        """Snap each column with Eq. (10). Shape preserved ``(T, K)``."""
        arr = np.asarray(power_watts, dtype=np.float64)
        squeeze = False
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
            squeeze = True
        if arr.shape[1] != len(self.appliances):
            raise ValueError(
                f"Expected {len(self.appliances)} columns, got {arr.shape[1]}"
            )
        out = arr.copy()
        for i, name in enumerate(self.appliances):
            out[:, i] = self.apply_appliance(name, out[:, i])
        return out.reshape(-1) if squeeze else out

    def summary(self) -> dict[str, list[float]]:
        """JSON-friendly center table."""
        return {a: [float(v) for v in self.centers.get(a, [])] for a in self.appliances}


def parse_active_state_fcm_config(block: Mapping[str, object] | None) -> ActiveStateFCMConfig:
    """Read optional yaml ``evaluation.active_state_fcm:`` dict."""
    if not isinstance(block, Mapping):
        return ActiveStateFCMConfig()
    n_clusters = block.get("n_clusters", None)
    return ActiveStateFCMConfig(
        on_threshold_watts=float(block.get("on_threshold_watts", 5.0)),
        eps_watts=float(block.get("eps_watts", block.get("epsilon_watts", 5.0))),
        use_ifcm=bool(block.get("use_ifcm", True)),
        n_clusters=None if n_clusters is None else int(n_clusters),
        c_min=int(block.get("c_min", 2)),
        c_max=int(block.get("c_max", 8)),
        mu_th_watts=float(block.get("mu_th_watts", 50.0)),
        fuzzifier_m=float(block.get("fuzzifier_m", 2.0)),
        max_iter=int(block.get("max_iter", 200)),
        seed=int(block.get("seed", 0)),
        per_appliance=block.get("per_appliance"),  # type: ignore[arg-type]
    )

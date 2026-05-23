"""
Feature Selection — Stage 01: Cleaning + Correlation Filter
===========================================================
Combines former Stage 0 and Stage 1 into one pipeline per appliance.

Stage 0 (cleaning) — target-agnostic:
  - Drop near-constant columns (variance < threshold)
  - Drop excessive NaN/Inf (ratio > threshold)
  - Median-fill remaining invalid values in kept columns

Stage 1 (correlation) — target-driven:
  - For pairs with |Pearson| or |Spearman| > threshold, drop the less
    relevant feature to {appliance}_power; tie-break with DOMAIN_PRIORITY

Outputs per appliance (feature_selection_outputs/{appliance}/)
  stage01_cleaning_report.csv      — Stage 0 metrics and drop reasons
  stage01_target_correlations.csv  — |r| to target (Pearson + Spearman)
  stage01_correlation_pairs.csv    — all redundant pairs (before greedy drop)
  stage01_correlation_report.csv   — greedy drop decisions
  stage01_feature_summary.csv      — final kept/dropped + stage reason
  stage01_explanation.txt          — human-readable summary
  stage01_matrix_pearson_pre.csv / _post.csv — correlation matrices (numeric)
  stage01_matrix_{pearson|spearman}_{pre|post}_filter.csv — numeric matrices
  stage01_corr_matrix_pearson.png  — lower-triangle heatmap, pre | post (600 dpi)
  stage01_corr_matrix_spearman.png — lower-triangle heatmap, pre | post (600 dpi)
  stage01_matrix_README.txt          — how to read the figures

Cross-appliance:
  feature_selection_outputs/stage01_summary.csv
  feature_selection_outputs/cross_appliance/  (fig01–fig08; run stage01_cross_report.py)

Usage
-----
  python feature_selection/stage01_filter.py
  python feature_selection/stage01_filter.py --data_dir path/to/csvs
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

HF_FEATURES = [
    "V_rms", "I_rms", "P_active", "S_apparent", "PF", "Fcv", "Fci",
    "I_skew", "I_kurt", "V_skew", "I_std", "V_std",
    "I1", "V1", "I3", "V3", "I5", "V5", "I7", "V7", "I9", "V9",
    "I11", "V11", "I13", "V13", "I15", "V15",
    "IH", "VH", "THDI", "THDV",
    "I_BP_low", "I_BP_mid", "I_BP_high", "V_BP_low",
    "I_spec_entropy",
    "I_env_0", "I_env_1", "I_env_2", "I_env_3", "I_env_4",
    "I_env_5", "I_env_6", "I_env_7",
    "DWT_E0", "DWT_E1", "DWT_E2", "DWT_E3", "DWT_E4",
]

FEATURE_DOMAIN = {
    **{f: "time_domain" for f in ["V_rms", "I_rms", "P_active", "S_apparent", "PF", "Fcv", "Fci"]},
    **{f: "shape_statistics" for f in ["I_skew", "I_kurt", "V_skew", "I_std", "V_std"]},
    **{
        f: "harmonics"
        for f in [
            "I1", "V1", "I3", "V3", "I5", "V5", "I7", "V7", "I9", "V9",
            "I11", "V11", "I13", "V13", "I15", "V15",
        ]
    },
    **{f: "distortion" for f in ["IH", "VH", "THDI", "THDV"]},
    **{f: "band_power" for f in ["I_BP_low", "I_BP_mid", "I_BP_high", "V_BP_low"]},
    **{f: "spectral_descriptors" for f in ["I_spec_entropy"]},
    **{f: "spectral_envelope" for f in [f"I_env_{i}" for i in range(8)]},
    **{f: "wavelet" for f in [f"DWT_E{i}" for i in range(5)]},
}

NON_FEATURE_COLS = {"readable_time", "aggregate", "on_off"}

NEAR_CONSTANT_VAR_THRESHOLD = 1e-8
MAX_INVALID_RATIO = 0.05
CORRELATION_THRESHOLD = 0.95
TARGET_CORR_TIE_EPS = 0.01

DOMAIN_PRIORITY = {
    "time_domain": 10,
    "distortion": 9,
    "harmonics": 8,
    "wavelet": 7,
    "band_power": 6,
    "shape_statistics": 5,
    "spectral_descriptors": 4,
    "spectral_envelope": 3,
}

# Tick label colors by feature domain (academic legend-style)
DOMAIN_COLORS = {
    "time_domain": "#1f77b4",
    "shape_statistics": "#ff7f0e",
    "harmonics": "#2ca02c",
    "distortion": "#d62728",
    "band_power": "#9467bd",
    "spectral_descriptors": "#8c564b",
    "spectral_envelope": "#e377c2",
    "wavelet": "#17becf",
    "unknown": "#7f7f7f",
}


def _feature_priority(feat: str) -> int:
    return DOMAIN_PRIORITY.get(FEATURE_DOMAIN.get(feat, "unknown"), 0)


def get_hf_columns(df: pd.DataFrame) -> list[str]:
    exclude = set(NON_FEATURE_COLS)
    for col in df.columns:
        if col.endswith("_power"):
            exclude.add(col)
    return [c for c in df.columns if c not in exclude]


def _sort_features_by_domain(feats: list[str]) -> list[str]:
    """Cluster features by domain (high DOMAIN_PRIORITY first) then HF_FEATURES order."""
    hf_idx = {f: i for i, f in enumerate(HF_FEATURES)}

    def key(f: str) -> tuple:
        dom = FEATURE_DOMAIN.get(f, "unknown")
        return (-DOMAIN_PRIORITY.get(dom, 0), hf_idx.get(f, 999), f)

    return sorted(feats, key=key)


def _correlation_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    """
    Discrete diverging map with explicit bands at ±0.95 (Stage 1 redundancy threshold).
    """
    bounds = [-1.0, -0.95, -0.5, 0.0, 0.5, 0.95, 1.0]
    colors = [
        "#2166ac",  # strong negative
        "#67a9cf",  # near -0.95
        "#d1e5f0",  # weak negative
        "#f7f7f7",  # near zero
        "#fddbc7",  # weak positive
        "#ef8a62",  # near +0.95
        "#b2182b",  # strong positive
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _cell_size_inches(n: int) -> float:
    if n <= 34:
        return 0.55
    if n <= 50:
        return 0.45
    return 0.36


def _tick_fontsize(n: int) -> int:
    if n <= 34:
        return 16
    if n <= 50:
        return 14
    return 12


def _draw_corr_panel(
    ax: plt.Axes,
    corr: pd.DataFrame,
    panel_title: str,
    corr_threshold: float,
    highlight_dropped: set[str],
    show_cbar: bool,
) -> None:
    """Lower-triangle correlation heatmap (symmetric matrix, no duplicate upper half)."""
    n = len(corr.columns)
    cmap, norm = _correlation_cmap()
    vals = corr.values.astype(float)
    mask_upper = np.triu(np.ones_like(vals, dtype=bool), k=1)

    hm = sns.heatmap(
        corr,
        mask=mask_upper,
        cmap=cmap,
        norm=norm,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.25,
        linecolor="#ffffff",
        cbar=show_cbar,
        cbar_kws={
            "label": "Correlation",
            "shrink": 0.75,
            "ticks": [-1, -0.95, 0, 0.95, 1],
            "aspect": 30,
        } if show_cbar else None,
        ax=ax,
    )

    if show_cbar and hm.collections[0].colorbar is not None:
        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=25, width=1.5, length=6)
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_fontweight("bold")
        cbar.set_label("Correlation", fontsize=26, fontweight="bold", labelpad=12)

    # Black border on redundant pairs in lower triangle only
    for i in range(n):
        for j in range(n):
            if j >= i:
                continue
            if abs(vals[i, j]) >= corr_threshold:
                ax.add_patch(
                    Rectangle(
                        (j, i), 1, 1, fill=False, edgecolor="#111111",
                        linewidth=2.0, zorder=10,
                    )
                )

    tick_feats = list(corr.columns)
    tick_fs = _tick_fontsize(n)
    tick_colors = [
        "#b2182b" if f in highlight_dropped
        else DOMAIN_COLORS.get(FEATURE_DOMAIN.get(f, "unknown"), "#111111")
        for f in tick_feats
    ]

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(tick_feats, rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticklabels(tick_feats, rotation=0, fontsize=tick_fs)
    for lbl, color in zip(ax.get_xticklabels(), tick_colors):
        lbl.set_color(color)
        lbl.set_fontweight("bold")
    for lbl, color in zip(ax.get_yticklabels(), tick_colors):
        lbl.set_color(color)
        lbl.set_fontweight("bold")

    ax.set_title(panel_title, fontsize=28, fontweight="bold", pad=10)
    ax.set_aspect("equal", adjustable="box")


def _plot_combined_correlation_figure(
    corr_pre: pd.DataFrame,
    corr_post: pd.DataFrame,
    output_path: str,
    appliance: str,
    method: str,
    corr_threshold: float,
    dropped_feats: set[str],
) -> str:
    """Side-by-side pre/post lower-triangle heatmaps (single PNG)."""
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.weight": "bold",
    })

    n_pre, n_post = len(corr_pre.columns), len(corr_post.columns)
    cell = max(_cell_size_inches(n_pre), _cell_size_inches(n_post))
    panel_side = max(n_pre, n_post) * cell
    fig_h = panel_side + 1.2
    fig_w = panel_side * 2 + 1.8
    fig, axes = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.18},
    )

    app = appliance.replace("_", " ").title()
    meth = method.capitalize()
    _draw_corr_panel(
        axes[0], corr_pre,
        f"Before filter (n={n_pre})",
        corr_threshold, dropped_feats, show_cbar=False,
    )
    _draw_corr_panel(
        axes[1], corr_post,
        f"After filter (n={n_post})",
        corr_threshold, set(), show_cbar=True,
    )

    fig.suptitle(
        f"{app} — {meth} correlation (lower triangle)\n"
        f"Black box = redundant pair (|r| ≥ {corr_threshold})",
        fontsize=80, fontweight="bold", y=1.02,
    )

    domain_order = [
        "time_domain", "distortion", "harmonics", "wavelet", "band_power",
        "shape_statistics", "spectral_descriptors", "spectral_envelope",
    ]
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", linestyle="",
                   markerfacecolor=DOMAIN_COLORS[d], markersize=12,
                   label=d.replace("_", " "))
        for d in domain_order
    ]
    handles.append(
        plt.Line2D([0], [0], color="w", marker="s", markerfacecolor="#b2182b",
                   markersize=12, label="removed in filter"),
    )
    handles.append(
        plt.Line2D([0], [0], color="#111111", lw=2.5, label=f"|r| ≥ {corr_threshold}"),
    )
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=5, frameon=False, fontsize=60,
        prop={"weight": "bold"},
    )

    fig.subplots_adjust(left=0.10, right=0.94, top=0.88, bottom=0.18, wspace=0.22)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _write_matrix_readme(output_dir: str, corr_threshold: float) -> str:
    path = os.path.join(output_dir, "stage01_matrix_README.txt")
    text = f"""How to read Stage 01 correlation figures
==========================================

Files (2 PNG per appliance, not 4):
  stage01_corr_matrix_pearson.png   — linear correlation, before | after filter
  stage01_corr_matrix_spearman.png  — rank (monotonic) correlation, before | after

CSV tables (numeric values for thesis tables):
  stage01_matrix_pearson_pre_filter.csv / _post_filter.csv
  stage01_matrix_spearman_pre_filter.csv / _post_filter.csv

Why two methods?
  Pearson: linear co-movement (good for power-like features).
  Spearman: monotonic co-movement (curve-shaped but still synchronized).
  Stage 1 flags a pair redundant if EITHER |Pearson| or |Spearman| > {corr_threshold}.

Why lower triangle (not full square)?
  The matrix is symmetric: cell (A,B) = cell (B,A). Only the lower triangle is
  drawn so the figure does not repeat the same information twice.

How to read colors:
  Blue = negative correlation, white ≈ 0, red = positive correlation.
  Color bands change at ±0.5 and ±{corr_threshold} on the colorbar.
  Black box around a cell = |r| ≥ {corr_threshold} → redundant pair in Stage 1.

Left panel = all features after cleaning (usually 50).
Right panel = features kept after correlation filter (usually 34).
Red feature names (left panel) = removed in Stage 1.

Why only 2 PNG files now?
  Previously: 4 separate images (pearson/spearman × pre/post) were repetitive.
  Now: one image per method with before/after side-by-side for compact papers.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _save_correlation_matrix_artifacts(
    df_pre: pd.DataFrame,
    feats_pre: list[str],
    df_post: pd.DataFrame,
    feats_post: list[str],
    output_dir: str,
    appliance: str,
    corr_threshold: float,
    dropped_feats: list[str] | None = None,
) -> list[str]:
    """Save CSV matrices + 2 combined PNG figures (Pearson and Spearman)."""
    feats_pre = _sort_features_by_domain([f for f in feats_pre if f in df_pre.columns])
    feats_post = _sort_features_by_domain([f for f in feats_post if f in df_post.columns])
    if len(feats_pre) < 2 or len(feats_post) < 2:
        return []

    os.makedirs(output_dir, exist_ok=True)
    dropped = set(dropped_feats or ())
    X_pre = df_pre[feats_pre].replace([np.inf, -np.inf], np.nan)
    X_post = df_post[feats_post].replace([np.inf, -np.inf], np.nan)
    saved_paths: list[str] = []

    for method in ("pearson", "spearman"):
        corr_pre = X_pre.corr(method=method)
        corr_post = X_post.corr(method=method)

        for label, corr in (("pre_filter", corr_pre), ("post_filter", corr_post)):
            csv_path = os.path.join(output_dir, f"stage01_matrix_{method}_{label}.csv")
            corr.to_csv(csv_path, float_format="%.8f")
            saved_paths.append(csv_path)

        png_path = os.path.join(output_dir, f"stage01_corr_matrix_{method}.png")
        saved_paths.append(
            _plot_combined_correlation_figure(
                corr_pre, corr_post, png_path, appliance, method,
                corr_threshold, dropped,
            )
        )

    saved_paths.append(_write_matrix_readme(output_dir, corr_threshold))
    return saved_paths


def _safe_corrwith(X: pd.DataFrame, y: pd.Series, method: str) -> pd.Series:
    """Column-wise correlation with target; NaN where undefined."""
    out = {}
    y_clean = y.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        x = X[col].replace([np.inf, -np.inf], np.nan)
        if method == "pearson":
            r = x.corr(y_clean)
        else:
            r = x.corr(y_clean, method="spearman")
        out[col] = abs(float(r)) if pd.notna(r) else np.nan
    return pd.Series(out)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: cleaning
# ─────────────────────────────────────────────────────────────────────────────
def run_cleaning(
    df: pd.DataFrame,
    appliance: str,
    var_threshold: float,
    invalid_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    feat_cols = get_hf_columns(df)
    records = []
    dropped: list[str] = []
    kept: list[str] = []

    print(f"\n[Stage 01 — Cleaning] {appliance.upper()}")
    print(f"  rows={len(df)}  hf_features={len(feat_cols)}")
    print(f"  var_threshold={var_threshold:.0e}  invalid_threshold={invalid_threshold:.0%}")
    print("  " + "-" * 72)

    for feat in feat_cols:
        col = df[feat]
        n_total = len(col)
        n_nan = int(col.isna().sum())
        n_inf = int(np.isinf(col.replace([np.nan], 0)).sum())
        n_invalid = n_nan + n_inf
        invalid_ratio = n_invalid / n_total if n_total else 0.0

        finite_vals = col.replace([np.inf, -np.inf], np.nan).dropna()
        variance = float(finite_vals.var()) if len(finite_vals) > 1 else 0.0

        if invalid_ratio > invalid_threshold:
            status, reason = "dropped", (
                f"invalid_ratio={invalid_ratio:.6f} > {invalid_threshold} "
                f"(n_nan={n_nan}, n_inf={n_inf}, n_total={n_total})"
            )
            dropped.append(feat)
        elif variance < var_threshold:
            status, reason = "dropped", (
                f"variance={variance:.6e} < {var_threshold:.0e} (near-constant)"
            )
            dropped.append(feat)
        else:
            status, reason = "kept", "passed variance and invalid checks"
            kept.append(feat)

        records.append({
            "feature": feat,
            "domain": FEATURE_DOMAIN.get(feat, "unknown"),
            "n_total": n_total,
            "n_nan": n_nan,
            "n_inf": n_inf,
            "n_invalid": n_invalid,
            "invalid_ratio": round(invalid_ratio, 8),
            "variance": variance,
            "var_threshold": var_threshold,
            "invalid_threshold": invalid_threshold,
            "status": status,
            "reason": reason,
        })

        tag = "DROP" if status == "dropped" else "keep"
        print(
            f"  [{tag}] {feat:<20} var={variance:12.4e}  "
            f"invalid={invalid_ratio:.6f} ({n_invalid}/{n_total})"
        )
        print(f"         -> {reason}")

    df_clean = df.copy()
    fill_log = []
    for feat in kept:
        col = df_clean[feat]
        has_nan = col.isna().any()
        has_inf = np.isinf(col.replace([np.nan], 0)).any()
        if has_nan or has_inf:
            col_finite = col.replace([np.inf, -np.inf], np.nan)
            median_val = col_finite.median()
            n_filled = int(col_finite.isna().sum())
            df_clean[feat] = col_finite.fillna(median_val)
            fill_log.append(f"    filled {feat}: {n_filled} values -> median={median_val:.6g}")

    df_clean.drop(columns=dropped, inplace=True, errors="ignore")

    if fill_log:
        print("\n  Median fills applied:")
        for line in fill_log:
            print(line)

    print(f"\n  Cleaning result: {len(feat_cols)} -> kept {len(kept)}, dropped {len(dropped)}")
    if dropped:
        print(f"  Dropped: {dropped}")

    report_df = pd.DataFrame(records)
    return df_clean, report_df, kept, dropped


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: correlation filter
# ─────────────────────────────────────────────────────────────────────────────
def run_correlation_filter(
    df_clean: pd.DataFrame,
    kept_after_cleaning: list[str],
    appliance: str,
    corr_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    target_col = f"{appliance}_power"
    feats = [f for f in kept_after_cleaning if f in df_clean.columns]
    X = df_clean[feats].copy()

    print(f"\n[Stage 01 — Correlation] {appliance.upper()}")
    print(f"  features_entering={len(feats)}  corr_threshold={corr_threshold}")
    print(f"  target={target_col}")
    print("  " + "-" * 72)

    if target_col in df_clean.columns:
        y = df_clean[target_col]
        target_pearson = _safe_corrwith(X, y, "pearson")
        target_spearman = _safe_corrwith(X, y, "spearman")
    else:
        print(f"  [WARNING] Target '{target_col}' missing — tie-break uses domain priority only")
        target_pearson = pd.Series(0.0, index=feats)
        target_spearman = pd.Series(0.0, index=feats)

    target_corr_df = pd.DataFrame({
        "feature": feats,
        "domain": [FEATURE_DOMAIN.get(f, "unknown") for f in feats],
        "domain_priority": [_feature_priority(f) for f in feats],
        "target_pearson_abs": [round(float(target_pearson.get(f, 0) or 0), 8) for f in feats],
        "target_spearman_abs": [round(float(target_spearman.get(f, 0) or 0), 8) for f in feats],
    }).sort_values("target_pearson_abs", ascending=False)

    print("\n  Target correlations (|r| to {0}):".format(target_col))
    print(target_corr_df.to_string(index=False))

    pearson_matrix = X.corr(method="pearson").abs()
    spearman_matrix = X.corr(method="spearman").abs()

    pair_records = []
    feat_list = feats
    for i in range(len(feat_list)):
        for j in range(i + 1, len(feat_list)):
            fi, fj = feat_list[i], feat_list[j]
            r_p = float(pearson_matrix.loc[fi, fj])
            r_s = float(spearman_matrix.loc[fi, fj])
            if r_p > corr_threshold or r_s > corr_threshold:
                pair_records.append({
                    "feature_a": fi,
                    "feature_b": fj,
                    "domain_a": FEATURE_DOMAIN.get(fi, "unknown"),
                    "domain_b": FEATURE_DOMAIN.get(fj, "unknown"),
                    "pearson_abs": round(r_p, 8),
                    "spearman_abs": round(r_s, 8),
                    "max_abs_corr": round(max(r_p, r_s), 8),
                    "above_threshold": True,
                })

    pairs_df = pd.DataFrame(pair_records)
    if len(pairs_df):
        pairs_df = pairs_df.sort_values("max_abs_corr", ascending=False)

    print(f"\n  Redundant pairs (|Pearson| or |Spearman| > {corr_threshold}): {len(pairs_df)}")
    if len(pairs_df):
        print(pairs_df.to_string(index=False))

    # Greedy elimination: process pairs by max(|r|) descending
    greedy_pairs = sorted(
        pair_records,
        key=lambda r: r["max_abs_corr"],
        reverse=True,
    )

    alive = set(feats)
    drop_records = []
    dropped: list[str] = []

    for rec in greedy_pairs:
        fi, fj = rec["feature_a"], rec["feature_b"]
        if fi not in alive or fj not in alive:
            continue

        tc_i = float(target_pearson.get(fi, 0) or 0)
        tc_j = float(target_pearson.get(fj, 0) or 0)
        ts_i = float(target_spearman.get(fi, 0) or 0)
        ts_j = float(target_spearman.get(fj, 0) or 0)

        if abs(tc_i - tc_j) >= TARGET_CORR_TIE_EPS:
            if tc_i >= tc_j:
                keep_feat, drop_feat = fi, fj
                reason = (
                    f"higher |Pearson| to target ({tc_i:.6f} vs {tc_j:.6f}); "
                    f"|Spearman| {ts_i:.6f} vs {ts_j:.6f}"
                )
            else:
                keep_feat, drop_feat = fj, fi
                reason = (
                    f"higher |Pearson| to target ({tc_j:.6f} vs {tc_i:.6f}); "
                    f"|Spearman| {ts_j:.6f} vs {ts_i:.6f}"
                )
        else:
            pri_i, pri_j = _feature_priority(fi), _feature_priority(fj)
            if pri_i >= pri_j:
                keep_feat, drop_feat = fi, fj
                reason = (
                    f"target |Pearson| tied within {TARGET_CORR_TIE_EPS} "
                    f"({tc_i:.6f} vs {tc_j:.6f}); "
                    f"domain_priority {FEATURE_DOMAIN.get(fi)}({pri_i}) >= "
                    f"{FEATURE_DOMAIN.get(fj)}({pri_j})"
                )
            else:
                keep_feat, drop_feat = fj, fi
                reason = (
                    f"target |Pearson| tied within {TARGET_CORR_TIE_EPS} "
                    f"({tc_j:.6f} vs {tc_i:.6f}); "
                    f"domain_priority {FEATURE_DOMAIN.get(fj)}({pri_j}) > "
                    f"{FEATURE_DOMAIN.get(fi)}({pri_i})"
                )

        alive.discard(drop_feat)
        dropped.append(drop_feat)

        drop_records.append({
            "dropped_feature": drop_feat,
            "kept_feature": keep_feat,
            "pearson_abs_pair": rec["pearson_abs"],
            "spearman_abs_pair": rec["spearman_abs"],
            "max_abs_corr_pair": rec["max_abs_corr"],
            "target_pearson_dropped": round(float(target_pearson.get(drop_feat, 0) or 0), 8),
            "target_pearson_kept": round(float(target_pearson.get(keep_feat, 0) or 0), 8),
            "target_spearman_dropped": round(float(target_spearman.get(drop_feat, 0) or 0), 8),
            "target_spearman_kept": round(float(target_spearman.get(keep_feat, 0) or 0), 8),
            "domain_dropped": FEATURE_DOMAIN.get(drop_feat, "unknown"),
            "domain_kept": FEATURE_DOMAIN.get(keep_feat, "unknown"),
            "reason": reason,
        })

        print(
            f"\n  DROP {drop_feat}  (keep {keep_feat})"
            f"\n    pair |Pearson|={rec['pearson_abs']:.6f}  |Spearman|={rec['spearman_abs']:.6f}"
            f"\n    {reason}"
        )

    kept = [f for f in feats if f in alive]
    corr_report_df = pd.DataFrame(drop_records)

    print(f"\n  Correlation result: {len(feats)} -> kept {len(kept)}, dropped {len(dropped)}")
    if dropped:
        print(f"  Dropped: {dropped}")

    df_out = df_clean.drop(columns=dropped, errors="ignore")
    return df_out, target_corr_df, pairs_df, corr_report_df, kept, dropped


def _write_explanation(
    path: str,
    appliance: str,
    n_rows: int,
    cleaning_report: pd.DataFrame,
    target_corr_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    corr_report: pd.DataFrame,
    kept_final: list[str],
    dropped_clean: list[str],
    dropped_corr: list[str],
    args,
) -> None:
    lines = [
        f"Stage 01 Feature Filter — {appliance}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Source rows: {n_rows}",
        "",
        "=== Thresholds ===",
        f"  variance < {args.var_threshold:.0e}  -> drop (near-constant)",
        f"  invalid_ratio > {args.invalid_threshold:.0%}  -> drop (NaN/Inf)",
        f"  |Pearson| or |Spearman| > {args.corr_threshold}  -> redundant pair",
        f"  target |Pearson| tie epsilon: {TARGET_CORR_TIE_EPS}",
        "",
        "=== Stage 0: Cleaning (why each feature was dropped) ===",
    ]

    for _, row in cleaning_report.iterrows():
        lines.append(
            f"  [{row['status'].upper():6}] {row['feature']:<20} "
            f"domain={row['domain']}"
        )
        lines.append(f"           {row['reason']}")

    lines.extend(["", "=== Target relevance (all kept-after-cleaning features) ==="])
    if len(target_corr_df):
        lines.append(target_corr_df.to_string(index=False))
    else:
        lines.append("  (none)")

    lines.extend([
        "",
        f"=== Stage 1: Correlation — {len(pairs_df)} redundant pairs found ===",
    ])
    if len(pairs_df):
        lines.append(pairs_df.to_string(index=False))
    else:
        lines.append("  (no pairs above threshold)")

    lines.extend(["", "=== Greedy drop decisions ==="])
    if len(corr_report):
        for _, row in corr_report.iterrows():
            lines.append(
                f"  DROP {row['dropped_feature']}  keep {row['kept_feature']}"
            )
            lines.append(
                f"       pair |r|: pearson={row['pearson_abs_pair']:.6f}  "
                f"spearman={row['spearman_abs_pair']:.6f}"
            )
            lines.append(f"       {row['reason']}")
    else:
        lines.append("  (no features dropped in correlation stage)")

    lines.extend([
        "",
        "=== Final feature set ===",
        f"  After cleaning : {len(cleaning_report[cleaning_report['status'] == 'kept'])}",
        f"  After correlation drops: {len(kept_final)}",
        f"  Dropped in cleaning: {dropped_clean or 'none'}",
        f"  Dropped in correlation: {dropped_corr or 'none'}",
        f"  Kept: {kept_final}",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _build_feature_summary(
    all_features: list[str],
    dropped_clean: list[str],
    dropped_corr: list[str],
    kept_final: list[str],
) -> pd.DataFrame:
    rows = []
    for feat in all_features:
        if feat in dropped_clean:
            final_status, stage, reason = "dropped", "cleaning", "failed Stage 0 checks"
        elif feat in dropped_corr:
            final_status, stage, reason = "dropped", "correlation", "redundant with higher-target feature"
        elif feat in kept_final:
            final_status, stage, reason = "kept", "passed", "retained through Stage 01"
        else:
            final_status, stage, reason = "missing", "n/a", "not in CSV"
        rows.append({
            "feature": feat,
            "domain": FEATURE_DOMAIN.get(feat, "unknown"),
            "final_status": final_status,
            "dropped_at_stage": stage,
            "reason_summary": reason,
        })
    return pd.DataFrame(rows)


def run_one_appliance(
    csv_path: str,
    appliance: str,
    output_dir: str,
    args,
) -> dict | None:
    if not os.path.exists(csv_path):
        print(f"\n[WARNING] CSV not found: {csv_path} — skipping {appliance}")
        return None

    df = pd.read_csv(csv_path)
    all_features = get_hf_columns(df)
    os.makedirs(output_dir, exist_ok=True)

    df_clean, cleaning_report, kept_s0, dropped_s0 = run_cleaning(
        df, appliance, args.var_threshold, args.invalid_threshold,
    )

    df_out, target_corr_df, pairs_df, corr_report, kept_final, dropped_s1 = run_correlation_filter(
        df_clean, kept_s0, appliance, args.corr_threshold,
    )

    matrix_paths: list[str] = []
    if not getattr(args, "no_plots", False):
        matrix_paths.extend(
            _save_correlation_matrix_artifacts(
                df_pre=df_clean,
                feats_pre=kept_s0,
                df_post=df_out,
                feats_post=kept_final,
                output_dir=output_dir,
                appliance=appliance,
                corr_threshold=args.corr_threshold,
                dropped_feats=dropped_s1,
            )
        )

    summary_df = _build_feature_summary(all_features, dropped_s0, dropped_s1, kept_final)

    cleaning_path = os.path.join(output_dir, "stage01_cleaning_report.csv")
    target_corr_path = os.path.join(output_dir, "stage01_target_correlations.csv")
    pairs_path = os.path.join(output_dir, "stage01_correlation_pairs.csv")
    corr_path = os.path.join(output_dir, "stage01_correlation_report.csv")
    summary_path = os.path.join(output_dir, "stage01_feature_summary.csv")
    explain_path = os.path.join(output_dir, "stage01_explanation.txt")

    cleaning_report.to_csv(cleaning_path, index=False)
    target_corr_df.to_csv(target_corr_path, index=False)
    pairs_df.to_csv(pairs_path, index=False)
    corr_report.to_csv(corr_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    _write_explanation(
        explain_path, appliance, len(df), cleaning_report,
        target_corr_df, pairs_df, corr_report, kept_final,
        dropped_s0, dropped_s1, args,
    )

    print(f"\n  Reports saved under {output_dir}:")
    report_paths = [
        cleaning_path, target_corr_path, pairs_path, corr_path,
        summary_path, explain_path, *matrix_paths,
    ]
    for p in report_paths:
        print(f"    {p}")

    return {
        "appliance": appliance,
        "n_rows": len(df),
        "n_original": len(all_features),
        "kept_final": kept_final,
        "dropped_clean": dropped_s0,
        "dropped_corr": dropped_s1,
        "summary_df": summary_df,
    }


def run_all_appliances(args) -> dict:
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    output_root = os.path.join(PROJECT_ROOT, args.output_dir)

    print("\n" + "=" * 72)
    print("  FEATURE SELECTION — STAGE 01: CLEANING + CORRELATION FILTER")
    print("=" * 72)
    print(f"  Data dir    : {data_dir}")
    print(f"  Output dir  : {output_root}")
    print(f"  Appliances  : {args.appliances}")
    print(f"  House/week  : {args.house} / {args.week}")
    print("=" * 72)

    results = {}
    cross_records = []

    for appliance in args.appliances:
        csv_name = f"{appliance}_{args.house}_{args.week}.csv"
        csv_path = os.path.join(data_dir, csv_name)
        app_output = os.path.join(output_root, appliance)

        res = run_one_appliance(csv_path, appliance, app_output, args)
        if res is None:
            continue

        results[appliance] = res
        for _, row in res["summary_df"].iterrows():
            cross_records.append({
                "appliance": appliance,
                "feature": row["feature"],
                "domain": row["domain"],
                "final_status": row["final_status"],
                "dropped_at_stage": row["dropped_at_stage"],
            })

    if not results:
        print("\nNo appliances processed.")
        return {}

    cross_df = pd.DataFrame(cross_records)
    pivot = cross_df.pivot_table(
        index=["feature", "domain"],
        columns="appliance",
        values="final_status",
        aggfunc="first",
    ).reset_index()

    app_cols = [c for c in pivot.columns if c not in ("feature", "domain")]
    pivot["n_kept"] = (pivot[app_cols] == "kept").sum(axis=1)
    pivot["n_dropped"] = (pivot[app_cols] == "dropped").sum(axis=1)
    pivot["globally_kept"] = pivot["n_kept"] == len(app_cols)

    summary_path = os.path.join(output_root, "stage01_summary.csv")
    pivot.to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("  STAGE 01 CROSS-APPLIANCE SUMMARY")
    print("=" * 72)
    print("\n  Per-appliance counts:")
    for app, res in results.items():
        print(
            f"    {app:<16}  original={res['n_original']:<3}  "
            f"final_kept={len(res['kept_final']):<3}  "
            f"dropped_clean={len(res['dropped_clean']):<2}  "
            f"dropped_corr={len(res['dropped_corr'])}"
        )

    global_kept = pivot[pivot["globally_kept"]]["feature"].tolist()
    print(f"\n  Kept in ALL appliances ({len(global_kept)}):")
    for f in global_kept:
        print(f"    {f}  [{FEATURE_DOMAIN.get(f, '?')}]")

    print(f"\n  Cross-appliance table: {summary_path}")
    print("=" * 72)

    return results


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Stage 01: HF feature cleaning + correlation filter"
    )
    parser.add_argument(
        "--data_dir",
        default="dataset_preprocess/high_frequency_data_extract/output",
        help="Folder with fused appliance CSVs (relative to project root)",
    )
    parser.add_argument(
        "--output_dir",
        default="feature_selection_outputs",
        help="Root folder for outputs (relative to project root)",
    )
    parser.add_argument(
        "--appliances",
        nargs="+",
        default=["kettle", "fridge", "microwave", "dishwasher", "washingmachine"],
    )
    parser.add_argument("--house", default="house2")
    parser.add_argument("--week", default="wk30")
    parser.add_argument("--var_threshold", type=float, default=NEAR_CONSTANT_VAR_THRESHOLD)
    parser.add_argument("--invalid_threshold", type=float, default=MAX_INVALID_RATIO)
    parser.add_argument("--corr_threshold", type=float, default=CORRELATION_THRESHOLD)
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip correlation matrix CSV/PNG export",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_all_appliances(get_arguments())

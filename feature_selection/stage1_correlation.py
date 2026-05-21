"""
Feature Selection — Stage 1: Correlation & Collinearity Filtering
=================================================================
Removes redundant HF features that are highly correlated with each other.

Logic
-----
  For every pair (A, B) where abs(Pearson) > threshold:
    - Keep the feature with higher absolute Pearson correlation to the
      regression target ({appliance}_power).
    - If target relevance is very similar (<0.01 difference), keep the
      more physically interpretable feature based on DOMAIN_PRIORITY.
    - Drop the other one.

  Uses a greedy approach:
    Sort all pairs by correlation descending.
    For each pair, if both features are still "alive", drop the weaker one.

Does NOT use on_off or aggregate as targets — only {appliance}_power is used
as the tiebreaker. The main decision (drop vs keep) is purely feature-vs-feature.

Output per appliance
--------------------
  feature_selection_outputs/{appliance}/stage1_correlation_report.csv
      dropped_feature | kept_feature | pearson_corr | target_corr_dropped
      | target_corr_kept | reason

  feature_selection_outputs/stage1_summary.csv
      Cross-appliance view of which features survived correlation filtering.

Usage
-----
  # Run standalone (calls Stage 0 internally):
  python feature_selection/stage1_correlation.py

  # Or import and call from another script:
  from feature_selection.stage1_correlation import run_stage1
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ── allow running from project root or feature_selection/ ────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from stage0_cleaning import (
    FEATURE_DOMAIN,
    MAX_INVALID_RATIO,
    NEAR_CONSTANT_VAR_THRESHOLD,
    get_hf_columns,
    run_stage0,
)
from stage0_cleaning import (
    run_all_appliances as stage0_run_all,
)

# ── thresholds ────────────────────────────────────────────────────────────────
CORRELATION_THRESHOLD = 0.95  # abs(Pearson) > this → treat pair as redundant

# Domain priority for tiebreaking (higher = prefer to keep)
# Physically meaningful features are preferred over derived ones
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


def _feature_priority(feat: str) -> int:
    domain = FEATURE_DOMAIN.get(feat, "unknown")
    return DOMAIN_PRIORITY.get(domain, 0)


# ─────────────────────────────────────────────────────────────────────────────
def run_stage1(
    df_clean: pd.DataFrame,
    kept_after_stage0: list[str],
    appliance: str,
    output_dir: str,
    corr_threshold: float = CORRELATION_THRESHOLD,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Run Stage 1 correlation filtering on one appliance DataFrame.

    Parameters
    ----------
    df_clean          : cleaned DataFrame from Stage 0 (bad features already dropped)
    kept_after_stage0 : list of HF feature names that survived Stage 0
    appliance         : appliance name string (e.g. 'kettle')
    output_dir        : folder to save the report CSV
    corr_threshold    : abs(Pearson) above this triggers redundancy check

    Returns
    -------
    df_out      : DataFrame with redundant features dropped
    kept        : list of feature names that survived Stage 1
    dropped     : list of feature names dropped in Stage 1
    """
    target_col = f"{appliance}_power"

    print(
        f"\n[Stage 1] {appliance.upper()} — {len(kept_after_stage0)} features entering"
    )
    print(f"          corr_threshold={corr_threshold}")
    print("          " + "─" * 56)

    feats = [f for f in kept_after_stage0 if f in df_clean.columns]
    X = df_clean[feats].copy()

    # ── compute target correlation (tiebreaker only) ──────────────────────────
    if target_col in df_clean.columns:
        target_corr = X.corrwith(df_clean[target_col]).abs()
    else:
        print(
            f"  ⚠️  Target column '{target_col}' not found — using domain priority only for tiebreaking"
        )
        target_corr = pd.Series(0.0, index=feats)

    # ── compute full Pearson correlation matrix ───────────────────────────────
    corr_matrix = X.corr().abs()

    # ── collect all redundant pairs above threshold ───────────────────────────
    # Upper triangle only to avoid duplicates
    pairs = []
    feat_list = feats
    for i in range(len(feat_list)):
        for j in range(i + 1, len(feat_list)):
            fi, fj = feat_list[i], feat_list[j]
            c = corr_matrix.loc[fi, fj]
            if c > corr_threshold:
                pairs.append((c, fi, fj))

    # Sort by correlation descending (most redundant pairs first)
    pairs.sort(key=lambda x: x[0], reverse=True)

    print(f"  Found {len(pairs)} pairs with |Pearson| > {corr_threshold}")

    # ── greedy elimination ────────────────────────────────────────────────────
    alive = set(feats)  # features still in the running
    records = []
    dropped = []

    for corr_val, fi, fj in pairs:
        # Skip if either was already dropped by a previous pair
        if fi not in alive or fj not in alive:
            continue

        # Tiebreaker 1: target correlation
        tc_i = float(target_corr.get(fi, 0.0))
        tc_j = float(target_corr.get(fj, 0.0))

        if abs(tc_i - tc_j) >= 0.01:
            # Clear winner: keep the one more correlated with target
            if tc_i >= tc_j:
                keep_feat, drop_feat = fi, fj
                reason = f"higher target_corr ({tc_i:.4f} vs {tc_j:.4f})"
            else:
                keep_feat, drop_feat = fj, fi
                reason = f"higher target_corr ({tc_j:.4f} vs {tc_i:.4f})"
        else:
            # Target correlations too similar → use domain priority
            pri_i = _feature_priority(fi)
            pri_j = _feature_priority(fj)
            if pri_i >= pri_j:
                keep_feat, drop_feat = fi, fj
                reason = f"domain_priority ({FEATURE_DOMAIN.get(fi, '?')} >= {FEATURE_DOMAIN.get(fj, '?')})"
            else:
                keep_feat, drop_feat = fj, fi
                reason = f"domain_priority ({FEATURE_DOMAIN.get(fj, '?')} > {FEATURE_DOMAIN.get(fi, '?')})"

        alive.discard(drop_feat)
        dropped.append(drop_feat)

        records.append(
            {
                "dropped_feature": drop_feat,
                "kept_feature": keep_feat,
                "pearson_corr": round(float(corr_val), 6),
                "target_corr_dropped": round(float(target_corr.get(drop_feat, 0.0)), 6),
                "target_corr_kept": round(float(target_corr.get(keep_feat, 0.0)), 6),
                "domain_dropped": FEATURE_DOMAIN.get(drop_feat, "unknown"),
                "domain_kept": FEATURE_DOMAIN.get(keep_feat, "unknown"),
                "reason": reason,
            }
        )

        print(
            f"  DROP  {drop_feat:<20}  ← kept: {keep_feat:<20}  "
            f"|r|={corr_val:.4f}  reason: {reason}"
        )

    kept = [f for f in feats if f in alive]

    # ── save report ───────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "stage1_correlation_report.csv")
    if records:
        pd.DataFrame(records).to_csv(report_path, index=False)
    else:
        # Write empty report with correct columns
        pd.DataFrame(
            columns=[
                "dropped_feature",
                "kept_feature",
                "pearson_corr",
                "target_corr_dropped",
                "target_corr_kept",
                "domain_dropped",
                "domain_kept",
                "reason",
            ]
        ).to_csv(report_path, index=False)

    # ── drop columns from df ──────────────────────────────────────────────────
    df_out = df_clean.drop(columns=dropped, errors="ignore")

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n  [Stage 1 Result] {appliance}")
    print(f"  Features entering : {len(feats)}")
    print(f"  Dropped           : {len(dropped)}  →  {dropped if dropped else 'none'}")
    print(f"  Kept              : {len(kept)}")
    print(f"  Report saved      : {report_path}")

    return df_out, kept, dropped


# ─────────────────────────────────────────────────────────────────────────────
def run_all_appliances(args) -> dict:
    """
    Run Stage 0 → Stage 1 for every appliance and produce a cross-appliance summary.
    """
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    output_root = os.path.join(PROJECT_ROOT, args.output_dir)

    print("\n" + "═" * 60)
    print("  FEATURE SELECTION — STAGE 1: CORRELATION FILTERING")
    print("═" * 60)

    # ── Stage 0 first ────────────────────────────────────────────────────────
    stage0_results = stage0_run_all(args)
    if not stage0_results:
        print("[Stage 1] No Stage 0 results — nothing to do.")
        return {}

    results = {}
    all_records = []

    for appliance, s0 in stage0_results.items():
        app_output_dir = os.path.join(output_root, appliance)

        df_out, kept, dropped = run_stage1(
            df_clean=s0["df_clean"],
            kept_after_stage0=s0["kept"],
            appliance=appliance,
            output_dir=app_output_dir,
            corr_threshold=args.corr_threshold,
        )

        results[appliance] = {
            "df_out": df_out,
            "kept_stage0": s0["kept"],
            "dropped_stage0": s0["dropped"],
            "kept_stage1": kept,
            "dropped_stage1": dropped,
        }

        for feat in s0["kept"]:
            all_records.append(
                {
                    "appliance": appliance,
                    "feature": feat,
                    "domain": FEATURE_DOMAIN.get(feat, "unknown"),
                    "stage1": "dropped" if feat in dropped else "kept",
                }
            )

    # ── cross-appliance summary ───────────────────────────────────────────────
    if all_records:
        summary_df = (
            pd.DataFrame(all_records)
            .pivot_table(
                index=["feature", "domain"],
                columns="appliance",
                values="stage1",
                aggfunc="first",
            )
            .reset_index()
        )
        app_cols = [c for c in summary_df.columns if c not in ("feature", "domain")]
        summary_df["n_dropped"] = (summary_df[app_cols] == "dropped").sum(axis=1)
        summary_df["globally_dropped"] = summary_df["n_dropped"] == len(app_cols)

        summary_path = os.path.join(output_root, "stage1_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "═" * 60)
        print("  STAGE 1 CROSS-APPLIANCE SUMMARY")
        print("═" * 60)

        globally_dropped = summary_df[summary_df["globally_dropped"]][
            "feature"
        ].tolist()
        print(f"\n  Dropped in ALL appliances ({len(globally_dropped)}):")
        for f in globally_dropped:
            print(f"    ✗  {f:<20}  [{FEATURE_DOMAIN.get(f, '?')}]")
        if not globally_dropped:
            print("    (none)")

        print(f"\n  Per-appliance feature counts after Stage 0 + Stage 1:")
        for app, res in results.items():
            print(
                f"    {app:<15}  stage0_kept={len(res['kept_stage0']):<3}  "
                f"stage1_kept={len(res['kept_stage1']):<3}  "
                f"stage1_dropped={res['dropped_stage1']}"
            )

        print(f"\n  Summary saved: {summary_path}")
        print("═" * 60)

    return results


# ─────────────────────────────────────────────────────────────────────────────
def get_arguments():
    parser = argparse.ArgumentParser(
        description="Stage 1: Correlation filtering — remove redundant HF features"
    )
    parser.add_argument(
        "--data_dir",
        default="dataset_preprocess/high_frequency_data_extract/output",
    )
    parser.add_argument("--output_dir", default="feature_selection_outputs")
    parser.add_argument(
        "--appliances",
        nargs="+",
        default=["kettle", "fridge", "microwave", "dishwasher", "washingmachine"],
    )
    parser.add_argument("--house", default="house2")
    parser.add_argument("--week", default="wk30")
    parser.add_argument(
        "--corr_threshold",
        type=float,
        default=CORRELATION_THRESHOLD,
        help=f"Pearson |r| above this → redundant pair  (default: {CORRELATION_THRESHOLD})",
    )
    # Pass-through Stage 0 thresholds
    parser.add_argument(
        "--var_threshold", type=float, default=NEAR_CONSTANT_VAR_THRESHOLD
    )
    parser.add_argument("--invalid_threshold", type=float, default=MAX_INVALID_RATIO)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    results = run_all_appliances(args)

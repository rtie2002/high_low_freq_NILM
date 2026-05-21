"""
Feature Selection — Stage 0: Feature Cleaning
==============================================
Checks each HF feature column for:
  1. Near-constant variance  (var < threshold  → drop)
  2. Excessive NaN / Inf     (ratio > threshold → drop)
  3. Remaining NaN / Inf     (below threshold   → fill with median)

Does NOT look at any target column (appliance_power, on_off, aggregate).
Run once per appliance CSV, then compare results across appliances.

Output per appliance
--------------------
  feature_selection_outputs/{appliance}/stage0_cleaning_report.csv
      feature | variance | nan_inf_ratio | status | reason

  feature_selection_outputs/stage0_summary.csv
      One row per feature, showing status across all appliances.

Usage
-----
  python feature_selection/stage0_cleaning.py
  python feature_selection/stage0_cleaning.py --data_dir path/to/csvs --appliances kettle fridge
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ── all 50 HF feature names (in extraction order) ────────────────────────────
HF_FEATURES = [
    # time_domain (7)
    "V_rms",
    "I_rms",
    "P_active",
    "S_apparent",
    "PF",
    "Fcv",
    "Fci",
    # shape_statistics (5)
    "I_skew",
    "I_kurt",
    "V_skew",
    "I_std",
    "V_std",
    # harmonics (16)
    "I1",
    "V1",
    "I3",
    "V3",
    "I5",
    "V5",
    "I7",
    "V7",
    "I9",
    "V9",
    "I11",
    "V11",
    "I13",
    "V13",
    "I15",
    "V15",
    # distortion (4)
    "IH",
    "VH",
    "THDI",
    "THDV",
    # band_power (4)
    "I_BP_low",
    "I_BP_mid",
    "I_BP_high",
    "V_BP_low",
    # spectral_descriptors (1)
    "I_spec_entropy",
    # spectral_envelope (8)
    "I_env_0",
    "I_env_1",
    "I_env_2",
    "I_env_3",
    "I_env_4",
    "I_env_5",
    "I_env_6",
    "I_env_7",
    # wavelet (5)
    "DWT_E0",
    "DWT_E1",
    "DWT_E2",
    "DWT_E3",
    "DWT_E4",
]

# ── domain label map ─────────────────────────────────────────────────────────
FEATURE_DOMAIN = {
    **{
        f: "time_domain"
        for f in ["V_rms", "I_rms", "P_active", "S_apparent", "PF", "Fcv", "Fci"]
    },
    **{f: "shape_statistics" for f in ["I_skew", "I_kurt", "V_skew", "I_std", "V_std"]},
    **{
        f: "harmonics"
        for f in [
            "I1",
            "V1",
            "I3",
            "V3",
            "I5",
            "V5",
            "I7",
            "V7",
            "I9",
            "V9",
            "I11",
            "V11",
            "I13",
            "V13",
            "I15",
            "V15",
        ]
    },
    **{f: "distortion" for f in ["IH", "VH", "THDI", "THDV"]},
    **{f: "band_power" for f in ["I_BP_low", "I_BP_mid", "I_BP_high", "V_BP_low"]},
    **{f: "spectral_descriptors" for f in ["I_spec_entropy"]},
    **{f: "spectral_envelope" for f in [f"I_env_{i}" for i in range(8)]},
    **{f: "wavelet" for f in [f"DWT_E{i}" for i in range(5)]},
}

# ── non-feature columns that must be excluded from analysis ──────────────────
NON_FEATURE_COLS = {"readable_time", "aggregate", "on_off"}
# appliance_power columns like "kettle_power" are detected dynamically

# ── thresholds (from PROJECT_PLANNING.md Stage 0) ────────────────────────────
NEAR_CONSTANT_VAR_THRESHOLD = 1e-8
MAX_INVALID_RATIO = 0.05


# ─────────────────────────────────────────────────────────────────────────────
def get_hf_columns(df: pd.DataFrame) -> list[str]:
    """
    Return only the HF feature columns present in df.
    Excludes readable_time, aggregate, on_off, and any *_power column.
    Also handles the case where some features are missing from the CSV
    (e.g. DWT columns absent when pywt was not installed).
    """
    exclude = set(NON_FEATURE_COLS)
    for col in df.columns:
        if col.endswith("_power"):
            exclude.add(col)
    return [c for c in df.columns if c not in exclude]


def run_stage0(
    df: pd.DataFrame,
    appliance: str,
    output_dir: str,
    var_threshold: float = NEAR_CONSTANT_VAR_THRESHOLD,
    invalid_threshold: float = MAX_INVALID_RATIO,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Run Stage 0 cleaning on one appliance DataFrame.

    Parameters
    ----------
    df               : raw fused CSV loaded as DataFrame
    appliance        : appliance name (used only for printing)
    output_dir       : folder to save the report CSV
    var_threshold    : variance below this → near-constant → drop
    invalid_threshold: NaN/Inf ratio above this → drop

    Returns
    -------
    df_clean   : DataFrame with bad features dropped and remaining NaN/Inf filled
    kept       : list of kept feature names
    dropped    : list of dropped feature names
    """
    feat_cols = get_hf_columns(df)
    print(
        f"\n[Stage 0] {appliance.upper()} — {len(df)} rows, {len(feat_cols)} HF features"
    )
    print(
        f"          var_threshold={var_threshold:.0e}  |  invalid_threshold={invalid_threshold:.0%}"
    )
    print("          " + "─" * 56)

    records = []
    dropped = []
    kept = []

    for feat in feat_cols:
        col = df[feat]

        # ── count NaN and Inf ────────────────────────────────────────────────
        n_total = len(col)
        n_nan = col.isna().sum()
        n_inf = np.isinf(col.replace([np.nan], 0)).sum()
        n_invalid = int(n_nan + n_inf)
        invalid_ratio = n_invalid / n_total

        # ── variance (computed on finite values only) ────────────────────────
        finite_vals = col.replace([np.inf, -np.inf], np.nan).dropna()
        variance = float(finite_vals.var()) if len(finite_vals) > 1 else 0.0

        # ── decision logic ───────────────────────────────────────────────────
        if invalid_ratio > invalid_threshold:
            status = "dropped"
            reason = f"invalid_ratio={invalid_ratio:.3f} > {invalid_threshold}"
            dropped.append(feat)

        elif variance < var_threshold:
            status = "dropped"
            reason = f"variance={variance:.2e} < {var_threshold:.0e} (near-constant)"
            dropped.append(feat)

        else:
            status = "kept"
            reason = "ok"
            kept.append(feat)

        records.append(
            {
                "feature": feat,
                "domain": FEATURE_DOMAIN.get(feat, "unknown"),
                "n_total": n_total,
                "n_invalid": n_invalid,
                "invalid_ratio": round(invalid_ratio, 6),
                "variance": round(variance, 10),
                "status": status,
                "reason": reason,
            }
        )

        # print one line per feature
        tag = "  DROP" if status == "dropped" else "  keep"
        print(
            f"  {tag}  {feat:<20}  var={variance:>12.4e}  "
            f"invalid={invalid_ratio:.3f}  → {reason}"
        )

    # ── fill remaining NaN/Inf in kept features ──────────────────────────────
    df_clean = df.copy()
    for feat in kept:
        col = df_clean[feat]
        has_nan = col.isna().any()
        has_inf = np.isinf(col.replace([np.nan], 0)).any()
        if has_nan or has_inf:
            median_val = col.replace([np.inf, -np.inf], np.nan).median()
            df_clean[feat] = col.replace([np.inf, -np.inf], np.nan).fillna(median_val)

    # ── drop bad feature columns from df_clean ───────────────────────────────
    df_clean.drop(columns=dropped, inplace=True, errors="ignore")

    # ── save per-appliance report ─────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    report_df = pd.DataFrame(records)
    report_path = os.path.join(output_dir, "stage0_cleaning_report.csv")
    report_df.to_csv(report_path, index=False)

    # ── summary print ─────────────────────────────────────────────────────────
    print(f"\n  [Stage 0 Result] {appliance}")
    print(f"  Original features : {len(feat_cols)}")
    print(f"  Dropped           : {len(dropped)}  →  {dropped if dropped else 'none'}")
    print(f"  Kept              : {len(kept)}")
    print(f"  Report saved      : {report_path}")

    return df_clean, kept, dropped


# ─────────────────────────────────────────────────────────────────────────────
def run_all_appliances(args) -> dict:
    """
    Run Stage 0 for every appliance CSV and produce a cross-appliance summary.

    Returns
    -------
    results : dict  { appliance: {"df_clean": ..., "kept": [...], "dropped": [...]} }
    """
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    output_root = os.path.join(PROJECT_ROOT, args.output_dir)

    print("\n" + "═" * 60)
    print("  FEATURE SELECTION — STAGE 0: FEATURE CLEANING")
    print("═" * 60)
    print(f"  Data dir   : {data_dir}")
    print(f"  Output dir : {output_root}")
    print(f"  Appliances : {args.appliances}")
    print("═" * 60)

    results = {}
    all_records = []  # for cross-appliance summary

    for appliance in args.appliances:
        # ── find CSV ──────────────────────────────────────────────────────────
        csv_name = f"{appliance}_{args.house}_{args.week}.csv"
        csv_path = os.path.join(data_dir, csv_name)
        if not os.path.exists(csv_path):
            print(f"\n[Stage 0] ⚠️  CSV not found: {csv_path}  — skipping {appliance}")
            continue

        df = pd.read_csv(csv_path)
        app_output_dir = os.path.join(output_root, appliance)

        df_clean, kept, dropped = run_stage0(
            df,
            appliance,
            app_output_dir,
            var_threshold=args.var_threshold,
            invalid_threshold=args.invalid_threshold,
        )

        results[appliance] = {
            "df_clean": df_clean,
            "kept": kept,
            "dropped": dropped,
        }

        # collect for summary
        for feat in get_hf_columns(df):
            status = "dropped" if feat in dropped else "kept"
            all_records.append(
                {
                    "appliance": appliance,
                    "feature": feat,
                    "domain": FEATURE_DOMAIN.get(feat, "unknown"),
                    "status": status,
                }
            )

    # ── cross-appliance summary ───────────────────────────────────────────────
    if all_records:
        summary_df = (
            pd.DataFrame(all_records)
            .pivot_table(
                index=["feature", "domain"],
                columns="appliance",
                values="status",
                aggfunc="first",
            )
            .reset_index()
        )

        # add a column: how many appliances DROPPED this feature
        app_cols = [c for c in summary_df.columns if c not in ("feature", "domain")]
        summary_df["n_dropped"] = (summary_df[app_cols] == "dropped").sum(axis=1)
        summary_df["globally_dropped"] = summary_df["n_dropped"] == len(app_cols)

        summary_path = os.path.join(output_root, "stage0_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "═" * 60)
        print("  STAGE 0 CROSS-APPLIANCE SUMMARY")
        print("═" * 60)

        globally_dropped = summary_df[summary_df["globally_dropped"]][
            "feature"
        ].tolist()
        partially_dropped = summary_df[
            (summary_df["n_dropped"] > 0) & (~summary_df["globally_dropped"])
        ][["feature", "domain", "n_dropped"]].sort_values("n_dropped", ascending=False)

        print(f"\n  Dropped in ALL appliances ({len(globally_dropped)}):")
        if globally_dropped:
            for f in globally_dropped:
                print(f"    ✗  {f}  [{FEATURE_DOMAIN.get(f, '?')}]")
        else:
            print("    (none)")

        print(f"\n  Dropped in SOME appliances:")
        if len(partially_dropped):
            print(partially_dropped.to_string(index=False))
        else:
            print("    (none)")

        print(f"\n  Summary saved: {summary_path}")
        print("═" * 60)

    return results


# ─────────────────────────────────────────────────────────────────────────────
def get_arguments():
    parser = argparse.ArgumentParser(
        description="Stage 0: Feature Cleaning — variance and NaN/Inf check"
    )
    parser.add_argument(
        "--data_dir",
        default="dataset_preprocess/high_frequency_data_extract/output",
        help="Folder containing the fused appliance CSVs (relative to project root)",
    )
    parser.add_argument(
        "--output_dir",
        default="feature_selection_outputs",
        help="Root folder for all feature selection output (relative to project root)",
    )
    parser.add_argument(
        "--appliances",
        nargs="+",
        default=["kettle", "fridge", "microwave", "dishwasher", "washingmachine"],
    )
    parser.add_argument("--house", default="house2")
    parser.add_argument("--week", default="wk30")
    parser.add_argument(
        "--var_threshold",
        type=float,
        default=NEAR_CONSTANT_VAR_THRESHOLD,
        help=f"Variance below this → near-constant → drop  (default: {NEAR_CONSTANT_VAR_THRESHOLD})",
    )
    parser.add_argument(
        "--invalid_threshold",
        type=float,
        default=MAX_INVALID_RATIO,
        help=f"NaN/Inf ratio above this → drop  (default: {MAX_INVALID_RATIO})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    results = run_all_appliances(args)

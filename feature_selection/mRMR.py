"""
mRMR feature selection for NILM HF features.

This script implements the mRMR idea used in "SVM-RFE with MRMR Filter":

    select features with high relevance to the target label
    while penalizing redundancy with already selected features

Paper equation mapping:
    Eq. (1) relevance R_S:
        I(label, feature)
        Implemented in relevance_scores()

    Eq. (2) redundancy Q_S,i:
        average I(candidate feature, selected features)
        Implemented in pairwise_redundancy() and used inside run_mrmr()

    Eq. (3) mRMR ranking:
        argmax relevance / redundancy
        Implemented in run_mrmr()

Default target:
    appliance_power regression.

Outputs:
    feature_selection/results/mrmr/{dataset_name}/{appliance}/mrmr_ranking.csv
    feature_selection/results/mrmr/{dataset_name}/mrmr_summary.csv

Usage:
    python feature_selection/mRMR.py
    python feature_selection/mRMR.py --dataset_name full_wk30_only
    python feature_selection/mRMR.py --dataset_name full_wk30_only --no_balance
    python feature_selection/mRMR.py --target power --top_k 20
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

APPLIANCES = ["kettle", "fridge", "microwave", "dishwasher", "washingmachine"]

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


def find_appliance_csv(data_dir: str, appliance: str) -> str:
    matches = [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.lower().endswith(".csv") and name.startswith(f"{appliance}_")
    ]
    if not matches:
        raise FileNotFoundError(f"No CSV found for {appliance} in {data_dir}")
    if len(matches) > 1:
        print(f"[warning] multiple CSVs for {appliance}; using {os.path.basename(matches[0])}")
    return matches[0]


def get_target_column(df: pd.DataFrame, appliance: str, target: str) -> str:
    if target == "on_off":
        return "on_off"
    if target == "power":
        return f"{appliance}_power"
    raise ValueError(f"Unknown target: {target}")


def balance_on_off(df: pd.DataFrame, target_col: str, random_state: int) -> pd.DataFrame:
    off_df = df[df[target_col].fillna(0).astype(int).eq(0)]
    on_df = df[df[target_col].fillna(0).astype(int).eq(1)]
    n = min(len(off_df), len(on_df))
    if n == 0:
        return df
    return pd.concat(
        [
            off_df.sample(n=n, random_state=random_state),
            on_df.sample(n=n, random_state=random_state),
        ],
        axis=0,
    ).sample(frac=1, random_state=random_state).reset_index(drop=True)


def sample_rows(df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if sample_size <= 0 or len(df) <= sample_size:
        return df
    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def prepare_xy(
    df: pd.DataFrame,
    appliance: str,
    target: str,
    balance: bool,
    sample_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    target_col = get_target_column(df, appliance, target)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    if target == "on_off" and balance:
        df = balance_on_off(df, target_col, random_state)

    df = sample_rows(df, sample_size, random_state)

    target_counts = {}
    if target == "on_off":
        target_counts = (
            df[target_col].fillna(0).astype(int).value_counts().sort_index().to_dict()
        )

    feature_cols = [f for f in HF_FEATURES if f in df.columns]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    y = df[target_col]
    if target == "on_off":
        y = y.fillna(0).astype(int)
    else:
        y = y.fillna(0.0).astype(float)

    return X, y, target_counts


def relevance_scores(X: pd.DataFrame, y: pd.Series, target: str, random_state: int) -> pd.Series:
    """
    Paper Eq. (1): relevance term.

    R_S is based on mutual information between the target label and each feature:

        I(Y; F_i) = sum_y sum_f p(y, f) log( p(y, f) / (p(y) p(f)) )

    For NILM:

        classification relevance = I(on_off; feature)
        regression relevance     = I(appliance_power; feature)

    sklearn is used to estimate I(Y; F_i) for continuous HF features.
    """
    X_scaled = StandardScaler().fit_transform(X)
    if target == "on_off":
        scores = mutual_info_classif(
            X_scaled, y, discrete_features=False, random_state=random_state
        )
    else:
        scores = mutual_info_regression(
            X_scaled, y, discrete_features=False, random_state=random_state
        )
    return pd.Series(scores, index=X.columns, name="relevance_mi")


def pairwise_redundancy(X: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """
    Paper Eq. (2): redundancy term.

    Q_S,i is based on mutual information between the candidate feature and
    other selected features:

        I(F_i; F_j) = sum_fi sum_fj p(fi, fj) log( p(fi, fj) / (p(fi) p(fj)) )

        Q_S,i = (1 / |S|) * sum_{F_j in S} I(F_i; F_j)

    Later, run_mrmr() averages I(F_i; F_j) over the already selected subset S.
    """
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    red = pd.DataFrame(0.0, index=X.columns, columns=X.columns)
    for feature in X.columns:
        y_feature = X_scaled[feature]
        other_features = [c for c in X.columns if c != feature]
        mi = mutual_info_regression(
            X_scaled[other_features],
            y_feature,
            discrete_features=False,
            random_state=random_state,
        )
        red.loc[feature, other_features] = mi
    red = (red + red.T) / 2.0
    np.fill_diagonal(red.values, 0.0)
    return red


def run_mrmr(X: pd.DataFrame, y: pd.Series, target: str, top_k: int, random_state: int) -> pd.DataFrame:
    """
    Paper Eq. (3): mRMR forward selection.

    Select the next feature by maximizing:

        F_i* = argmax_{F_i in remaining} R_i / Q_S,i

    where:

        R_i   = I(Y; F_i)
        Q_S,i = average I(F_i; F_j), for F_j already selected

    In code:
        rel = MI(feature, target)
        red = mean MI(feature, already selected features)
        score = rel / (red + eps)
    """
    relevance = relevance_scores(X, y, target, random_state)
    redundancy = pairwise_redundancy(X, random_state)

    selected: list[str] = []
    remaining = list(X.columns)
    rows = []

    if top_k <= 0:
        top_k = len(remaining)
    else:
        top_k = min(top_k, len(remaining))
    eps = 1e-9

    for rank in range(1, top_k + 1):
        candidates = []
        for feature in remaining:
            rel = float(relevance[feature])
            if selected:
                red = float(redundancy.loc[feature, selected].mean())
            else:
                red = 0.0

            # Paper Eq. (3): maximize relevance / redundancy.
            # If MI redundancy is estimated as zero, treat the candidate as
            # non-redundant and rank it by relevance instead of creating an
            # artificial near-infinite quotient.
            score = rel / red if selected and red > eps else rel
            candidates.append((feature, score, rel, red))

        feature, score, rel, red = max(candidates, key=lambda x: x[1])
        selected.append(feature)
        remaining.remove(feature)
        rows.append(
            {
                "rank": rank,
                "feature": feature,
                "mrmr_score": score,
                "relevance_mi_to_target": rel,
                "mean_redundancy_mi_to_selected": red,
            }
        )

    return pd.DataFrame(rows)


def save_ranking_plot(ranking: pd.DataFrame, output_path: str, appliance: str, target: str) -> None:
    plot_df = ranking.sort_values("rank").copy()
    labels = [f"{int(row.rank)}. {row.feature}" for row in plot_df.itertuples()]

    fig_h = max(8, len(plot_df) * 0.22)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.barh(labels, plot_df["mrmr_score"], color="#2f80ed")
    ax.invert_yaxis()
    ax.set_xlabel("mRMR score")
    ax.set_ylabel("Feature rank")
    ax.set_title(f"{appliance} mRMR feature ranking ({target})")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_cross_appliance_rank_heatmap(
    output_root: str,
    appliances: list[str],
    target: str,
) -> None:
    rank_data = {"feature": HF_FEATURES}
    for appliance in appliances:
        path = os.path.join(output_root, appliance, "mrmr_ranking.csv")
        if not os.path.exists(path):
            continue
        ranking = pd.read_csv(path)
        rank_map = dict(zip(ranking["feature"], ranking["rank"]))
        rank_data[appliance] = [rank_map.get(feature, np.nan) for feature in HF_FEATURES]

    rank_df = pd.DataFrame(rank_data)
    app_cols = [c for c in rank_df.columns if c != "feature"]
    if not app_cols:
        return

    csv_path = os.path.join(output_root, "mrmr_rank_matrix.csv")
    png_path = os.path.join(output_root, "mrmr_cross_appliance_rank_heatmap.png")
    rank_df.to_csv(csv_path, index=False)

    mat = rank_df[app_cols].astype(float).values
    annot = np.empty(mat.shape, dtype=object)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            annot[i, j] = "" if np.isnan(mat[i, j]) else str(int(mat[i, j]))

    fig_h = max(10, len(rank_df) * 0.24)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="viridis_r", vmin=1, vmax=len(HF_FEATURES))

    ax.set_xticks(np.arange(len(app_cols)))
    ax.set_xticklabels(app_cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(rank_df)))
    ax.set_yticklabels(rank_df["feature"])
    ax.set_title(f"Cross-appliance mRMR feature rank ({target})")
    ax.set_xlabel("Appliance")
    ax.set_ylabel("HF feature")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if annot[i, j]:
                val = mat[i, j]
                color = "white" if val > len(HF_FEATURES) * 0.55 else "black"
                ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("mRMR rank (1 = best)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nCross-appliance rank matrix saved: {csv_path}")
    print(f"Cross-appliance rank heatmap saved: {png_path}")

    summary = rank_df.copy()
    summary["mean_rank"] = summary[app_cols].mean(axis=1)
    summary["median_rank"] = summary[app_cols].median(axis=1)
    summary["top5_count"] = (summary[app_cols] <= 5).sum(axis=1)
    summary["top10_count"] = (summary[app_cols] <= 10).sum(axis=1)
    summary = summary.sort_values(
        ["top5_count", "top10_count", "mean_rank"],
        ascending=[False, False, True],
    )

    summary_csv = os.path.join(output_root, "mrmr_overall_feature_importance.csv")
    summary_png = os.path.join(output_root, "mrmr_overall_top_features.png")
    summary.to_csv(summary_csv, index=False)

    top_n = min(20, len(summary))
    plot_df = summary.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.32)))
    colors = plt.cm.YlGnBu(np.linspace(0.35, 0.9, top_n))
    ax.barh(plot_df["feature"], plot_df["top10_count"], color=colors)
    for y_pos, row in enumerate(plot_df.itertuples()):
        ax.text(
            row.top10_count + 0.05,
            y_pos,
            f"top5={row.top5_count}, mean rank={row.mean_rank:.1f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, len(app_cols) + 1.6)
    ax.set_xlabel(f"Number of appliances ranked in top 10 (out of {len(app_cols)})")
    ax.set_ylabel("HF feature")
    ax.set_title(f"Overall mRMR important features across appliances ({target})")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(summary_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Overall feature importance saved: {summary_csv}")
    print(f"Overall top-feature plot saved: {summary_png}")


def run_one_appliance(args, data_dir: str, output_root: str, appliance: str) -> dict:
    csv_path = find_appliance_csv(data_dir, appliance)
    df = pd.read_csv(csv_path)

    X, y, target_counts = prepare_xy(
        df=df,
        appliance=appliance,
        target=args.target,
        balance=args.balance,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    ranking = run_mrmr(
        X=X,
        y=y,
        target=args.target,
        top_k=args.top_k,
        random_state=args.random_state,
    )

    app_dir = os.path.join(output_root, appliance)
    os.makedirs(app_dir, exist_ok=True)
    ranking_path = os.path.join(app_dir, "mrmr_ranking.csv")
    plot_path = os.path.join(app_dir, "mrmr_ranking.png")
    ranking.to_csv(ranking_path, index=False)
    save_ranking_plot(ranking, plot_path, appliance, args.target)

    print(f"\n[{appliance}]")
    print(f"  file       : {os.path.basename(csv_path)}")
    print(f"  rows used  : {len(X)}")
    if target_counts:
        print(f"  off/on     : {target_counts.get(0, 0)} / {target_counts.get(1, 0)}")
    print(f"  features   : {len(X.columns)}")
    print(f"  saved      : {ranking_path}")
    print(f"  plot       : {plot_path}")
    print(ranking.head(min(10, len(ranking))).to_string(index=False))

    return {
        "appliance": appliance,
        "rows_used": len(X),
        "off_rows": target_counts.get(0, np.nan),
        "on_rows": target_counts.get(1, np.nan),
        "n_features": len(X.columns),
        "ranking_path": ranking_path,
        "top_features": ", ".join(ranking["feature"].head(10).tolist()),
    }


def get_arguments():
    parser = argparse.ArgumentParser(description="mRMR feature selection for NILM HF features")
    parser.add_argument("--dataset_name", default="output(On_only_wk30_wk31)")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--appliances", nargs="+", default=APPLIANCES)
    parser.add_argument("--target", choices=["on_off", "power", "both"], default="power")
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Number of features to rank. Use 0 to rank all available features.",
    )
    parser.add_argument("--sample_size", type=int, default=50000)
    parser.set_defaults(balance=True)
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Use balanced ON/OFF samples for on_off target. Enabled by default.",
    )
    parser.add_argument(
        "--no_balance",
        action="store_false",
        dest="balance",
        help="Use all rows without ON/OFF balancing.",
    )
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_arguments()

    data_dir = args.data_dir or os.path.join(
        PROJECT_ROOT, "dataset_preprocess", "high_frequency_data_extract", args.dataset_name
    )
    base_output_root = args.output_dir or os.path.join(
        PROJECT_ROOT, "feature_selection", "results", "mrmr", args.dataset_name
    )
    targets = ["on_off", "power"] if args.target == "both" else [args.target]

    for target in targets:
        run_args = argparse.Namespace(**vars(args))
        run_args.target = target
        output_root = os.path.join(base_output_root, target)
        os.makedirs(output_root, exist_ok=True)

        print("=" * 72)
        print("mRMR FEATURE SELECTION")
        print("=" * 72)
        print(f"data_dir    : {data_dir}")
        print(f"output_dir  : {output_root}")
        print(f"target      : {run_args.target}")
        print(f"top_k       : {run_args.top_k}")
        print(f"sample_size : {run_args.sample_size}")
        print(f"balance     : {run_args.balance if target == 'on_off' else 'not used'}")
        print("=" * 72)

        summary = []
        for appliance in run_args.appliances:
            summary.append(run_one_appliance(run_args, data_dir, output_root, appliance))

        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(output_root, "mrmr_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved: {summary_path}")
        save_cross_appliance_rank_heatmap(output_root, run_args.appliances, target)


if __name__ == "__main__":
    main()

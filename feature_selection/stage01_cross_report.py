"""
Stage 01 cross-appliance report — figures and summary tables from existing outputs.

Reads feature_selection_outputs/{appliance}/stage01_*.csv (no correlation re-run).
Writes feature_selection_outputs/cross_appliance/*.png and companion CSVs.

Usage (project root):
  python feature_selection/stage01_cross_report.py
  python feature_selection/stage01_cross_report.py --output_dir feature_selection_outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from stage01_filter import DOMAIN_COLORS, FEATURE_DOMAIN, HF_FEATURES  # noqa: E402

DEFAULT_APPLIANCES = ["kettle", "fridge", "microwave", "dishwasher", "washingmachine"]
FLIP_FEATURES = [
    "I_rms", "P_active", "I_skew", "I3", "IH", "THDI",
    "I_env_0", "DWT_E0", "DWT_E3", "DWT_E4",
]
GLOBAL_DROPS = [
    "S_apparent", "I_kurt", "I_std", "V_std", "I1", "V1",
    "THDV", "I_BP_low", "I_BP_high", "I_env_7",
]
ON_RATES = {
    "kettle": 0.62,
    "fridge": 70.1,
    "microwave": 0.41,
    "dishwasher": 3.17,
    "washingmachine": 2.62,
}
APP_LABELS = {
    "kettle": "Kettle",
    "fridge": "Fridge",
    "microwave": "Microwave",
    "dishwasher": "Dishwasher",
    "washingmachine": "Washing machine",
}


def _load_appliance_data(output_root: Path, appliances: list[str]) -> dict:
    data = {}
    for app in appliances:
        app_dir = output_root / app
        summary_path = app_dir / "stage01_feature_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing {summary_path}")
        data[app] = {
            "summary": pd.read_csv(summary_path),
            "target": pd.read_csv(app_dir / "stage01_target_correlations.csv"),
            "corr_report": pd.read_csv(app_dir / "stage01_correlation_report.csv"),
        }
    return data


def _tier_label(n_kept: int) -> str:
    if n_kept == 5:
        return "A_universal"
    if n_kept == 0:
        return "D_never"
    if n_kept >= 3:
        return "B_majority"
    return "C_minority"


def build_summary_pivot(data: dict, appliances: list[str]) -> pd.DataFrame:
    records = []
    for app in appliances:
        for _, row in data[app]["summary"].iterrows():
            records.append({
                "feature": row["feature"],
                "domain": row["domain"],
                "appliance": app,
                "final_status": row["final_status"],
            })
    cross = pd.DataFrame(records)
    pivot = cross.pivot_table(
        index=["feature", "domain"],
        columns="appliance",
        values="final_status",
        aggfunc="first",
    ).reset_index()
    app_cols = [c for c in pivot.columns if c not in ("feature", "domain")]
    pivot["n_kept"] = (pivot[app_cols] == "kept").sum(axis=1)
    pivot["n_dropped"] = (pivot[app_cols] == "dropped").sum(axis=1)
    pivot["globally_kept"] = pivot["n_kept"] == len(app_cols)
    pivot["globally_dropped"] = pivot["n_dropped"] == len(app_cols)
    pivot["tier"] = pivot["n_kept"].map(_tier_label)
    return pivot.sort_values(["n_kept", "feature"], ascending=[False, True])


def build_global_drop_partners(data: dict, appliances: list[str]) -> pd.DataFrame:
    rows = []
    for feat in GLOBAL_DROPS:
        row = {"feature": feat, "domain": FEATURE_DOMAIN.get(feat, "?")}
        partners = []
        for app in appliances:
            match = data[app]["corr_report"]
            match = match[match["dropped_feature"] == feat]
            if len(match):
                m = match.iloc[0]
                row[f"{app}_kept_partner"] = m["kept_feature"]
                row[f"{app}_pair_r"] = m["pearson_abs_pair"]
                partners.append(m["kept_feature"])
            else:
                row[f"{app}_kept_partner"] = ""
                row[f"{app}_pair_r"] = np.nan
        row["partner_consistent"] = len(set(partners)) == 1
        rows.append(row)
    return pd.DataFrame(rows)


def build_target_matrix(data: dict, appliances: list[str]) -> pd.DataFrame:
    rows = []
    for feat in HF_FEATURES:
        row = {"feature": feat}
        for app in appliances:
            t = data[app]["target"]
            m = t[t["feature"] == feat]
            row[app] = m["target_pearson_abs"].iloc[0] if len(m) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_jaccard(data: dict, appliances: list[str]) -> pd.DataFrame:
    kept_sets = {
        app: set(data[app]["summary"].loc[
            data[app]["summary"]["final_status"] == "kept", "feature"
        ])
        for app in appliances
    }
    n = len(appliances)
    mat = np.zeros((n, n))
    for i, a in enumerate(appliances):
        for j, b in enumerate(appliances):
            inter = len(kept_sets[a] & kept_sets[b])
            union = len(kept_sets[a] | kept_sets[b])
            mat[i, j] = inter / union if union else 0.0
    return pd.DataFrame(mat, index=appliances, columns=appliances)


def build_drop_rules(data: dict, appliances: list[str]) -> pd.DataFrame:
    rows = []
    for app in appliances:
        rpt = data[app]["corr_report"]
        priority = int(rpt["reason"].str.contains("domain_priority", na=False).sum())
        rows.append({
            "appliance": app,
            "target_driven": len(rpt) - priority,
            "domain_priority": priority,
            "total_drops": len(rpt),
            "on_rate_pct": ON_RATES.get(app, np.nan),
        })
    return pd.DataFrame(rows)


def _save_caption(out_dir: Path, fig_id: str, text: str) -> None:
    (out_dir / f"{fig_id}_caption.txt").write_text(text, encoding="utf-8")


def fig01_pipeline_counts(data: dict, appliances: list[str], out_dir: Path) -> None:
    kept = [(data[a]["summary"]["final_status"] == "kept").sum() for a in appliances]
    labels = [APP_LABELS.get(a, a) for a in appliances]
    x = np.arange(len(appliances))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, [50] * len(appliances), w, label="Input (50)", color="#bdbdbd")
    ax.bar(x + w / 2, kept, w, label="Kept (34)", color="#2ca02c")
    for i, k in enumerate(kept):
        ax.text(i + w / 2, k + 0.5, str(k), ha="center", fontsize=10)
        ax.text(i - w / 2, 51, "50", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Feature count")
    ax.set_title("Stage 01 outcome per appliance (wk30)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 55)
    fig.tight_layout()
    fig.savefig(out_dir / "fig01_pipeline_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    _save_caption(out_dir, "fig01", "Figure 1. HF feature counts before and after Stage 01.")


def fig02_stability_heatmap(pivot: pd.DataFrame, appliances: list[str], out_dir: Path) -> None:
    feats = pivot["feature"].tolist()
    mat = np.zeros((len(feats), len(appliances)))
    for i, feat in enumerate(feats):
        for j, app in enumerate(appliances):
            mat[i, j] = 1.0 if pivot.loc[pivot["feature"] == feat, app].iloc[0] == "kept" else 0.0
    fig_h = max(10, len(feats) * 0.22)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    sns.heatmap(
        mat,
        xticklabels=[APP_LABELS.get(a, a) for a in appliances],
        yticklabels=feats,
        cmap=sns.color_palette(["#d62728", "#2ca02c"], as_cmap=True),
        cbar_kws={"ticks": [0.25, 0.75]},
        linewidths=0.3,
        linecolor="#eee",
        ax=ax,
    )
    ax.collections[0].colorbar.set_ticklabels(["Dropped", "Kept"])
    ax.set_title("Cross-appliance feature stability (sorted by n_kept)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig02_stability_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    _save_caption(out_dir, "fig02", "Figure 2. Kept vs dropped per feature and appliance.")


def fig03_target_heatmap(target_df: pd.DataFrame, appliances: list[str], out_dir: Path) -> None:
    feats = target_df["feature"].tolist()
    mat = target_df[appliances].values
    fig_h = max(10, len(feats) * 0.22)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    sns.heatmap(
        mat,
        xticklabels=[APP_LABELS.get(a, a) for a in appliances],
        yticklabels=feats,
        cmap="YlOrRd",
        vmin=0,
        vmax=0.75,
        cbar_kws={"label": "|Pearson| to appliance power"},
        ax=ax,
    )
    ax.set_title("Target relevance (same HF, different sub-meter labels)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig03_target_relevance_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    _save_caption(out_dir, "fig03", "Figure 3. Target |Pearson| heatmap across appliances.")


def fig04_flip_spotlight(target_df: pd.DataFrame, pivot: pd.DataFrame, appliances: list[str], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), sharey=True)
    axes = axes.flatten()
    for idx, feat in enumerate(FLIP_FEATURES):
        ax = axes[idx]
        vals = [float(target_df.loc[target_df["feature"] == feat, app].iloc[0]) for app in appliances]
        colors = [
            "#2ca02c" if pivot.loc[pivot["feature"] == feat, app].iloc[0] == "kept" else "#d62728"
            for app in appliances
        ]
        x = np.arange(len(appliances))
        ax.bar(x, vals, color=colors, edgecolor="#333", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([a[:4] for a in appliances], fontsize=7, rotation=45)
        ax.set_title(feat, fontsize=9)
        ax.set_ylim(0, 0.8)
        if idx % 5 == 0:
            ax.set_ylabel("|r| to target")
    fig.suptitle("Flip features: target |Pearson| (green=kept, red=dropped)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig04_flip_features_spotlight.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    _save_caption(out_dir, "fig04", "Figure 4. Ten appliance-dependent features.")


def fig05_global_drop_partners(partners_df: pd.DataFrame, appliances: list[str], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = np.arange(len(GLOBAL_DROPS))
    for j, app in enumerate(appliances):
        rs = partners_df[f"{app}_pair_r"].values.astype(float)
        offset = (j - len(appliances) / 2 + 0.5) * 0.08
        ax.barh(y_pos + offset, rs, height=0.07, label=APP_LABELS.get(app, app))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(GLOBAL_DROPS)
    ax.axvline(0.95, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Pair |Pearson| (dropped vs kept partner)")
    ax.set_title("Globally dropped features: redundancy with greedy winner")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig05_global_drop_partners.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    _save_caption(out_dir, "fig05", "Figure 5. Pair |Pearson| for universal drops.")


def fig06_domain_survival(pivot: pd.DataFrame, appliances: list[str], out_dir: Path) -> None:
    domains = sorted(
        set(FEATURE_DOMAIN.values()),
        key=lambda d: sum(1 for f in HF_FEATURES if FEATURE_DOMAIN.get(f) == d),
        reverse=True,
    )
    pct = np.zeros((len(domains), len(appliances)))
    for i, dom in enumerate(domains):
        feats = [f for f in HF_FEATURES if FEATURE_DOMAIN.get(f) == dom]
        sub = pivot[pivot["feature"].isin(feats)]
        for j, app in enumerate(appliances):
            pct[i, j] = 100 * (sub[app] == "kept").mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(appliances))
    w = 0.8 / len(domains)
    for i, dom in enumerate(domains):
        ax.bar(x + i * w, pct[i], w, label=dom, color=DOMAIN_COLORS.get(dom, "#888"))
    ax.set_xticks(x + w * (len(domains) - 1) / 2)
    ax.set_xticklabels([APP_LABELS.get(a, a) for a in appliances], rotation=15, ha="right")
    ax.set_ylabel("% features kept in domain")
    ax.set_title("Domain-level survival after Stage 01")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out_dir / "fig06_domain_survival.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    _save_caption(out_dir, "fig06", "Figure 6. Domain survival rates.")


def fig07_jaccard(jaccard: pd.DataFrame, out_dir: Path) -> None:
    labels = [APP_LABELS.get(a, a) for a in jaccard.index]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        jaccard.values,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.5,
        vmax=1.0,
        ax=ax,
    )
    ax.set_title("Jaccard similarity of final kept sets")
    fig.tight_layout()
    fig.savefig(out_dir / "fig07_appliance_similarity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    _save_caption(out_dir, "fig07", "Figure 7. Jaccard similarity between kept sets.")


def fig08_drop_rules(rules_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    apps = rules_df["appliance"].tolist()
    labels = [APP_LABELS.get(a, a) for a in apps]
    x = np.arange(len(apps))
    axes[0].bar(x, rules_df["target_driven"], label="Target |r| wins", color="#1f77b4")
    axes[0].bar(
        x, rules_df["domain_priority"],
        bottom=rules_df["target_driven"],
        label="Domain priority tie",
        color="#ff7f0e",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Drop decisions (of 16)")
    axes[0].set_title("Greedy drop rule per appliance")
    axes[0].legend()
    tot_p = int(rules_df["domain_priority"].sum())
    tot_t = int(rules_df["target_driven"].sum())
    axes[1].pie(
        [tot_t, tot_p],
        labels=[f"Target-driven\n({tot_t})", f"Domain priority\n({tot_p})"],
        autopct="%1.1f%%",
        colors=["#1f77b4", "#ff7f0e"],
        startangle=90,
    )
    axes[1].set_title(f"All appliances pooled (n={tot_t + tot_p})")
    fig.tight_layout()
    fig.savefig(out_dir / "fig08_drop_decision_rules.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    _save_caption(out_dir, "fig08", "Figure 8. Drop decision rules pooled and per appliance.")


def run_report(output_root: Path, appliances: list[str]) -> None:
    cross_dir = output_root / "cross_appliance"
    cross_dir.mkdir(parents=True, exist_ok=True)

    data = _load_appliance_data(output_root, appliances)
    pivot = build_summary_pivot(data, appliances)
    pivot.to_csv(output_root / "stage01_summary.csv", index=False)

    pivot[["feature", "domain", "n_kept", "n_dropped", "tier"]].to_csv(
        cross_dir / "feature_stability_tiers.csv", index=False
    )
    build_global_drop_partners(data, appliances).to_csv(
        cross_dir / "global_drop_partners.csv", index=False
    )
    build_target_matrix(data, appliances).to_csv(
        cross_dir / "target_pearson_matrix.csv", index=False
    )
    build_jaccard(data, appliances).to_csv(cross_dir / "appliance_jaccard.csv")
    build_drop_rules(data, appliances).to_csv(
        cross_dir / "drop_decision_rules.csv", index=False
    )

    fig01_pipeline_counts(data, appliances, cross_dir)
    fig02_stability_heatmap(pivot, appliances, cross_dir)
    fig03_target_heatmap(build_target_matrix(data, appliances), appliances, cross_dir)
    fig04_flip_spotlight(
        pd.read_csv(cross_dir / "target_pearson_matrix.csv"),
        pivot, appliances, cross_dir,
    )
    fig05_global_drop_partners(
        pd.read_csv(cross_dir / "global_drop_partners.csv"), appliances, cross_dir,
    )
    fig06_domain_survival(pivot, appliances, cross_dir)
    fig07_jaccard(pd.read_csv(cross_dir / "appliance_jaccard.csv", index_col=0), cross_dir)
    fig08_drop_rules(pd.read_csv(cross_dir / "drop_decision_rules.csv"), cross_dir)

    inter = pivot[pivot["globally_kept"]]["feature"].tolist()
    union = set()
    for app in appliances:
        s = data[app]["summary"]
        union |= set(s.loc[s["final_status"] == "kept", "feature"])
    _write_appendix(pivot, data, appliances, _SCRIPT_DIR / "stage01_results_appendix.md")

    print(f"Wrote: {cross_dir}")
    print(f"  Universal kept (5/5): {len(inter)}")
    print(f"  Union kept: {len(union)}")


def _write_appendix(pivot: pd.DataFrame, data: dict, appliances: list[str], out_path: Path) -> None:
    lines = [
        "# Stage 01 Results Appendix",
        "",
        "Audit tables and greedy logs. Main narrative: [feature_selection.md](feature_selection.md).",
        "",
        "## Master feature status (50 x 5)",
        "",
        "Legend: K = kept, D = dropped.",
        "",
        "| Feature | Domain | " + " | ".join(appliances) + " | n_kept |",
        "|---------|--------|" + "|".join([":---:"] * len(appliances)) + "|:------:|",
    ]
    for _, row in pivot.iterrows():
        cells = ["K" if row[a] == "kept" else "D" for a in appliances]
        lines.append(
            f"| `{row['feature']}` | {row['domain']} | "
            + " | ".join(cells)
            + f" | {int(row['n_kept'])} |"
        )
    lines += ["", "## Greedy elimination logs (16 steps each)", ""]
    for app in appliances:
        lines += [f"### {app}", ""]
        rpt = data[app]["corr_report"]
        lines += [
            "| Step | Dropped | Kept | Pair |r| | Reason |",
            "|------|---------|------|--------|------|",
        ]
        for i, r in rpt.iterrows():
            pr = max(r["pearson_abs_pair"], r["spearman_abs_pair"])
            reason = "priority" if "domain_priority" in str(r["reason"]) else "target"
            lines.append(
                f"| {i+1} | `{r['dropped_feature']}` | `{r['kept_feature']}` | {pr:.3f} | {reason} |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def get_arguments():
    p = argparse.ArgumentParser(description="Stage 01 cross-appliance figures and tables")
    p.add_argument("--output_dir", default="feature_selection_outputs")
    p.add_argument("--appliances", nargs="+", default=DEFAULT_APPLIANCES)
    return p.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    root = Path(args.output_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / root
    run_report(root, args.appliances)

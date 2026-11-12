"""Render high-resolution publication-quality benchmark comparison table.

Matches the reference layout:
  UK-DALE · House 2          REFIT · House 20
  [Metric, Kettle, Fridge, Dishwasher, Washer, Microwave, Avg.,
   Kettle, Fridge, Dishwasher, Washer, Microwave, Avg.]
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


METRIC_SPECS = [
    ("MAE (W)",    "mae",               "{:.2f}"),
    ("SAE (W)",    "sae",               "{:.2f}"),
    ("ON MAE (W)", "on_mae",            "{:.2f}"),
    ("OFF MAE (W)","off_mae",           "{:.2f}"),
    ("AP",         "average_precision", "{:.3f}"),
    ("Precision",  "precision",         "{:.3f}"),
    ("Recall",     "recall",            "{:.3f}"),
    ("F1",         "f1",                "{:.3f}"),
]

APPLIANCE_KEYS = [
    ("kettle",         "Kettle"),
    ("fridge",         "Fridge"),
    ("dishwasher",     "Dishwasher"),
    ("washingmachine", "Washer"),
    ("microwave",      "Microwave"),
    ("overall",        "Avg."),
]


def extract_house_metrics(metrics_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Extract metrics mapped by metric_label -> dict of appliance -> formatted value."""
    df = metrics_df.set_index("appliance")
    result = {}
    for metric_label, col_name, fmt in METRIC_SPECS:
        row_vals = {}
        for app_key, app_label in APPLIANCE_KEYS:
            if app_key in df.index and col_name in df.columns:
                val = df.loc[app_key, col_name]
                try:
                    v_float = float(val)
                    row_vals[app_label] = fmt.format(v_float) if np.isfinite(v_float) else "—"
                except (ValueError, TypeError):
                    row_vals[app_label] = "—"
            else:
                row_vals[app_label] = "—"
        result[metric_label] = row_vals
    return result


def plot_benchmark_table_png(
    ukdale_metrics_path: Path | str,
    refit_metrics_path: Path | str,
    output_png_path: Path | str,
    dpi: int = 300,
) -> Path:
    """Render the exact two-house benchmark comparison table as high-res PNG."""
    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    df_ukdale = pd.read_csv(ukdale_metrics_path) if Path(ukdale_metrics_path).exists() else pd.DataFrame()
    df_refit = pd.read_csv(refit_metrics_path) if Path(refit_metrics_path).exists() else pd.DataFrame()

    m_ukdale = extract_house_metrics(df_ukdale) if not df_ukdale.empty else {}
    m_refit = extract_house_metrics(df_refit) if not df_refit.empty else {}

    col_headers_ukdale = [label for _, label in APPLIANCE_KEYS]
    col_headers_refit = [label for _, label in APPLIANCE_KEYS]

    all_cols = ["Metric"] + col_headers_ukdale + col_headers_refit
    n_cols = len(all_cols)
    n_rows = len(METRIC_SPECS)

    # Palette
    c_header_bg = "#1A365D"       # Deep navy blue header bar
    c_header_fg = "#FFFFFF"       # White text
    c_avg_bg    = "#EBF3FB"       # Soft blue tint for Avg. columns
    c_row_alt   = "#F8FAFC"       # Very subtle zebra row
    c_row_white = "#FFFFFF"
    c_border    = "#CBD5E1"       # Subtle border
    c_text      = "#0F172A"       # Dark text

    # Column widths (relative)
    # Metric column is slightly wider
    col_widths = [1.5] + [1.0] * 6 + [1.0] * 6
    total_w = sum(col_widths)
    norm_widths = [w / total_w for w in col_widths]

    fig_w = 14.0
    fig_h = 4.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Coordinates
    top_y = 0.95
    super_header_h = 0.09
    col_header_h = 0.08
    row_h = (top_y - super_header_h - col_header_h - 0.04) / n_rows

    # Calculate column X positions
    x_positions = [0.02]
    curr_x = 0.02
    avail_w = 0.96
    norm_w_pixels = [w * avail_w for w in norm_widths]
    for w in norm_w_pixels:
        curr_x += w
        x_positions.append(curr_x)

    # 1. Super Headers: UK-DALE · House 2 and REFIT · House 20
    # UK-DALE spans col 1 to 6 (0-indexed col 1 to 6)
    ukdale_x0 = x_positions[1]
    ukdale_x1 = x_positions[7]
    refit_x0  = x_positions[7]
    refit_x1  = x_positions[13]

    super_y = top_y - super_header_h
    ax.text(
        (ukdale_x0 + ukdale_x1) / 2,
        super_y + super_header_h * 0.45,
        "UK-DALE · House 2",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color=c_text,
    )
    ax.text(
        (refit_x0 + refit_x1) / 2,
        super_y + super_header_h * 0.45,
        "REFIT · House 20",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color=c_text,
    )

    # 2. Main Column Headers Bar
    header_y = super_y - col_header_h
    # Draw navy bar for the whole row
    bar_rect = patches.Rectangle(
        (x_positions[0], header_y),
        avail_w, col_header_h,
        facecolor=c_header_bg, edgecolor="none", zorder=2
    )
    ax.add_patch(bar_rect)

    # Text for headers
    for c_idx, col_name in enumerate(all_cols):
        cx = (x_positions[c_idx] + x_positions[c_idx + 1]) / 2
        cy = header_y + col_header_h / 2
        ax.text(
            cx, cy, col_name,
            ha="center", va="center",
            fontsize=10, fontweight="bold", color=c_header_fg,
            zorder=3
        )

    # 3. Data Rows
    for r_idx, (m_label, _, _) in enumerate(METRIC_SPECS):
        y_bottom = header_y - (r_idx + 1) * row_h
        y_center = y_bottom + row_h / 2
        bg_color = c_row_alt if r_idx % 2 == 1 else c_row_white

        # Draw base row background
        row_rect = patches.Rectangle(
            (x_positions[0], y_bottom),
            avail_w, row_h,
            facecolor=bg_color, edgecolor="none", zorder=1
        )
        ax.add_patch(row_rect)

        # Highlight Avg. columns with c_avg_bg
        for avg_c_idx in (7, 13):
            avg_rect = patches.Rectangle(
                (x_positions[avg_c_idx - 1], y_bottom),
                norm_w_pixels[avg_c_idx - 1], row_h,
                facecolor=c_avg_bg, edgecolor="none", zorder=2
            )
            ax.add_patch(avg_rect)

        # Subtle bottom line
        line = plt.Line2D(
            [x_positions[0], x_positions[-1]], [y_bottom, y_bottom],
            color=c_border, linewidth=0.5, zorder=3
        )
        ax.add_line(line)

        # Cell values:
        # Col 0: Metric label
        ax.text(
            x_positions[0] + 0.015, y_center,
            m_label,
            ha="left", va="center",
            fontsize=9.5, fontweight="bold" if "F1" in m_label or "MAE" in m_label else "normal",
            color=c_text, zorder=4
        )

        # UK-DALE values (cols 1..6)
        uk_dict = m_ukdale.get(m_label, {})
        for a_idx, app_label in enumerate(col_headers_ukdale):
            col_pos = 1 + a_idx
            cx = (x_positions[col_pos] + x_positions[col_pos + 1]) / 2
            val_str = uk_dict.get(app_label, "—")
            is_avg = (app_label == "Avg.")
            ax.text(
                cx, y_center, val_str,
                ha="center", va="center",
                fontsize=9.5,
                fontweight="bold" if is_avg else "normal",
                color=c_text, zorder=4
            )

        # REFIT values (cols 7..12)
        rf_dict = m_refit.get(m_label, {})
        for a_idx, app_label in enumerate(col_headers_refit):
            col_pos = 7 + a_idx
            cx = (x_positions[col_pos] + x_positions[col_pos + 1]) / 2
            val_str = rf_dict.get(app_label, "—")
            is_avg = (app_label == "Avg.")
            ax.text(
                cx, y_center, val_str,
                ha="center", va="center",
                fontsize=9.5,
                fontweight="bold" if is_avg else "normal",
                color=c_text, zorder=4
            )

    # Vertical subtle separator between UK-DALE and REFIT
    sep_line = plt.Line2D(
        [x_positions[7], x_positions[7]], [header_y - n_rows * row_h, header_y + col_header_h],
        color=c_border, linewidth=1.2, zorder=5
    )
    ax.add_line(sep_line)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return output_png_path


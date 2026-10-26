"""Plot four frozen seed-2026 models on identical, documented example intervals.

Uses downloaded prediction bundles only: no fitting, inference, or smoothing.
The multi-task bundle must be the uniform-decoding re-evaluation, as in Table III.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
APPLIANCES = ("kettle", "fridge", "dishwasher", "washingmachine", "microwave")
NAMES = ("Kettle", "Fridge", "Dishwasher", "Washing machine", "Microwave")
MODELS = ("unetnilm", "mat_nilm", "multi_task", "mart_nilm")
LABELS = ("UNet-NILM", "MAT-Conv", "Multi-task net.", "MART-NILM")
COLORS = ("#D89000", "#009E73", "#A45A99", "#0072B2")
SPLITS = ("ukdale_house2", "refit_house20")
HOUSES = ("UK-DALE · House 2", "REFIT · House 20")


def load_aligned(source: Path, data_root: Path, split: str, limits: dict) -> dict:
    bundles = {}
    paths = {}
    common = None
    for model in MODELS:
        path = source / model / "test" / split / "predictions.npz"
        with np.load(path, allow_pickle=False) as data:
            bundle = {key: data[key] for key in data.files}
        times = bundle["csv_timesteps"].astype(np.int64)
        if len(times) == 0 or np.any(np.diff(times) <= 0):
            raise ValueError(f"Missing/non-unique chronological CSV indices: {path}")
        bundles[model] = bundle
        paths[model] = str(path.resolve())
        common = times if common is None else np.intersect1d(common, times)

    columns = ["readable_time", "aggregate", "sequence_id"]
    columns += [f"{app}_{suffix}" for app in APPLIANCES for suffix in ("power", "on")]
    frame = pd.read_csv(data_root / "testing" / split / "multi_appliance_testing.csv")
    required = ["aggregate"] + [f"{app}_{s}" for app in APPLIANCES for s in ("power", "on")]
    frame = frame.dropna(subset=required).reset_index(drop=True)
    timeline = frame.iloc[common][columns].reset_index(drop=True)
    truth = timeline[[f"{app}_power" for app in APPLIANCES]].to_numpy()
    predictions = {}
    for model, bundle in bundles.items():
        rows = np.searchsorted(bundle["csv_timesteps"], common)
        app_indices = [bundle["appliances"].tolist().index(app) for app in APPLIANCES]
        reference = bundle["y_true_watts"][rows][:, app_indices]
        # Target standardization/denormalization may introduce float32 roundoff.
        if not np.allclose(reference, np.maximum(truth, 0), atol=0.02, rtol=1e-5):
            raise ValueError(f"Ground truth is misaligned for {model}/{split}")
        pred = np.maximum(bundle["y_pred_watts"][rows][:, app_indices], 0)
        # The saved bundles already contain validation-calibrated state-to-power
        # masking. Apply only the same final physical limits used by Table III.
        pred[pred < limits["min_power_watts"]] = 0
        cap = [limits["max_on_power_watts"][app] for app in APPLIANCES]
        predictions[model] = np.minimum(pred, cap)
    unix_time = frame.iloc[common]["unix_time"].to_numpy()
    boundary = np.r_[True, (np.diff(common) != 1) | (np.diff(unix_time) != 8)
                     | (np.diff(timeline["sequence_id"].to_numpy()) != 0)]
    return dict(timeline=timeline, truth=truth, predictions=predictions,
                csv_rows=common, segments=np.cumsum(boundary), sources=paths)


def select_interval(data: dict, app_i: int, success_case: bool = False) -> tuple[int, int, dict]:
    """Choose clear examples; explicitly label optional performance-selected cases."""
    app = APPLIANCES[app_i]
    on = data["timeline"][f"{app}_on"].to_numpy().astype(bool)
    segments = data["segments"]
    boundary = np.r_[True, np.diff(segments) != 0]
    starts = np.flatnonzero(on & (boundary | np.r_[True, ~on[:-1]]))
    ends = np.flatnonzero(on & (np.r_[boundary[1:], True] | np.r_[~on[1:], True])) + 1
    events = [(int(s), int(e)) for s, e in zip(starts, ends)
              if s > 0 and e < len(on) and not on[s - 1] and not on[e]
              and segments[s - 1] == segments[e] and e - s >= 2]
    if not events:
        raise ValueError(f"No complete ON events for {app}")
    energies = np.array([data["truth"][s:e, app_i].sum() for s, e in events])
    typical = (energies >= np.quantile(energies, .25)) & (energies <= np.quantile(energies, .75))
    candidates = []
    aggregate = data["timeline"]["aggregate"].to_numpy()
    truth = data["truth"][:, app_i]
    for i, (s, e) in enumerate(events):
        if not typical[i]:
            continue
        background = np.maximum(aggregate[s:e] - truth[s:e], 0).mean()
        candidates.append((background, s, e))
    # The lower-quartile background is a clear illustrative condition, rather
    # than the easiest extreme. No model outputs enter this choice.
    candidates.sort()
    background, s, e = candidates[(len(candidates) - 1) // 4]
    if success_case:
        def relative_error(candidate: tuple) -> float:
            _, start, end = candidate
            pad = 30 if app == "fridge" else max(45, int((end - start) * .22))
            a, b = max(0, start - pad), min(len(truth), end + pad)
            prediction = data["predictions"]["mart_nilm"][a:b, app_i]
            return float(np.abs(prediction - truth[a:b]).sum() / max(truth[a:b].sum(), 1))

        background, s, e = min(candidates, key=relative_error)
    margin = max(45, int((e - s) * .22))
    if app == "fridge":
        pairs = []
        for i in range(len(events) - 1):
            s1, e1 = events[i]
            s2, e2 = events[i + 1]
            if not (typical[i] and typical[i + 1]) or segments[s1] != segments[e2]:
                continue
            duration_ratio = (e1 - s1) / (e2 - s2)
            power_ratio = truth[s1:e1].mean() / max(truth[s2:e2].mean(), 1)
            if not (.75 <= duration_ratio <= 1.33 and .8 <= power_ratio <= 1.25):
                continue
            if not (150 <= s2 - e1 <= 600):  # Separate cycles by 20--80 min.
                continue
            background = np.maximum(aggregate[s1:e2] - truth[s1:e2], 0).mean()
            pairs.append((background, s1, e2))
        if not pairs:
            raise ValueError("No two regular complete fridge cycles found")
        pairs.sort()
        background, s, e = pairs[(len(pairs) - 1) // 4]
        if success_case:
            # Keep aggregate context on the same watt axis while retaining
            # visible compressor detail: prefer pairs without >800 W demand.
            readable_pairs = [p for p in pairs
                              if aggregate[max(0, p[1] - 30):p[2] + 30].max() <= 800]
            background, s, e = min(readable_pairs or pairs, key=relative_error)
        margin = 30  # Four minutes around the pair; avoid a third partial cycle.
    segment_rows = np.flatnonzero(segments == segments[s])
    left = max(int(segment_rows[0]), s - margin)
    right = min(int(segment_rows[-1]) + 1, e + margin)
    metadata = dict(appliance=app, complete_event_count=len(events),
                    selection=("Selected MART-NILM success case: minimum relative waveform L1 error among typical-energy candidates; fridge: two regular consecutive cycles, preferring aggregate peaks <=800 W for legibility"
                               if success_case else "Typical-energy events at lower-quartile interfering demand; fridge: two regular consecutive cycles"),
                    mean_interfering_power_watts=float(background),
                    event_csv_start=int(data["csv_rows"][s]),
                    event_csv_end_inclusive=int(data["csv_rows"][e - 1]),
                    crop_csv_start=int(data["csv_rows"][left]),
                    crop_csv_end_inclusive=int(data["csv_rows"][right - 1]),
                    start_time=str(data["timeline"].iloc[left]["readable_time"]),
                    end_time=str(data["timeline"].iloc[right - 1]["readable_time"]))
    return left, right, metadata


def draw(data_by_split: dict, crops: dict, output: Path, context: bool = False) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "mathtext.fontset": "stix", "font.size": 7,
                         "axes.labelsize": 7, "axes.titlesize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.linewidth": .5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.16, 2.65))
    handles = []
    for row, split in enumerate(SPLITS):
        data = data_by_split[split]
        shift = row * .435
        fig.add_artist(Rectangle((.006, .510 - shift), .958, .425,
                                 transform=fig.transFigure, fill=False,
                                 edgecolor=".50", linewidth=.55,
                                 linestyle=(0, (4, 3))))
        fig.text(.019, .899 - shift, HOUSES[row].replace(" · ", " — "),
                 fontsize=7.5, weight="bold", va="center")
        fig.text(.014, .738 - shift, "Power (W)", rotation=90,
                 ha="center", va="center", fontsize=7)
        for col, name in enumerate(NAMES):
            ax = fig.add_axes([.067 + col * .185, .620 - shift, .145, .240])
            left, right, _ = crops[split][col]
            minutes = np.arange(right - left) * 8 / 60
            sl = slice(left, right)
            if context:
                aggregate_watts = data["timeline"]["aggregate"].to_numpy()[sl]
                ax.fill_between(minutes, 0, aggregate_watts, color=".5", alpha=.06, zorder=0)
                ax.plot(minutes, aggregate_watts, color=".60",
                        linestyle=(0, (3, 2)), lw=.50, zorder=1)
            for i, (model, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
                line, = ax.plot(minutes, data["predictions"][model][sl, col],
                                color=color, lw=.68 if i < 3 else .85, label=label,
                                alpha=1, zorder=3 if i < 3 else 4)
                if row == 0 and col == 0:
                    handles.append(line)
            gt, = ax.plot(minutes, data["truth"][sl, col], color="#202020",
                          lw=.65, linestyle=(0, (3, 2)), label="Ground truth", zorder=5)
            if row == 0 and col == 0:
                handles.insert(0, gt)
            letter = chr(ord("a") + row * 5 + col)
            box = ax.get_position()
            fig.text((box.x0 + box.x1) / 2, .540 - shift,
                     f"({letter}) {name}", ha="center", va="center", fontsize=7)
            ax.set_xlim(minutes[0], minutes[-1])
            ax.set_ylim(bottom=-.025 * ax.get_ylim()[1])
            ax.xaxis.set_major_locator(MaxNLocator(3, min_n_ticks=2))
            ax.yaxis.set_major_locator(MaxNLocator(3, min_n_ticks=2))
            ax.tick_params(length=2, width=.45, pad=1.5)
            ax.grid(axis="y", color=".93", linewidth=.35, zorder=0)
            for spine in ax.spines.values():
                spine.set_color(".20")
    if context:
        handles.append(Line2D([], [], color=".60", linestyle="--", label="Aggregate"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.51, .990),
               ncol=len(handles), fontsize=6.5, frameon=False, handlelength=2.0,
               columnspacing=.9, handletextpad=.4, borderaxespad=0)
    stem = "baseline_waveforms_context" if context else "baseline_waveforms"
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=.03)
    fig.savefig(output / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/paper_5w/waveform_comparison")
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "datasets/mixed_ukdale_refit_5w_all_house_split")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "paper/figures")
    args = parser.parse_args()
    config_path = args.source / "mart_nilm/config_merged.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["experiment"]["evaluation"]
    limits = dict(min_power_watts=config["power_postprocess"]["min_power_watts"],
                  max_on_power_watts=config["max_on_power_watts"])
    for model in MODELS:
        saved_config = yaml.safe_load((args.source / model / "config_merged.yaml").read_text(encoding="utf-8"))
        evaluation = saved_config["experiment"]["evaluation"]
        if (evaluation["power_postprocess"] != config["power_postprocess"]
                or evaluation["max_on_power_watts"] != config["max_on_power_watts"]):
            raise ValueError(f"Physical output limits differ for {model}")
    data = {split: load_aligned(args.source, args.data_root, split, limits) for split in SPLITS}
    crops = {split: [select_interval(data[split], i,
                                    success_case=split == "refit_house20" and i in (1, 3))
                    for i in range(5)] for split in SPLITS}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw(data, crops, args.output_dir)
    draw(data, crops, args.output_dir, context=True)
    metadata = dict(seed=2026, sample_seconds=8, smoothing="none",
                    selection="Illustrative examples, not a random sample or performance estimate. REFIT fridge and washing-machine intervals are explicitly selected MART-NILM success cases; see per-panel selection records. Other intervals use typical energy and lower-quartile interfering demand.",
                    original_run_paths={model: (
                        "runs/paper_5w/postprocess_audit/transfer_multi_appliance/calibrated/seed_2026"
                        if model == "multi_task" else f"runs/paper_5w/table_v/{model}/seed_2026"
                    ) for model in MODELS},
                    power_limits=limits,
                    sources={split: data[split]["sources"] for split in SPLITS},
                    intervals={split: [item[2] for item in crops[split]] for split in SPLITS})
    (args.output_dir / "baseline_waveforms_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    selected_traces = []
    for split in SPLITS:
        for i, (left, right, _) in enumerate(crops[split]):
            sample = pd.DataFrame({"house": split, "appliance": APPLIANCES[i],
                                   "csv_row": data[split]["csv_rows"][left:right],
                                   "elapsed_minutes": np.arange(right - left) * 8 / 60,
                                   "aggregate_watts": data[split]["timeline"]["aggregate"].to_numpy()[left:right],
                                   "ground_truth_watts": data[split]["truth"][left:right, i]})
            for model in MODELS:
                sample[model] = data[split]["predictions"][model][left:right, i]
            selected_traces.append(sample)
    pd.concat(selected_traces, ignore_index=True).to_csv(
        args.source / "selected_waveform_values.csv", index=False)
    print(f"Saved waveform figures and provenance to {args.output_dir}")


if __name__ == "__main__":
    main()

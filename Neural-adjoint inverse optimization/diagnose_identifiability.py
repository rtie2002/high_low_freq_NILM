"""Frozen-checkpoint diagnostics for source identifiability (D1--D3).

This script is deliberately diagnostic only.  It never trains a model, changes
the inverse objective, or evaluates the held-out test partition.  It uses
real validation appliance traces only to create known-valid encoder codes and
to score the *oracle* trajectory; labels never enter an inverse update.

D1: keep each encoded validation code fixed and switch only its decoder ID.
D2: estimate decoder-tangent overlap with random Jacobian-vector products.
D3: compare training codes, deployed train-mean starts, and oracle-inversion
    codes at steps 3, 25 and 150 against sampled training-code references.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

# The local Anaconda build loads both PyTorch and NumPy OpenMP libraries.  This
# diagnostic has no parallel numerical reduction whose result depends on that
# implementation choice; permit both libraries so saved figures are produced.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import Tensor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forward_model.cnn import ApplianceConditionedCNN, build_forward_model
from utils.config import load_yaml
from utils.protocol import load_protocol


CODE_STEPS = (0, 3, 25, 150)
N_EVENTS_PER_SOURCE = 24
N_TRAIN_REFERENCE_WINDOWS = 128
N_JVP_PROBES = 16
EPS = 1e-8


def decode_one(model: ApplianceConditionedCNN, code: Tensor, identity: int, length: int) -> Tensor:
    """Decode one [Z, L] code through exactly one requested appliance ID."""
    packed = code.new_zeros((1, len(model.APPLIANCES), *code.shape))
    packed[:, identity] = code
    return model.decode(packed, output_length=length)[0, identity]


def selected_on_windows(protocol, split, length: int, per_source: int) -> dict[int, np.ndarray]:
    """Replicate the deterministic ON-event selection used by Experiment C."""
    raw = protocol.loader.get_split(split)
    starts = protocol.loader.build_dataset(split).indices
    answer: dict[int, np.ndarray] = {}
    offsets = np.arange(length)
    for i, app in enumerate(protocol.appliances):
        rows = starts[:, None] + offsets
        states = raw.states[rows, i].astype(bool)
        active = states & (raw.targets[rows, i] >= protocol.thresholds_watts[i])
        eligible = starts[(states[:, 1:] & ~states[:, :-1]).any(axis=1) & (active.sum(axis=1) >= 2)]
        nonoverlap: list[int] = []
        for start in eligible:
            if not nonoverlap or start >= nonoverlap[-1] + length:
                nonoverlap.append(int(start))
        if not nonoverlap:
            raise ValueError(f"No eligible ON-event window for {app}.")
        picks = np.linspace(0, len(nonoverlap) - 1, min(per_source, len(nonoverlap)), dtype=int)
        answer[i] = np.asarray(nonoverlap, dtype=np.int64)[picks]
    return answer


def encode_source_windows(model, raw, starts: np.ndarray, source: int, length: int, scale: float) -> Tensor:
    """Return valid [N, Z, L] codes from one source's actual appliance windows."""
    rows = starts[:, None] + np.arange(length)
    trace = torch.as_tensor(raw.targets[rows, source] / scale, dtype=torch.float32)
    tiled = trace[:, None, :].expand(-1, len(model.APPLIANCES), -1)
    with torch.no_grad():
        return model.encode(tiled)[:, 0].cpu()


def normalised_difference(left: Tensor, right: Tensor) -> Tensor:
    """||left-right|| / (||left||+||right||), calculated per waveform."""
    return (left - right).norm(dim=-1) / (left.norm(dim=-1) + right.norm(dim=-1) + EPS)


def run_d1(model, protocol, raw, codes_by_source, length, scale, output: Path) -> pd.DataFrame:
    """Fixed-code identity sensitivity.  Rows are code origins, columns decoder IDs."""
    apps = protocol.appliances
    norm = np.zeros((len(apps), len(apps)), dtype=np.float64)
    rms_w = np.zeros_like(norm)
    rows: list[dict[str, object]] = []
    for origin, codes in codes_by_source.items():
        # Same code; only the fixed one-hot decoder condition changes.
        outputs = []
        for code in codes:
            outputs.append(torch.stack([decode_one(model, code, j, length) for j in range(len(apps))]))
        wave = torch.stack(outputs)  # [event, decoder-ID, T]
        own = wave[:, origin]
        for target in range(len(apps)):
            difference = normalised_difference(own, wave[:, target])
            delta_rms = (own - wave[:, target]).square().mean(dim=-1).sqrt() * scale
            norm[origin, target] = float(difference.median())
            rms_w[origin, target] = float(delta_rms.median())
            rows.append({
                "code_origin": apps[origin], "decoder_identity": apps[target],
                "median_normalized_difference": norm[origin, target],
                "median_difference_rms_w": rms_w[origin, target],
                "event_count": len(codes),
            })
    table = pd.DataFrame(rows)
    table.to_csv(output / "D1_identity_sensitivity_long.csv", index=False)
    pd.DataFrame(norm, index=apps, columns=apps).to_csv(output / "D1_identity_sensitivity_normalized.csv")
    pd.DataFrame(rms_w, index=apps, columns=apps).to_csv(output / "D1_identity_sensitivity_rms_w.csv")
    return table


def random_tangent_basis(
    model: ApplianceConditionedCNN, code: Tensor, identity: int, length: int, probes: int, generator: torch.Generator
) -> Tensor:
    """Return an orthonormal basis for sampled J_i directions without dense J_i.

    For each normalized random vector v in latent space, autograd computes the
    Jacobian-vector product J_i v.  QR of those output vectors approximates the
    reachable local waveform tangent space under this finite probe budget.
    """
    directions: list[Tensor] = []
    for _ in range(probes):
        vector = torch.randn(code.shape, generator=generator, dtype=code.dtype)
        vector = vector / vector.norm().clamp_min(EPS)
        _, jvp = torch.autograd.functional.jvp(
            lambda z: decode_one(model, z, identity, length), (code,), (vector,), create_graph=False
        )
        if float(jvp.norm()) > 1e-10:
            directions.append(jvp.detach())
    if not directions:
        return torch.empty(length, 0)
    matrix = torch.stack(directions, dim=1)  # [time, probes]
    return torch.linalg.qr(matrix, mode="reduced").Q


def run_d2(model, apps, codes_by_source, length, output: Path) -> pd.DataFrame:
    """Approximate cross-ID tangent-space overlap and cancellation residuals."""
    generator = torch.Generator().manual_seed(20261112)
    # Eight deterministic valid codes per true appliance keep this diagnostic compact.
    selected = [(source, code) for source, codes in codes_by_source.items() for code in codes[:8]]
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    for sample, (origin, code) in enumerate(selected):
        bases = [random_tangent_basis(model, code, identity, length, N_JVP_PROBES, generator)
                 for identity in range(len(apps))]
        for left in range(len(apps)):
            for right in range(left + 1, len(apps)):
                cross = bases[left].T @ bases[right]
                # Largest singular value is the closest cosine between the two
                # sampled tangent subspaces.  sqrt(1-s^2) is the residual after
                # the best unit-norm two-source cancellation in this subspace.
                singular = torch.linalg.svdvals(cross)
                closest_cosine = float(singular.max()) if len(singular) else float("nan")
                cancellation = math.sqrt(max(0.0, 1.0 - closest_cosine**2))
                records.append({
                    "sample": sample, "code_origin": apps[origin],
                    "left_identity": apps[left], "right_identity": apps[right],
                    "left_rank": bases[left].shape[1], "right_rank": bases[right].shape[1],
                    "closest_tangent_cosine": closest_cosine,
                    "best_two_source_cancellation_residual": cancellation,
                })
    result = pd.DataFrame(records)
    result.to_csv(output / "D2_jacobian_pair_samples.csv", index=False)
    summary = result.groupby(["left_identity", "right_identity"])[
        ["closest_tangent_cosine", "best_two_source_cancellation_residual"]
    ].agg(["median", "mean", "min", "max"])
    summary.to_csv(output / "D2_jacobian_pair_summary.csv")
    (output / "D2_runtime.json").write_text(json.dumps({
        "valid_codes": len(selected), "probes_per_identity": N_JVP_PROBES,
        "pairs_per_code": 10, "elapsed_seconds": time.perf_counter() - started,
        "method": "random JVPs followed by QR bases and principal-angle SVD",
    }, indent=2))
    return result


def optimize_oracle_trajectory(model, initial: Tensor, meter: Tensor, length: int) -> dict[int, Tensor]:
    """Replay the unchanged no-background oracle objective solely to save codes."""
    codes = initial.detach().clone().requires_grad_()
    optimizer = torch.optim.Adam([codes], lr=0.05)
    snapshots = {0: codes.detach().cpu().clone()}
    for step in range(1, 151):
        power = model.decode(codes, output_length=length)
        loss = (power.sum(dim=1) - meter).square().mean(dim=-1).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in CODE_STEPS:
            snapshots[step] = codes.detach().cpu().clone()
    return snapshots


def nearest_distance(codes: Tensor, references: Tensor) -> Tensor:
    """Nearest Euclidean distance to sampled valid train codes, source by source."""
    # codes [N,5,Z,L], references [R,5,Z,L].  No all-pairs dense Jacobian;
    # this small R=128 code reference is intentionally sampled and explicit.
    flat_codes = codes.flatten(2)
    flat_ref = references.flatten(2)
    return torch.stack([
        torch.cdist(flat_codes[:, i], flat_ref[:, i]).min(dim=1).values
        for i in range(flat_codes.shape[1])
    ], dim=1)


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Small dependency-free Spearman approximation, robust to ties."""
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(pd.Series(left).rank().corr(pd.Series(right).rank(), method="pearson"))


def run_d3(model, protocol, raw, train_codes, event_starts, length, scale, output: Path) -> pd.DataFrame:
    """Measure support distance and output/error drift along the oracle trajectory."""
    # Match Experiment B's 48 saved validation starts whenever available.  The
    # manifest is a diagnostic provenance source; it supplies no labels to Adam.
    archive_manifest = Path(output).parent / "zero_background_diagnostic/run_manifest.json"
    if archive_manifest.exists():
        starts = np.asarray(json.loads(archive_manifest.read_text())["validation_window_starts"], dtype=np.int64)
    else:
        starts = np.unique(np.concatenate(list(event_starts.values())))[:48]
    rows = starts[:, None] + np.arange(length)
    truth = torch.as_tensor(raw.targets[rows] / scale, dtype=torch.float32).transpose(1, 2)
    states = torch.as_tensor(raw.states[rows].astype(bool)).transpose(1, 2)
    with torch.no_grad():
        initial = model.encode(truth)
        meter = truth.sum(dim=1)
        train_center = train_codes.mean(dim=(0, 3), keepdim=True)
        deployed_start = train_center.expand_as(initial).cpu()

    batches = []
    for initial_batch, meter_batch in zip(initial.split(8), meter.split(8)):
        batches.append(optimize_oracle_trajectory(model, initial_batch, meter_batch, length))
    snapshots = {step: torch.cat([batch[step] for batch in batches], dim=0) for step in CODE_STEPS}
    snapshots["deployed_train_mean"] = deployed_start

    records: list[dict[str, object]] = []
    for stage, codes in snapshots.items():
        code_distance = nearest_distance(codes, train_codes)
        code_norm = codes.square().mean(dim=(2, 3)).sqrt()
        initial_move = (codes - initial.cpu()).square().mean(dim=(2, 3)).sqrt()
        with torch.no_grad():
            output_power = model.decode(codes, output_length=length).cpu()
        output_change = (output_power - model.decode(initial, output_length=length).cpu()).square().mean(dim=-1).sqrt()
        abs_error = (output_power - truth.cpu()).abs().mean(dim=-1) * scale
        active = states.cpu()
        on_error = ((output_power - truth.cpu()).abs() * active).sum(dim=-1) / active.sum(dim=-1).clamp_min(1)
        on_error = on_error * scale
        for window in range(len(starts)):
            for source, app in enumerate(protocol.appliances):
                records.append({
                    "stage": str(stage), "csv_start": int(starts[window]), "appliance": app,
                    # A whole-window flag is only for support diagnostics.  It
                    # never enters the inverse solver or model input.
                    "window_contains_on": bool(active[window, source].any()),
                    "nearest_train_code_distance": float(code_distance[window, source]),
                    "code_rms": float(code_norm[window, source]),
                    "movement_from_oracle_initial_rms": float(initial_move[window, source]),
                    "decoded_change_from_oracle_initial_rms_w": float(output_change[window, source] * scale),
                    "mae_w": float(abs_error[window, source]),
                    "on_mae_w": float(on_error[window, source]),
                    "on_samples": int(active[window, source].sum()),
                })
    result = pd.DataFrame(records)
    result.to_csv(output / "D3_latent_validity_per_window.csv", index=False)
    summary = result.groupby(["stage", "appliance"]).median(numeric_only=True)
    summary.to_csv(output / "D3_latent_validity_summary.csv")
    result.groupby(["stage", "appliance", "window_contains_on"]).median(numeric_only=True).to_csv(
        output / "D3_latent_validity_on_off_summary.csv"
    )
    # Conditional correlation is reported only for sources/windows that are ON.
    on = result[result["on_samples"] > 0]
    correlations = []
    for (stage, app), group in on.groupby(["stage", "appliance"]):
        correlations.append({
            "stage": stage, "appliance": app, "on_windows": len(group),
            "spearman_nearest_distance_vs_on_mae": rank_correlation(
                group["nearest_train_code_distance"].to_numpy(), group["on_mae_w"].to_numpy()
            ),
            "spearman_movement_vs_on_mae": rank_correlation(
                group["movement_from_oracle_initial_rms"].to_numpy(), group["on_mae_w"].to_numpy()
            ),
        })
    pd.DataFrame(correlations).to_csv(output / "D3_on_error_correlations.csv", index=False)
    return result


def plot(output: Path, apps: list[str]) -> None:
    sensitivity = pd.read_csv(output / "D1_identity_sensitivity_normalized.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    image = ax.imshow(sensitivity, cmap="magma", vmin=0)
    fig.colorbar(image, ax=ax, label="median normalized waveform difference")
    ax.set_xticks(range(len(apps)), apps, rotation=30, ha="right")
    ax.set_yticks(range(len(apps)), apps)
    ax.set(xlabel="decoder identity substituted", ylabel="encoded waveform origin", title="D1: fixed-code identity sensitivity")
    for i, j in np.ndindex(sensitivity.shape):
        ax.text(j, i, f"{sensitivity.iloc[i, j]:.3f}", ha="center", va="center", color="white")
    fig.savefig(output / "D1_identity_sensitivity.png", dpi=240)
    plt.close(fig)

    pairs = pd.read_csv(output / "D2_jacobian_pair_samples.csv")
    labels = pairs["left_identity"] + "–" + pairs["right_identity"]
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    pairs.assign(pair=labels).boxplot(column="closest_tangent_cosine", by="pair", ax=ax, rot=35)
    ax.set(title="D2: sampled decoder-tangent alignment", xlabel="identity pair", ylabel="closest tangent cosine")
    plt.suptitle("")
    fig.savefig(output / "D2_jacobian_alignment.png", dpi=240)
    plt.close(fig)

    validity = pd.read_csv(output / "D3_latent_validity_per_window.csv")
    summary = validity.groupby("stage")["nearest_train_code_distance"].median().reindex(
        ["0", "3", "25", "150", "deployed_train_mean"]
    )
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(summary.index, summary.values, marker="o")
    ax.set(title="D3: median nearest sampled training-code distance", xlabel="trajectory stage", ylabel="latent Euclidean distance")
    fig.savefig(output / "D3_latent_support_distance.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="rerun_... archive containing forward/best.pt")
    parser.add_argument("output", type=Path, help="new or empty output directory")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parent
    cfg_path = project / "forward_training/train_forward.yaml"
    cfg = load_yaml(cfg_path)
    checkpoint = torch.load(args.archive / "forward/best.pt", map_location="cpu", weights_only=True)
    model = build_forward_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().requires_grad_(False)
    length = int(checkpoint["window_length"])
    scale = float(checkpoint["power_scale_watts"])
    protocol = load_protocol(experiment_path=cfg_path.parent / cfg["experiment"], window_length=length,
                             stride=int(cfg["data"]["window_stride"]), batch_size=8)
    validation = protocol.loader.get_split("validation")
    training = protocol.loader.get_split("train")

    event_starts = selected_on_windows(protocol, "validation", length, N_EVENTS_PER_SOURCE)
    codes_by_source = {i: encode_source_windows(model, validation, starts, i, length, scale)
                       for i, starts in event_starts.items()}
    train_starts = protocol.loader.build_dataset("train").indices
    sampled = train_starts[np.linspace(0, len(train_starts) - 1, min(N_TRAIN_REFERENCE_WINDOWS, len(train_starts)), dtype=int)]
    train_rows = sampled[:, None] + np.arange(length)
    train_power = torch.as_tensor(training.targets[train_rows] / scale, dtype=torch.float32).transpose(1, 2)
    with torch.no_grad():
        train_codes = model.encode(train_power).cpu()

    run_d1(model, protocol, validation, codes_by_source, length, scale, args.output)
    run_d2(model, protocol.appliances, codes_by_source, length, args.output)
    run_d3(model, protocol, validation, train_codes, event_starts, length, scale, args.output)
    plot(args.output, protocol.appliances)
    manifest = {
        "checkpoint": str((args.archive / "forward/best.pt").resolve()), "split": "validation",
        "test_evaluated": False, "training_performed": False, "inverse_objective_modified": False,
        "event_windows_per_source": N_EVENTS_PER_SOURCE, "train_reference_windows": len(sampled),
        "jvp_probes_per_identity": N_JVP_PROBES, "oracle_steps": list(CODE_STEPS),
        "device": "cpu", "seed": 20261112,
    }
    (args.output / "D_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved D1--D3 diagnostics to {args.output}")


if __name__ == "__main__":
    main()

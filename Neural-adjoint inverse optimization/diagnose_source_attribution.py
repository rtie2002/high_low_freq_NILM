"""Read saved validation reconstructions and diagnose the frozen decoder."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def compute(archive, output):
    import numpy as np
    import pandas as pd
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook

    from diagnose_zero_background import score
    from forward_model.cnn import build_forward_model
    from step2_inverse_optimization import solve_windows
    from utils.config import load_yaml
    from utils.protocol import load_protocol
    from utils.metrics import _event_onset_metrics

    cfg_path = Path("forward_training/train_forward.yaml")
    cfg = load_yaml(cfg_path)
    saved_b = archive / "zero_background_diagnostic"
    manifest = json.loads((saved_b / "run_manifest.json").read_text())
    length, stride = cfg["data"]["window_length"], cfg["data"]["window_stride"]
    protocol = load_protocol(experiment_path=cfg_path.parent / cfg["experiment"],
                             window_length=length, stride=stride, batch_size=8)
    thresholds = protocol.thresholds_watts
    apps = protocol.appliances
    ckpt = torch.load(archive / "forward/best.pt", map_location="cpu", weights_only=True)
    model = build_forward_model(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().requires_grad_(False)
    scale = ckpt["power_scale_watts"]

    recon_dir = archive / "reconstruction/validation"
    with np.load(recon_dir / "reconstruction.npz") as r:
        pred, actual, on = r["reconstructed_watts"], r["input_watts"], r["true_on"]
        decoded_on = pred >= thresholds
        bundle = SimpleNamespace(csv_timesteps=r["csv_timesteps"], segment_ids=r["segment_ids"])
        eval_cfg = protocol.experiment["evaluation"]
        tolerance = round(eval_cfg["event_tolerance_seconds"] / protocol.experiment["csv"]["sample_seconds"])
        _, recall, _ = _event_onset_metrics(bundle, on, decoded_on, tolerance)
        table = pd.read_csv(recon_dir / "reconstruction_metrics.csv")
        table["f1"] = score(torch.from_numpy((pred / scale).T[None]),
                            torch.from_numpy((actual / scale).T[None]), on, thresholds, scale)["f1"]
        table["event_recall"] = recall
        table["on_mean_true_watts"] = [actual[on[:, i], i].mean() for i in range(5)]
        table["on_mean_pred_watts"] = [pred[on[:, i], i].mean() for i in range(5)]
        table.to_csv(output / "A_reconstruction.csv", index=False)

    val = protocol.loader.get_split("validation")
    rows = np.array(manifest["validation_window_starts"])[:, None] + np.arange(length)
    truth = torch.from_numpy(val.targets[rows] / scale).transpose(1, 2)
    states = val.states[rows].astype(bool)
    initial = model.encode(truth)
    meter = truth.sum(1)
    trace_batches, movement_batches = [], []
    for x, y in zip(initial.split(8), meter.split(8)):
        trace = [model.decode(x, length)]
        movement = [torch.zeros_like(x).square().mean((2, 3))]

        @torch.no_grad()
        def observe(opt, _args, _kwargs):
            z = opt.param_groups[0]["params"][0]
            trace.append(model.decode(z, length))
            movement.append((z - x).square().mean((2, 3)))

        handle = register_optimizer_step_post_hook(observe)
        try:
            solve_windows(model, y, steps=manifest["steps"], learning_rate=manifest["learning_rate"],
                          generator=torch.Generator().manual_seed(manifest["seed"]),
                          initial_codes=x, fit_background=False)
        finally:
            handle.remove()
        trace_batches.append(torch.stack(trace))
        movement_batches.append(torch.stack(movement))
    trace = torch.cat(trace_batches, dim=1)
    movement = torch.cat(movement_batches, dim=1).mean(1).sqrt().numpy()
    del trace_batches
    summary, per_app = [], []
    for iteration, power in enumerate(trace):
        pred = (power.transpose(1, 2) * scale).numpy()
        err = (power - truth).abs().transpose(1, 2).numpy() * scale
        state_pred = pred >= thresholds
        f1 = 2 * (states & state_pred).sum((0, 1)) / np.maximum((states.sum((0, 1)) + state_pred.sum((0, 1))), 1)
        mae = err.mean((0, 1))
        on_mae = (err * states).sum((0, 1)) / np.maximum(states.sum((0, 1)), 1)
        rmse = float((power.sum(1) - meter).square().mean().sqrt() * scale)
        summary.append([iteration, rmse, mae.mean(), on_mae.mean(), f1.mean(), movement[iteration].mean()])
        per_app.extend([iteration, a, mae[i], on_mae[i], f1[i], movement[iteration, i]] for i, a in enumerate(apps))
    columns = ["iteration", "meter_rmse_w", "mae_w", "on_mae_w", "macro_f1", "latent_rms"]
    pd.DataFrame(summary, columns=columns).to_csv(output / "B_oracle_trajectory.csv", index=False)
    pd.DataFrame(per_app, columns=["iteration", "appliance", "mae_w", "on_mae_w", "f1", "latent_rms"]).to_csv(
        output / "B_per_appliance.csv", index=False)
    del trace
    print("A metrics and B oracle trajectory saved", flush=True)

    starts = protocol.loader.build_dataset("validation").indices
    # Nonoverlapping candidates within each appliance. Require a labelled ON
    # onset and at least two measured above-threshold samples in the same window.
    fit_rows, examples = [], []
    for i, appliance in enumerate(apps):
        windows = starts[:, None] + np.arange(length)
        flags = val.states[windows, i].astype(bool)
        active = flags & (val.targets[windows, i] >= thresholds[i])
        eligible = starts[(flags[:, 1:] & ~flags[:, :-1]).any(1) & (active.sum(1) >= 2)]
        distinct = []
        for start in eligible:
            if not distinct or start >= distinct[-1] + length:
                distinct.append(start)
        take = np.linspace(0, len(distinct) - 1, min(len(distinct), 24), dtype=int)
        if not len(take):
            raise ValueError(f"No eligible ON windows for {appliance}")
        selected = np.array(distinct)[take]
        rows = selected[:, None] + np.arange(length)
        target = torch.from_numpy(val.targets[rows, i] / scale)
        active = val.states[rows, i].astype(bool)
        powers = []
        for batch in target.split(8):
            # The encoder shares weights across identities. Tile the same
            # target and encoded initialization into every candidate slot.
            x = batch[:, None].expand(-1, 5, -1)
            encoded = model.encode(x)
            best_error = torch.full(x.shape[:2], float("inf"))
            best = torch.zeros_like(x)
            for initial_code in (torch.zeros_like(encoded), encoded):
                z = initial_code.detach().clone().requires_grad_()
                optimizer = torch.optim.Adam([z], lr=.05)
                for k in range(151):
                    p = model.decode(z, length)
                    err = (p - x).square().mean(-1)
                    with torch.no_grad():
                        better = err < best_error
                        best_error = torch.minimum(best_error, err)
                        best = torch.where(better[..., None], p, best)
                    if k < 150:
                        optimizer.zero_grad(set_to_none=True)
                        err.sum().backward()
                        optimizer.step()
            powers.append(best.numpy() * scale)
        pred = np.concatenate(powers)
        truth_w = target.numpy() * scale
        for n, start in enumerate(selected):
            error = pred[n] - truth_w[n]
            on_rmse = np.sqrt((error[:, active[n]] ** 2).mean(1))
            rmse = np.sqrt((error ** 2).mean(1))
            norm = np.sqrt((truth_w[n, active[n]] ** 2).mean())
            fit_rows.extend([appliance, app, start, active[n].sum(), rmse[j],
                             on_rmse[j], on_rmse[j] / norm, on_rmse[j] / max(on_rmse[i], 1e-6)]
                            for j, app in enumerate(apps))
        # The first selected event is reproducible and chosen independently of error.
        examples.append(np.vstack((truth_w[0], pred[0])))
        print(f"C {appliance}: {len(selected)} windows", flush=True)
    fits = pd.DataFrame(fit_rows, columns=["target", "decoder", "csv_start", "on_samples",
                                           "rmse_w", "on_rmse_w", "normalized_on_rmse", "relative_to_correct"])
    fits.to_csv(output / "C_window_errors.csv", index=False)
    for metric in ["on_rmse_w", "normalized_on_rmse", "relative_to_correct"]:
        fits.groupby(["target", "decoder"])[metric].median().unstack().loc[apps, apps].to_csv(output / f"C_{metric}.csv")
    np.savez_compressed(output / "C_examples.npz", appliances=apps, watts=examples)


def plot(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    curve = pd.read_csv(output / "B_oracle_trajectory.csv")
    fig, axes = plt.subplots(5, figsize=(9, 10), sharex=True, layout="constrained")
    for ax, col in zip(axes, curve.columns[1:]):
        ax.plot(curve["iteration"], curve[col])
        ax.set_ylabel(col.replace("_", " "))
    axes[-1].set_xlabel("Adam update")
    fig.savefig(output / "B_oracle_convergence.png", dpi=180)
    plt.close(fig)
    table = pd.read_csv(output / "C_normalized_on_rmse.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    fig.colorbar(ax.imshow(table, cmap="Blues"), ax=ax)
    ax.set_xticks(range(5), table.columns, rotation=25)
    ax.set_yticks(range(5), table.index)
    ax.set(xlabel="Decoder identity", ylabel="True appliance", title="Median ON RMSE / true ON RMS")
    for i, j in np.ndindex(table.shape):
        value = table.iloc[i, j]
        ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                color="white" if value > .65 * table.to_numpy().max() else "black")
    fig.savefig(output / "C_confusability.png", dpi=180)
    plt.close(fig)
    with np.load(output / "C_examples.npz") as data:
        fig, axes = plt.subplots(5, figsize=(10, 11), layout="constrained")
        for i, ax in enumerate(axes):
            for name, signal in zip(["Submeter", *data["appliances"]], data["watts"][i]):
                ax.plot(signal, label=name, alpha=.8)
            ax.set_ylabel(f"{data['appliances'][i]} (W)")
        axes[0].legend(ncol=3)
        axes[-1].set_xlabel("Sample in window")
        fig.savefig(output / "C_waveform_examples.png", dpi=180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot(args.output)
    else:
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        args.output.mkdir()
        compute(args.archive, args.output)
        subprocess.run([sys.executable, __file__, str(args.archive), str(args.output), "--plot-only"], check=True)

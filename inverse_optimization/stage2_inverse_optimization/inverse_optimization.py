"""Stage 2: recover five appliance powers from one aggregate window.

The Stage-1 LSTM is frozen. During inverse optimisation, Adam updates the
candidate appliance-power input instead of the LSTM weights.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn


# Support both module execution and an IDE Run button.
if __package__:
    from ..data_loader import APPLIANCES, POWER_COLUMNS, TIME_FEATURE_COLUMNS, load_windows
    from ..metrics import average_precision, f1, mae, off_mae, on_mae, precision, recall, sae
    from ..model import build_model_from_checkpoint
    from ..waveform_plots import save_appliance_event_waveforms
    from .optimization_loss import inverse_optimization_loss
else:
    project_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_dir))
    from data_loader import APPLIANCES, POWER_COLUMNS, TIME_FEATURE_COLUMNS, load_windows
    from metrics import average_precision, f1, mae, off_mae, on_mae, precision, recall, sae
    from model import build_model_from_checkpoint
    from waveform_plots import save_appliance_event_waveforms
    from stage2_inverse_optimization.optimization_loss import inverse_optimization_loss


STAGE2_DIR = Path(__file__).resolve().parent
with (STAGE2_DIR / "config.yaml").open(encoding="utf-8") as file:
    config = yaml.safe_load(file)


def plot_loss_curve(meter_history, total_history, run_dir):
    """Save the best restart's loss during inverse optimisation."""
    plot_path = run_dir / "figures" / "loss" / "inverse_loss_curve.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(meter_history, label="Best meter MSE")
    plt.plot(total_history, label="Best total loss")
    plt.xlabel("Optimisation step")
    plt.ylabel("Loss")
    plt.title("Stage-2 Inverse Optimisation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path


def plot_inverse_result(
    observed_aggregate,
    predicted_aggregate,
    true_appliance,
    estimated_appliance,
    run_dir,
):
    """Plot the aggregate fit and all five recovered appliance waveforms."""
    plot_path = run_dir / "figures" / "overview" / "inverse_waveforms.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True)

    axes[0].plot(observed_aggregate, label="Observed aggregate", linewidth=1)
    axes[0].plot(predicted_aggregate, label="Forward prediction", linewidth=1)
    axes[0].set_ylabel("Power (W)")
    axes[0].set_title("Aggregate: Observed vs. Re-simulated")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for appliance_index, appliance in enumerate(APPLIANCES):
        axis = axes[appliance_index + 1]
        axis.plot(
            true_appliance[:, appliance_index],
            label="Ground truth",
            linewidth=1,
        )
        axis.plot(
            np.maximum(estimated_appliance[:, appliance_index], 0),
            label="Inverse estimate (clipped at 0 W for display)",
            linewidth=1,
        )
        axis.set_ylabel("Power (W)")
        axis.set_title(appliance.capitalize())
        axis.legend()
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time step (8 seconds per step)")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=300)
    plt.close(figure)
    return plot_path


def plot_metrics_table(metric_values, split, window_index, run_dir):
    """Save a per-appliance NILM metric table as a PNG image."""
    rows = [
        ("MAE (W)", "mae", 2),
        ("SAE", "sae", 3),
        ("ON MAE (W)", "on_mae", 2),
        ("OFF MAE (W)", "off_mae", 2),
        ("AP", "average_precision", 3),
        ("Precision", "precision", 3),
        ("Recall", "recall", 3),
        ("F1", "f1", 3),
    ]
    table_rows = []
    for label, key, decimals in rows:
        values = metric_values[key].detach().cpu()
        cells = [f"{value:.{decimals}f}" for value in values]
        table_rows.append([label, *cells, f"{torch.nanmean(values):.{decimals}f}"])

    table_path = run_dir / "figures" / "metrics" / "inverse_metrics.png"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=["Metric", *[name.capitalize() for name in APPLIANCES], "Avg."],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    last_column = len(APPLIANCES) + 1
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f3b63")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#edf2f7")
        if column == last_column and row > 0:
            cell.set_facecolor("#dbe8f5")
            cell.set_text_props(weight="bold")

    axis.set_title(f"Stage-2 NILM Metrics · {split} split · window {window_index}", weight="bold", pad=18)
    figure.tight_layout()
    figure.savefig(table_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return table_path


def _state_events(state, min_samples):
    """Return inclusive start/end indices for contiguous ON events."""
    state = np.asarray(state, dtype=bool)
    starts = np.flatnonzero(state & np.r_[True, ~state[:-1]])
    ends = np.flatnonzero(state & np.r_[~state[1:], True])
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start + 1 >= min_samples
    ]


def _shade_state(axis, state, color, label):
    """Shade every contiguous ON run in a waveform plot."""
    for index, (start, end) in enumerate(_state_events(state, min_samples=1)):
        axis.axvspan(start, end + 1, color=color, alpha=0.18, label=label if index == 0 else None)


def plot_appliance_on_events(
    true_appliance,
    estimated_appliance,
    true_state,
    on_thresholds,
    margin_samples,
    min_on_samples,
    run_dir,
    appliance_indices=None,
    observed_aggregate=None,
    window_index=None,
    split_name=None,
):
    """Save the same focused ON-event and 10x-context layout as MART-NILM."""
    indices = list(range(len(APPLIANCES))) if appliance_indices is None else appliance_indices
    names = [APPLIANCES[index] for index in indices]
    return save_appliance_event_waveforms(
        run_dir / "figures" / "waveforms" / (split_name or config["inverse"]["split"]),
        appliances=names,
        true_power=true_appliance[:, indices],
        predicted_power=estimated_appliance[:, indices],
        true_state=true_state[:, indices],
        thresholds=[on_thresholds[index] for index in indices],
        aggregate=observed_aggregate,
        seed=config["inverse"]["seed"] + (window_index or 0),
        file_prefix="inverse",
        title_prefix=f"Inverse optimisation | window {window_index} | ",
        min_duration=min_on_samples,
        margin_min=margin_samples,
    )


def create_run_dir(experiment_name):
    """Create results/<experiment_name>, adding (2), (3), ... if needed."""
    results_dir = Path(__file__).resolve().parents[1] / "results"
    run_dir = results_dir / experiment_name
    number = 2
    while run_dir.exists():
        run_dir = results_dir / f"{experiment_name} ({number})"
        number += 1
    run_dir.mkdir(parents=True)
    return run_dir


def main():
    # /************************************************************
    #  * STEP 0 - Read config and choose the Stage-1 checkpoint
    #  ************************************************************/
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional Stage-1 checkpoint path that overrides inverse.checkpoint_path.",
    )
    parser.add_argument("--split", choices=["validating", "testing"])
    parser.add_argument("--dataset", help="Optional dataset filter, used for a test-house report.")
    parser.add_argument("--house", type=int, help="Optional house filter, used for a test-house report.")
    parser.add_argument("--window-stride", type=int, help="Optional Stage-2 stride override in samples.")
    parser.add_argument("--num-windows", type=int, help="Optional limit for a short diagnostic run.")
    args = parser.parse_args()

    settings = config["inverse"]
    split_name = args.split or settings["split"]
    window_stride = args.window_stride or settings["window_stride"]
    requested_windows = args.num_windows if args.num_windows is not None else settings["num_windows"]
    torch.manual_seed(settings["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = (
        STAGE2_DIR / settings["checkpoint_path"]
        if args.checkpoint is None else args.checkpoint
    )
    checkpoint_path = checkpoint_path.expanduser().resolve()

    # /************************************************************
    #  * STEP 1 - Load and freeze the forward model
    #  * $$\widehat{\mathbf y}=F_{\hat\theta}(\mathbf Z)$$
    #  * The LSTM parameters \hat\theta never change in Stage 2.
    #  * Autograd only calculates gradients with respect to Z.
    #  ************************************************************/
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    assert list(checkpoint["appliances"]) == APPLIANCES
    if checkpoint["config"]["data"].get("use_state_features", False):
        raise ValueError(
            "This checkpoint uses ground-truth ON/OFF state features. "
            "Stage 2 cannot use it because those states are unknown during inverse optimisation."
        )
    use_calendar_features = checkpoint["x_mean"].numel() == len(POWER_COLUMNS) + len(TIME_FEATURE_COLUMNS)

    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    # cuDNN needs a recurrent module in train mode to obtain its input
    # gradient. Dropout is set to zero, so this does not change its prediction.
    for module in model.modules():
        if isinstance(module, nn.RNNBase):
            module.dropout = 0.0
            module.train()

    # /************************************************************
    #  * STEP 2 - Load the data used in Stage 2
    #  *
    #  * observed aggregate: y_obs [number of windows, T]
    #  * true appliances:   P_true[number of windows, T, 5]
    #  *
    #  * P_true and states are never used during optimisation.
    #  * They are retained only for evaluation after inference.
    #  ************************************************************/
    input_windows, aggregate_windows, state_windows = load_windows(
        split_name, stride=window_stride, dataset=args.dataset, house=args.house,
        include_calendar_features=use_calendar_features,
    )
    training_input, _, _ = load_windows(
        "training", include_calendar_features=use_calendar_features
    )
    appliance_windows = input_windows[:, :, :len(POWER_COLUMNS)]
    training_power = torch.from_numpy(training_input[:, :, :len(POWER_COLUMNS)])

    x_mean = checkpoint["x_mean"].to(device)
    x_std = checkpoint["x_std"].to(device)
    appliance_x_mean = x_mean[:len(POWER_COLUMNS)]
    appliance_x_std = x_std[:len(POWER_COLUMNS)]
    y_mean = checkpoint["y_mean"].to(device)
    y_std = checkpoint["y_std"].to(device)

    # Training extrema are used only by the optional soft boundary loss.
    power_min = training_power.amin(dim=(0, 1)).to(device)
    power_max = training_power.amax(dim=(0, 1)).to(device)

    # Adam stores Z in z-score coordinates. Convert the configured physical
    # watt limits into that same coordinate system once.
    appliance_min = torch.tensor(
        settings["appliance_min_w"], dtype=appliance_x_mean.dtype, device=device
    ).view(1, 1, -1)
    appliance_max = torch.tensor(
        settings["appliance_max_w"], dtype=appliance_x_mean.dtype, device=device
    ).view(1, 1, -1)
    # $$Z_{i,\min}=(P_{i,\min}-\mu_i)/\sigma_i$$
    # $$Z_{i,\max}=(P_{i,\max}-\mu_i)/\sigma_i$$
    candidate_min = (appliance_min - appliance_x_mean) / appliance_x_std
    candidate_max = (appliance_max - appliance_x_mean) / appliance_x_std

    start = settings["window_index"]
    available = len(aggregate_windows) - start
    count = available if requested_windows is None else min(
        requested_windows, available
    )
    window_indices = np.arange(start, start + count, dtype=int)

    # Labels choose one useful plot only. They are not passed to the optimiser.
    active_counts = (state_windows[window_indices].sum(axis=1) > 0).sum(axis=1)
    plot_index = int(window_indices[active_counts.argmax()])

    run_dir = create_run_dir(settings["experiment_name"].strip())

    # /************************************************************
    #  * STEP 3 - Prepare disk storage for every inferred window
    #  *
    #  * estimated_appliances[window, time, appliance] is in watts.
    #  ************************************************************/
    prediction_path = run_dir / "estimated_appliances.npy"
    predictions = np.lib.format.open_memmap(
        prediction_path,
        mode="w+",
        dtype=np.float32,
        shape=(count, appliance_windows.shape[1], appliance_windows.shape[2]),
    )
    np.save(run_dir / "window_indices.npy", window_indices)
    rows = []

    # /************************************************************
    #  * STEP 4 - Inverse optimisation, evaluation, and saving
    #  *
    #  * For each aggregate window:
    #  * 1. Start R candidate appliance combinations from training traces.
    #  * 2. Update every candidate for K Adam steps.
    #  * 3. Retain the candidate with the smallest final inverse loss.
    #  * 4. Convert it to watts and evaluate it against held-out labels.
    #  ************************************************************/
    batch_size = settings["batch_windows"]
    restarts = settings["restarts"]
    steps = settings["steps"]

    for offset in range(0, count, batch_size):
        batch_indices = window_indices[offset:offset + batch_size]
        batch_count = len(batch_indices)
        print(
            f"\nBatch {offset // batch_size + 1}: windows {batch_indices.tolist()}",
            flush=True,
        )

        # ------------------------------------------------------------
        # STEP 4A - Read B observed aggregate windows and normalize them.
        #
        # observed: [B, T]
        # target:   [B*R, T], one identical meter target for each restart.
        #
        # $$\widetilde{\mathbf y}_{\rm obs}=(\mathbf y_{\rm obs}-\mu_y)/\sigma_y$$
        # ------------------------------------------------------------
        # Read B measured aggregate windows from the selected split.
        # observed[b, t] is y_obs,b(t) in watts. Shape: [B, T].
        observed = torch.from_numpy(aggregate_windows[batch_indices]).to(device)

        # Apply the same aggregate z-score normalization used in Stage 1,
        # then repeat each meter trace R times: one target for each restart.
        # target[r, t] is \tilde y_obs(t). Shape: [B*R, T].
        target = ((observed - y_mean) / y_std).repeat_interleave(restarts, dim=0)
        fixed_calendar = None
        if use_calendar_features:
            calendar = torch.from_numpy(
                input_windows[batch_indices, :, len(POWER_COLUMNS):]
            ).to(device)
            fixed_calendar = ((calendar - x_mean[len(POWER_COLUMNS):]) / x_std[len(POWER_COLUMNS):]).repeat_interleave(
                restarts, dim=0
            )

        # ------------------------------------------------------------
        # STEP 4B - Initialise B*R five-appliance candidates.
        #
        # Each restart begins as one complete five-appliance training window.
        # candidate: [B*R, T, 5] in z-score coordinates.
        #
        # $$\mathbf Z^{(0)}=(\mathbf P^{(0)}-\boldsymbol\mu_x)/\boldsymbol\sigma_x$$
        # ------------------------------------------------------------
        # Draw B*R training-window indices. This supplies different initial
        # appliance combinations, rather than different final answers.
        initial_indices = torch.randint(len(training_power), (batch_count, restarts))
        initial_power = training_power[initial_indices].to(device)

        # Convert watts to the z-score coordinates expected by the frozen LSTM,
        # then collapse [B, R, T, 5] into [B*R, T, 5].
        # candidate is the ONLY tensor Adam is allowed to change.
        candidate = nn.Parameter(
            ((initial_power - appliance_x_mean) / appliance_x_std).reshape(
                -1, initial_power.shape[2], initial_power.shape[3]
            ).clone()
        )

        # Project the initial candidates into the physical appliance box.
        with torch.no_grad():
            candidate.copy_(
                torch.maximum(torch.minimum(candidate, candidate_max), candidate_min)
            )

        # Adam holds momentum and variance estimates for candidate only.
        # The frozen LSTM parameters are absent from this optimizer.
        optimizer = torch.optim.Adam([candidate], lr=settings["learning_rate"])
        histories = [[[], []] for _ in range(batch_count)]

        # ------------------------------------------------------------
        # STEP 4C - Repeatedly update candidate appliance powers.
        #
        # $$\widehat{\mathbf y}^{(k)}=F_{\hat\theta}(\mathbf Z^{(k)})$$
        # $$L_{\rm meter}=T^{-1}\|\widehat{\mathbf y}^{(k)}-\widetilde{\mathbf y}_{\rm obs}\|_2^2$$
        # $$\mathbf Z^{(k+1)}=\Pi_{\mathcal C}(\operatorname{AdamStep}(\mathbf Z^{(k)},\nabla_{\mathbf Z}L))$$
        #
        # F_hat's weights are frozen. optimizer.step() changes only candidate.
        # ------------------------------------------------------------
        for step in range(steps):
            # (1) Forward simulation: pass every candidate appliance set
            # Z^(k) through F_hat to obtain a predicted aggregate.
            # prediction shape: [B*R, T].
            # $$\widehat{\mathbf y}^{(k)}=F_{\hat\theta}(\mathbf Z^{(k)})$$
            forward_input = candidate if fixed_calendar is None else torch.cat(
                [candidate, fixed_calendar], dim=-1
            )
            prediction = model(forward_input)

            # (2) Convert the same candidate to watts for physical penalties.
            # This is P^(k) = Z^(k) * sigma_x + mu_x, shape [B*R, T, 5].
            candidate_power = candidate * appliance_x_std + appliance_x_mean

            # (3) Calculate the scalar objective L used for Adam.
            # For each restart r, L_meter,r is the mean time-wise squared
            # difference between its forward prediction and the observed meter.
            # L is the mean of the B*R restart losses.
            # $$L_{\rm meter,r}=T^{-1}\sum_t(\hat y_r(t)-\tilde y_{obs}(t))^2$$
            loss, meter_loss, _, total_loss = inverse_optimization_loss(
                prediction,
                target,
                candidate_power,
                power_min,
                power_max,
                settings["boundary_weight"],
            )

            # (4) Delete the previous derivative dL/dZ from candidate.grad.
            optimizer.zero_grad()

            # (5) Backpropagate through the frozen LSTM.
            # This calculates dL/dZ, not dL/dtheta, because theta has
            # requires_grad=False. No appliance ground-truth label is used.
            loss.backward()

            # (6) Adam uses dL/dZ and its accumulated moments to propose an
            # updated candidate, Z_tilde^(k+1). It does not change F_hat.
            optimizer.step()

            # (7) Projected Adam: clip Z_tilde^(k+1) back into C, the allowed
            # appliance-specific watt ranges. candidate remains z-score data.
            # $$\mathbf Z^{(k+1)}=\Pi_{\mathcal C}(\widetilde{\mathbf Z}^{(k+1)})$$
            with torch.no_grad():
                candidate.copy_(
                    torch.maximum(torch.minimum(candidate, candidate_max), candidate_min)
                )

            # (8) Record the lowest loss among the R restarts for each of the
            # B windows. This is for the loss curve only; the winning restart
            # is selected after the final update below.
            meter_values = (
                meter_loss.detach().reshape(batch_count, restarts)
                .min(dim=1).values.cpu().tolist()
            )
            total_values = (
                total_loss.detach().reshape(batch_count, restarts)
                .min(dim=1).values.cpu().tolist()
            )
            for position in range(batch_count):
                histories[position][0].append(meter_values[position])
                histories[position][1].append(total_values[position])

            if (
                step == 0
                or (step + 1) % settings["print_every"] == 0
                or step + 1 == steps
            ):
                print(
                    f"Step {step + 1:04d}/{steps} | "
                    f"mean best meter MSE {np.mean(meter_values):.6f}",
                    flush=True,
                )

        # ------------------------------------------------------------
        # STEP 4D - Evaluate every restart after its final update, then select
        # the winning candidate separately for each aggregate window.
        #
        # $$r^*=\arg\min_{r=1,\ldots,R} L(\mathbf Z_r^{(K)})$$
        # ------------------------------------------------------------
        # Recompute the complete objective after the last projection. The
        # stored loss inside the loop was measured before Adam's last update.
        with torch.no_grad():
            forward_input = candidate if fixed_calendar is None else torch.cat(
                [candidate, fixed_calendar], dim=-1
            )
            final_prediction = model(forward_input)
            _, _, _, final_total_loss = inverse_optimization_loss(
                final_prediction,
                target,
                candidate * appliance_x_std + appliance_x_mean,
                power_min,
                power_max,
                settings["boundary_weight"],
            )
            # Regroup [B*R] losses as [B, R] and retain the lowest-loss restart
            # independently for every aggregate window.
            best_restarts = final_total_loss.reshape(
                batch_count, restarts
            ).argmin(dim=1)
            # Select its five appliance trajectories. Shape: [B, T, 5].
            best_candidates = candidate.reshape(
                batch_count, restarts, *candidate.shape[1:]
            )[
                torch.arange(batch_count, device=device), best_restarts
            ].detach()

        # ------------------------------------------------------------
        # STEP 4E - Turn each selected candidate into final NILM output.
        #
        # This section does not optimize. It only converts to watts, evaluates
        # held-out labels, optionally plots one window, and saves results.
        # ------------------------------------------------------------
        for position, window_index in enumerate(batch_indices):
            # The selected candidate is the Stage-2 output in normalized form.
            # It is not a new prediction made by the LSTM.
            candidate_normalized = best_candidates[position]
            selected_restart = best_restarts[position].item()

            # True appliance signals and states appear only after optimization.
            true_appliance = torch.from_numpy(
                appliance_windows[window_index]
            ).to(device)
            observed_aggregate = torch.from_numpy(
                aggregate_windows[window_index]
            ).to(device)
            true_state = torch.from_numpy(
                state_windows[window_index]
            ).to(device)

            # Convert the selected candidate back to watts for NILM output:
            # $$\widehat{\mathbf P}=\widehat{\mathbf Z}\odot\sigma_x+\mu_x$$
            model.eval()
            with torch.no_grad():
                estimated_appliance = candidate_normalized * appliance_x_std + appliance_x_mean
                calendar_for_window = None
                if use_calendar_features:
                    calendar_for_window = (
                        torch.from_numpy(input_windows[window_index, :, len(POWER_COLUMNS):]).to(device)
                        - x_mean[len(POWER_COLUMNS):]
                    ) / x_std[len(POWER_COLUMNS):]
                    calendar_for_window = calendar_for_window.unsqueeze(0)
                forward_input = candidate_normalized.unsqueeze(0)
                if calendar_for_window is not None:
                    forward_input = torch.cat([forward_input, calendar_for_window], dim=-1)
                predicted_normalized = model(forward_input)[0]
                predicted_aggregate = predicted_normalized * y_std + y_mean

                # Oracle diagnostic: even true appliances may not reproduce the
                # meter accurately because unmonitored load remains in y_obs.
                oracle_input = ((true_appliance - appliance_x_mean) / appliance_x_std).unsqueeze(0)
                if calendar_for_window is not None:
                    oracle_input = torch.cat([oracle_input, calendar_for_window], dim=-1)
                oracle_prediction = model(oracle_input)[0]
                oracle_aggregate = oracle_prediction * y_std + y_mean

            metric_values = {
                "mae": mae(estimated_appliance, true_appliance),
                "sae": sae(estimated_appliance, true_appliance),
                "on_mae": on_mae(
                    estimated_appliance, true_appliance, true_state
                ),
                "off_mae": off_mae(
                    estimated_appliance, true_appliance, true_state
                ),
                "average_precision": average_precision(
                    estimated_appliance, true_state
                ),
                "precision": precision(
                    estimated_appliance, true_state, settings["on_thresholds"]
                ),
                "recall": recall(
                    estimated_appliance, true_state, settings["on_thresholds"]
                ),
                "f1": f1(
                    estimated_appliance, true_state, settings["on_thresholds"]
                ),
            }
            aggregate_mae = (
                predicted_aggregate - observed_aggregate
            ).abs().mean()
            oracle_mae = (oracle_aggregate - observed_aggregate).abs().mean()

            print(
                f"Window {window_index} | restart {selected_restart:02d} | "
                f"aggregate MAE {aggregate_mae.item():.2f} W",
                flush=True,
            )

            if window_index == plot_index:
                loss_plot = plot_loss_curve(
                    histories[position][0], histories[position][1], run_dir
                )
                waveform_plot = plot_inverse_result(
                    observed_aggregate.cpu().numpy(),
                    predicted_aggregate.cpu().numpy(),
                    true_appliance.cpu().numpy(),
                    estimated_appliance.cpu().numpy(),
                    run_dir,
                )
                metrics_plot = plot_metrics_table(
                    metric_values, split_name, window_index, run_dir
                )
                print(f"Loss plot: {loss_plot}")
                print(f"Waveform plot: {waveform_plot}")
                print(f"Metrics table: {metrics_plot}")

            row = {
                "split": split_name,
                "dataset": args.dataset or "all",
                "house": args.house if args.house is not None else "all",
                "window_index": int(window_index),
                "selected_restart": selected_restart,
                "aggregate_mae_w": aggregate_mae.item(),
                "forward_oracle_mae_w": oracle_mae.item(),
            }
            for name, values in metric_values.items():
                for appliance, value in zip(APPLIANCES, values):
                    row[f"{appliance}_{name}"] = value.item()

            rows.append(row)
            predictions[offset + position] = estimated_appliance.cpu().numpy()

        # Commit the full solved batch to estimated_appliances.npy.
        predictions.flush()

        # Re-enable the recurrent input-gradient mode before next batch.
        for module in model.modules():
            if isinstance(module, nn.RNNBase):
                module.train()

    # /************************************************************
    #  * STEP 5 - Plot one representative ON event per appliance
    #  ************************************************************/
    for appliance_index, appliance in enumerate(APPLIANCES):
        best_position = None
        best_length = 0
        for position, window_index in enumerate(window_indices):
            state = state_windows[window_index, :, appliance_index]
            events = _state_events(state, settings["scenario_min_on_samples"])
            complete = [
                (start, end) for start, end in events
                if start > 0 and end < len(state) - 1
            ]
            if complete:
                length = max(end - start + 1 for start, end in complete)
                if length > best_length:
                    best_position, best_length = position, length
        if best_position is None:
            print(f"No complete ON event for {appliance} in selected windows")
            continue

        window_index = int(window_indices[best_position])
        paths = plot_appliance_on_events(
            appliance_windows[window_index],
            predictions[best_position],
            state_windows[window_index],
            settings["on_thresholds"],
            settings["scenario_margin_samples"],
            settings["scenario_min_on_samples"],
            run_dir,
            appliance_indices=[appliance_index],
            observed_aggregate=aggregate_windows[window_index],
            window_index=window_index,
            split_name=split_name,
        )
        for path in paths:
            print(f"ON-event waveform plot: {path}")

    # /************************************************************
    #  * STEP 6 - Save per-window metrics
    #  ************************************************************/
    summary_path = run_dir / "window_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # These are overlapping-window diagnostics, not full-timeline metrics.
    average_path = run_dir / "window_averages.csv"
    metric_names = [
        name for name in rows[0]
        if name not in {"split", "dataset", "house", "window_index", "selected_restart"}
    ]
    with average_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "mean_over_windows", "valid_windows"])
        for name in metric_names:
            values = np.asarray([row[name] for row in rows], dtype=float)
            valid = values[np.isfinite(values)]
            writer.writerow([
                name,
                float(valid.mean()) if len(valid) else "",
                len(valid),
            ])

    print(f"Window-level diagnostics: {summary_path}")
    print(f"Window averages: {average_path}")
    print(f"All estimated windows: {prediction_path}")
    print("These rows overlap in time; they are not full-timeline metrics.")

if __name__ == "__main__":
    main()

"""Stage 2: recover five appliance powers from one aggregate window.

The Stage-1 LSTM is frozen. During inverse optimisation, Adam updates the
candidate appliance-power input instead of the LSTM weights.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn


# Support both module execution and an IDE Run button.
if __package__:
    from ..data_loader import APPLIANCES, load_windows
    from ..metrics import average_precision, f1, mae, off_mae, on_mae, precision, recall, sae
    from ..model import build_model_from_checkpoint
    from .optimization_loss import inverse_optimization_loss
else:
    project_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_dir))
    from data_loader import APPLIANCES, load_windows
    from metrics import average_precision, f1, mae, off_mae, on_mae, precision, recall, sae
    from model import build_model_from_checkpoint
    from stage2_inverse_optimization.optimization_loss import inverse_optimization_loss


STAGE2_DIR = Path(__file__).resolve().parent
with (STAGE2_DIR / "config.yaml").open(encoding="utf-8") as file:
    config = yaml.safe_load(file)


def plot_loss_curve(meter_history, total_history, run_dir):
    """Save the best restart's loss during inverse optimisation."""
    plot_path = run_dir / "inverse_loss_curve.png"
    plt.figure(figsize=(8, 5))
    plt.plot(meter_history, label="Best meter MSE")
    plt.plot(total_history, label="Best total loss")
    plt.xlabel("Optimisation step")
    plt.ylabel("Loss")
    plt.title("Stage-2 Inverse Optimisation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
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
    plot_path = run_dir / "inverse_waveforms.png"
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
            estimated_appliance[:, appliance_index],
            label="Inverse estimate",
            linewidth=1,
        )
        axis.set_ylabel("Power (W)")
        axis.set_title(appliance.capitalize())
        axis.legend()
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time step (8 seconds per step)")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200)
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

    table_path = run_dir / "inverse_metrics.png"
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
    figure.savefig(table_path, dpi=200, bbox_inches="tight")
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
    seed,
    run_dir,
):
    """Save one full true-ON event plus OFF context for each appliance."""
    rng = np.random.default_rng(seed)
    plot_paths = []
    total_steps = len(true_appliance)

    for appliance_index, appliance in enumerate(APPLIANCES):
        state = true_state[:, appliance_index].astype(bool)
        events = _state_events(state, min_on_samples)
        plot_path = run_dir / f"inverse_{appliance}_on_event.png"
        figure, axis = plt.subplots(figsize=(10, 4))

        if not events:
            axis.text(0.5, 0.5, "No complete ON event in this inverse window", ha="center", va="center")
            axis.set_axis_off()
        else:
            # Use a random complete ON event; retain the full event and add
            # OFF context on both sides, rather than selecting one ON sample.
            event_start, event_end = events[rng.integers(len(events))]
            event_length = event_end - event_start + 1
            margin = max(margin_samples, int(round(0.08 * event_length)))
            start = max(0, event_start - margin)
            end = min(total_steps, event_end + margin + 1)
            time = np.arange(start, end) - start
            true_view = true_appliance[start:end, appliance_index]
            estimated_view = estimated_appliance[start:end, appliance_index]
            true_on_view = state[start:end]
            predicted_on_view = estimated_view >= on_thresholds[appliance_index]

            # MART-NILM-style state bands: true-only, predicted-only, overlap.
            _shade_state(axis, true_on_view & ~predicted_on_view, "#6baed6", "True ON only")
            _shade_state(axis, predicted_on_view & ~true_on_view, "#7ad66d", "Predicted ON only")
            _shade_state(axis, true_on_view & predicted_on_view, "#9b59b6", "ON overlap")
            axis.plot(time, true_view, color="#1f77b4", linewidth=1.8, label="Ground truth")
            axis.plot(time, estimated_view, color="#d62728", linewidth=1.5, label="Inverse estimate")
            axis.set_xlabel("Time step (8 seconds per step)")
            axis.set_ylabel("Appliance power (W)")
            axis.set_title(f"ON event [{event_start}:{event_end}] with OFF context")
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=8)

        figure.suptitle(f"{appliance.capitalize()}: Inverse Estimate Around a True ON Event", weight="bold")
        figure.tight_layout()
        figure.savefig(plot_path, dpi=200)
        plt.close(figure)
        plot_paths.append(plot_path)

    return plot_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional Stage-1 checkpoint path that overrides inverse.checkpoint_path.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use two restarts and three optimisation steps.",
    )
    args = parser.parse_args()

    settings = config["inverse"]
    torch.manual_seed(settings["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The config path is relative to stage2_inverse_optimization/config.yaml.
    # A command-line --checkpoint path, when supplied, takes priority.
    if args.checkpoint is None:
        checkpoint_path = STAGE2_DIR / settings["checkpoint_path"]
    else:
        checkpoint_path = args.checkpoint
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Stage-1 checkpoint not found: {checkpoint_path}. "
            "Set inverse.checkpoint_path in stage2_inverse_optimization/config.yaml."
        )

    # STEP 1: Load the trained Stage-1 forward model.
    # The checkpoint also contains the training-only normalization statistics.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    assert list(checkpoint["appliances"]) == APPLIANCES

    # checkpoint["model"] chooses the registered model class. The matching
    # checkpoint["model_settings"] reconstructs the exact trained structure.
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # STEP 2: Freeze every model weight.
    # Autograd will pass through the model, but Adam will update only the
    # candidate input created in STEP 6. cuDNN requires an LSTM to be in
    # training mode when backpropagating to its input. Only recurrent modules
    # are switched, while the complete model otherwise remains in eval mode.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.RNNBase):
            # eval mode normally disables recurrent dropout. cuDNN requires
            # train mode for input gradients, so set dropout to zero directly
            # before enabling that computation mode.
            module.dropout = 0.0
            module.train()

    # STEP 3: Load one observed aggregate window from the configured split.
    # Ground-truth appliance powers are retained only for final evaluation.
    appliance_windows, aggregate_windows, state_windows = load_windows(settings["split"])
    window_index = settings["window_index"]
    true_appliance = torch.from_numpy(appliance_windows[window_index]).to(device)
    observed_aggregate = torch.from_numpy(aggregate_windows[window_index]).to(device)
    true_state = torch.from_numpy(state_windows[window_index]).to(device)
    print(f"Evaluating {settings['split']} window {window_index} of {len(aggregate_windows)}")
    print("ON samples:", dict(zip(APPLIANCES, true_state.sum(dim=0).tolist())))

    # STEP 4: Reuse the Stage-1 training statistics exactly.
    # The frozen LSTM expects normalized appliance inputs and produces a
    # normalized aggregate output.
    x_mean = checkpoint["x_mean"].to(device)
    x_std = checkpoint["x_std"].to(device)
    y_mean = checkpoint["y_mean"].to(device)
    y_std = checkpoint["y_std"].to(device)
    target_normalized = (observed_aggregate - y_mean) / y_std

    # STEP 5: Load training appliance windows for two training-only purposes:
    # random initial candidates and the boundary-loss power range.
    training_power, _, _ = load_windows("training")
    training_power = torch.from_numpy(training_power)
    power_min = training_power.amin(dim=(0, 1)).to(device)
    power_max = training_power.amax(dim=(0, 1)).to(device)

    restarts = 2 if args.smoke else settings["restarts"]
    steps = 3 if args.smoke else settings["steps"]
    initial_indices = torch.randint(len(training_power), (restarts,))
    initial_power = training_power[initial_indices].to(device)

    # STEP 6: Create the variables that Stage 2 will optimise.
    # We optimise normalized candidates because that is the coordinate system
    # learned by the LSTM. Raw watts are recovered whenever they are needed.
    candidate_normalized = nn.Parameter(
        ((initial_power - x_mean) / x_std).clone()
    )
    optimizer = torch.optim.Adam(
        [candidate_normalized],
        lr=settings["learning_rate"],
    )

    meter_history = []
    total_history = []

    # STEP 7: Neural-Adjoint inverse optimisation.
    # Paper Eq. (4): x_(k+1) = x_k - alpha * gradient_x L(F(x_k), y).
    # The LSTM weights remain fixed. Only candidate_normalized changes.
    for step in range(steps):
        candidate_power = candidate_normalized * x_std + x_mean
        predicted_normalized = model(candidate_normalized)

        # L is one scalar for Adam; L_meter, L_bnd, and L_total contain one
        # value per random restart r:
        # L_meter[r] = aggregate reconstruction MSE,
        # L_bnd[r] = appliance-power boundary penalty, and
        # L_total[r] = L_meter[r] + boundary_weight * L_bnd[r].
        L, L_meter, L_bnd, L_total = (
            inverse_optimization_loss(
                predicted_normalized=predicted_normalized,
                target_normalized=target_normalized,
                candidate_power=candidate_power,
                power_min=power_min,
                power_max=power_max,
                boundary_weight=settings["boundary_weight"],
            )
        )

        optimizer.zero_grad()
        # Compute gradient_x L(F(x), y), then Adam updates candidate_normalized.
        L.backward()  # calculate ∂L / ∂x
        optimizer.step() # update x using the gradient: x_(k+1) = x_k - alpha * gradient_x L(F(x_k), y).

        meter_history.append(L_meter.min().item())
        total_history.append(L_total.min().item())

        if (step == 0 or (step + 1) % settings["print_every"] == 0 or step + 1 == steps):
            best_restart = L_total.argmin().item()
            print(
                f"Step {step + 1:04d}/{steps} | "
                f"best restart {best_restart:02d} | "
                f"meter MSE {L_meter[best_restart].item():.6f} | "
                f"boundary {L_bnd[best_restart].item():.6f}"
            )

    # STEP 8: Re-evaluate all restarts after the last Adam update and retain
    # the candidate with the smallest complete inverse objective.
    model.eval()
    with torch.no_grad():
        candidate_power = candidate_normalized * x_std + x_mean
        predicted_normalized = model(candidate_normalized)
        _, L_meter, L_bnd, L_total = (
            inverse_optimization_loss(
                predicted_normalized=predicted_normalized,
                target_normalized=target_normalized,
                candidate_power=candidate_power,
                power_min=power_min,
                power_max=power_max,
                boundary_weight=settings["boundary_weight"],
            )
        )

        # Retain the restart with the smallest complete objective L_total.
        best_index = L_total.argmin()
        estimated_appliance = candidate_power[best_index]
        predicted_aggregate = (
            predicted_normalized[best_index] * y_std + y_mean
        )

    # STEP 9: Evaluate in watts. The true appliance powers are used here only;
    # they never entered the inverse objective or its gradient.
    metric_values = {
        "mae": mae(estimated_appliance, true_appliance),
        "sae": sae(estimated_appliance, true_appliance),
        "on_mae": on_mae(estimated_appliance, true_appliance, true_state),
        "off_mae": off_mae(estimated_appliance, true_appliance, true_state),
        "average_precision": average_precision(estimated_appliance, true_state),
        "precision": precision(estimated_appliance, true_state, settings["on_thresholds"]),
        "recall": recall(estimated_appliance, true_state, settings["on_thresholds"]),
        "f1": f1(estimated_appliance, true_state, settings["on_thresholds"]),
    }
    aggregate_mae = (predicted_aggregate - observed_aggregate).abs().mean()
    # Diagnostic only: the true appliance signals never enter the inverse
    # objective. If the frozen forward model cannot fit these perfect inputs,
    # reducing inverse meter loss cannot guarantee correct appliance estimates.
    with torch.no_grad():
        oracle_aggregate = model(((true_appliance - x_mean) / x_std).unsqueeze(0))[0]
        oracle_aggregate = oracle_aggregate * y_std + y_mean
        oracle_mae = (oracle_aggregate - observed_aggregate).abs().mean()

    print()
    print(f"Device: {device}")
    print(f"Forward model: {checkpoint['model']}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Selected restart: {best_index.item()}")
    print(f"Aggregate re-simulation MAE: {aggregate_mae.item():.2f} W")
    print(f"Forward MAE with true appliance inputs: {oracle_mae.item():.2f} W")
    for appliance, value in zip(APPLIANCES, metric_values["mae"]):
        print(f"{appliance:15s} MAE: {value.item():.2f} W")

    # STEP 10: Save numerical results and figures for inspection.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        Path(__file__).resolve().parents[1]
        / "results"
        / f"{settings['experiment_name']}_{timestamp}"
    )
    run_dir.mkdir(parents=True)

    result_path = run_dir / "inverse_result.pt"
    torch.save(
        {
            "checkpoint": str(checkpoint_path),
            "config": config,
            "window_index": window_index,
            "observed_aggregate": observed_aggregate.cpu(),
            "predicted_aggregate": predicted_aggregate.cpu(),
            "true_appliance": true_appliance.cpu(),
            "true_state": true_state.cpu(),
            "estimated_appliance": estimated_appliance.cpu(),
            "metrics": {name: value.cpu() for name, value in metric_values.items()},
            "aggregate_mae": aggregate_mae.cpu(),
            "forward_oracle_mae": oracle_mae.cpu(),
            "selected_restart": best_index.cpu(),
            "final_meter_mse": L_meter[best_index].cpu(),
            "final_boundary_loss": L_bnd[best_index].cpu(),
        },
        result_path,
    )

    loss_plot = plot_loss_curve(meter_history, total_history, run_dir)
    waveform_plot = plot_inverse_result(
        observed_aggregate.cpu().numpy(),
        predicted_aggregate.cpu().numpy(),
        true_appliance.cpu().numpy(),
        estimated_appliance.cpu().numpy(),
        run_dir,
    )
    metrics_plot = plot_metrics_table(
        metric_values, settings["split"], window_index, run_dir
    )
    scenario_plots = plot_appliance_on_events(
        true_appliance.cpu().numpy(),
        estimated_appliance.cpu().numpy(),
        true_state.cpu().numpy(),
        settings["on_thresholds"],
        settings["scenario_margin_samples"],
        settings["scenario_min_on_samples"],
        settings["seed"],
        run_dir,
    )
    print(f"Result: {result_path}")
    print(f"Loss plot: {loss_plot}")
    print(f"Waveform plot: {waveform_plot}")
    print(f"Metrics table: {metrics_plot}")
    for plot_path in scenario_plots:
        print(f"ON-event waveform plot: {plot_path}")


if __name__ == "__main__":
    main()

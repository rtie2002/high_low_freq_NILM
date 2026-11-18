"""Stage 2: recover five appliance powers from one aggregate window.

The Stage-1 LSTM is frozen. During inverse optimisation, Adam updates the
candidate appliance-power input instead of the LSTM weights.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn


# Support both module execution and an IDE Run button.
if __package__:
    from ..data_loader import APPLIANCES, load_windows
    from ..model.lstm import LSTM_NILM
    from .optimization_loss import inverse_optimization_loss
else:
    project_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_dir))
    from data_loader import APPLIANCES, load_windows
    from model.lstm import LSTM_NILM
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the Stage-1 best.pt checkpoint.",
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

    # STEP 1: Load the trained Stage-1 forward model.
    # The checkpoint also contains the training-only normalization statistics.
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )
    assert list(checkpoint["appliances"]) == APPLIANCES

    model = LSTM_NILM(**checkpoint["model_settings"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # STEP 2: Freeze every LSTM weight.
    # Autograd will pass through the model, but Adam will update only the
    # candidate input created in STEP 6.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # STEP 3: Load one observed aggregate window from the configured split.
    # Ground-truth appliance powers are retained only for final evaluation.
    appliance_windows, aggregate_windows, _ = load_windows(settings["split"])
    window_index = settings["window_index"]
    true_appliance = torch.from_numpy(appliance_windows[window_index]).to(device)
    observed_aggregate = torch.from_numpy(aggregate_windows[window_index]).to(device)

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

        total_loss, meter_per_restart, boundary_per_restart, total_per_restart = (
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
        total_loss.backward()
        optimizer.step()

        meter_history.append(meter_per_restart.min().item())
        total_history.append(total_per_restart.min().item())

        if (
            step == 0
            or (step + 1) % settings["print_every"] == 0
            or step + 1 == steps
        ):
            best_restart = total_per_restart.argmin().item()
            print(
                f"Step {step + 1:04d}/{steps} | "
                f"best restart {best_restart:02d} | "
                f"meter MSE {meter_per_restart[best_restart].item():.6f} | "
                f"boundary {boundary_per_restart[best_restart].item():.6f}"
            )

    # STEP 8: Re-evaluate all restarts after the last Adam update and retain
    # the candidate with the smallest complete inverse objective.
    with torch.no_grad():
        candidate_power = candidate_normalized * x_std + x_mean
        predicted_normalized = model(candidate_normalized)
        _, meter_per_restart, boundary_per_restart, total_per_restart = (
            inverse_optimization_loss(
                predicted_normalized=predicted_normalized,
                target_normalized=target_normalized,
                candidate_power=candidate_power,
                power_min=power_min,
                power_max=power_max,
                boundary_weight=settings["boundary_weight"],
            )
        )

        best_index = total_per_restart.argmin()
        estimated_appliance = candidate_power[best_index]
        predicted_aggregate = (
            predicted_normalized[best_index] * y_std + y_mean
        )

    # STEP 9: Evaluate in watts. The true appliance powers are used here only;
    # they never entered the inverse objective or its gradient.
    appliance_mae = (estimated_appliance - true_appliance).abs().mean(dim=0)
    aggregate_mae = (predicted_aggregate - observed_aggregate).abs().mean()

    print()
    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Selected restart: {best_index.item()}")
    print(f"Aggregate re-simulation MAE: {aggregate_mae.item():.2f} W")
    for appliance, mae in zip(APPLIANCES, appliance_mae):
        print(f"{appliance:15s} MAE: {mae.item():.2f} W")

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
            "checkpoint": str(args.checkpoint),
            "config": config,
            "window_index": window_index,
            "observed_aggregate": observed_aggregate.cpu(),
            "predicted_aggregate": predicted_aggregate.cpu(),
            "true_appliance": true_appliance.cpu(),
            "estimated_appliance": estimated_appliance.cpu(),
            "appliance_mae": appliance_mae.cpu(),
            "aggregate_mae": aggregate_mae.cpu(),
            "selected_restart": best_index.cpu(),
            "final_meter_mse": meter_per_restart[best_index].cpu(),
            "final_boundary_loss": boundary_per_restart[best_index].cpu(),
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
    print(f"Result: {result_path}")
    print(f"Loss plot: {loss_plot}")
    print(f"Waveform plot: {waveform_plot}")


if __name__ == "__main__":
    main()

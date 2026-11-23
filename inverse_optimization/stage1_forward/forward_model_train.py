"""Train the Stage 1 model on five appliance traces and total-meter targets."""

import csv
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Support both `python -m inverse_optimization...` and an IDE Run button that
# executes this file directly. Direct execution has no package context, so add
# the project directory to Python's import path before using absolute imports.
if __package__:
    from ..data_loader import (
        APPLIANCES,
        config,
        load_random_appliance_event_windows,
        load_random_continuous_interval,
        load_windows,
    )
    from ..model import build_model
    from ..waveform_plots import save_forward_event_waveforms
else:
    project_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_dir))
    from data_loader import (
        APPLIANCES,
        config,
        load_random_appliance_event_windows,
        load_random_continuous_interval,
        load_windows,
    )
    from model import build_model
    from waveform_plots import save_forward_event_waveforms


def plot_loss_curve(training_losses, validation_losses, test_mse, best_epoch, run_dir):
    """Save training and validation loss per epoch plus one final test point."""
    plot_path = run_dir / "figures" / "loss" / "loss_curve.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(training_losses) + 1), training_losses, label="Training MSE")
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation MSE")
    plt.scatter(best_epoch, test_mse, color="red", label="Test MSE (best epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE")
    plt.title("Validation and Test Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path


def plot_aggregate_predictions(week_aggregate, week_prediction, seed, run_dir):
    """Save 1,024-sample, one-day, and one-week waveform comparisons."""
    sample_seconds = 8
    week_samples = len(week_aggregate)
    plot_lengths = {
        "1024_samples": 1024,
        "one_day": 24 * 60 * 60 // sample_seconds,
        "one_week": week_samples,
    }
    rng = np.random.default_rng(seed)
    plot_paths = []

    for plot_name, length in plot_lengths.items():
        start = rng.integers(week_samples - length + 1)
        end = start + length
        waveform_path = run_dir / "figures" / "aggregate" / f"{plot_name}.png"
        waveform_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 5))
        plt.plot(week_aggregate[start:end], label="Real aggregate power", linewidth=1)
        plt.plot(week_prediction[start:end], label="Predicted aggregate power", linewidth=1)
        plt.xlabel("Time step (8 seconds per step)")
        plt.ylabel("Power (W)")
        plt.title(f"Test Aggregate Power: Real vs. Predicted ({plot_name})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(waveform_path, dpi=300)
        plt.close()
        plot_paths.append(waveform_path)

    return plot_paths


def plot_appliance_scenarios(examples, predictions, run_dir):
    """Save MART-NILM-style focused and context aggregate figures by appliance."""
    return save_forward_event_waveforms(
        run_dir / "figures" / "waveforms" / "testing",
        examples,
        predictions,
        config["training"]["seed"],
    )


def aggregate_accuracy_metrics(predicted_watts, true_watts):
    """Measure final aggregate-meter accuracy in physical units, not z-scores."""
    predicted = predicted_watts.reshape(-1).float()
    true = true_watts.reshape(-1).float()
    error = predicted - true

    # One sample represents 8 seconds of power. W × hours becomes Wh.
    sample_hours = 8 / 3600
    relative_energy_error = error.sum().abs() / true.sum().abs().clamp_min(1e-6)
    centered_true = true - true.mean()
    centered_predicted = predicted - predicted.mean()
    correlation = (centered_true * centered_predicted).sum() / (
        centered_true.square().sum().sqrt()
        * centered_predicted.square().sum().sqrt()
    ).clamp_min(1e-6)
    r_squared = 1 - error.square().sum() / centered_true.square().sum().clamp_min(1e-6)

    return {
        "MAE (W)": error.abs().mean().item(),
        "RMSE (W)": error.square().mean().sqrt().item(),
        "Absolute energy error (kWh)": (error.sum().abs() * sample_hours / 1000).item(),
        "Relative energy error (%)": (relative_energy_error * 100).item(),
        "R2": r_squared.item(),
        "Pearson correlation": correlation.item(),
    }


def save_aggregate_metrics(metrics, normalized_mse, training_seconds, run_dir):
    """Save the final held-out aggregate-meter metrics as CSV and PNG."""
    metrics_dir = run_dir / "figures" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "Stage 1 training time (minutes)": training_seconds / 60,
        "Normalized test MSE": normalized_mse,
        **metrics,
    }

    csv_path = metrics_dir / "forward_test_aggregate_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(values.items())

    table_path = metrics_dir / "forward_test_aggregate_metrics.png"
    rows = [[name, f"{value:.5f}"] for name, value in values.items()]
    figure, axis = plt.subplots(figsize=(7, 3.6))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Forward-model test metric", "Value"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f3b63")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#edf2f7")
    axis.set_title("Stage 1: Aggregate-Meter Accuracy on Held-Out Test Data", weight="bold", pad=14)
    figure.tight_layout()
    figure.savefig(table_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return csv_path, table_path


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
    settings = config["training"]
    use_calendar_features = config["data"]["use_calendar_features"]
    clean_background = config["data"]["clean_background"]
    background_std_multiplier = config["data"]["background_std_multiplier"]
    torch.manual_seed(settings["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = perf_counter()

    # 1. Load aligned windows in watts. Fit z-score statistics on training only.
    x_train, y_train, _ = load_windows(
        "training",
        include_calendar_features=use_calendar_features,
        clean_background=clean_background,
        background_std_multiplier=background_std_multiplier,
    )
    del _
    x_val, y_val, _ = load_windows(
        "validating",
        include_calendar_features=use_calendar_features,
    )
    del _
    x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_val, y_val = torch.from_numpy(x_val), torch.from_numpy(y_val)

    # Calculate the z-score statistics on training data only.
    x_mean = x_train.mean(dim=(0, 1))
    x_std = x_train.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    y_mean = y_train.mean()
    y_std = y_train.std(unbiased=False).clamp_min(1e-6)
    print(
        f"Training aggregate: mean {y_mean.item():.2f} W | "
        f"std {y_std.item():.2f} W"
    )
    print(
        f"Validation aggregate: mean {y_val.mean().item():.2f} W | "
        f"std {y_val.std(unbiased=False).item():.2f} W"
    )

    # Normalize the input data and create PyTorch Dataset and DataLoader objects.
    train = TensorDataset((x_train - x_mean) / x_std, (y_train - y_mean) / y_std)
    val = TensorDataset((x_val - x_mean) / x_std, (y_val - y_mean) / y_std)
    train_loader = DataLoader(train, batch_size=settings["batch_size"], shuffle=True)
    val_loader = DataLoader(val, batch_size=settings["batch_size"])

    # 2. Predict the total-meter z-score at every time point.
    model = build_model(config["model"]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{device} | {parameter_count:,} trainable parameters")

    # Keep the folder name easy to read. Existing experiments are preserved as
    # "name (2)", "name (3)", and so on instead of adding a timestamp.
    experiment_name = settings["experiment_name"].strip()
    run_dir = create_run_dir(experiment_name)
    checkpoint_path = run_dir / "best.pt"
    print(f"Checkpoint: {checkpoint_path}")

    best_val_mse = float("inf")
    best_epoch = 0
    training_losses = []
    validation_losses = []
    epochs = settings["epochs"]

    #Training Loop
    for epoch in range(1, epochs + 1):
        # 3. Train: compare the prediction with the meter and update the weights.
        model.train()
        train_error, train_points = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            train_error += loss.item() * y.numel()
            train_points += y.numel()

        # 4. Validate: compute the same loss with the weights fixed.
        model.eval()
        val_error, val_points = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_error += loss_fn(model(x), y).item() * y.numel()
                val_points += y.numel()

        train_mse = train_error / train_points
        val_mse = val_error / val_points
        training_losses.append(train_mse)
        validation_losses.append(val_mse)
        train_rmse_w = train_mse**0.5 * y_std.item()
        val_rmse_w = val_mse**0.5 * y_std.item()
        print(
            f"Epoch {epoch:02d} | train MSE {train_mse:.5f} | "
            f"val MSE {val_mse:.5f} | train RMSE {train_rmse_w:.2f} W | "
            f"val RMSE {val_rmse_w:.2f} W"
        )
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model": model.model_name,
                "model_settings": model.model_settings,
                "appliances": APPLIANCES,
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std,
                "config": config,
                "epoch": epoch, "val_mse": val_mse,
                "parameter_count": parameter_count,
            }, checkpoint_path)

    # This records Stage-1 setup, training, validation, and checkpoint saving.
    # It excludes the held-out test evaluation and figure generation below.
    training_seconds = perf_counter() - started

    # Test the checkpoint chosen by validation loss only once. This keeps
    # the test split independent from model selection during training.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    x_test, y_test, _ = load_windows(
        "testing",
        include_calendar_features=use_calendar_features,
    )
    test = TensorDataset(
        (torch.from_numpy(x_test) - x_mean) / x_std,
        (torch.from_numpy(y_test) - y_mean) / y_std,
    )
    test_loader = DataLoader(test, batch_size=settings["batch_size"])

    model.eval()
    test_error, test_points = 0.0, 0
    predicted_watts, true_watts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            predicted_normalized = model(x)
            test_error += loss_fn(predicted_normalized, y).item() * y.numel()
            test_points += y.numel()
            predicted_watts.append((predicted_normalized * y_std + y_mean).cpu())
            true_watts.append((y * y_std + y_mean).cpu())
    test_mse = test_error / test_points
    print(f"Best epoch {best_epoch:02d} | test MSE {test_mse:.5f}")

    aggregate_metrics = aggregate_accuracy_metrics(
        torch.cat(predicted_watts), torch.cat(true_watts)
    )
    metrics_csv, metrics_plot = save_aggregate_metrics(
        aggregate_metrics, test_mse, training_seconds, run_dir
    )
    print(f"Stage 1 training time: {training_seconds / 60:.2f} minutes")
    for name, value in aggregate_metrics.items():
        print(f"Test aggregate {name}: {value:.5f}")
    print(f"Aggregate metrics CSV: {metrics_csv}")
    print(f"Aggregate metrics table: {metrics_plot}")

    # Test MSE is a single point because the test split is evaluated only
    # for the checkpoint selected using validation loss.
    plot_path = plot_loss_curve(
        training_losses, validation_losses, test_mse, best_epoch, run_dir
    )
    print(f"Loss plot: {plot_path}")

    # Select one continuous random week. The same week's shorter portions
    # are used for the 1,024-sample and one-day plots.
    sample_seconds = 8
    week_samples = 7 * 24 * 60 * 60 // sample_seconds
    week_power, week_aggregate = load_random_continuous_interval(
        "testing", week_samples, settings["seed"],
        include_calendar_features=use_calendar_features,
    )

    # Predict the selected week, then convert predictions from normalized
    # values back to real aggregate power in watts. The LSTM is trained
    # on fixed-length windows, so predict the long plotting interval in
    # consecutive windows rather than giving cuDNN a 7-day sequence.
    with torch.no_grad():
        week_input = (torch.from_numpy(week_power[None]) - x_mean) / x_std
        predicted_normalized = torch.cat([
            model(chunk.contiguous().to(device)).cpu()
            for chunk in week_input.split(config["data"]["window_length"], dim=1)
        ], dim=1).squeeze(0)
    week_prediction = (predicted_normalized * y_std + y_mean).numpy()

    for waveform_path in plot_aggregate_predictions(
        week_aggregate, week_prediction, settings["seed"], run_dir
    ):
        print(f"Aggregate prediction plot: {waveform_path}")

    # Select an 8,192-sample continuous test interval per appliance. The
    # focused ON-event plot and its 10x context are cropped from this interval.
    event_examples = load_random_appliance_event_windows(
        "testing", 8192, settings["seed"],
        include_calendar_features=use_calendar_features,
    )
    event_power = np.stack([event_examples[appliance][0] for appliance in APPLIANCES])
    with torch.no_grad():
        event_input = (torch.from_numpy(event_power) - x_mean) / x_std
        event_prediction = torch.cat([
            model(chunk.contiguous().to(device)).cpu()
            for chunk in event_input.split(config["data"]["window_length"], dim=1)
        ], dim=1)
    event_predictions = {
        appliance: (event_prediction[index] * y_std + y_mean).numpy()
        for index, appliance in enumerate(APPLIANCES)
    }
    for plot_path in plot_appliance_scenarios(event_examples, event_predictions, run_dir):
        print(f"Appliance scenario plot: {plot_path}")

    print(f"Elapsed time: {perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()

"""Train the Stage 1 model on five appliance traces and total-meter targets."""

import argparse 
import sys
from datetime import datetime
from itertools import islice
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


def plot_loss_curve(training_losses, validation_losses, test_mse, best_epoch, run_dir):
    """Save training and validation loss per epoch plus one final test point."""
    plot_path = run_dir / "loss_curve.png"
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
    plt.savefig(plot_path, dpi=150)
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
        waveform_path = run_dir / f"test_aggregate_prediction_{plot_name}.png"

        plt.figure(figsize=(12, 5))
        plt.plot(week_aggregate[start:end], label="Real aggregate power", linewidth=1)
        plt.plot(week_prediction[start:end], label="Predicted aggregate power", linewidth=1)
        plt.xlabel("Time step (8 seconds per step)")
        plt.ylabel("Power (W)")
        plt.title(f"Test Aggregate Power: Real vs. Predicted ({plot_name})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(waveform_path, dpi=150)
        plt.close()
        plot_paths.append(waveform_path)

    return plot_paths


def plot_appliance_scenarios(examples, predictions, run_dir):
    """Save one real-versus-predicted aggregate plot for each active appliance."""
    plot_paths = []
    for appliance, (power, aggregate) in examples.items():
        appliance_index = APPLIANCES.index(appliance)
        plot_path = run_dir / f"test_aggregate_prediction_{appliance}.png"
        figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        axes[0].plot(aggregate, label="Real aggregate power", linewidth=1)
        axes[0].plot(predictions[appliance], label="Predicted aggregate power", linewidth=1)
        axes[0].set_ylabel("Aggregate power (W)")
        axes[0].set_title(f"{appliance.capitalize()} ON scenario: Real vs. Predicted Aggregate")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(power[:, appliance_index], color="tab:green", label=f"{appliance} power")
        axes[1].set_xlabel("Time step (8 seconds per step)")
        axes[1].set_ylabel("Appliance power (W)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        figure.tight_layout()
        figure.savefig(plot_path, dpi=150)
        plt.close(figure)
        plot_paths.append(plot_path)

    return plot_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run two batches per split.")
    args = parser.parse_args()
    settings = config["training"]
    torch.manual_seed(settings["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = perf_counter()

    # 1. Load aligned windows in watts. Fit z-score statistics on training only.
    x_train, y_train, _ = load_windows("training")
    del _
    x_val, y_val, _ = load_windows("validating")
    del _
    x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_val, y_val = torch.from_numpy(x_val), torch.from_numpy(y_val)

    # Calculate the z-score statistics on training data only.
    x_mean = x_train.mean(dim=(0, 1))
    x_std = x_train.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    y_mean = y_train.mean()
    y_std = y_train.std(unbiased=False).clamp_min(1e-6)

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

    if not args.smoke:
        # Keep every experiment under results/<experiment_name>_<timestamp>/.
        # The timestamp prevents a second run with the same name from overwriting
        # the earlier checkpoint.
        experiment_name = settings["experiment_name"].strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = (
            Path(__file__).resolve().parents[1]
            / "results"
            / f"{experiment_name}_{timestamp}"
        )
        run_dir.mkdir(parents=True)
        checkpoint_path = run_dir / "best.pt"
        print(f"Checkpoint: {checkpoint_path}")

    best_val_mse = float("inf")
    best_epoch = 0
    training_losses = []
    validation_losses = []
    epochs = 1 if args.smoke else settings["epochs"]
    batch_limit = 2 if args.smoke else None

    #Training Loop
    for epoch in range(1, epochs + 1):
        # 3. Train: compare the prediction with the meter and update the weights.
        model.train()
        train_error, train_points = 0.0, 0
        for x, y in islice(train_loader, batch_limit):
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
            for x, y in islice(val_loader, batch_limit):
                x, y = x.to(device), y.to(device)
                val_error += loss_fn(model(x), y).item() * y.numel()
                val_points += y.numel()

        train_mse = train_error / train_points
        val_mse = val_error / val_points
        training_losses.append(train_mse)
        validation_losses.append(val_mse)
        print(f"Epoch {epoch:02d} | train MSE {train_mse:.5f} | val MSE {val_mse:.5f}")
        if not args.smoke and val_mse < best_val_mse:
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

    if not args.smoke:
        # Test the checkpoint chosen by validation loss only once. This keeps
        # the test split independent from model selection during training.
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        x_test, y_test, _ = load_windows("testing")
        test = TensorDataset(
            (torch.from_numpy(x_test) - x_mean) / x_std,
            (torch.from_numpy(y_test) - y_mean) / y_std,
        )
        test_loader = DataLoader(test, batch_size=settings["batch_size"])

        model.eval()
        test_error, test_points = 0.0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                test_error += loss_fn(model(x), y).item() * y.numel()
                test_points += y.numel()
        test_mse = test_error / test_points
        print(f"Best epoch {best_epoch:02d} | test MSE {test_mse:.5f}")

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
            "testing", week_samples, settings["seed"]
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

        # Select one 1,024-sample test window for each appliance where its
        # ground-truth ON label is active, then plot that appliance's scenario.
        event_examples = load_random_appliance_event_windows("testing", 1024, settings["seed"])
        event_power = np.stack([event_examples[appliance][0] for appliance in APPLIANCES])
        with torch.no_grad():
            event_input = (torch.from_numpy(event_power) - x_mean) / x_std
            event_prediction = model(event_input.to(device)).cpu()
        event_predictions = {
            appliance: (event_prediction[index] * y_std + y_mean).numpy()
            for index, appliance in enumerate(APPLIANCES)
        }
        for plot_path in plot_appliance_scenarios(event_examples, event_predictions, run_dir):
            print(f"Appliance scenario plot: {plot_path}")

    print(f"Elapsed time: {perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()

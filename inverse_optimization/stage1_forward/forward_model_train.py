"""Train the Stage 1 model on five appliance traces and total-meter targets."""

import argparse 
from datetime import datetime
from itertools import islice
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..data_loader import APPLIANCES, config, load_windows
from ..model import build_model


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

        val_mse = val_error / val_points
        validation_losses.append(val_mse)
        print(f"Epoch {epoch:02d} | train MSE {train_error / train_points:.5f} | val MSE {val_mse:.5f}")
        if not args.smoke and val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model": config["model"]["name"],
                "model_settings": {
                    "input_size": model.lstm.input_size,
                    "hidden_size": model.lstm.hidden_size,
                    "num_layers": model.lstm.num_layers,
                    "batch_first": model.lstm.batch_first,
                    "bidirectional": model.lstm.bidirectional,
                    "dropout": model.lstm.dropout,
                },
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

        # The validation curve has one value per epoch. Test MSE is a single
        # point because the test set is evaluated only for the best checkpoint.
        plot_path = run_dir / "loss_curve.png"
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epochs + 1), validation_losses, label="Validation MSE")
        plt.scatter(best_epoch, test_mse, color="red", label="Test MSE (best epoch)")
        plt.xlabel("Epoch")
        plt.ylabel("Normalized MSE")
        plt.title("Validation and Test Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Loss plot: {plot_path}")

    print(f"Elapsed time: {perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()

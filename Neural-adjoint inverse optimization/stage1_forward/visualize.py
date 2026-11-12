"""Small Stage-1 plotting helpers; no training logic is hidden here."""

from pathlib import Path


def save_loss_curve(history: list[dict[str, float]], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot([row["epoch"] for row in history], [row["train_mse"] for row in history], label="train MSE")
    plt.plot([row["epoch"] for row in history], [row["val_mse"] for row in history], label="validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled power MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()

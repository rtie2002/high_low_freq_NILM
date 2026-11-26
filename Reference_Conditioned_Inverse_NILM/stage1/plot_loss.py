"""Save Stage 1 loss curves in a process that does not import PyTorch."""

import json
import sys

from matplotlib.figure import Figure


if __name__ == "__main__":
    history = json.load(sys.stdin)
    epochs = range(1, len(history["train"]) + 1)

    figure = Figure(figsize=(8, 4.5), layout="constrained")
    axes = figure.subplots()
    axes.plot(epochs, history["train"], label="Training loss")
    axes.plot(epochs, history["validation"], label="Validation loss")

    best_epoch = history["best_epoch"]
    if best_epoch is not None:
        best_loss = history["validation"][best_epoch - 1]
        axes.scatter(best_epoch, best_loss, label=f"Best checkpoint (epoch {best_epoch})")

    axes.set(title="Stage 1 training and validation loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes.grid(alpha=0.25)
    axes.legend()
    figure.savefig(sys.argv[1], dpi=150)

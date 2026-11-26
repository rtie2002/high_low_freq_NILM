"""Train the shared Stage 1 reference encoder with five-class supervision."""

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if __package__:
    from .model import APPLIANCES, ApplianceReferenceEncoder
else:
    from model import APPLIANCES, ApplianceReferenceEncoder


# ============================================================
# Edit the experiment settings here.
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "dataset" / "appliance_waveform reference"
RESULTS_ROOT = PROJECT_ROOT / "results" / "stage1"
EXPERIMENT_NAME = "reference_encoder"

SEED = 2026
CHECK_DATA_ONLY = False           # Print data counts without starting training.
EPOCHS = 100
BATCH_SIZE = 20
TRAIN_STEPS_PER_EPOCH = 100
VALIDATION_EVENTS_PER_CLASS_HOUSE = 200  # None scores every validation event
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0                   # Keep waveform arrays in one process on Windows
SAMPLE_SECONDS = 8

STAGE1_VALIDATION_HOUSES = {("refit", 6), ("refit", 10)}
MART_RESERVED_HOUSES = {("refit", 2), ("refit", 20), ("ukdale", 2), ("ukdale", 5)}


# ============================================================
# Read the prepared reference files.
# ============================================================
@dataclass(frozen=True)
class ReferenceEvent:
    power_watts: np.ndarray
    class_id: int
    house: tuple[str, int]  # (dataset, house number)
    event_id: str


def load_reference_events(root: Path) -> list[ReferenceEvent]:
    """Read prepared events using the row ranges saved by the extractor."""
    manifest = pd.read_csv(root / "manifest.csv")
    manifest["file"] = manifest["file"].str.replace("\\", "/", regex=False)

    # Older exports can contain incomplete events. Ask for a fresh export
    # instead of applying a second set of event-selection rules in training.
    incomplete_events = (
        manifest["duration_censored"]
        | manifest["off_pre_samples"].lt(1)
        | manifest["off_post_samples"].lt(1)
    )

    events = []
    for class_id, appliance in enumerate(APPLIANCES):
        folder = appliance.replace("_", "")
        relative_path = f"{folder}/events.csv"
        file_events = manifest[manifest["file"] == relative_path].sort_values("sample_start_row")
        event_lengths = file_events["samples"].to_numpy()
        first_rows = file_events["sample_start_row"].to_numpy()
        last_rows = file_events["sample_end_row"].to_numpy()  # Inclusive.

        print(f"Reading {relative_path} ...", flush=True)
        samples = pd.read_csv(
            root / relative_path,
            usecols=["appliance_power_w"],
            dtype={"appliance_power_w": np.float32},
        )

        # A row range represents one complete event. Keep its original length.
        previous_count = len(events)
        for event in file_events.itertuples(index=False):
            house = (event.dataset, int(event.house))
            if house in MART_RESERVED_HOUSES:
                continue  # Keep MART's held-out houses out of Stage 1.

            first_row = event.sample_start_row
            stop_row = event.sample_end_row + 1
            waveform = power_watts[first_row:stop_row]
            events.append(ReferenceEvent(waveform, class_id, house, event.event_id))
        print(f"  Loaded {len(events) - previous_count:,} {appliance} events.", flush=True)
    return events


# ============================================================
# Prepare variable-length batches and apply Z-score.
# ============================================================
class ReferenceDataset(Dataset):
    """Give the DataLoader one normalized waveform and its class label at a time."""

    def __init__(self, events: list[ReferenceEvent], normalization: dict):
        self.events = events
        self.mean_watts = normalization["offset"]
        self.std_watts = normalization["scale"]

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        event = self.events[index]
        power_watts = torch.from_numpy(event.power_watts)
        normalized_power = (power_watts - self.mean_watts) / self.std_watts
        return normalized_power, event.class_id


def collate_events(batch) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Keep the waveforms in a list so each event can have a different length."""
    waveforms, class_ids = zip(*batch)
    return list(waveforms), torch.tensor(class_ids, dtype=torch.long)


# ============================================================
# Validation: calculate loss and accuracy without updating weights.
# ============================================================
def evaluate(model, loader, loss_function, device, batch_limit=None):
    model.eval()
    loss_sum = 0.0
    correct_count = 0
    event_count = 0
    with torch.no_grad():
        for waveforms, true_classes in islice(loader, batch_limit):
            waveforms = [waveform.to(device) for waveform in waveforms]
            true_classes = true_classes.to(device)
            _, class_scores = model(waveforms)
            loss = loss_function(class_scores, true_classes)
            predictions = class_scores.argmax(dim=1)

            # The loss is a batch mean. Multiply by the batch size before summing.
            loss_sum += loss.item() * len(true_classes)
            correct_count += (predictions == true_classes).sum().item()
            event_count += len(true_classes)

    return loss_sum / event_count, correct_count / event_count


# ============================================================
# Main program: read -> split -> normalize -> train -> save.
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run one train and one validation batch.")
    parser.add_argument("--check-data-only", action="store_true", help="Read data and print counts.")
    args = parser.parse_args()
    check_data_only = CHECK_DATA_ONLY or args.check_data_only

    torch.manual_seed(SEED)

    # Step 1. Read already-selected events and hold out whole houses.
    events = load_reference_events(REFERENCE_ROOT)
    train_events = [event for event in events if event.house not in STAGE1_VALIDATION_HOUSES]
    validation_events = [event for event in events if event.house in STAGE1_VALIDATION_HOUSES]

    train_class_counts = np.bincount(
        [event.class_id for event in train_events], minlength=len(APPLIANCES)
    )
    validation_class_counts = np.bincount(
        [event.class_id for event in validation_events], minlength=len(APPLIANCES)
    )
    for split_name, class_counts in (
        ("Train", train_class_counts), ("Validation", validation_class_counts),
    ):
        counts_by_appliance = dict(zip(APPLIANCES, class_counts.tolist()))

        print(f"{split_name}: {class_counts.sum():,} events | {counts_by_appliance}", flush=True)

    first_event = train_events[0]
    sample_count = len(first_event.power_watts)
    print(
        f"First training event: {first_event.event_id} | {sample_count} points | "
        f"{sample_count * SAMPLE_SECONDS} seconds including context | "
        f"first values (W): {first_event.power_watts[:6].tolist()}",
        flush=True,
    )
    if check_data_only:
        print("Data check complete.", flush=True)
        return

    device = torch.device("cuda")
    print(f"Stage 1 | CUDA: {torch.cuda.get_device_name(device)} | normalization: Z-score", flush=True)

    # Step 2. Select the validation subset once, then reuse it in every epoch.
    if VALIDATION_EVENTS_PER_CLASS_HOUSE is not None:
        validation_groups = defaultdict(list)
        for event in validation_events:
            validation_groups[event.class_id, event.house].append(event)

        validation_random = random.Random(SEED)
        validation_events = []
        for group in validation_groups.values():
            count = min(len(group), VALIDATION_EVENTS_PER_CLASS_HOUSE)
            validation_events.extend(validation_random.sample(group, count))
        validation_random.shuffle(validation_events)
    print(f"Validation used each epoch: {len(validation_events):,} events.", flush=True)

    # Step 3. Fit one Z-score using all training time points.
    # Both DataLoaders use this same mean and standard deviation.
    sample_count = sum(event.power_watts.size for event in train_events)
    total_power = sum(event.power_watts.sum(dtype=np.float64) for event in train_events)
    mean_watts = total_power / sample_count
    squared_differences = sum(
        np.square(event.power_watts.astype(np.float64) - mean_watts).sum()
        for event in train_events
    )
    std_watts = float(np.sqrt(squared_differences / sample_count))
   
    # The checkpoint uses offset for the training mean and scale for the training std.
    normalization = {"method": "zscore", "offset": float(mean_watts), "scale": std_watts}

    # Step 4. Give rare classes larger sampling weights. Across many draws,
    # each class is equally likely. The class counts in one batch can vary.
    event_weights = [1.0 / train_class_counts[event.class_id] for event in train_events]
    train_sampler = WeightedRandomSampler(
        event_weights, num_samples=BATCH_SIZE * TRAIN_STEPS_PER_EPOCH, replacement=True
    )
    train_loader = DataLoader(
        ReferenceDataset(train_events, normalization),
        batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_events,
        num_workers=NUM_WORKERS,
    )
    validation_loader = DataLoader(
        ReferenceDataset(validation_events, normalization),
        batch_size=BATCH_SIZE, collate_fn=collate_events, num_workers=NUM_WORKERS,
    )

    # Step 5. Create the model, classification loss, and optimizer.
    model = ApplianceReferenceEncoder().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_validation_loss = float("inf")
    best_epoch = None
    train_losses = []
    validation_losses = []
    batch_limit = 1 if args.smoke else None  # None means use every batch.
    epoch_count = 1 if args.smoke else EPOCHS

    # Give each full run its own directory. A smoke test skips checkpoint saving.
    checkpoint = None
    if not args.smoke:
        run_dir = RESULTS_ROOT / EXPERIMENT_NAME
        run_number = 2
        while run_dir.exists():
            run_dir = RESULTS_ROOT / f"{EXPERIMENT_NAME} ({run_number})"
            run_number += 1
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "best.pt"
        print(f"Checkpoint: {checkpoint}", flush=True)

    for epoch in range(1, epoch_count + 1):
        # Step 6. Train. Each batch gives one update to the model weights.
        model.train()
        train_loss_sum = 0.0
        train_event_count = 0
        for waveforms, true_classes in islice(train_loader, batch_limit):
            waveforms = [waveform.to(device) for waveform in waveforms]
            true_classes = true_classes.to(device)

            optimizer.zero_grad(set_to_none=True)  # Clear gradients from the previous batch.
            _, class_scores = model(waveforms)    # Forward pass: predict five scores per event.
            loss = loss_function(class_scores, true_classes)  # Compare scores with true labels.
            loss.backward()                       # Compute the gradient of the loss for each weight.
            optimizer.step()                      # AdamW uses those gradients to update the weights.

            train_loss_sum += loss.item() * len(true_classes)
            train_event_count += len(true_classes)

        train_loss = train_loss_sum / train_event_count

        # Step 7. Measure loss and accuracy on held-out houses.
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, loss_function, device, batch_limit
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        print(
            f"Epoch {epoch:02d} | train CE {train_loss:.4f} | "
            f"val CE {validation_loss:.4f} | val accuracy {validation_accuracy:.3f}",
            flush=True,
        )
        # Step 8. Keep the weights from the epoch with the lowest validation loss.
        if checkpoint is not None and validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "appliances": APPLIANCES,
                    "normalization": normalization,
                    "epoch": epoch,
                    "val_loss": validation_loss,
                    "val_accuracy": validation_accuracy,
                    "train_houses": sorted({event.house for event in train_events}),
                    "validation_houses": sorted({event.house for event in validation_events}),
                    "sampling_interval_sec": SAMPLE_SECONDS,
                    "seed": SEED,
                },
                checkpoint,
            )

    if checkpoint is not None:
        # Step 9. Save both loss curves after training. Each point is one epoch.
        if best_epoch is not None:
            print(f"Best Stage 1 checkpoint: {checkpoint}", flush=True)
        plot_path = checkpoint.with_name("loss_curve.png")
        # Use a separate process to avoid PyTorch/Matplotlib OpenMP conflicts on Windows.
        plot_script = Path(__file__).with_name("plot_loss.py")
        subprocess.run(
            [sys.executable, str(plot_script), str(plot_path)],
            input=json.dumps({"train": train_losses, "validation": validation_losses, "best_epoch": best_epoch}),
            text=True,
            check=True,
        )
        print(f"Loss plot: {plot_path}", flush=True)


if __name__ == "__main__":
    main()

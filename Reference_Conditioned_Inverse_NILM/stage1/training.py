"""Train the shared Stage 1 reference encoder with five-class supervision."""

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
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
EXPERIMENT_NAME = "reference_encoder_8_3_3"

SEED = 2026
CHECK_DATA_ONLY = False           # Print data counts without starting training.
EPOCHS = 100
EARLY_STOP_PATIENCE = 12          # Stop after this many epochs without lower validation loss.
USE_BATCH_NORM = False           # Events are encoded one at a time, even within a batch.
BATCH_SIZE = 20
TRAIN_STEPS_PER_EPOCH = 100
VALIDATION_EVENTS_PER_CLASS_HOUSE = 200  # None scores every validation event
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0                   # Keep waveform arrays in one process on Windows
SAMPLE_SECONDS = 8
REFERENCES_PER_CLASS = 5        # Fixed before test evaluation.

STAGE1_TRAIN_HOUSES = {
    ("refit", 1), ("refit", 8), ("refit", 12), ("refit", 17),
    ("refit", 18), ("refit", 19), ("refit", 21), ("ukdale", 3),
}
STAGE1_VALIDATION_HOUSES = {("refit", 6), ("refit", 7), ("refit", 10)}
STAGE1_TEST_HOUSES = {("refit", 4), ("refit", 13), ("refit", 16)}
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
    if incomplete_events.any():
        raise ValueError(
            f"Reference manifest contains {int(incomplete_events.sum()):,} incomplete events. "
            "Regenerate the reference export with applainces_event_selection.py before training."
        )

    events = []
    for class_id, appliance in enumerate(APPLIANCES):
        folder = appliance.replace("_", "")
        relative_path = f"{folder}/events.csv"
        file_events = manifest[manifest["file"] == relative_path].sort_values("sample_start_row")
        event_lengths = file_events["samples"].to_numpy()
        first_rows = file_events["sample_start_row"].to_numpy()
        last_rows = file_events["sample_end_row"].to_numpy()  # Inclusive.
        if (file_events.empty
                or not file_events["sample_period_sec"].eq(SAMPLE_SECONDS).all()
                or (event_lengths < 2).any()
                or not np.array_equal(first_rows, np.cumsum(event_lengths) - event_lengths)
                or not np.array_equal(last_rows, first_rows + event_lengths - 1)):
            raise ValueError(f"Invalid reference manifest rows for {relative_path}.")

        print(f"Reading {relative_path} ...", flush=True)
        samples = pd.read_csv(
            root / relative_path,
            usecols=["appliance_power_w"],
            dtype={"appliance_power_w": np.float32},
        )
        power_watts = samples["appliance_power_w"].to_numpy()
        if len(power_watts) != event_lengths.sum() or not np.isfinite(power_watts).all():
            raise ValueError(f"Invalid power data or CSV/manifest row count mismatch in {relative_path}.")

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


def select_validation_events(events, limit, seed):
    """Use the same fixed sample for training-time and later validation."""
    if limit is None:
        return events
    groups = defaultdict(list)
    for event in events:
        groups[event.class_id, event.house].append(event)

    validation_random = random.Random(seed)
    selected_events = []
    for group in groups.values():
        selected_events.extend(validation_random.sample(group, min(len(group), limit)))
    validation_random.shuffle(selected_events)
    return selected_events


def reference_manifest_hash(root: Path) -> str:
    """Identify the event manifest used by a checkpoint."""
    return sha256((root / "manifest.csv").read_bytes()).hexdigest()


# ============================================================
# Validation: calculate loss, accuracy and per-appliance F1.
# ============================================================
def evaluate(model, loader, loss_function, device, batch_limit=None):
    model.eval()
    loss_sum = 0.0
    event_count = 0
    class_count = len(APPLIANCES)
    confusion = torch.zeros((class_count, class_count), dtype=torch.long, device=device)
    with torch.no_grad():
        for waveforms, true_classes in islice(loader, batch_limit):
            waveforms = [waveform.to(device) for waveform in waveforms]
            true_classes = true_classes.to(device)
            _, class_scores = model(waveforms)
            loss = loss_function(class_scores, true_classes)
            predictions = class_scores.argmax(dim=1)

            # The loss is a batch mean. Multiply by the batch size before summing.
            loss_sum += loss.item() * len(true_classes)
            # Rows are true classes. Columns are predicted classes.
            pairs = true_classes * class_count + predictions
            confusion += torch.bincount(pairs, minlength=class_count ** 2).reshape(class_count, class_count)
            event_count += len(true_classes)

    correct_by_class = confusion.diag()
    true_counts = confusion.sum(dim=1)
    predicted_counts = confusion.sum(dim=0)
    f1_by_class = 2 * correct_by_class / (true_counts + predicted_counts).clamp_min(1)
    accuracy = correct_by_class.sum().item() / event_count
    return loss_sum / event_count, accuracy, f1_by_class.tolist(), confusion.tolist()


# ============================================================
# Main program: read -> split -> normalize -> train -> save.
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run one train and one validation batch.")
    parser.add_argument("--check-data-only", action="store_true", help="Read data and print counts.")
    args = parser.parse_args()
    check_data_only = CHECK_DATA_ONLY or args.check_data_only

    if not check_data_only and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Stage 1 training.")
    all_split_houses = STAGE1_TRAIN_HOUSES | STAGE1_VALIDATION_HOUSES | STAGE1_TEST_HOUSES
    split_house_count = len(STAGE1_TRAIN_HOUSES) + len(STAGE1_VALIDATION_HOUSES) + len(STAGE1_TEST_HOUSES)
    if len(all_split_houses) != split_house_count or all_split_houses & MART_RESERVED_HOUSES:
        raise ValueError("Stage 1 house splits must be disjoint and exclude MART-reserved houses.")
    torch.manual_seed(SEED)

    # Step 1. Assign whole houses before fitting anything. The test partition
    # is used here only to check coverage, and never enters a DataLoader.
    manifest_hash = reference_manifest_hash(REFERENCE_ROOT)
    events = load_reference_events(REFERENCE_ROOT)
    train_events = [event for event in events if event.house in STAGE1_TRAIN_HOUSES]
    validation_events = [event for event in events if event.house in STAGE1_VALIDATION_HOUSES]
    test_events = [event for event in events if event.house in STAGE1_TEST_HOUSES]

    for split_name, split_events, expected_houses in (
        ("Train", train_events, STAGE1_TRAIN_HOUSES),
        ("Validation", validation_events, STAGE1_VALIDATION_HOUSES),
        ("Test (held out)", test_events, STAGE1_TEST_HOUSES),
    ):
        class_counts = np.bincount(
            [event.class_id for event in split_events], minlength=len(APPLIANCES)
        )
        present_houses = {event.house for event in split_events}
        if present_houses != expected_houses:
            raise ValueError(f"{split_name} has missing houses: {sorted(expected_houses - present_houses)}")
        counts_by_appliance = dict(zip(APPLIANCES, class_counts.tolist()))
        if (class_counts == 0).any():
            raise ValueError(f"{split_name} must contain all five classes: {counts_by_appliance}")
        print(f"{split_name}: {len(present_houses)} houses | {len(split_events):,} events | {counts_by_appliance}", flush=True)
        print(f"  Houses: {sorted(present_houses)}", flush=True)
        if split_name != "Train":
            houses_by_class = [set() for _ in APPLIANCES]
            for event in split_events:
                houses_by_class[event.class_id].add(event.house)
            for appliance, houses in zip(APPLIANCES, houses_by_class):
                print(f"  {appliance}: {sorted(houses)}", flush=True)
                if len(houses) < 2:
                    raise ValueError(f"{split_name} needs at least two houses for {appliance}.")
    del test_events, split_events, events

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
    print(
        f"Stage 1 | CUDA: {torch.cuda.get_device_name(device)} | "
        f"normalization: Z-score | BatchNorm: {USE_BATCH_NORM}",
        flush=True,
    )

    # Step 2. Select the validation subset once, then reuse it in every epoch.
    validation_events = select_validation_events(validation_events, VALIDATION_EVENTS_PER_CLASS_HOUSE, SEED)
    print(f"Validation used each epoch: {len(validation_events):,} events.", flush=True)

    # Step 3. Give equal weight to each appliance, then each house in that
    # appliance, then each event from that house. A large fridge file should
    # not dominate either the Z-score or the training batches.
    train_group_counts = defaultdict(int)
    train_houses_by_class = [set() for _ in APPLIANCES]
    for event in train_events:
        train_group_counts[event.class_id, event.house] += 1
        train_houses_by_class[event.class_id].add(event.house)
    event_weights = [
        1.0 / (len(train_houses_by_class[event.class_id]) * train_group_counts[event.class_id, event.house])
        for event in train_events
    ]

    # Fit one shared Z-score from training data. First take the mean within
    # each event, so a long event does not get extra weight just for its length.
    total_weight = sum(event_weights)
    mean_watts = sum(
        weight * event.power_watts.mean(dtype=np.float64)
        for event, weight in zip(train_events, event_weights)
    ) / total_weight
    squared_differences = sum(
        weight * np.square(event.power_watts.astype(np.float64) - mean_watts).mean()
        for event, weight in zip(train_events, event_weights)
    )
    std_watts = float(np.sqrt(squared_differences / total_weight))
    if not np.isfinite(std_watts) or std_watts <= 0:
        raise ValueError("Training power must have a finite, positive standard deviation.")
    normalization = {
        "method": "zscore", "offset": float(mean_watts), "scale": std_watts,
        "weighting": "equal_class_then_house_then_event",
    }
    print(f"Training scale: mean {mean_watts:.2f} W | std {std_watts:.2f} W", flush=True)

    # Step 4. Use the same weights for drawing batches. Each appliance gets
    # 20% of the probability, and houses within that appliance are equally likely.
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
    model = ApplianceReferenceEncoder(use_batch_norm=USE_BATCH_NORM).to(device)
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
        validation_loss, validation_accuracy, validation_f1, validation_confusion = evaluate(
            model, validation_loader, loss_function, device, batch_limit
        )
        if not np.isfinite(validation_loss):
            raise ValueError("Validation loss is not finite. Check the input waveforms and Z-score.")
        validation_macro_f1 = sum(validation_f1) / len(APPLIANCES)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        print(
            f"Epoch {epoch:02d} | train CE {train_loss:.4f} | "
            f"val CE {validation_loss:.4f} | val accuracy {validation_accuracy:.3f} | "
            f"val macro-F1 {validation_macro_f1:.3f}",
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
                    "val_macro_f1": validation_macro_f1,
                    "val_f1_by_appliance": dict(zip(APPLIANCES, validation_f1)),
                    "val_confusion_matrix": validation_confusion,
                    "use_batch_norm": USE_BATCH_NORM,
                    "train_houses": sorted({event.house for event in train_events}),
                    "validation_houses": sorted({event.house for event in validation_events}),
                    "validation_event_ids": [event.event_id for event in validation_events],
                    "test_houses": sorted(STAGE1_TEST_HOUSES),
                    "reference_manifest_sha256": manifest_hash,
                    "references_per_class": REFERENCES_PER_CLASS,
                    "sampling_interval_sec": SAMPLE_SECONDS,
                    "seed": SEED,
                },
                checkpoint,
            )
            print(f"  Best epoch F1: {dict(zip(APPLIANCES, [round(f1, 3) for f1 in validation_f1]))}", flush=True)

        if best_epoch is not None and epoch - best_epoch >= EARLY_STOP_PATIENCE:
            print(f"Early stop: validation loss has not improved for {EARLY_STOP_PATIENCE} epochs.", flush=True)
            break

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

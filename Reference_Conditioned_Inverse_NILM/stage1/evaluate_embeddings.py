"""Compare Stage 1 classification scores and reference-vector matching."""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader

if __package__:
    from . import training
else:
    import training


DEFAULT_CHECKPOINT = training.RESULTS_ROOT / training.EXPERIMENT_NAME / "best.pt"
REFERENCES_PER_CLASS = 5


def select_references(train_events, references_per_class, seed):
    """Pick independent events from different houses before reusing a house."""
    groups = defaultdict(lambda: defaultdict(list))
    for event in train_events:
        groups[event.class_id][event.house].append(event)

    random_picker = random.Random(seed)
    references = []
    for class_id, appliance in enumerate(training.APPLIANCES):
        houses = sorted(groups[class_id])
        if not houses:
            raise ValueError(f"No training-house references for {appliance}.")
        random_picker.shuffle(houses)
        events_by_house = [
            random_picker.sample(groups[class_id][house], min(references_per_class, len(groups[class_id][house])))
            for house in houses
        ]
        # Take one event from every house before taking a second event from any house.
        selected = [
            events[event_number]
            for event_number in range(references_per_class)
            for events in events_by_house
            if event_number < len(events)
        ][:references_per_class]
        references.append(selected)
    return references


def measure_confusion(matrix):
    """Convert confusion counts into per-class and overall scores."""
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = matrix.diagonal()
    recall = correct / np.maximum(actual, 1)
    precision = correct / np.maximum(predicted, 1)
    f1 = 2 * correct / np.maximum(actual + predicted, 1)
    return actual, correct, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--references-per-class", type=int, default=REFERENCES_PER_CLASS)
    args = parser.parse_args()
    if args.references_per_class < 1:
        parser.error("--references-per-class must be at least 1.")

    # Step 1. Read the model's original house split and normalization.
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    appliances = tuple(checkpoint["appliances"])
    if appliances != training.APPLIANCES:
        raise ValueError("The checkpoint's appliance order does not match the reference dataset.")
    train_houses = {tuple(house) for house in checkpoint["train_houses"]}
    validation_houses = {tuple(house) for house in checkpoint["validation_houses"]}
    if train_houses & validation_houses:
        raise ValueError("Training and validation houses overlap in the checkpoint.")

    events = training.load_reference_events(training.REFERENCE_ROOT)
    train_events = [event for event in events if event.house in train_houses]
    validation_events = [event for event in events if event.house in validation_houses]
    saved_ids = checkpoint.get("validation_event_ids")
    if saved_ids is None:
        validation_events = training.select_validation_events(
            validation_events, training.VALIDATION_EVENTS_PER_CLASS_HOUSE, checkpoint["seed"]
        )
        validation_note = "Reconstructed from the saved houses and seed using the current validation limit."
    else:
        events_by_id = {event.event_id: event for event in validation_events}
        if any(event_id not in events_by_id for event_id in saved_ids):
            raise ValueError("Some saved validation events are missing. Use the original reference export.")
        validation_events = [events_by_id[event_id] for event_id in saved_ids]
        validation_note = "Used the validation event IDs saved in the checkpoint."
    if {event.class_id for event in validation_events} != set(range(len(appliances))):
        raise ValueError("The validation data must contain all five appliance classes.")

    # Step 2. Build one prototype per class from training-house reference events.
    # Normalize each reference vector, average within its class, then normalize the mean.
    reference_groups = select_references(train_events, args.references_per_class, checkpoint["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = training.ApplianceReferenceEncoder(use_batch_norm=checkpoint.get("use_batch_norm", True))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # Step 3. Make both predictions for the exact same validation waveforms.
    normalization = checkpoint["normalization"]
    validation_loader = DataLoader(
        training.ReferenceDataset(validation_events, normalization),
        batch_size=training.BATCH_SIZE,
        collate_fn=training.collate_events,
    )
    class_count = len(appliances)
    classifier_confusion = torch.zeros((class_count, class_count), dtype=torch.long, device=device)
    embedding_confusion = torch.zeros_like(classifier_confusion)
    with torch.inference_mode():
        prototypes = []
        for reference_events in reference_groups:
            reference_data = training.ReferenceDataset(reference_events, normalization)
            vectors = torch.stack([model.encode(waveform.to(device)) for waveform, _ in reference_data])
            prototypes.append(normalize(vectors, dim=1).mean(dim=0))
        prototypes = normalize(torch.stack(prototypes), dim=1)

        for waveforms, true_classes in validation_loader:
            vectors, class_scores = model([waveform.to(device) for waveform in waveforms])
            cosine_scores = normalize(vectors, dim=1) @ prototypes.T
            true_classes = true_classes.to(device)
            for scores, confusion in ((class_scores, classifier_confusion), (cosine_scores, embedding_confusion)):
                predictions = scores.argmax(dim=1)
                pairs = true_classes * class_count + predictions
                confusion += torch.bincount(pairs, minlength=class_count ** 2).reshape(class_count, class_count)

    # Step 4. Write the report and machine-readable confusion counts beside best.pt.
    classifier_confusion = classifier_confusion.cpu().numpy()
    embedding_confusion = embedding_confusion.cpu().numpy()
    saved_confusion = checkpoint.get("val_confusion_matrix")
    if saved_confusion is not None:
        if np.array_equal(classifier_confusion, saved_confusion):
            validation_note += " The classifier confusion matrix matches the checkpoint."
        else:
            validation_note += " Warning: the classifier confusion matrix differs from the checkpoint."

    report = [
        "# Stage 1 validation: classifier and embedding matching", "",
        f"Checkpoint: `{args.checkpoint.name}` (epoch {checkpoint['epoch']}).",
        f"Validation events: **{len(validation_events):,}**. {validation_note}",
        f"Validation houses: {', '.join(f'{dataset} H{house}' for dataset, house in sorted(validation_houses))}.",
        "", "Each validation event receives one predicted appliance label.",
        "This evaluates labeled events; it does not detect ON/OFF times in aggregate power.",
        "", "## Training-house reference bank", "",
        "| Appliance | Reference events | Reference houses |", "|---|---:|---:|",
    ]
    for appliance, references in zip(appliances, reference_groups):
        report.append(f"| {appliance} | {len(references)} | {len({event.house for event in references})} |")

    for title, name, matrix in (
        ("Five-class classifier", "classifier", classifier_confusion),
        ("Cosine matching with the 128-dimensional embedding", "embedding", embedding_confusion),
    ):
        actual, correct, precision, recall, f1 = measure_confusion(matrix)
        accuracy = correct.sum() / actual.sum()
        report.extend([
            "", f"## {title}", "",
            f"Overall accuracy: **{accuracy:.2%}** ({correct.sum():,}/{actual.sum():,}). "
            f"Macro-F1: **{f1.mean():.3f}**.",
            "", "| Appliance | True events | Correct | Recall | Precision | F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for index, appliance in enumerate(appliances):
            report.append(
                f"| {appliance} | {actual[index]} | {correct[index]} | {recall[index]:.2%} | "
                f"{precision[index]:.2%} | {f1[index]:.3f} |"
            )

        report.extend([
            "", "Confusion matrix: rows are true classes and columns are predictions.", "",
            "| True / predicted | " + " | ".join(appliances) + " |",
            "|---|" + "---:|" * class_count,
        ])
        for appliance, row in zip(appliances, matrix):
            report.append(f"| {appliance} | " + " | ".join(str(count) for count in row) + " |")
        pd.DataFrame(matrix, index=appliances, columns=appliances).to_csv(
            args.checkpoint.with_name(f"{name}_confusion.csv"), index_label="true_label"
        )
        print(f"{title}: {correct.sum():,}/{actual.sum():,} correct | accuracy {accuracy:.2%} | macro-F1 {f1.mean():.3f}")

    report_path = args.checkpoint.with_name("embedding_validation.md")
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

"""Remove incomplete saved dishwasher references without redetecting events."""
from pathlib import Path
import pandas as pd
import numpy as np

root = Path(__file__).resolve().parents[2] / "Reference_Conditioned_Inverse_NILM/dataset/appliance_waveform reference"
source, manifest_path = root / "dishwasher/events.csv", root / "manifest.csv"
manifest = pd.read_csv(manifest_path)
dish = manifest[manifest.event_id.str.contains("_dishwasher_")].copy()
valid = ~dish.duration_censored.astype(bool) & (dish.off_pre_samples >= 16) & (dish.off_post_samples >= 16)
kept = dish[valid].copy()
ids = set(kept.event_id)
staged = source.with_name("events.complete.tmp")
signature = (source.stat().st_size, source.stat().st_mtime_ns,
             manifest_path.stat().st_size, manifest_path.stat().st_mtime_ns)
count = 0
with staged.open("w", encoding="utf-8", newline="") as output:
    for chunk in pd.read_csv(source, chunksize=100000):
        selected = chunk[chunk.event_id.isin(ids)]
        if len(selected):
            selected.to_csv(output, index=False, header=(count == 0))
            count += len(selected)
assert count == kept.samples.sum()
kept["sample_start_row"] = np.r_[0, kept.samples.cumsum().to_numpy()[:-1]]
kept["sample_end_row"] = kept.sample_start_row + kept.samples - 1
updated = pd.concat([manifest[~manifest.event_id.str.contains("_dishwasher_")], kept], ignore_index=True)
manifest_stage = manifest_path.with_name("manifest.complete.tmp")
updated.to_csv(manifest_stage, index=False)
assert signature == (source.stat().st_size, source.stat().st_mtime_ns,
                     manifest_path.stat().st_size, manifest_path.stat().st_mtime_ns)
backup = root / "_before_dishwasher_completeness"
backup.mkdir(exist_ok=False)
for path in (source, manifest_path, staged, manifest_stage):
    assert path.resolve().is_relative_to(root.resolve())
source.replace(backup / "dishwasher_events.csv")
manifest_path.replace(backup / "manifest.csv")
staged.replace(source)
manifest_stage.replace(manifest_path)
print(f"Dishwasher: {len(dish)} -> {len(kept)} events; {len(dish)-len(kept)} incomplete references removed.")
print(f"Sample rows retained: {count:,}. Backup: {backup}")

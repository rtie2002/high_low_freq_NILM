"""Compact existing reference tables without rerunning event extraction."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "Reference_Conditioned_Inverse_NILM/dataset/appliance_waveform reference"
BACKUP = ROOT / "_schema_backup"
KEEP = ["event_id", "dataset", "house", "channel", "event_on", "above_threshold",
        "source_idx", "timestamp", "unix_time", "appliance_power_w", "aggregate_power_w"]
DROP_METRICS = ["refit_quality_checked", "longest_severe_mismatch_sec",
                "severe_active_fraction", "refit_issue_fraction"]
META = ["event_id", "dataset", "house", "channel", "on_start_idx", "on_end_idx",
        "ref_start_idx", "ref_end_idx", "off_pre_samples", "off_post_samples",
        "duration_sec", "peak_power_w", "mean_on_power_w", "duration_censored"]


def main():
    if BACKUP.exists():
        raise RuntimeError("Backup directory already exists; preserve previous backup.")
    manifest_rows = []
    signatures = {}
    staging = ROOT / "_schema_staging"
    staging.mkdir(exist_ok=True)
    end_times = {}
    old_bytes = new_bytes = 0
    for app in ["kettle", "microwave", "dishwasher", "washingmachine", "fridge"]:
        source = ROOT / app / "events.csv"
        destination = staging / app / "events.csv"
        destination.parent.mkdir(exist_ok=True)
        signatures[source] = (source.stat().st_size, source.stat().st_mtime_ns)
        header = pd.read_csv(source, nrows=0).columns
        extras = [c for c in ["on_end_idx", "sample_end_unix"] if c in header]
        columns = list(dict.fromkeys(KEEP + META + extras))
        if "event_on" not in header:
            columns[columns.index("event_on")] = "is_on"
        rows = 0
        records = {}
        with destination.open("w", encoding="utf-8", newline="") as stream:
            for chunk in pd.read_csv(source, usecols=columns, chunksize=100000):
                if "event_on" not in chunk:
                    chunk.rename(columns={"is_on": "event_on"}, inplace=True)
                if "on_end_idx" in chunk and "sample_end_unix" in chunk:
                    ends = chunk[chunk.source_idx == chunk.on_end_idx]
                    end_times.update(zip(ends.event_id, ends.sample_end_unix))
                chunk.reset_index(drop=True, inplace=True)
                for event_id, positions in chunk.groupby("event_id", sort=False).indices.items():
                    first, last = int(positions[0]), int(positions[-1])
                    assert last - first + 1 == len(positions)
                    if event_id not in records:
                        record = chunk.iloc[first][META].to_dict()
                        record.update(samples=0, sample_start_row=rows + first,
                                      sample_end_row=rows + first - 1, file=f"{app}/events.csv")
                        records[event_id] = record
                    record = records[event_id]
                    assert record['sample_end_row'] + 1 == rows + first
                    record['samples'] += len(positions)
                    record['sample_end_row'] = rows + last
                chunk[KEEP].to_csv(stream, index=False, header=(rows == 0))
                rows += len(chunk)
        for record in records.values():
            assert record['samples'] == record['ref_end_idx'] - record['ref_start_idx'] + 1, record['event_id']
        manifest_rows.extend(records.values())
        old_bytes += source.stat().st_size
        new_bytes += destination.stat().st_size
        print(f"{app}: {rows:,} rows preserved | {source.stat().st_size/1e6:.1f} -> {destination.stat().st_size/1e6:.1f} MB", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    if end_times:
        manifest["on_end_unix"] = manifest.event_id.map(end_times)
        assert manifest.on_end_unix.notna().all()
        manifest["on_end_unix"] = manifest.on_end_unix.astype("int64")
    manifest.to_csv(staging / "manifest.csv", index=False)
    for source, signature in signatures.items():
        assert signature == (source.stat().st_size, source.stat().st_mtime_ns), 'Source changed during conversion'

    # Exact targets stay inside this dataset directory. Original files are moved
    # to a recoverable backup before the staged files replace them.
    targets = [ROOT / app / "events.csv" for app in ["kettle", "microwave", "dishwasher", "washingmachine", "fridge"]]
    targets.append(ROOT / "manifest.csv")
    for target in targets:
        assert target.resolve().is_relative_to(ROOT.resolve())
    BACKUP.mkdir()
    for target in targets:
        relative = target.relative_to(ROOT)
        backup_file = BACKUP / relative
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        target.replace(backup_file)
        (staging / relative).replace(target)
    print(f"TOTAL waveform size: {old_bytes/1e9:.3f} -> {new_bytes/1e9:.3f} GB", flush=True)
    print(f"Backup: {BACKUP}", flush=True)


if __name__ == "__main__":
    main()

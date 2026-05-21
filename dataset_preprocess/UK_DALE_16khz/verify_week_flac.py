"""
Validate a UK-DALE 16 kHz week folder (hourly vi-*.flac files).

Run interactively (prompts for folder path) or pass --path explicitly.
Optionally compares file count and sizes against CEDA data.ceda.ac.uk.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

_TZ_CACHE: dict[str, timezone] = {}
_BST_FALLBACK_NOTE: str | None = None


def get_tz(tz_name: str) -> timezone:
    """Resolve timezone; on Windows without tzdata, fall back to UTC+1 for UK summer."""
    global _BST_FALLBACK_NOTE
    if tz_name == "UTC":
        return timezone.utc
    if tz_name in _TZ_CACHE:
        return _TZ_CACHE[tz_name]
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(tz_name)
    except Exception:
        if tz_name in ("Europe/London", "Europe/London/BST"):
            tz = timezone(timedelta(hours=1))
            _BST_FALLBACK_NOTE = (
                "tzdata not installed; UK times shown as UTC+1 (BST, correct for Jul 2013 wk30)."
            )
        else:
            raise
    _TZ_CACHE[tz_name] = tz
    return tz

try:
    import requests
except ImportError:
    requests = None

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("ERROR: soundfile is required. Install with: pip install soundfile")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
FLAC_PATTERN = re.compile(r"^vi-(\d+)_\d+\.flac$", re.IGNORECASE)
EXPECTED_FS = 16000
EXPECTED_GAP_SEC = 3600
EXPECTED_FILES_PER_WEEK = 168
TINY_FILE_BYTES = 1_000_000
CEDA_JSON_TEMPLATE = (
    "https://data.ceda.ac.uk/edc/d1/887733b3-4c04-471f-9404-9f7459c4a1a0"
    "/data/version_0/{house}/{year}/{week}/?json"
)


def prompt_week_path() -> Path:
    print("\nUK-DALE 16 kHz — Week folder validator")
    print("=" * 50)
    print("Enter the folder to analyse, for example:")
    print(r"  dataset_preprocess\UK_DALE_16khz\house_2\2013\wk30")
    print("Press Enter to use that example path.\n")
    raw = input("Week folder path: ").strip().strip('"').strip("'")
    if not raw:
        raw = r"dataset_preprocess\UK_DALE_16khz\house_2\2013\wk30"
    path = Path(raw)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
        if not path.exists():
            path = (PROJECT_ROOT / raw).resolve()
    return path


def parse_week_meta(week_dir: Path) -> tuple[str, str, str] | None:
    """Return (house, year, week) from path like .../house_2/2013/wk30."""
    parts = week_dir.parts
    for i in range(len(parts) - 2, -1, -1):
        if parts[i].startswith("house_") and re.match(r"^\d{4}$", parts[i + 1]):
            house, year = parts[i], parts[i + 1]
            week = parts[i + 2] if i + 2 < len(parts) else week_dir.name
            if week.startswith("wk"):
                return house, year, week
    return None


def ceda_listing_url(week_dir: Path) -> str | None:
    meta = parse_week_meta(week_dir)
    if not meta:
        return None
    house, year, week = meta
    return CEDA_JSON_TEMPLATE.format(house=house, year=year, week=week)


def fetch_remote_flacs(url: str) -> dict[str, int] | None:
    if requests is None:
        return None
    try:
        resp = requests.get(url, timeout=45)
        resp.raise_for_status()
        data = resp.json()
        out = {}
        for item in data.get("items", []):
            name = item.get("name", "")
            if name.endswith(".flac"):
                size = item.get("size") or item.get("file_size")
                if size is not None:
                    out[name] = int(size)
        return out
    except Exception:
        return None


def ts_from_name(name: str) -> int | None:
    m = FLAC_PATTERN.match(name)
    return int(m.group(1)) if m else None


def fmt_dt(ts: int, tz_name: str = "UTC") -> str:
    tz = get_tz(tz_name)
    label = "BST" if tz_name == "Europe/London" and _BST_FALLBACK_NOTE else "%Z"
    return datetime.fromtimestamp(ts, tz=tz).strftime(f"%Y-%m-%d %H:%M:%S {label}")


def six_second_windows(frames: int, fs: int = EXPECTED_FS) -> int:
    return int(frames / fs / 6)


def audio_minutes(frames: int, fs: int = EXPECTED_FS) -> float:
    return frames / fs / 60.0


def format_duration_cluster_table(frame_groups: Counter[int]) -> list[str]:
    """Build rows for the duration-cluster summary table."""
    rows: list[tuple[str, int, str, int]] = []
    for frames, count in sorted(frame_groups.items()):
        duration_s = frames / EXPECTED_FS
        rows.append(
            (
                f"~{duration_s:.2f} s",
                count,
                f"~{audio_minutes(frames):.2f} min",
                six_second_windows(frames),
            )
        )
    if not rows:
        return ["  (no audio files analysed)"]

    headers = ("Duration cluster", "# files", "Audio length", "6 s windows")
    col_w = [
        max(len(headers[0]), *(len(r[0]) for r in rows)),
        max(len(headers[1]), *(len(str(r[1])) for r in rows)),
        max(len(headers[2]), *(len(r[2]) for r in rows)),
        max(len(headers[3]), *(len(str(r[3])) for r in rows)),
    ]
    sep = "  " + "  ".join("-" * w for w in col_w)
    header_line = "  " + "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
    out = [header_line, sep]
    for dur, n, amin, wins in rows:
        out.append(
            "  "
            + "  ".join(
                [
                    dur.ljust(col_w[0]),
                    str(n).rjust(col_w[1]),
                    amin.ljust(col_w[2]),
                    str(wins).rjust(col_w[3]),
                ]
            )
        )
    return out


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.ok = True

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def section(self, title: str) -> None:
        self.add()
        self.add(title)
        self.add("-" * len(title))

    def pass_fail(self, condition: bool, ok_msg: str, fail_msg: str) -> None:
        if condition:
            self.add(f"  [OK]   {ok_msg}")
        else:
            self.ok = False
            self.add(f"  [FAIL] {fail_msg}")  # caller builds fail_msg only when needed

    def dump(self) -> str:
        return "\n".join(self.lines)


def validate_week(week_dir: Path, check_remote: bool = True) -> Report:
    r = Report()
    r.section("UK-DALE Week FLAC Validation")

    if not week_dir.is_dir():
        r.pass_fail(False, "", f"Folder does not exist: {week_dir}")
        return r

    r.add(f"Folder: {week_dir}")
    meta = parse_week_meta(week_dir)
    if meta:
        r.add(f"Parsed: {meta[0]} / {meta[1]} / {meta[2]}")
    else:
        r.add("Parsed: (could not infer house/year/week from path)")

    flacs = sorted(week_dir.glob("*.flac"))
    other = [p.name for p in week_dir.iterdir() if p.is_file() and not p.name.endswith(".flac")]

    r.section("1. Folder layout")
    subdirs = [p.name for p in week_dir.iterdir() if p.is_dir()]
    r.add(f"  FLAC files:     {len(flacs)}")
    r.add(f"  Subfolders:     {subdirs if subdirs else '(none — flat week folder)'}")
    r.add(f"  Other files:    {other if other else '(none)'}")

    if not flacs:
        r.pass_fail(False, "", "No .flac files found.")
        return r

    r.section("2. Weekly coverage (1 file per hour)")
    timestamps = []
    bad_names = []
    for f in flacs:
        ts = ts_from_name(f.name)
        if ts is None:
            bad_names.append(f.name)
        else:
            timestamps.append((ts, f.name))

    r.pass_fail(not bad_names, "All filenames match vi-<unix>_<us>.flac", f"Bad filenames: {bad_names[:5]}")

    ts_sorted = sorted(t for t, _ in timestamps)
    gaps = [ts_sorted[i + 1] - ts_sorted[i] for i in range(len(ts_sorted) - 1)]
    span_h = (ts_sorted[-1] - ts_sorted[0]) / 3600 if len(ts_sorted) > 1 else 0

    r.add(f"  First file (UTC): {flacs[0].name}")
    r.add(f"                    {fmt_dt(ts_sorted[0])}")
    r.add(f"  First file (BST): {fmt_dt(ts_sorted[0], 'Europe/London')}")
    if _BST_FALLBACK_NOTE:
        r.add(f"  ({_BST_FALLBACK_NOTE})")
    r.add(f"  Last start (UTC): {flacs[-1].name}")
    r.add(f"                    {fmt_dt(ts_sorted[-1])}")
    r.add(f"  Last start (BST): {fmt_dt(ts_sorted[-1], 'Europe/London')}")
    r.add(f"  Span (hours):     {span_h:.0f} h between first and last start")
    r.add(f"  Expected files:   {EXPECTED_FILES_PER_WEEK} (= 7 days × 24 h)")

    r.pass_fail(
        len(flacs) == EXPECTED_FILES_PER_WEEK,
        f"File count is {len(flacs)} (full week)",
        f"File count is {len(flacs)}, expected {EXPECTED_FILES_PER_WEEK}",
    )

    if gaps:
        wrong = [(ts_sorted[i], ts_sorted[i + 1], gaps[i]) for i in range(len(gaps)) if gaps[i] != EXPECTED_GAP_SEC]
        r.add(f"  Gap between starts: min={min(gaps)}s max={max(gaps)}s median={statistics.median(gaps)}s")
        if wrong:
            r.pass_fail(
                False,
                "",
                f"{len(wrong)} gap(s) are not 3600 s (first: {wrong[0][2]} s)",
            )
        else:
            r.pass_fail(True, "All hourly gaps are exactly 3600 s", "")

    r.section("3. File sizes on disk")
    sizes = [f.stat().st_size for f in flacs]
    tiny = [(f.name, f.stat().st_size) for f in flacs if f.stat().st_size < TINY_FILE_BYTES]
    r.add(f"  Size (MB): min={min(sizes)/1e6:.2f}  median={statistics.median(sizes)/1e6:.2f}  max={max(sizes)/1e6:.2f}")
    r.pass_fail(not tiny, "No suspiciously small files (< 1 MB)", f"{len(tiny)} file(s) under 1 MB")

    r.section("4. Audio integrity (soundfile)")
    read_errors = []
    durations = []
    frame_groups: Counter[int] = Counter()
    sr_set: set[int] = set()
    nan_files: list[str] = []
    failed_files: set[str] = set()
    peak_min, peak_max = 1.0, 0.0
    probe_frames = 16_000  # 1 s at 16 kHz — start + end of each file
    n_analysed = len(flacs)
    file_meta: dict[str, dict] = {}

    for f in flacs:
        try:
            info = sf.info(str(f))
            sr_set.add(info.samplerate)
            durations.append(info.duration)
            frame_groups[info.frames] += 1
            file_meta[f.name] = {"duration": info.duration, "frames": info.frames}
            if info.samplerate != EXPECTED_FS:
                read_errors.append((f.name, f"sr={info.samplerate}"))
                failed_files.add(f.name)

            with sf.SoundFile(str(f)) as snd:
                head = snd.read(min(probe_frames, len(snd)), dtype="float32", always_2d=True)
                if len(snd) > probe_frames:
                    snd.seek(max(0, len(snd) - probe_frames))
                    tail = snd.read(probe_frames, dtype="float32", always_2d=True)
                else:
                    tail = head
            block = np.vstack([head, tail]) if head.size and tail.size else head
            if block.size:
                peak = float(np.max(np.abs(block)))
                peak_min = min(peak_min, peak)
                peak_max = max(peak_max, peak)
                if np.isnan(block).any():
                    nan_files.append(f.name)
                    failed_files.add(f.name)
        except Exception as exc:
            read_errors.append((f.name, str(exc)))
            failed_files.add(f.name)

    deep_check_names = {flacs[0].name, flacs[len(flacs) // 2].name, flacs[-1].name}
    for f in flacs:
        if f.name not in deep_check_names:
            continue
        try:
            data, _ = sf.read(str(f), dtype="float32", always_2d=True)
            if data.size and np.isnan(data).any():
                nan_files.append(f.name)
            if data.size:
                peak = float(np.max(np.abs(data)))
                peak_min = min(peak_min, peak)
                peak_max = max(peak_max, peak)
        except Exception as exc:
            read_errors.append((f.name, f"deep read: {exc}"))
            failed_files.add(f.name)
    r.add(f"  Deep full-file scan: {sorted(deep_check_names)}")

    for name, meta in file_meta.items():
        if name in failed_files:
            continue
        dur = meta["duration"]
        if dur <= 3500 or dur >= 3650:
            failed_files.add(name)

    n_failed = len(failed_files)
    n_ok = n_analysed - n_failed

    r.pass_fail(not read_errors, f"All {n_analysed} files opened successfully", f"{len(read_errors)} read error(s)")
    if read_errors:
        for name, err in read_errors[:5]:
            r.add(f"         {name}: {err}")

    r.add(f"  Sample rate:      {sorted(sr_set)}")
    r.pass_fail(sr_set == {EXPECTED_FS}, f"All files are {EXPECTED_FS} Hz", f"Unexpected sample rates: {sr_set}")

    if durations:
        r.add(f"  Duration (s):     min={min(durations):.3f}  median={statistics.median(durations):.3f}  max={max(durations):.3f}")
        r.add("  Note: UK-DALE hourly FLACs are ~3594–3598 s, not exactly 3600 s (normal).")
        r.pass_fail(
            min(durations) > 3500 and max(durations) < 3650,
            "Durations are in the expected ~1-hour range",
            "Some durations look abnormal (possible truncation or corruption)",
        )

    r.add("")
    r.add("  Duration cluster summary (by frame count):")
    for line in format_duration_cluster_table(frame_groups):
        r.add(line)
    r.add(f"  Cluster total:    {sum(frame_groups.values())} file(s) in {len(frame_groups)} duration group(s)")

    if peak_max > 0:
        r.add(f"  Signal peak:      {peak_min:.4f} – {peak_max:.4f}")
    r.pass_fail(not nan_files, "No NaN samples in waveforms", f"NaN in: {nan_files[:5]}")

    r.section("4b. Files analysed — summary")
    r.add(f"  Total FLAC files analysed:  {n_analysed}")
    r.add(f"  Passed all audio checks:      {n_ok}")
    r.add(f"  Failed / flagged:             {n_failed}")
    if failed_files:
        r.add("  Failed files:")
        for name in sorted(failed_files)[:10]:
            r.add(f"    - {name}")
        if len(failed_files) > 10:
            r.add(f"    ... and {len(failed_files) - 10} more")
    r.pass_fail(
        n_failed == 0 and n_ok == n_analysed,
        f"All {n_analysed} files passed analysis",
        f"{n_failed} of {n_analysed} file(s) failed or were flagged",
    )

    r.section("5. CEDA remote comparison (optional)")
    remote = None
    if check_remote:
        url = ceda_listing_url(week_dir)
        if url:
            r.add(f"  JSON URL: {url}")
            remote = fetch_remote_flacs(url)
        else:
            r.add("  Skipped: could not build CEDA URL from folder path.")

    if remote is None:
        r.add("  Remote check: not available (offline, requests missing, or CEDA error).")
    else:
        local_names = {f.name for f in flacs}
        remote_names = set(remote)
        missing = sorted(remote_names - local_names)
        extra = sorted(local_names - remote_names)
        r.add(f"  Remote FLAC count: {len(remote)}")
        r.pass_fail(not missing, "All remote files present locally", f"Missing locally: {len(missing)}")
        if missing:
            for n in missing[:5]:
                r.add(f"         - {n}")
        r.pass_fail(not extra, "No extra local files vs remote", f"Extra locally: {len(extra)}")
        if extra:
            for n in extra[:5]:
                r.add(f"         + {n}")

        size_mismatch = []
        for f in flacs:
            if f.name in remote and f.stat().st_size != remote[f.name]:
                size_mismatch.append((f.name, f.stat().st_size, remote[f.name]))
        r.pass_fail(
            not size_mismatch,
            "All local file sizes match CEDA",
            f"{len(size_mismatch)} file(s) differ in size from server",
        )
        if size_mismatch:
            for n, loc, rem in size_mismatch[:5]:
                r.add(f"         {n}: local={loc} remote={rem}")

    r.section("6. Verdict")
    r.add(f"  Analysed: {n_analysed} FLAC file(s)  |  OK: {n_ok}  |  Failed: {n_failed}")
    if r.ok:
        r.add("  WEEK FOLDER LOOKS COMPLETE AND READY FOR PREPROCESSING.")
        r.add(f"  - {n_analysed} hourly files with continuous 3600 s start gaps.")
        r.add("  - ~1 hour of 16 kHz stereo audio per file (~599 six-second windows).")
    else:
        r.add("  ISSUES FOUND — review [FAIL] lines above before batch extraction.")

    return r


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate UK-DALE 16 kHz week FLAC folder")
    parser.add_argument("--path", type=str, default=None, help="Week folder (skip interactive prompt)")
    parser.add_argument("--no-remote", action="store_true", help="Skip CEDA size/count check")
    parser.add_argument("--save", action="store_true", help="Write validation_report.txt into the week folder")
    args = parser.parse_args()

    week_dir = Path(args.path).resolve() if args.path else prompt_week_path()

    if not week_dir.exists():
        print(f"\nERROR: Path not found: {week_dir}")
        sys.exit(1)

    report = validate_week(week_dir, check_remote=not args.no_remote)
    text = report.dump()
    print("\n" + text)

    if args.save:
        out_path = week_dir / "validation_report.txt"
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"\nReport saved to: {out_path}")

    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()

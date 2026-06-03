"""
UK-DALE 16kHz FLAC Downloader
==============================
Single-file downloader. Everything is self-contained — no external helper
files needed.

What it does automatically
---------------------------
  1. Installs missing Python packages (soundfile)
  2. Installs wget if not found (scoop → choco → winget → direct .exe download)
  3. Fetches a CEDA Bearer token using username/password, caches it for 71h
  4. Runs wget to mirror the full week folder from CEDA
  5. Validates every downloaded FLAC (duration check)
  6. Re-runs wget if any files are corrupt/incomplete

Usage
-----
    python download_ukdale_flac.py                        # downloads DEFAULT_WEEKS
    python download_ukdale_flac.py --weeks 31
    python download_ukdale_flac.py --weeks 30,31,32
    python download_ukdale_flac.py --weeks 31 --check_only   # validate only
    python download_ukdale_flac.py --weeks 31 --no_validate  # skip validation
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure virtualenv's bin/Scripts directory is in PATH so tools like wget.exe are found
_py_dir = os.path.dirname(sys.executable)
if _py_dir and _py_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _py_dir + os.pathsep + os.environ.get("PATH", "")

# ── auto-install Python packages ──────────────────────────────────────────────
for _pkg in ("soundfile",):
    try:
        __import__(_pkg)
    except ImportError:
        print(f"[deps] Installing {_pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg])

import soundfile as sf  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — edit these to change defaults
# ─────────────────────────────────────────────────────────────────────────────
CEDA_USERNAME = "rtie2002"
CEDA_PASSWORD = "RtiE2002"

DEFAULT_HOUSE = "2"
DEFAULT_YEAR = "2013"
DEFAULT_WEEKS = ["28", "29"]

EXPECTED_DURATION_SEC = 3600  # each UK-DALE FLAC is ~1 hour
DURATION_TOLERANCE = 5  # seconds
MIN_EXPECTED_SIZE_MB = 50
SIZE_TOLERANCE_BYTES = 1024 * 1024
MAX_DOWNLOAD_ROUNDS = 0  # 0 means unlimited outer retry rounds

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_SAVE = SCRIPT_DIR  # files land in dataset_preprocess/UK_DALE_16khz/

CEDA_BASE = (
    "https://data.ceda.ac.uk/edc/d1/887733b3-4c04-471f-9404-9f7459c4a1a0/data/version_0"
)

TOKEN_URL = "https://services.ceda.ac.uk/api/token/create/"
TOKEN_CACHE_FILE = os.path.join(SCRIPT_DIR, ".ceda_token_cache")
TOKEN_LIFETIME_SEC = 3 * 24 * 3600 - 3600  # 71 h  (tokens last 3 days)
WGET_EXE = "wget"


# 
# SECTION 1 — CEDA TOKEN
# 


def _fetch_token_from_api(username: str, password: str) -> Optional[str]:
    """POST credentials to CEDA token API, return token string or None."""
    import base64
    import urllib.request

    auth_str = f"{username}:{password}"
    auth_b64 = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")

    req = urllib.request.Request(
        TOKEN_URL,
        method="POST",
        headers={
            "Authorization": f"Basic {auth_b64}",
            "User-Agent": "NILM-downloader/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8").strip()
    except Exception as e:
        print(f"[auth]   Token request failed: {e}")
        return None

    # Response can be plain-text token or JSON {"token": "..."}
    if body.startswith("{"):
        try:
            data = json.loads(body)
            token = data.get("token") or data.get("access_token", "")
        except Exception:
            token = ""
    else:
        token = body

    if token and len(token) > 20:
        return token

    print(f"[auth] ⚠  Unexpected API response: {body[:200]}")
    return None


def _load_cached_token() -> Optional[str]:
    """Return cached token if still fresh, else None."""
    if not os.path.exists(TOKEN_CACHE_FILE):
        return None
    try:
        with open(TOKEN_CACHE_FILE, "r") as f:
            data = json.load(f)
        age = time.time() - data.get("fetched_at", 0)
        token = data.get("token", "")
        if age < TOKEN_LIFETIME_SEC and token:
            print(f"[auth] ✅  Using cached token  (age {age / 3600:.1f}h)")
            return token
    except Exception:
        pass
    return None


def _save_token_cache(token: str) -> None:
    try:
        with open(TOKEN_CACHE_FILE, "w") as f:
            json.dump({"token": token, "fetched_at": time.time()}, f)
    except Exception:
        pass


def get_token() -> Optional[str]:
    """
    Return a valid CEDA Bearer token.
    Uses cache if fresh; otherwise fetches a new one from the API.
    """
    token = _load_cached_token()
    if token:
        return token

    print(f"[auth] 🔑  Fetching token for '{CEDA_USERNAME}'...")
    token = _fetch_token_from_api(CEDA_USERNAME, CEDA_PASSWORD)
    if token:
        _save_token_cache(token)
        print("[auth] ✅  Token obtained and cached")
        return token

    print("[auth]   Could not obtain token — downloads may fail for restricted files")
    return None


# 
# SECTION 2 — WGET INSTALL
# 


def _tool_on_path(name: str) -> bool:
    try:
        r = subprocess.run([name, "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, PermissionError, OSError):
        return False


def _prepend_path_once(path: str) -> None:
    path = os.path.abspath(path)
    path_parts = [
        os.path.abspath(p)
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p
    ]
    if path not in path_parts:
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")


def _wget_version_line(wget_cmd: str) -> Optional[str]:
    try:
        r = subprocess.run(
            [wget_cmd, "--version"], capture_output=True, text=True, timeout=5
        )
    except (FileNotFoundError, PermissionError, OSError):
        return None
    if r.returncode != 0:
        return None
    lines = (r.stdout or r.stderr).splitlines()
    return lines[0] if lines else wget_cmd


def _use_wget(wget_cmd: str, source: str) -> bool:
    global WGET_EXE

    version = _wget_version_line(wget_cmd)
    if not version:
        return False

    WGET_EXE = wget_cmd
    wget_dir = os.path.dirname(os.path.abspath(wget_cmd)) if os.path.dirname(wget_cmd) else ""
    if wget_dir:
        _prepend_path_once(wget_dir)
    print(f"[wget] ✅  {version}  ({source})")
    return True


def _local_wget_candidates() -> List[str]:
    candidates = [
        os.path.join(SCRIPT_DIR, "wget.exe"),
        os.path.join(os.path.dirname(sys.executable), "wget.exe"),
        os.path.join(PROJECT_ROOT, ".venv", "Scripts", "wget.exe"),
        os.path.join(PROJECT_ROOT, ".venv", "bin", "wget"),
    ]
    seen = set()
    unique = []
    for path in candidates:
        norm = os.path.abspath(path)
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)
    return unique


def _find_local_wget() -> bool:
    for path in _local_wget_candidates():
        if os.path.isfile(path) and _use_wget(path, f"local: {path}"):
            return True
    return False


def _install_wget_windows() -> bool:
    """Try scoop → choco → winget → direct .exe download."""
    print("[wget] Attempting automatic installation...")

    if _find_local_wget():
        return True

    # 1. scoop
    if _tool_on_path("scoop"):
        r = subprocess.run(["scoop", "install", "wget"], timeout=120)
        if r.returncode == 0 and _use_wget("wget", "scoop"):
            return True

    # 2. chocolatey
    if _tool_on_path("choco"):
        r = subprocess.run(["choco", "install", "-y", "wget"], timeout=120)
        if r.returncode == 0 and _use_wget("wget", "chocolatey"):
            return True

    # 3. winget
    if _tool_on_path("winget"):
        r = subprocess.run(
            [
                "winget",
                "install",
                "--id",
                "GnuWin32.Wget",
                "--silent",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ],
            timeout=180,
        )
        if r.returncode == 0:
            gnuwin = r"C:\Program Files (x86)\GnuWin32\bin"
            if os.path.isfile(os.path.join(gnuwin, "wget.exe")):
                _prepend_path_once(gnuwin)
            if _use_wget("wget", "winget"):
                return True

    # 4. direct binary download (last resort)
    try:
        import urllib.request

        print("  → downloading wget.exe directly from eternallybored.org ...")
        wget_url = "https://eternallybored.org/misc/wget/1.21.4/64/wget.exe"
        candidates = [
            SCRIPT_DIR,
            os.path.join(PROJECT_ROOT, ".venv", "Scripts"),
            os.path.join(PROJECT_ROOT, ".venv", "bin"),
        ]
        for dest_dir in candidates:
            if not os.path.isdir(dest_dir):
                continue
            dest = os.path.join(dest_dir, "wget.exe")
            try:
                urllib.request.urlretrieve(wget_url, dest)
            except PermissionError:
                print(f"  cannot write to {dest_dir}; trying another folder")
                continue
            if _use_wget(dest, f"downloaded: {dest}"):
                print(f"  ✅ wget.exe saved to {dest}")
                return True
    except Exception as e:
        print(f"  direct download failed: {e}")

    return False


def ensure_wget() -> bool:
    """Make sure wget is on PATH, installing it if necessary."""
    if _find_local_wget():
        return True

    if _use_wget("wget", "PATH"):
        return True

    print("[wget] ⚠  wget not found — attempting auto-install...")

    if os.name == "nt":
        success = _install_wget_windows()
    else:
        success = False
        for mgr in ("apt-get", "apt", "brew"):
            if _tool_on_path(mgr):
                try:
                    r = subprocess.run(
                        ["sudo", mgr, "install", "-y", "wget"], timeout=120
                    )
                    if r.returncode == 0 and _use_wget("wget", mgr):
                        success = True
                        break
                except Exception:
                    continue

    if success and _use_wget(WGET_EXE, "installed"):
        return True

    print("[wget]   Auto-install failed. Install manually:")
    print("    Windows : scoop install wget  |  choco install wget")
    print("    Linux   : sudo apt install wget")
    print("    macOS   : brew install wget")
    return False


# 
# SECTION 3 — FLAC VALIDATION
# 


def validate_flac(
    path: str,
    strict: bool = False,
    expected_size: Optional[int] = None,
) -> Tuple[bool, str]:
    try:
        local_size = os.path.getsize(path)
        size_mb = local_size / 1024 / 1024
        if expected_size is not None:
            remote_mb = expected_size / 1024 / 1024
            if local_size + SIZE_TOLERANCE_BYTES < expected_size:
                return False, f"incomplete size {size_mb:.1f} MB (server {remote_mb:.1f} MB)"
            if local_size > expected_size + SIZE_TOLERANCE_BYTES:
                return False, f"larger than server {size_mb:.1f} MB (server {remote_mb:.1f} MB)"
        if size_mb < MIN_EXPECTED_SIZE_MB:
            return False, f"suspiciously small {size_mb:.1f} MB"

        with sf.SoundFile(path) as f:
            expected_frames = f.frames
            samplerate = f.samplerate
            duration = expected_frames / samplerate

            if strict:
                decoded_frames = 0
                for block in f.blocks(blocksize=262144, dtype="float32"):
                    decoded_frames += len(block)
                if decoded_frames != expected_frames:
                    return (
                        False,
                        f"incomplete decode {decoded_frames}/{expected_frames} frames",
                    )
        if abs(duration - EXPECTED_DURATION_SEC) > DURATION_TOLERANCE:
            return (
                False,
                f"bad duration {duration:.1f}s (expected ~{EXPECTED_DURATION_SEC}s)",
            )
        mode = "strict" if strict else "fast"
        return True, f"{duration:.1f}s OK ({mode})"
    except Exception as e:
        return False, str(e)


def validate_week(
    week_dir: str,
    skip_files: Optional[set] = None,
    strict: bool = False,
    remote_meta: Optional[Dict[str, dict]] = None,
) -> dict:
    """Validate all FLAC files in a week directory. Returns summary dict."""
    skip_files = skip_files or set()
    flac_files = sorted(
        os.path.join(week_dir, f)
        for f in os.listdir(week_dir)
        if f.endswith(".flac") and f not in skip_files
    )
    if not flac_files:
        print(f"  [validate] No FLAC files found in {week_dir}")
        return {"total": 0, "ok": 0, "bad": []}

    ok_count = 0
    bad_files = []
    bad_details = []
    for i, path in enumerate(flac_files, 1):
        name = os.path.basename(path)
        expected_size = (remote_meta or {}).get(name, {}).get("size")
        valid, msg = validate_flac(path, strict=strict, expected_size=expected_size)
        tag = "✅" if valid else ""
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  [{i:03d}/{len(flac_files)}] {tag}  {name}  {msg}  ({size_mb:.1f} MB)")
        if valid:
            ok_count += 1
        else:
            bad_files.append(name)
            bad_details.append({"file": name, "reason": msg})

    print(
        f"\n  [validate] {ok_count}/{len(flac_files)} OK"
        + (f"  |  BAD: {bad_files}" if bad_files else "")
    )
    return {
        "total": len(flac_files),
        "ok": ok_count,
        "bad": bad_files,
        "bad_details": bad_details,
    }


# 
# SECTION 4 — DOWNLOAD
# 


def _list_flac_files(base_url: str, token: Optional[str]) -> List[str]:
    """
    Query the CEDA JSON directory index and return a sorted list of .flac filenames.
    The CEDA UI is JS-rendered, so wget -r never finds .flac href links in the HTML.
    We must use the '?json' endpoint to enumerate files.
    """
    import urllib.request as _urlreq

    json_url = base_url.rstrip("/") + "/?json"
    headers = {"User-Agent": "NILM-downloader/1.0"}
    if token:
        token_clean = re.sub(r"[^A-Za-z0-9\-_=.+/]", "", token)
        headers["Authorization"] = f"Bearer {token_clean}"

    req = _urlreq.Request(json_url, headers=headers)
    try:
        with _urlreq.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [list]   Could not fetch directory listing: {e}")
        return []

    items = data.get("items", [])
    flac_names = sorted(
        item["name"] for item in items
        if item.get("type") == "file" and item.get("name", "").endswith(".flac")
    )
    print(f"  [list] 📋  Found {len(flac_names)} FLAC files on server")
    return flac_names


def _metadata_size_bytes(item: dict) -> Optional[int]:
    for key in ("size", "bytes", "length", "content_length", "contentLength"):
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _list_flac_metadata(base_url: str, token: Optional[str]) -> Dict[str, dict]:
    """Return server-side FLAC metadata by filename from the CEDA JSON endpoint."""
    import urllib.request as _urlreq

    json_url = base_url.rstrip("/") + "/?json"
    headers = {"User-Agent": "NILM-downloader/1.0"}
    if token:
        token_clean = re.sub(r"[^A-Za-z0-9\-_=.+/]", "", token)
        headers["Authorization"] = f"Bearer {token_clean}"

    req = _urlreq.Request(json_url, headers=headers)
    try:
        with _urlreq.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [list]   Could not fetch directory listing: {e}")
        return {}

    flac_meta = {}
    for item in data.get("items", []):
        name = item.get("name", "")
        if item.get("type") == "file" and name.endswith(".flac"):
            flac_meta[name] = {"size": _metadata_size_bytes(item)}
    print(f"  [list] Found {len(flac_meta)} FLAC files on server")
    return flac_meta


def _build_wget_single(file_url: str, save_dir: str, token: Optional[str]) -> List[str]:
    """Build a wget command to download one file directly."""
    cmd = [
        WGET_EXE,
        "--no-verbose",
        "--show-progress",
        "--timeout=60",
        "--tries=5",
        "--waitretry=5",
        "-c",           # resume partial downloads
        f"-P{save_dir}",
        file_url,
    ]
    if token:
        token_clean = re.sub(r"[^A-Za-z0-9\-_=.+/]", "", token)
        cmd.insert(1, f"--header=Authorization: Bearer {token_clean}")
    return cmd


def _quarantine_bad_file(path: str) -> Optional[str]:
    """Move a failed local FLAC aside before clean re-download."""
    if not os.path.exists(path):
        return None

    bad_dir = os.path.join(os.path.dirname(path), "_bad_flac_retry")
    os.makedirs(bad_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    quarantined = os.path.join(bad_dir, f"{os.path.basename(path)}.{stamp}.bad")
    shutil.move(path, quarantined)
    return quarantined


def _bad_backup_paths(week_dir: str, name: str) -> set:
    bad_dir = os.path.join(week_dir, "_bad_flac_retry")
    if not os.path.isdir(bad_dir):
        return set()
    prefix = f"{name}."
    return {
        os.path.join(bad_dir, f)
        for f in os.listdir(bad_dir)
        if f.startswith(prefix) and f.endswith(".bad")
    }


def _download_single_flac(
    base_url: str,
    name: str,
    week_dir: str,
    token: Optional[str],
    clean: bool = False,
) -> int:
    """Download one FLAC. If clean=True, move existing local file aside first."""
    dest = os.path.join(week_dir, name)
    if clean:
        quarantined = _quarantine_bad_file(dest)
        if quarantined:
            print(f"    moved bad local file aside: {quarantined}")

    file_url = f"{base_url}/{name}"
    cmd = _build_wget_single(file_url, week_dir, token)
    result = subprocess.run(cmd)
    return result.returncode


def _retry_bad_files_clean(
    bad_details: list[dict],
    base_url: str,
    week_dir: str,
    token: Optional[str],
    strict: bool = False,
) -> list[dict]:
    """Force clean re-download for validation failures, then revalidate each file."""
    still_bad = []
    if not bad_details:
        return still_bad

    print(f"\n  [repair] {len(bad_details)} bad files found; forcing clean re-download ...")
    for rec in bad_details:
        name = rec["file"]
        print(f"  [repair] {name}")
        print(f"    original validation error: {rec['reason']}")
        old_backups = _bad_backup_paths(week_dir, name)
        rc = _download_single_flac(base_url, name, week_dir, token, clean=True)
        new_backups = _bad_backup_paths(week_dir, name) - old_backups
        if rc not in (0, 8):
            still_bad.append({
                "file": name,
                "reason": f"wget exit code {rc} during clean retry",
            })
            continue

        valid, msg = validate_flac(os.path.join(week_dir, name), strict=strict)
        if valid:
            print(f"    repaired OK: {msg}")
            for backup in new_backups:
                try:
                    os.remove(backup)
                    print(f"    deleted old .bad backup: {backup}")
                except OSError as e:
                    print(f"    warning: could not delete .bad backup {backup}: {e}")
        else:
            print(f"    still bad after clean re-download: {msg}")
            still_bad.append({"file": name, "reason": msg})

    return still_bad


def _missing_or_empty_files(expected_names: List[str], week_dir: str) -> List[str]:
    """Return expected server files that are absent locally or have zero bytes."""
    missing = []
    for name in expected_names:
        path = os.path.join(week_dir, name)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            missing.append(name)
    return missing


def _present_expected_files(expected_names: List[str], week_dir: str) -> List[str]:
    return [
        name for name in expected_names
        if os.path.exists(os.path.join(week_dir, name))
        and os.path.getsize(os.path.join(week_dir, name)) > 0
    ]


def _download_file_list(
    names: List[str],
    base_url: str,
    week_dir: str,
    token: Optional[str],
    clean: bool = False,
) -> List[str]:
    """Download a list of files and return names where wget reported failure."""
    failed = []
    if not names:
        return failed

    for i, name in enumerate(names, 1):
        dest = os.path.join(week_dir, name)
        already_present = os.path.exists(dest) and os.path.getsize(dest) > 0
        tag = "(resume)" if already_present and not clean else ""
        print(f"  [{i:03d}/{len(names)}] {name} {tag}")
        rc = _download_single_flac(base_url, name, week_dir, token, clean=clean)
        if rc not in (0, 8):
            print(f"    [warning] wget exit code {rc}")
            failed.append(name)
    return failed


def download_week(
    house: str,
    year: str,
    week: str,
    save_dir: str,
    token: Optional[str],
    check_after: bool = True,
    max_rounds: int = MAX_DOWNLOAD_ROUNDS,
    strict_validate: bool = False,
) -> bool:
    """Download one week of FLAC files. Returns True only if all server files validate."""
    week_str = f"wk{str(week).zfill(2)}"
    base_url = f"{CEDA_BASE}/house_{house}/{year}/{week_str}"
    week_dir = os.path.join(save_dir, f"house_{house}", year, week_str)
    os.makedirs(week_dir, exist_ok=True)

    print()
    print("=" * 65)
    print(f"  DOWNLOADING  house={house}  year={year}  week={week_str}")
    print("=" * 65)
    print(f"  Base URL : {base_url}/")
    print(f"  Save dir : {week_dir}")
    print()

    # Step 1: enumerate filenames and metadata via JSON API
    remote_meta = _list_flac_metadata(base_url, token)
    flac_names = sorted(remote_meta.keys())
    if not flac_names:
        print("  [error] No FLAC files found on server - check URL or token.")
        return False

    t0 = time.time()
    round_idx = 0
    source_bad_names = set()

    while True:
        round_idx += 1
        round_label = f"{round_idx}/{max_rounds}" if max_rounds > 0 else f"{round_idx}/unlimited"
        print()
        print(f"  [round {round_label}] Checking local files against server list ...")

        missing = _missing_or_empty_files(flac_names, week_dir)
        present = _present_expected_files(flac_names, week_dir)
        print(
            f"  [pre-validate] Server files: {len(flac_names)} | "
            f"local present: {len(present)} | missing/empty: {len(missing)}"
        )

        if missing:
            print(f"  [dl] Downloading {len(missing)} missing/empty files ...")
            failed = _download_file_list(missing, base_url, week_dir, token)
            if not check_after:
                return len(failed) == 0

            missing = _missing_or_empty_files(flac_names, week_dir)
            present = _present_expected_files(flac_names, week_dir)
            print(
                f"  [pre-validate] Server files: {len(flac_names)} | "
                f"local present: {len(present)} | missing/empty: {len(missing)}"
            )
            if missing:
                print("  [pre-validate] Still missing locally; validation skipped.")
                for name in missing[:20]:
                    print(f"    - {name}")
                if len(missing) > 20:
                    print(f"    ... plus {len(missing) - 20} more")
                if max_rounds > 0 and round_idx >= max_rounds:
                    break
                continue
        else:
            print("  [dl] All server-listed files already exist locally.")

        if not check_after:
            return True

        mode = "strict full-decode" if strict_validate else "fast header-duration"
        print(f"\n  [validate] Checking {week_dir} ({mode}) ...")
        summary = validate_week(
            week_dir,
            skip_files=source_bad_names,
            strict=strict_validate,
            remote_meta=remote_meta,
        )
        if not summary["bad"]:
            elapsed = time.time() - t0
            if source_bad_names:
                print(
                    f"\n  Download finished in {elapsed / 60:.1f} min, "
                    f"but {len(source_bad_names)} source-side file issue(s) were skipped:"
                )
                for name in sorted(source_bad_names):
                    print(f"    {name}")
                return False
            print(f"\n  Download + validation succeeded in {elapsed / 60:.1f} min")
            return True

        retry_bad = []
        for rec in summary.get("bad_details", []):
            name = rec["file"]
            local_path = os.path.join(week_dir, name)
            local_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
            server_size = remote_meta.get(name, {}).get("size")
            if server_size is not None and abs(local_size - server_size) <= SIZE_TOLERANCE_BYTES:
                source_bad_names.add(name)
                print(
                    f"  [source-check] {name}: {rec['reason']} "
                    f"(local size matches server; skip as source issue)"
                )
            else:
                retry_bad.append(rec)

        if retry_bad:
            _retry_bad_files_clean(
                retry_bad, base_url, week_dir, token, strict=strict_validate
            )

        if not retry_bad and source_bad_names:
            elapsed = time.time() - t0
            print(
                f"\n  Download + validation finished in {elapsed / 60:.1f} min "
                f"({len(source_bad_names)} source-side file issue(s) skipped)"
            )
            return False

        if max_rounds > 0 and round_idx >= max_rounds:
            break

    elapsed = time.time() - t0
    print(f"\n  Download stopped after {round_idx} round(s) in {elapsed / 60:.1f} min")

    missing = _missing_or_empty_files(flac_names, week_dir)
    if missing:
        print("\n  [source-check] Files listed by server but still missing locally:")
        for name in missing:
            print(f"    {name}")

    if source_bad_names:
        print("\n  [source-check] Source-side file issue(s) skipped:")
        for name in sorted(source_bad_names):
            print(f"    {name}")

    return False

# SECTION 5 — MAIN
# 


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Download UK-DALE 16kHz FLAC files using wget"
    )
    parser.add_argument("--house", default=DEFAULT_HOUSE)
    parser.add_argument("--year", default=DEFAULT_YEAR)
    parser.add_argument(
        "--weeks",
        default=",".join(DEFAULT_WEEKS),
        help="Comma-separated week numbers, e.g. 30,31,32",
    )
    parser.add_argument(
        "--save_dir",
        default=DEFAULT_SAVE,
        help="Root directory to save downloaded files",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Skip download, only validate existing FLAC files",
    )
    parser.add_argument(
        "--no_validate", action="store_true", help="Skip post-download FLAC validation"
    )
    parser.add_argument(
        "--strict_validate",
        action="store_true",
        help="Decode the full FLAC during validation. Slower, but catches block-level corruption.",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=MAX_DOWNLOAD_ROUNDS,
        help=(
            "Maximum outer retry rounds for server-listed files that are missing, "
            "empty, or fail FLAC validation. Use 0 for unlimited retry."
        ),
    )
    return parser.parse_args()


def main():
    args = get_arguments()
    weeks = [w.strip() for w in args.weeks.split(",") if w.strip()]

    print()
    print("=" * 65)
    print("  UK-DALE 16kHz FLAC DOWNLOADER")
    print("=" * 65)
    print(f"  House    : {args.house}")
    print(f"  Year     : {args.year}")
    print(f"  Weeks    : {weeks}")
    print(f"  Save dir : {args.save_dir}")
    print("=" * 65)

    # check_only — just validate, no download
    if args.check_only:
        for week in weeks:
            week_str = f"wk{str(week).zfill(2)}"
            week_dir = os.path.join(
                args.save_dir, f"house_{args.house}", args.year, week_str
            )
            print(f"\n[check] {week_str}  →  {week_dir}")
            if not os.path.isdir(week_dir):
                print("  Directory not found — skipping")
                continue
            validate_week(week_dir, strict=args.strict_validate)
        return

    # ensure wget is installed
    if not ensure_wget():
        sys.exit(1)

    # get CEDA token
    token = get_token()
    if not token:
        print("\n[auth] ⚠  Proceeding without token (only public files will download)")

    # download each week
    for week in weeks:
        ok = download_week(
            house=args.house,
            year=args.year,
            week=week,
            save_dir=args.save_dir,
            token=token,
            check_after=not args.no_validate,
            max_rounds=args.max_rounds,
            strict_validate=args.strict_validate,
        )
        status = "✅  all files validated" if ok else "⚠  week not fully validated"
        print(f"\n  Week {week}: {status}")

    print()
    print("=" * 65)
    print("  DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()

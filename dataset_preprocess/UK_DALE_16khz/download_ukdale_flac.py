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
import subprocess
import sys
import time
from typing import List, Optional, Tuple

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
DEFAULT_WEEKS = ["32"]

EXPECTED_DURATION_SEC = 3600  # each UK-DALE FLAC is ~1 hour
DURATION_TOLERANCE = 5  # seconds

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


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CEDA TOKEN
# ═════════════════════════════════════════════════════════════════════════════


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
        print(f"[auth] ❌  Token request failed: {e}")
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

    print(f"[auth] ⚠️  Unexpected API response: {body[:200]}")
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

    print("[auth] ❌  Could not obtain token — downloads may fail for restricted files")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — WGET INSTALL
# ═════════════════════════════════════════════════════════════════════════════


def _tool_on_path(name: str) -> bool:
    try:
        r = subprocess.run([name, "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def _install_wget_windows() -> bool:
    """Try scoop → choco → winget → direct .exe download."""
    print("[wget] Attempting automatic installation...")

    # 1. scoop
    if _tool_on_path("scoop"):
        r = subprocess.run(["scoop", "install", "wget"], timeout=120)
        if r.returncode == 0 and _tool_on_path("wget"):
            return True

    # 2. chocolatey
    if _tool_on_path("choco"):
        r = subprocess.run(["choco", "install", "-y", "wget"], timeout=120)
        if r.returncode == 0 and _tool_on_path("wget"):
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
                os.environ["PATH"] = gnuwin + os.pathsep + os.environ["PATH"]
            if _tool_on_path("wget"):
                return True

    # 4. direct binary download (last resort)
    try:
        import urllib.request

        print("  → downloading wget.exe directly from eternallybored.org ...")
        wget_url = "https://eternallybored.org/misc/wget/1.21.4/64/wget.exe"
        candidates = [
            os.path.join(PROJECT_ROOT, ".venv", "Scripts"),
            os.path.join(PROJECT_ROOT, ".venv", "bin"),
            SCRIPT_DIR,
        ]
        dest_dir = next((d for d in candidates if os.path.isdir(d)), SCRIPT_DIR)
        dest = os.path.join(dest_dir, "wget.exe")
        urllib.request.urlretrieve(wget_url, dest)
        os.environ["PATH"] = dest_dir + os.pathsep + os.environ["PATH"]
        if _tool_on_path("wget"):
            print(f"  ✅ wget.exe saved to {dest}")
            return True
    except Exception as e:
        print(f"  direct download failed: {e}")

    return False


def ensure_wget() -> bool:
    """Make sure wget is on PATH, installing it if necessary."""
    if _tool_on_path("wget"):
        r = subprocess.run(
            ["wget", "--version"], capture_output=True, text=True, timeout=5
        )
        print(f"[wget] ✅  {r.stdout.splitlines()[0]}")
        return True

    print("[wget] ⚠️  wget not found — attempting auto-install...")

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
                    if r.returncode == 0 and _tool_on_path("wget"):
                        success = True
                        break
                except Exception:
                    continue

    if success and _tool_on_path("wget"):
        r = subprocess.run(
            ["wget", "--version"], capture_output=True, text=True, timeout=5
        )
        print(f"[wget] ✅  Installed: {r.stdout.splitlines()[0]}")
        return True

    print("[wget] ❌  Auto-install failed. Install manually:")
    print("    Windows : scoop install wget  |  choco install wget")
    print("    Linux   : sudo apt install wget")
    print("    macOS   : brew install wget")
    return False


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FLAC VALIDATION
# ═════════════════════════════════════════════════════════════════════════════


def validate_flac(path: str) -> Tuple[bool, str]:
    try:
        with sf.SoundFile(path) as f:
            expected_frames = f.frames
            samplerate = f.samplerate
            duration = expected_frames / samplerate

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
        return True, f"{duration:.1f}s OK"
    except Exception as e:
        return False, str(e)


def validate_week(week_dir: str) -> dict:
    """Validate all FLAC files in a week directory. Returns summary dict."""
    flac_files = sorted(
        os.path.join(week_dir, f) for f in os.listdir(week_dir) if f.endswith(".flac")
    )
    if not flac_files:
        print(f"  [validate] No FLAC files found in {week_dir}")
        return {"total": 0, "ok": 0, "bad": []}

    ok_count = 0
    bad_files = []
    for i, path in enumerate(flac_files, 1):
        valid, msg = validate_flac(path)
        tag = "✅" if valid else "❌"
        print(f"  [{i:03d}/{len(flac_files)}] {tag}  {os.path.basename(path)}  {msg}")
        if valid:
            ok_count += 1
        else:
            bad_files.append(os.path.basename(path))

    print(
        f"\n  [validate] {ok_count}/{len(flac_files)} OK"
        + (f"  |  BAD: {bad_files}" if bad_files else "")
    )
    return {"total": len(flac_files), "ok": ok_count, "bad": bad_files}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════


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
        print(f"  [list] ❌  Could not fetch directory listing: {e}")
        return []

    items = data.get("items", [])
    flac_names = sorted(
        item["name"] for item in items
        if item.get("type") == "file" and item.get("name", "").endswith(".flac")
    )
    print(f"  [list] 📋  Found {len(flac_names)} FLAC files on server")
    return flac_names


def _build_wget_single(file_url: str, save_dir: str, token: Optional[str]) -> List[str]:
    """Build a wget command to download one file directly."""
    cmd = [
        "wget",
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


def download_week(
    house: str,
    year: str,
    week: str,
    save_dir: str,
    token: Optional[str],
    check_after: bool = True,
) -> bool:
    """Download one week of FLAC files. Returns True if all files validated OK."""
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

    # Step 1: enumerate filenames via JSON API
    flac_names = _list_flac_files(base_url, token)
    if not flac_names:
        print("  ❌  No FLAC files found on server — check URL or token.")
        return False

    # Step 2: determine which files still need downloading
    existing = set(os.listdir(week_dir))
    to_download = []
    for name in flac_names:
        dest = os.path.join(week_dir, name)
        if name in existing and os.path.getsize(dest) > 0:
            pass  # already present — wget -c will skip/resume anyway
        to_download.append(name)

    print(f"  [dl] Downloading {len(to_download)} files ...")
    t0 = time.time()
    failed = []
    for i, name in enumerate(to_download, 1):
        file_url = f"{base_url}/{name}"
        dest = os.path.join(week_dir, name)
        already_done = name in existing and os.path.getsize(dest) > 0
        tag = "(resume)" if already_done else ""
        print(f"  [{i:03d}/{len(to_download)}] {name} {tag}")
        cmd = _build_wget_single(file_url, week_dir, token)
        result = subprocess.run(cmd)
        if result.returncode not in (0, 8):
            print(f"    ⚠️  wget exit code {result.returncode}")
            failed.append(name)

    elapsed = time.time() - t0
    print(f"\n  Download finished in {elapsed / 60:.1f} min  ({len(failed)} failures)")

    if not check_after or not os.path.isdir(week_dir):
        return len(failed) == 0

    print(f"\n  [validate] Checking {week_dir} ...")
    summary = validate_week(week_dir)

    if summary["bad"]:
        print(f"\n  ⚠️  {len(summary['bad'])} bad files — retrying ...")
        for name in summary["bad"]:
            file_url = f"{base_url}/{name}"
            cmd = _build_wget_single(file_url, week_dir, token)
            subprocess.run(cmd)
        summary = validate_week(week_dir)

    return len(summary["bad"]) == 0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN
# ═════════════════════════════════════════════════════════════════════════════


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
            validate_week(week_dir)
        return

    # ensure wget is installed
    if not ensure_wget():
        sys.exit(1)

    # get CEDA token
    token = get_token()
    if not token:
        print("\n[auth] ⚠️  Proceeding without token (only public files will download)")

    # download each week
    for week in weeks:
        ok = download_week(
            house=args.house,
            year=args.year,
            week=week,
            save_dir=args.save_dir,
            token=token,
            check_after=not args.no_validate,
        )
        status = (
            "✅  all files validated" if ok else "⚠️  some files may need re-downloading"
        )
        print(f"\n  Week {week}: {status}")

    print()
    print("=" * 65)
    print("  DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()

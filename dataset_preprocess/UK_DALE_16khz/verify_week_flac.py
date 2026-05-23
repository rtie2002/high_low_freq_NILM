"""
UK-DALE 16kHz FLAC Verifier
============================
Standalone script to verify the integrity and duration of downloaded
UK-DALE 16kHz FLAC files without re-downloading anything.

What it does
------------
  1. Scans the target week directory for .flac files
  2. Checks each file is readable by soundfile
  3. Validates that each file is approximately 1 hour long (3600 s ±5 s)
  4. Optionally cross-checks against the CEDA JSON index to detect
     missing files (requires CEDA Bearer token)
  5. Prints a clear summary and exits with code 0 (all OK) or 1 (issues found)

Usage
-----
    python verify_week_flac.py                          # prompt for week(s)
    python verify_week_flac.py --weeks 31
    python verify_week_flac.py --weeks 30,31,32
    python verify_week_flac.py --weeks 31 --house 2 --year 2013
    python verify_week_flac.py --weeks 31 --no_remote   # skip CEDA cross-check
    python verify_week_flac.py --weeks 31 --save_dir /path/to/data
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure virtualenv's bin/Scripts directory is on PATH
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
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CEDA_USERNAME = "rtie2002"
CEDA_PASSWORD = "RtiE2002"

DEFAULT_HOUSE = "2"
DEFAULT_YEAR = "2013"

EXPECTED_DURATION_SEC = 3600  # each UK-DALE FLAC is ~1 hour
DURATION_TOLERANCE = 5        # seconds

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE = SCRIPT_DIR

CEDA_BASE = (
    "https://data.ceda.ac.uk/edc/d1/887733b3-4c04-471f-9404-9f7459c4a1a0/data/version_0"
)
TOKEN_URL = "https://services.ceda.ac.uk/api/token/create/"
TOKEN_CACHE_FILE = os.path.join(SCRIPT_DIR, ".ceda_token_cache")
TOKEN_LIFETIME_SEC = 3 * 24 * 3600 - 3600  # 71 h


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CEDA TOKEN (for remote cross-check)
# ═════════════════════════════════════════════════════════════════════════════


def _fetch_token_from_api(username: str, password: str) -> str | None:
    """POST credentials to CEDA token API using Basic auth, return token or None."""
    import base64
    import urllib.request

    auth_b64 = base64.b64encode(f"{username}:{password}".encode()).decode()
    req = urllib.request.Request(
        TOKEN_URL,
        method="POST",
        headers={
            "Authorization": f"Basic {auth_b64}",
            "User-Agent": "NILM-verifier/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8").strip()
    except Exception as e:
        print(f"[auth] ❌  Token request failed: {e}")
        return None

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


def _load_cached_token() -> str | None:
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


def get_token() -> str | None:
    token = _load_cached_token()
    if token:
        return token
    print(f"[auth] 🔑  Fetching token for '{CEDA_USERNAME}'...")
    token = _fetch_token_from_api(CEDA_USERNAME, CEDA_PASSWORD)
    if token:
        _save_token_cache(token)
        print("[auth] ✅  Token obtained and cached")
        return token
    print("[auth] ❌  Could not obtain token — remote cross-check will be skipped")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — REMOTE FILE LISTING
# ═════════════════════════════════════════════════════════════════════════════


def list_remote_flac(house: str, year: str, week_str: str, token: str | None) -> list[str]:
    """Fetch FLAC filenames from the CEDA JSON directory index."""
    import urllib.request

    base_url = f"{CEDA_BASE}/house_{house}/{year}/{week_str}"
    json_url = f"{base_url}/?json"
    headers = {"User-Agent": "NILM-verifier/1.0"}
    if token:
        token_clean = re.sub(r"[^A-Za-z0-9\-_=.+/]", "", token)
        headers["Authorization"] = f"Bearer {token_clean}"

    req = urllib.request.Request(json_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [remote] ⚠️  Could not fetch remote listing: {e}")
        return []

    items = data.get("items", [])
    return sorted(
        item["name"] for item in items
        if item.get("type") == "file" and item.get("name", "").endswith(".flac")
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOCAL FLAC VALIDATION
# ═════════════════════════════════════════════════════════════════════════════


def validate_flac(path: str) -> tuple[bool, str]:
    """Check that a FLAC file is readable and has the expected duration."""
    try:
        info = sf.info(path)
        duration = info.frames / info.samplerate
        if abs(duration - EXPECTED_DURATION_SEC) > DURATION_TOLERANCE:
            return (
                False,
                f"bad duration {duration:.1f}s (expected ~{EXPECTED_DURATION_SEC}s)",
            )
        return True, f"{duration:.1f}s OK"
    except Exception as e:
        return False, f"unreadable: {e}"


def verify_week(
    house: str,
    year: str,
    week: str,
    save_dir: str,
    remote_check: bool = True,
    token: str | None = None,
) -> dict:
    """
    Verify all FLAC files in a week directory.

    Returns a dict with keys:
        total_local   — number of local .flac files found
        ok            — number that passed duration check
        bad           — list of bad/corrupt filenames
        missing       — list of filenames present on CEDA but not locally
        extra         — list of local files not found on CEDA
    """
    week_str = f"wk{str(week).zfill(2)}"
    week_dir = os.path.join(save_dir, f"house_{house}", year, week_str)

    print()
    print("=" * 65)
    print(f"  VERIFYING  house={house}  year={year}  week={week_str}")
    print("=" * 65)
    print(f"  Directory : {week_dir}")

    if not os.path.isdir(week_dir):
        print(f"  ❌  Directory not found: {week_dir}")
        return {"total_local": 0, "ok": 0, "bad": [], "missing": [], "extra": []}

    # ── local scan ──────────────────────────────────────────────────────────
    local_files = sorted(f for f in os.listdir(week_dir) if f.endswith(".flac"))
    print(f"\n  Local FLAC files found : {len(local_files)}")

    ok_count = 0
    bad_files = []
    for i, fname in enumerate(local_files, 1):
        path = os.path.join(week_dir, fname)
        valid, msg = validate_flac(path)
        tag = "✅" if valid else "❌"
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  [{i:03d}/{len(local_files)}] {tag}  {fname}  {msg}  ({size_mb:.1f} MB)")
        if valid:
            ok_count += 1
        else:
            bad_files.append(fname)

    print(
        f"\n  Local check : {ok_count}/{len(local_files)} OK"
        + (f"  |  BAD: {bad_files}" if bad_files else "")
    )

    # ── remote cross-check ──────────────────────────────────────────────────
    missing = []
    extra = []
    if remote_check:
        print("\n  [remote] Fetching CEDA directory listing ...")
        remote_files = list_remote_flac(house, year, week_str, token)
        if remote_files:
            remote_set = set(remote_files)
            local_set = set(local_files)
            missing = sorted(remote_set - local_set)
            extra = sorted(local_set - remote_set)
            print(f"  [remote] Server has {len(remote_files)} FLAC files")
            if missing:
                print(f"  [remote] ⚠️  {len(missing)} files MISSING locally:")
                for f in missing:
                    print(f"    - {f}")
            else:
                print("  [remote] ✅  All server files are present locally")
            if extra:
                print(f"  [remote] ℹ️  {len(extra)} extra local files not on server:")
                for f in extra:
                    print(f"    + {f}")
        else:
            print("  [remote] ⚠️  Could not retrieve remote listing — skipping cross-check")

    return {
        "total_local": len(local_files),
        "ok": ok_count,
        "bad": bad_files,
        "missing": missing,
        "extra": extra,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MAIN
# ═════════════════════════════════════════════════════════════════════════════


def parse_weeks(weeks_text: str) -> list[str]:
    """Parse week input such as '30', 'wk30', '30,31', or a path ending in wk30."""
    weeks = []
    for item in weeks_text.split(","):
        item = item.strip()
        if not item:
            continue
        week_name = os.path.basename(os.path.normpath(item))
        match = re.fullmatch(r"(?:wk)?(\d+)", week_name, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid week value: {item!r}")
        weeks.append(match.group(1))
    if not weeks:
        raise ValueError("No week value was entered")
    return weeks


def prompt_for_weeks() -> list[str]:
    while True:
        raw = input(
            "Enter week folder(s) to verify, e.g. 30, wk30, 30,31, or a path ending in wk30: "
        )
        try:
            return parse_weeks(raw)
        except ValueError as e:
            print(f"  {e}. Please try again.")


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Verify UK-DALE 16kHz FLAC files for one or more weeks"
    )
    parser.add_argument("--house", default=DEFAULT_HOUSE, help="House number (default: 2)")
    parser.add_argument("--year", default=DEFAULT_YEAR, help="Year (default: 2013)")
    parser.add_argument(
        "--weeks",
        default=None,
        help="Comma-separated week numbers or week folders, e.g. 30,31,32 or wk30",
    )
    parser.add_argument(
        "--save_dir",
        default=DEFAULT_SAVE,
        help="Root directory of downloaded files",
    )
    parser.add_argument(
        "--no_remote",
        action="store_true",
        help="Skip the CEDA remote cross-check (faster, no token needed)",
    )
    return parser.parse_args()


def main():
    args = get_arguments()
    try:
        weeks = parse_weeks(args.weeks) if args.weeks else prompt_for_weeks()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(2)

    print()
    print("=" * 65)
    print("  UK-DALE 16kHz FLAC VERIFIER")
    print("=" * 65)
    print(f"  House    : {args.house}")
    print(f"  Year     : {args.year}")
    print(f"  Weeks    : {weeks}")
    print(f"  Save dir : {args.save_dir}")
    print(f"  Remote   : {'disabled' if args.no_remote else 'enabled'}")
    print("=" * 65)

    token = None
    if not args.no_remote:
        token = get_token()
        if not token:
            print("\n[auth] ⚠️  Proceeding without token — remote cross-check disabled\n")

    all_ok = True
    for week in weeks:
        result = verify_week(
            house=args.house,
            year=args.year,
            week=week,
            save_dir=args.save_dir,
            remote_check=(not args.no_remote),
            token=token,
        )
        week_str = f"wk{str(week).zfill(2)}"
        issues = result["bad"] + result["missing"]
        if issues:
            all_ok = False
            print(f"\n  Week {week_str}: ⚠️  {len(issues)} issue(s) found")
        else:
            print(f"\n  Week {week_str}: ✅  All {result['ok']} files OK")

    print()
    print("=" * 65)
    print("  DONE")
    print("=" * 65)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

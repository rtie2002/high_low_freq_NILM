import os
import sys
import re
import time
import threading
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# AUTO INSTALL DEPENDENCIES
# =========================================================

def ensure_package(pkg):
    try:
        __import__(pkg)
    except ImportError:
        print(f"[Auto-Install] Missing {pkg}, installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg]
        )

ensure_package("requests")
ensure_package("soundfile")   # ✅ NEW
ensure_package("rich")

import requests
import soundfile as sf

from rich.live import Live
from rich.table import Table
from rich.console import Console

# =========================================================
# CONFIG
# =========================================================

HOUSE = "2"
YEAR = "2013"
WEEKS = ["31"]

MAX_WORKERS = 180

EXPECTED_DURATION_SEC = 3600   # ✅ 1-hour UK-DALE check
DURATION_TOLERANCE = 5        # seconds

# =========================================================

download_status = {}
status_lock = threading.Lock()

console = Console()

# =========================================================
# CEDA AUTH
# =========================================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        'dataset_preprocess',
        'UK_DALE_16khz'
    )
)

try:
    from ceda_auth import get_ceda_token
except ImportError:
    print("Warning: ceda_auth not found.")
    get_ceda_token = lambda: None

# =========================================================
# STATUS
# =========================================================

def update_status(file_name, status, extra=""):
    with status_lock:
        download_status[file_name] = {
            "status": status,
            "extra": extra,
            "time": time.time()
        }

# =========================================================
# FLAC VALIDATION (IMPORTANT)
# =========================================================

def validate_flac(file_path):
    """
    Ensure file is not corrupted and is ~1 hour long
    """
    try:
        data, sr = sf.read(file_path)
        duration = len(data) / sr

        if len(data) == 0:
            return False, "empty file"

        if abs(duration - EXPECTED_DURATION_SEC) > DURATION_TOLERANCE:
            return False, f"bad duration {duration:.1f}s"

        return True, f"{duration:.1f}s OK"

    except Exception as e:
        return False, str(e)

# =========================================================
# MONITOR
# =========================================================

def monitor_status(total_files):

    while True:

        os.system('cls' if os.name == 'nt' else 'clear')

        print("=" * 90)
        print("UK-DALE DOWNLOAD + VALIDATION MONITOR")
        print("=" * 90)

        with status_lock:
            items = list(download_status.items())

        items.sort()

        done = failed = downloading = retrying = queued = 0

        for idx, (fname, info) in enumerate(items, 1):

            status = info["status"]
            extra = info["extra"]

            print(f"[{idx:03d}] {status:<14} {extra:<25} {fname}")

            if status == "DONE":
                done += 1
            elif status == "FAILED":
                failed += 1
            elif status == "DOWNLOADING":
                downloading += 1
            elif status == "RETRYING":
                retrying += 1
            elif status == "QUEUED":
                queued += 1

        print("\n" + "=" * 90)
        print(f"DONE:{done} | DOWN:{downloading} | RETRY:{retrying} | QUEUE:{queued} | FAIL:{failed}")
        print("=" * 90)

        if done + failed >= total_files:
            break

        time.sleep(1)

# =========================================================
# SESSION
# =========================================================

def create_session(headers):
    session = requests.Session()
    session.headers.update(headers)

    adapter = requests.adapters.HTTPAdapter(
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

# =========================================================
# TOKEN CLEAN
# =========================================================

def clean_token(token):
    token_clean = re.sub(r'[^A-Za-z0-9\-_=.+/]', '', token)

    if len(token_clean) > 4:
        second_start = token_clean.find('ey', 2)
        if second_start > 0 and token_clean[second_start:] == token_clean[:second_start]:
            token_clean = token_clean[:second_start]

    return token_clean

# =========================================================
# DOWNLOAD WORKER
# =========================================================

def download_single_file(file_info, target_dir, headers):

    f_name = file_info["name"]
    f_url = file_info["url"]
    f_size = file_info["size"]

    target_path = os.path.join(target_dir, f_name)

    session = create_session(headers)

    max_retries = 10
    retry_delay = 10

    for attempt in range(max_retries):

        try:
            update_status(f_name, "REQUESTING", f"try {attempt+1}")

            with session.get(f_url, stream=True, timeout=60) as r:

                if r.status_code == 503:
                    update_status(f_name, "RETRYING", "503 busy")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 300)
                    continue

                r.raise_for_status()

                total_size = int(r.headers.get("content-length", f_size))
                downloaded = 0
                start = time.time()

                with open(target_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            percent = downloaded / total_size * 100 if total_size else 0
                            speed = downloaded / (time.time() - start + 1e-6) / 1024 / 1024

                            update_status(
                                f_name,
                                "DOWNLOADING",
                                f"{percent:.1f}% {speed:.2f}MB/s"
                            )

                # =====================================================
                # FILE SIZE CHECK
                # =====================================================

                if total_size > 0 and os.path.getsize(target_path) != total_size:
                    update_status(f_name, "RETRYING", "size mismatch")
                    continue

                # =====================================================
                # FLAC VALIDATION (IMPORTANT)
                # =====================================================

                ok, msg = validate_flac(target_path)

                if not ok:
                    update_status(f_name, "RETRYING", f"corrupt: {msg}")
                    continue

                update_status(f_name, "DONE", msg)
                return

        except Exception as e:
            update_status(f_name, "RETRYING", str(e)[:30])
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 300)

    update_status(f_name, "FAILED")

    if os.path.exists(target_path):
        try:
            os.remove(target_path)
        except:
            pass

# =========================================================
# DOWNLOAD WEEK
# =========================================================

def download_week(house, year, week, base_dir):

    week_str = f"wk{str(week).zfill(2)}"

    base_url = (
        "https://data.ceda.ac.uk/edc/d1/"
        "887733b3-4c04-471f-9404-9f7459c4a1a0/"
        "data/version_0"
    )

    url = f"{base_url}/house_{house}/{year}/{week_str}/"

    target_dir = os.path.join(base_dir, f"house_{house}", str(year), week_str)
    os.makedirs(target_dir, exist_ok=True)

    token = get_ceda_token()
    headers = {}

    if token:
        headers["Authorization"] = f"Bearer {clean_token(token)}"

    data = requests.get(url + "?json", headers=headers).json()

    file_links = [
        {
            "name": x["name"],
            "url": x["download"].replace("dap.ceda.ac.uk", "data.ceda.ac.uk"),
            "size": x.get("size", 0)
        }
        for x in data.get("items", [])
        if x.get("type") == "file" and x.get("name", "").endswith(".flac")
    ]

    print(f"\nDetected {len(file_links)} files")

    for f in file_links:
        update_status(f["name"], "QUEUED")

    monitor_thread = threading.Thread(
        target=monitor_status,
        args=(len(file_links),),
        daemon=True
    )
    monitor_thread.start()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(download_single_file, f, target_dir, headers)
            for f in file_links
        ]

        for f in as_completed(futures):
            f.result()

# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--house", default=HOUSE)
    parser.add_argument("--year", default=YEAR)
    parser.add_argument("--weeks", default=",".join(WEEKS))
    args = parser.parse_args()

    weeks = [w.strip() for w in args.weeks.split(",")]

    base_dir = os.path.join(
        os.path.dirname(__file__),
        "dataset_preprocess",
        "UK_DALE_16khz"
    )

    for w in weeks:
        download_week(args.house, args.year, w, base_dir)

if __name__ == "__main__":
    main()
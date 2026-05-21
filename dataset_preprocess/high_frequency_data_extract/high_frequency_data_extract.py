import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..'))

# (import_name for importlib, pip package name)
_REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("yaml", "pyyaml"),
    ("soundfile", "soundfile"),
    ("scipy", "scipy"),
    ("tzdata", "tzdata"),
]


def ensure_dependencies() -> None:
    """Install missing Python packages into the current interpreter (.venv)."""
    import importlib

    missing_pip: list[str] = []
    for import_name, pip_name in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_pip.append(pip_name)

    if not missing_pip:
        return

    req_file = os.path.join(PROJECT_ROOT, "requirements.txt")
    print("\n[deps] Missing packages:", ", ".join(sorted(set(missing_pip))))
    print("[deps] Installing via pip (this may take a few minutes)...\n")

    if os.path.isfile(req_file):
        cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
    else:
        cmd = [sys.executable, "-m", "pip", "install", *sorted(set(missing_pip))]

    subprocess.check_call(cmd)

    still_missing = []
    for import_name, pip_name in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
        except ImportError:
            still_missing.append(pip_name)
    if still_missing:
        raise ImportError(
            "Could not import after pip install: "
            + ", ".join(still_missing)
            + "\nTry manually: pip install -r requirements.txt"
        )
    print("[deps] All required packages are ready.\n")


ensure_dependencies()

import numpy as np
import soundfile as sf
import datetime
from zoneinfo import ZoneInfo
import argparse
import yaml
import pandas as pd
import configparser
import math
import tempfile
from hf_feature import compute_hf_features

# --- Official UK-DALE Calibration Logic ---
ADC_SCALE = 2**31


def get_arguments():
    parser = argparse.ArgumentParser(
        description='NILM HF extractor + LF fusion. '
                    'With no --input_path, runs batch from hf_config.yaml (weeks + appliances).'
    )
    parser.add_argument(
        '--config', type=str,
        default=os.path.join(SCRIPT_DIR, 'hf_config.yaml'),
        help='Path to hf_config.yaml',
    )
    parser.add_argument(
        '--input_path', type=str, default=None,
        help='Single .flac or folder. If omitted, uses batch section in hf_config.yaml.',
    )
    parser.add_argument('--lf_config', type=str, default=None, help='Override ukdale.yaml path')
    parser.add_argument(
        '--weeks', type=str, default=None,
        help='Override batch weeks, e.g. wk30,wk31',
    )
    parser.add_argument(
        '--appliances', type=str, default=None,
        help='Override appliances, e.g. kettle,fridge (default: ukdale.yaml list)',
    )
    return parser.parse_args()


def load_hf_config(config_path: str) -> tuple[dict, str]:
    """Resolve hf_config.yaml whether you run from repo root or this script folder."""
    candidates = []
    if os.path.isabs(config_path):
        candidates.append(config_path)
    else:
        candidates.append(os.path.abspath(config_path))
        candidates.append(os.path.join(SCRIPT_DIR, config_path))
        candidates.append(os.path.join(SCRIPT_DIR, os.path.basename(config_path)))

    config_path = next((p for p in candidates if os.path.isfile(p)), None)
    if not config_path:
        tried = "\n  ".join(dict.fromkeys(candidates))
        raise FileNotFoundError(
            f"hf_config not found. Tried:\n  {tried}\n"
            f"Use --config or run from {SCRIPT_DIR}"
        )
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config_dir = os.path.dirname(config_path)
    resolve_config_paths(config, config_dir)
    return config, config_dir


def resolve_config_paths(config: dict, config_dir: str) -> None:
    paths = config.setdefault('paths', {})
    for key in ('save_path', 'data_root', 'lf_config'):
        val = paths.get(key)
        if val and not os.path.isabs(val):
            paths[key] = os.path.normpath(os.path.join(config_dir, val))
    if not os.path.isabs(paths.get('save_path', 'output')):
        paths['save_path'] = os.path.normpath(
            os.path.join(config_dir, paths.get('save_path', 'output'))
        )

def get_calibration(file_path, config_house_id=None):
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    house_id = config_house_id
    if not house_id:
        if 'house_2' in file_path.lower(): house_id = 2
        elif 'house_1' in file_path.lower(): house_id = 1
        elif 'house_5' in file_path.lower(): house_id = 5
        else:
            raise ValueError(f"CRITICAL: Could not determine house_id from path: {file_path}. "
                             "Please ensure the path contains 'house_1', 'house_2', or 'house_5', "
                             "or specify house_id in the configuration.")
        
    search_dir = parent_dir
    for _ in range(4):
        cal_file = os.path.join(search_dir, f"calibration_house_{house_id}.cfg")
        if os.path.exists(cal_file):
            cp = configparser.ConfigParser()
            cp.read(cal_file)
            v_step = float(cp.get('Calibration', 'volts_per_adc_step'))
            i_step = float(cp.get('Calibration', 'amps_per_adc_step'))
            return v_step, i_step, cal_file, house_id
        search_dir = os.path.dirname(search_dir)
        if not search_dir or search_dir == os.path.dirname(search_dir): break
    
    raise FileNotFoundError(f"CRITICAL: Calibration file 'calibration_house_{house_id}.cfg' not found "
                            f"in {parent_dir} or its parent directories up to 4 levels.")

def decode_unix_time(ts, tz_name="UTC"):
    return datetime.datetime.fromtimestamp(int(ts), tz=ZoneInfo(tz_name)).strftime('%Y-%m-%d %H:%M:%S')


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: HF + LF FUSION
# ─────────────────────────────────────────────────────────────────────────────
def fuse_with_lf(df_hf, start_unix, end_unix, house_id, hf_config, lf_config_path, appliances_filter=None):
    """
    Automatically calls ukdale_processing.py for the exact time window of the
    processed FLAC file, then performs a time-key inner join between:
        - df_hf  : 55 HF features at 6s resolution (readable_time = London-naive)
        - df_lf  : aggregate + appliance_power + on_off at 6s resolution (time = UTC+offset)
    
    Outputs one CSV per appliance into the HF save_path.
    
    Time Key Normalization:
        LF  "2013-07-21 23:00:00+00:00" (UTC)
        → converted to Europe/London naive → "2013-07-22 00:00:00"
        HF  "2013-07-22 00:00:00" (London-naive, already correct)
        ✓ Both become the same string → exact merge.
    """
    tz = hf_config['hyperparameters'].get('timezone', 'Europe/London')
    win_sec = hf_config['hyperparameters']['window_size_seconds']
    save_path = hf_config['paths']['save_path']

    # Convert unix window to London-time strings (matching ukdale.yaml format)
    start_str = datetime.datetime.fromtimestamp(start_unix, tz=ZoneInfo(tz)).strftime('%Y-%m-%d %H:%M:%S')
    end_str   = datetime.datetime.fromtimestamp(end_unix,   tz=ZoneInfo(tz)).strftime('%Y-%m-%d %H:%M:%S')

    print("\n" + "━"*60)
    print("  PHASE 2: LOW-FREQUENCY DATA FUSION")
    print("━"*60)
    print(f"  [FUSION] House:      {house_id}")
    print(f"  [FUSION] Window:     {start_str}  →  {end_str}")
    print(f"  [FUSION] LF Config:  {os.path.basename(lf_config_path)}")

    # ── Step 1: Run ukdale_processing.py in a temp directory ─────────────────
    # We override: house, time window, no train/val/test split, and save to temp
    lf_script = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ukdale_processing.py')
    )

    # Return dict: {appliance_name: df_merged} — caller handles saving
    fused_dfs = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, lf_script,
            '--config',              lf_config_path,
            '--override_start',      start_str,
            '--override_end',        end_str,
            '--override_house',      str(house_id),
            '--no_split',
            '--save_path_override',  tmpdir,
        ]
        # Force UTF-8 in the LF child process (Windows cp1252 console breaks on → etc.)
        lf_env = os.environ.copy()
        lf_env["PYTHONIOENCODING"] = "utf-8"
        lf_env["PYTHONUTF8"] = "1"

        print(f"  [FUSION] Running LF script... (may take 30-60s on first load)")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=lf_env,
        )

        if result.returncode != 0:
            print(f"  [FUSION] ⚠️  LF processing FAILED.")
            err_tail = (result.stderr or result.stdout or "")[-500:]
            print(f"  [FUSION] stderr: {err_tail}")
            return fused_dfs

        print(f"  [FUSION] LF processing complete.")

        with open(lf_config_path, 'r') as f:
            lf_cfg = yaml.safe_load(f)
        appliances = lf_cfg['global_params'].get('appliances_to_process',
                     ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine'])
        if appliances_filter:
            appliances = [a for a in appliances if a in appliances_filter]

        merged_count = 0
        for appliance in appliances:
            lf_file_real = os.path.join(tmpdir, f"{appliance}_training_real.csv")
            if not os.path.exists(lf_file_real):
                print(f"  [FUSION] ⚠️  No LF file for {appliance}")
                continue

            df_lf = pd.read_csv(lf_file_real)
            df_lf['time_key'] = (
                pd.to_datetime(df_lf['time'], utc=True)
                  .dt.tz_convert(tz)
                  .dt.strftime('%Y-%m-%d %H:%M:%S')
            )

            df_merged = pd.merge(
                df_hf,
                df_lf[['time_key', 'aggregate', appliance, 'on_off']].rename(
                    columns={appliance: f'{appliance}_power'}
                ),
                left_on='readable_time',
                right_on='time_key',
                how='inner'
            ).drop(columns=['time_key'])

            if df_merged.empty:
                print(f"  [FUSION] ⚠️  {appliance}: 0 rows matched.")
                continue

            fused_dfs[appliance] = df_merged
            merged_count += 1
            print(f"  [FUSION] ✅ {appliance:<15} ({len(df_merged)} rows fused)")

    print(f"\n  [FUSION] Done. {merged_count}/{len(appliances)} appliances fused.")
    print("━"*60 + "\n")
    return fused_dfs


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: HF FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def process_file(flac_path, config, lf_config_path=None, save_hf_csv=True, appliances_filter=None):
    basename = os.path.basename(flac_path)
    
    print("\n" + "━"*60)
    print("  NILM HIGH-FREQUENCY DATA PROCESSOR | PHASE 1: CALIBRATED EXTRACTION")
    print("━"*60)
    print(f"  [MOUNTING] File: {basename}")
    
    info = sf.info(flac_path)
    actual_sr = info.samplerate
    config_sr = config['hyperparameters']['high_frequency']['sampling_rate']
    win_sec = config['hyperparameters']['window_size_seconds']
    chunk_size = int(actual_sr * win_sec)
    
    print(f"  [INFO] Signal Properties: {actual_sr}Hz (Actual) vs {config_sr}Hz (Config)")
    
    # Calibration
    v_step, i_step, cal_src, house_id = get_calibration(flac_path, config['hyperparameters'].get('house_id'))
    print(f"  [CALIBRATION] House: {house_id} | Source: {os.path.basename(str(cal_src))}")

    # Time Sync
    try:
        start_unix = int(basename.split('-')[1].split('_')[0])
    except:
        start_unix = 0
    target_tz = config['hyperparameters'].get('timezone', 'Europe/London')
    
    total_windows = math.ceil(info.frames / chunk_size)
    end_unix = start_unix + (total_windows * win_sec)
    
    print(f"  [TIME] Start: {decode_unix_time(start_unix, target_tz)}")
    print(f"  [TIME] End:   {decode_unix_time(end_unix, target_tz)}")
    print(f"  [CONFIG] Window: {win_sec}s | Channels: V={config['hyperparameters']['channel_config']['voltage_idx']}, "
          f"I={config['hyperparameters']['channel_config']['current_idx']}")
    print("━"*60 + "\n")

    # Processing Loop
    features = []
    chunk_idx = 0
    full_blocks = 0
    partial_blocks = []
    
    print(f"  {'Index':<7} | {'Timestamp':<19} | {'V_rms':<8} | {'I_rms':<10} | {'P_active':<8}")
    print("  " + "-"*60)

    for block in sf.blocks(flac_path, blocksize=chunk_size):
        actual_len = len(block)
        current_unix = start_unix + (chunk_idx * win_sec)
        readable_time = decode_unix_time(current_unix, target_tz)

        if actual_len == chunk_size:
            full_blocks += 1
            feat = compute_hf_features(block, config, v_step, i_step)
            print(f"  [{chunk_idx:03d}]   | {readable_time} | {feat['V_rms']:.2f}V   | {feat['I_rms']:.4f}A  | {feat['P_active']:.1f}W")
            feat['readable_time'] = readable_time
            features.append(feat)
        else:
            # Incomplete window – still compute features (using available samples)
            partial_blocks.append((readable_time, actual_len))
            feat = compute_hf_features(block, config, v_step, i_step)
            print(f"  [{chunk_idx:03d}]*  | {readable_time} | {feat['V_rms']:.2f}V   | {feat['I_rms']:.4f}A  | {feat['P_active']:.1f}W   (partial)")
            feat['readable_time'] = readable_time
            features.append(feat)
            
        chunk_idx += 1

    # Phase 1 Summary
    print("\n" + "━"*60)
    print("  PHASE 1: FEATURE EXTRACTION SUMMARY")
    print("━"*60)
    print(f"  [TOTAL] 6s Windows Processed: {chunk_idx}")
    print(f"  [DIST]  Full Features:        {full_blocks}")
    if partial_blocks:
        print(f"  [DIST]  ⚠️  INCOMPLETE WINDOWS (PROCESSED):")
        for p_time, p_len in partial_blocks:
            print(f"          -> {p_time} (Samples: {p_len} / {chunk_size})")
    
    # Save HF-only Feature Matrix (skip in batch mode — already embedded in merged CSV)
    save_path = config['paths']['save_path']
    out_file = os.path.join(save_path, f"features_{basename.replace('.flac', '.csv')}")

    df_hf = pd.DataFrame(features)
    cols = ['readable_time'] + [c for c in df_hf.columns if c != 'readable_time']
    df_hf = df_hf[cols]
    feat_cols = [c for c in df_hf.columns if c != 'readable_time']
    df_hf[feat_cols] = df_hf[feat_cols].astype('float32')
    df_hf = df_hf.round(4)

    if save_hf_csv:
        os.makedirs(save_path, exist_ok=True)
        df_hf.to_csv(out_file, index=False)
        print(f"  [SAVE]  HF Feature Matrix: {os.path.basename(out_file)}")
    else:
        print(f"  [SKIP]  HF Feature Matrix not saved (batch mode — embedded in merged CSV)")

    print(f"  [SAVE]  Shape:             {len(features)} rows × {len(df_hf.columns)} columns")
    print("━"*60 + "\n")

    # --- Phase 2: Optional LF Fusion ---
    fused_dfs = {}
    if lf_config_path and os.path.exists(lf_config_path):
        fused_dfs = fuse_with_lf(
            df_hf, start_unix, end_unix, house_id, config, lf_config_path,
            appliances_filter=appliances_filter,
        )
    elif lf_config_path:
        print(f"  [FUSION] ⚠️  --lf_config path not found: {lf_config_path}. Skipping fusion.")
    
    return fused_dfs  # {appliance_name: df_merged}


def get_appliances_filter(config: dict, cli_appliances: str | None) -> list[str] | None:
    if cli_appliances:
        return [a.strip() for a in cli_appliances.split(',') if a.strip()]
    batch_apps = config.get('batch', {}).get('appliances') or []
    if batch_apps:
        return list(batch_apps)
    return None


def week_directory(config: dict, week: str) -> str:
    batch = config.get('batch', {})
    house = batch.get('house', config['hyperparameters'].get('house_id', 2))
    year = batch.get('year', 2013)
    data_root = config['paths']['data_root']
    return os.path.join(data_root, f"house_{house}", str(year), week)


def save_fused_chunk(
    app_name: str,
    df_chunk: pd.DataFrame,
    output_dir: str,
    house_id: int,
    week_label: str | None,
    batch_output_files: dict,
    single_file_mode: bool,
    flac_path: str,
) -> None:
    house_tag = f"house{house_id}"
    if single_file_mode:
        try:
            start_unix = int(os.path.basename(flac_path).split('-')[1].split('_')[0])
        except (IndexError, ValueError):
            start_unix = 0
        out_name = f"{app_name}_{house_tag}_{start_unix}.csv"
        out_path = os.path.join(output_dir, out_name)
        df_chunk.to_csv(out_path, index=False)
        print(f"  [SAVE] {app_name} → {out_name}  ({len(df_chunk)} rows)")
        return

    key = (app_name, week_label or 'batch')
    if key not in batch_output_files:
        if week_label:
            out_name = f"{app_name}_{house_tag}_{week_label}.csv"
        else:
            start_t = pd.to_datetime(df_chunk['readable_time'].iloc[0]).strftime('%Y-%m-%d')
            out_name = f"{app_name}_{house_tag}_batch_{start_t}.csv"
        out_path = os.path.join(output_dir, out_name)
        batch_output_files[key] = out_path
        df_chunk.to_csv(out_path, index=False, mode='w')
        print(f"  [BATCH] Created  → {out_name}")
    else:
        out_path = batch_output_files[key]
        df_chunk.to_csv(out_path, index=False, mode='a', header=False)
        print(f"  [BATCH] Appended → {os.path.basename(out_path)}  (+{len(df_chunk)} rows)")


def run_flac_pipeline(
    flac_files: list[str],
    config: dict,
    lf_config_path: str | None,
    appliances_filter: list[str] | None,
    save_hf_csv: bool,
    week_label: str | None = None,
) -> dict:
    """Process a list of FLAC paths; return batch output file map."""
    output_dir = config['paths']['save_path']
    os.makedirs(output_dir, exist_ok=True)
    house_id = config['hyperparameters'].get('house_id', 2)

    batch_mode = len(flac_files) > 1
    batch_output_files = {}

    for i, f_path in enumerate(flac_files):
        print(f"\n[{i+1}/{len(flac_files)}] Processing: {os.path.basename(f_path)}")
        fused_dfs = process_file(
            f_path, config, lf_config_path,
            save_hf_csv=save_hf_csv,
            appliances_filter=appliances_filter,
        )
        if not fused_dfs:
            continue
        for app_name, df_chunk in fused_dfs.items():
            save_fused_chunk(
                app_name, df_chunk, output_dir, house_id, week_label,
                batch_output_files, single_file_mode=not batch_mode, flac_path=f_path,
            )

    if batch_mode and batch_output_files:
        print("\n" + "━"*60)
        label = week_label or "batch"
        print(f"  WEEK/BATCH COMPLETE — {label}")
        print("━"*60)
        for (_, _), out_path in sorted(batch_output_files.items(), key=lambda x: x[1]):
            total = len(pd.read_csv(out_path))
            print(f"  ✅ {os.path.basename(out_path):<40} | rows: {total:>6}")
        print("━"*60)

    return batch_output_files


def run_batch_from_config(config: dict, lf_config_path: str | None, weeks_override: str | None,
                          appliances_filter: list[str] | None) -> None:
    batch = config.get('batch', {})
    if not batch.get('enabled', True):
        print("[BATCH] batch.enabled is false in hf_config.yaml. Nothing to run.")
        sys.exit(1)

    weeks = [w.strip() for w in (weeks_override or ','.join(batch.get('weeks', []))).split(',') if w.strip()]
    if not weeks:
        print("[BATCH] No weeks configured. Set batch.weeks in hf_config.yaml or use --weeks wk30")
        sys.exit(1)

    fuse_lf = batch.get('fuse_lf', True)
    lf_path = lf_config_path if lf_config_path else config['paths'].get('lf_config')
    if fuse_lf and (not lf_path or not os.path.exists(lf_path)):
        print(f"[BATCH] LF fusion enabled but lf_config not found: {lf_path}")
        sys.exit(1)
    if not fuse_lf:
        lf_path = None

    save_hf = batch.get('save_hf_csv_per_flac', False)
    apps = appliances_filter
    if apps:
        print(f"[BATCH] Appliances: {apps}")
    else:
        print("[BATCH] Appliances: all from ukdale.yaml")

    print("\n" + "═"*60)
    print("  CONFIG-DRIVEN BATCH — HF + LF FUSION")
    print("═"*60)
    print(f"  Data root:  {config['paths']['data_root']}")
    print(f"  Output:     {config['paths']['save_path']}")
    print(f"  Weeks:      {weeks}")
    print(f"  LF config:  {lf_path or '(disabled)'}")
    print("═"*60)

    all_outputs = {}
    for week in weeks:
        week_dir = week_directory(config, week)
        if not os.path.isdir(week_dir):
            print(f"\n[BATCH] ⚠️  Skip {week}: folder not found:\n         {week_dir}")
            continue
        flac_files = sorted(
            os.path.join(week_dir, f) for f in os.listdir(week_dir) if f.endswith('.flac')
        )
        if not flac_files:
            print(f"\n[BATCH] ⚠️  Skip {week}: no .flac files in {week_dir}")
            continue

        print(f"\n{'═'*60}\n  WEEK {week} — {len(flac_files)} FLAC file(s)\n{'═'*60}")
        out = run_flac_pipeline(
            flac_files, config, lf_path, apps, save_hf_csv=save_hf, week_label=week,
        )
        all_outputs.update(out)

    print("\n" + "═"*60)
    print("  ALL WEEKS FINISHED")
    print("═"*60)
    if all_outputs:
        for out_path in sorted(set(all_outputs.values())):
            print(f"  📄 {out_path}")
    else:
        print("  No output files were written.")
    print("═"*60 + "\n[DONE]")


if __name__ == "__main__":
    args = get_arguments()
    config, _ = load_hf_config(args.config)
    appliances_filter = get_appliances_filter(config, args.appliances)
    lf_override = args.lf_config

    if args.input_path:
        path = os.path.abspath(args.input_path)
        if os.path.isfile(path) and path.endswith('.flac'):
            flac_files = [path]
        elif os.path.isdir(path):
            flac_files = sorted(
                os.path.join(path, f) for f in os.listdir(path) if f.endswith('.flac')
            )
        else:
            print(f"Error: {path} is not a valid file or directory.")
            sys.exit(1)
        lf_path = lf_override or config['paths'].get('lf_config')
        batch_cfg = config.get('batch', {})
        save_hf = True if len(flac_files) == 1 else batch_cfg.get('save_hf_csv_per_flac', False)
        run_flac_pipeline(
            flac_files, config, lf_path, appliances_filter,
            save_hf_csv=save_hf,
            week_label=None,
        )
        print("\n[DONE] All tasks finished.")
    else:
        run_batch_from_config(config, lf_override, args.weeks, appliances_filter)

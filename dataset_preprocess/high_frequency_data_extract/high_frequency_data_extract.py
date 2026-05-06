import numpy as np
import soundfile as sf
import datetime
from zoneinfo import ZoneInfo
import os
import argparse
import yaml
import pandas as pd
import configparser
import math

def get_arguments():
    parser = argparse.ArgumentParser(description='NILM High-Frequency Feature Extractor (Calibrated & Rich UI)')
    parser.add_argument('--config', type=str, default='hf_config.yaml', help='Path to config')
    parser.add_argument('--input_path', type=str, required=True, help='Path to .flac file or directory')
    return parser.parse_args()

# --- Official UK-DALE Calibration Logic ---
ADC_SCALE = 2**31 

def get_calibration(file_path, config_house_id=None):
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    house_id = config_house_id
    if not house_id:
        if 'house_2' in file_path.lower(): house_id = 2
        elif 'house_1' in file_path.lower(): house_id = 1
        elif 'house_5' in file_path.lower(): house_id = 5
        else: house_id = 2 # Default to 2 for Week 30
        
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
    return 1.88296904357e-07, 4.77518864497e-08, "Defaults (H2)", 2

def decode_unix_time(ts, tz_name="UTC"):
    return datetime.datetime.fromtimestamp(int(ts), tz=ZoneInfo(tz_name)).strftime('%Y-%m-%d %H:%M:%S')

def compute_features(block, config, v_step, i_step):
    """
    Converts raw block to physical units and extracts a 12-dimensional feature vector.
    Features: V_rms, I_rms, P_active, Q_reactive, S_apparent, PF, THD, H3, H5, H7, H9, H11
    """
    v_col = config['hyperparameters']['channel_config']['voltage_idx']
    i_col = config['hyperparameters']['channel_config']['current_idx']
    
    # 1. Official UK-DALE Conversion: Extract Instantaneous Waveforms v(t), i(t)
    v_t = block[:, v_col] * ADC_SCALE * v_step
    i_t = block[:, i_col] * ADC_SCALE * i_step
    
    # 2. Fundamental Electrical Parameters (RMS and Power)
    V_rms = np.sqrt(np.mean(np.square(v_t))) #Root Mean Square of voltage
    I_rms = np.sqrt(np.mean(np.square(i_t))) #Root Mean Square of current

    P_active = np.mean(v_t * i_t) #Active Power, P(W)
    S_apparent = V_rms * I_rms #Apparent Power, S(VA)
    Q_reactive = np.sqrt(max(0, S_apparent**2 - P_active**2)) #Reactive Power, Q(VAR)
    
    #Power Factor, PF = cos(theta)
    PF = P_active / S_apparent if S_apparent > 1e-6 else 1.0
    
    # 3. Harmonic Content (Cycle-averaging Analysis)
    samples_per_cycle = 320 
    num_cycles = len(v_t) // samples_per_cycle
    i_cycles = i_t[:num_cycles*samples_per_cycle].reshape(num_cycles, samples_per_cycle)
    
    fft_avg = np.mean(np.abs(np.fft.rfft(i_cycles, axis=1)), axis=0)
    fund = fft_avg[1] if fft_avg[1] > 1e-6 else 1e-6
    
    # Extract ratios for odd harmonics (H3 to H11)
    H_ratios = {}
    for h in [3, 5, 7, 9, 11]:
        H_ratios[f'H{h}_ratio'] = round(fft_avg[h] / fund, 6)
        
    # 4. Total Harmonic Distortion (THD)
    harmonic_energy = np.sum(np.square(fft_avg[2:20])) 
    THD = np.sqrt(harmonic_energy) / fund
    
    return {
        'V_rms': round(V_rms, 4),
        'I_rms': round(I_rms, 6),
        'P_active': round(P_active, 2),
        'Q_reactive': round(Q_reactive, 2),
        'S_apparent': round(S_apparent, 2),
        'PF': round(PF, 4),
        'THD': round(THD, 6),
        **H_ratios
    }

def process_file(flac_path, config):
    basename = os.path.basename(flac_path)
    
    # 1. UI: Phase 1 Header
    print("\n" + "━"*60)
    print("  NILM HIGH-FREQUENCY DATA PROCESSOR | PHASE 1: CALIBRATED EXTRACTION")
    print("━"*60)
    print(f"  [MOUNTING] File: {basename}")
    
    info = sf.info(flac_path)
    actual_sr = info.samplerate
    config_sr = config['hyperparameters']['sampling_rate']
    win_sec = config['hyperparameters']['window_size_seconds']
    chunk_size = int(actual_sr * win_sec)
    
    print(f"  [INFO] Signal Properties: {actual_sr}Hz (Actual) vs {config_sr}Hz (Config)")
    
    # 2. Calibration Setup
    v_step, i_step, cal_src, house_id = get_calibration(flac_path, config['hyperparameters'].get('house_id'))
    print(f"  [CALIBRATION] House: {house_id} | Source: {os.path.basename(str(cal_src))}")

    # 3. Time Sync
    try:
        start_unix = int(basename.split('-')[1].split('_')[0])
    except:
        start_unix = 0
    target_tz = config['hyperparameters'].get('timezone', 'Europe/London')
    
    total_windows = math.ceil(info.frames / chunk_size)
    end_unix = start_unix + (total_windows * win_sec)
    
    print(f"  [TIME] Start Timestamp: {decode_unix_time(start_unix, target_tz)}")
    print(f"  [TIME] End Timestamp:   {decode_unix_time(end_unix, target_tz)}")
    print(f"  [CONFIG] Window: {win_sec}s | Channels: V={config['hyperparameters']['channel_config']['voltage_idx']}, I={config['hyperparameters']['channel_config']['current_idx']}")
    print("━"*60 + "\n")

    # 4. Processing Loop
    features = []
    chunk_idx = 0
    full_blocks = 0
    partial_blocks = []
    
    for block in sf.blocks(flac_path, blocksize=chunk_size):
        actual_len = len(block)
        current_unix = start_unix + (chunk_idx * win_sec)
        
        if actual_len == chunk_size:
            full_blocks += 1
            feat = compute_features(block, config, v_step, i_step)
            
            # --- VALIDATION SNAPSHOT (Print first window results) ---
            if full_blocks == 1:
                print(f"  [VALIDATION] First Window (6s) Snapshot:")
                print(f"               -> Voltage RMS: {feat['v_rms']} V")
                print(f"               -> Current RMS: {feat['i_rms']} A")
                print(f"               -> Real Power:  {feat['real_power']} W")
                if 220 < feat['v_rms'] < 260:
                    print(f"               -> Status:      ✅ CALIBRATION CORRECT")
                else:
                    print(f"               -> Status:      ⚠️ CHECK CALIBRATION")

            feat['timestamp'] = current_unix
            feat['readable_time'] = decode_unix_time(current_unix, target_tz)
            features.append(feat)
        else:
            readable_time = decode_unix_time(current_unix, target_tz)
            partial_blocks.append((readable_time, actual_len))
            
        chunk_idx += 1

    # 5. UI: Phase 1 Summary
    print("\n" + "━"*60)
    print("  PHASE 1: EXTRACTION & CALIBRATION SUMMARY")
    print("━"*60)
    print(f"  [TOTAL] Blocks Generated:  {chunk_idx}")
    print(f"  [DIST]  Full (Calibrated): {full_blocks}")
    if partial_blocks:
        print(f"  [DIST]  Problematic Blocks (Partial):")
        for p_time, p_len in partial_blocks:
            print(f"          -> {p_time} (Samples: {p_len})")
    else:
        print(f"  [DIST]  Partial:           0 blocks")
    
    # Save
    save_path = config['paths']['save_path']
    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, f"features_{basename.replace('.flac', '.csv')}")
    pd.DataFrame(features).to_csv(out_file, index=False)
    print(f"  [SAVE]  Feature Matrix:    {os.path.basename(out_file)}")
    print("━"*60 + "\n")

if __name__ == "__main__":
    args = get_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    path = args.input_path
    if os.path.isdir(path):
        files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.flac')])
        for f in files: process_file(f, config)
    else:
        process_file(path, config)

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Slider, Button
import soundfile as sf
from datetime import datetime, timedelta
import configparser

# ==========================================
# VI WAVEFORM VISUALIZER (CALIBRATED)
# ==========================================

def interactive_vi_viewer(file_path, forced_house=None):
    """
    Interactive viewer for UK-DALE VI .flac files with Auto-Calibration.
    """
    print(f"\n[LOADING] {file_path}")
    
    # 1. Basic Metadata
    info = sf.info(file_path)
    sr = info.samplerate
    total_frames = info.frames
    duration = total_frames / sr
    
    # 2. Timestamp Extraction
    try:
        base_ts = int(os.path.basename(file_path).split('-')[1].split('_')[0])
        start_time = datetime.fromtimestamp(base_ts)
        start_dt_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    except:
        start_time = datetime(2013, 1, 1)
        start_dt_str = "Unknown"

    # 3. Automatic Calibration Detection
    parent_dir = os.path.dirname(file_path)
    house_id = forced_house
    
    if not house_id:
        if 'house_2' in file_path.lower(): house_id = 2
        elif 'house_1' in file_path.lower(): house_id = 1
        elif 'house_5' in file_path.lower(): house_id = 5
        else:
            # Fallback: Check if any calibration file exists in the directory
            cal_files = [f for f in os.listdir(parent_dir) if f.startswith('calibration_house_') and f.endswith('.cfg')]
            if len(cal_files) == 1:
                house_id = int(cal_files[0].split('_')[2].split('.')[0])
                print(f"  [AUTO] Only one calibration file found, using House {house_id}")
            elif len(cal_files) > 1:
                # If multiple found and it's 2013 July, it's likely House 2 for Week 30
                house_id = 2 
                print(f"  [AUTO] Multiple calibrations found. Defaulting to House 2 for Week 30.")

    v_multiplier, i_multiplier = 1.0, 1.0
    adc_scale = 2**31 # Official UK-DALE 32-bit ADC half-range
    is_calibrated = False

    if house_id:
        # Search for calibration file (look up to 4 levels up)
        cal_file = None
        search_dir = parent_dir
        for _ in range(4):
            candidate = os.path.join(search_dir, f"calibration_house_{house_id}.cfg")
            if os.path.exists(candidate):
                cal_file = candidate
                break
            search_dir = os.path.dirname(search_dir)
            if not search_dir or search_dir == os.path.dirname(search_dir): break
        
        if cal_file and os.path.exists(cal_file):
            config = configparser.ConfigParser()
            config.read(cal_file)
            v_step = float(config.get('Calibration', 'volts_per_adc_step'))
            i_step = float(config.get('Calibration', 'amps_per_adc_step'))
            # --- Official UK-DALE Conversion Formula ---
            # "volts = value from WAV * volts per ADC step * 2^31 ADC steps"
            # 
            # Rationale: The recording software stores each sample as a 32-bit integer.
            # 2^32 steps for full range [-1, 1], so 2^31 steps for half range.
            v_multiplier = adc_scale * v_step
            i_multiplier = adc_scale * i_step
            is_calibrated = True
            
            print(f"  [SUCCESS] Found calibration at: {os.path.basename(cal_file)}")
            print(f"  [SUCCESS] Calibrated for House {house_id} (V_Mult: {v_multiplier:.2f})")

    # 4. Data Loading & Scaling
    print("  [READING] Data into memory...")
    raw_data, _ = sf.read(file_path)
    
    if is_calibrated:
        print(f"  [SCALING] Converting to physical units (V, A)...")
        data = np.zeros_like(raw_data)
        data[:, 0] = raw_data[:, 0] * v_multiplier
        data[:, 1] = raw_data[:, 1] * i_multiplier
    else:
        data = raw_data
        print("  [WARNING] No calibration applied. Showing raw normalized values [-1, 1].")

    # 5. UI Setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    plt.subplots_adjust(bottom=0.25, hspace=0.3, left=0.08, right=0.95)
    
    state = {'start_idx': 0, 'view_span': int(sr * 0.04)} # 40ms view
    
    def get_time_axis(s_idx, e_idx):
        offsets = np.arange(s_idx, e_idx) / sr
        return [start_time + timedelta(seconds=o) for o in offsets]

    end_idx = min(state['start_idx'] + state['view_span'], total_frames)
    t_axis = get_time_axis(state['start_idx'], end_idx)
    
    line_v, = ax1.plot(t_axis, data[state['start_idx']:end_idx, 0], color='#d62728', label='Voltage (V)', linewidth=1)
    line_i, = ax2.plot(t_axis, data[state['start_idx']:end_idx, 1], color='#1f77b4', label='Current (A)', linewidth=1)
    
    xfmt = mdates.DateFormatter('%H:%M:%S') 
    ax2.xaxis.set_major_formatter(xfmt)
    fig.autofmt_xdate()

    ax1.set_ylabel('Voltage (V)')
    ax2.set_ylabel('Current (A)')
    ax1.set_title(f"UK-DALE Calibrated Waveform | {os.path.basename(file_path)}")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Sliders
    ax_pos = plt.axes([0.1, 0.1, 0.45, 0.03])
    ax_span = plt.axes([0.1, 0.05, 0.45, 0.03])
    pos_slider = Slider(ax_pos, 'Start (s)', 0, max(0, duration - 0.01), valinit=0)
    span_slider = Slider(ax_span, 'Span (s)', 0.001, 1.0, valinit=0.04)

    def update(val):
        start_idx = int(pos_slider.val * sr)
        span_idx = int(span_slider.val * sr)
        end_idx = min(start_idx + span_idx, total_frames)
        
        t_seg = get_time_axis(start_idx, end_idx)
        line_v.set_xdata(t_seg)
        line_v.set_ydata(data[start_idx:end_idx, 0])
        line_i.set_xdata(t_seg)
        line_i.set_ydata(data[start_idx:end_idx, 1])
        
        ax1.set_xlim(t_seg[0], t_seg[-1])
        fig.canvas.draw_idle()

    pos_slider.on_changed(update)
    span_slider.on_changed(update)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='UK-DALE VI Waveform Visualizer')
    parser.add_argument('--path', type=str, help='Path to .flac file')
    parser.add_argument('--house', type=int, help='Force House ID (1, 2, or 5)')
    args = parser.parse_args()
    
    file_path = args.path
    if not file_path:
        # Use relative path from the root of the project
        default_dir = os.path.join("high_low_freq_NILM", "dataset_preprocess", "UK_DALE_16khz")
        files = sorted([f for f in os.listdir(default_dir) if f.endswith('.flac')]) if os.path.exists(default_dir) else []
        
        if not files:
            file_path = input("Enter full path to .flac file: ").strip().strip('"')
        else:
            for i, f in enumerate(files[:20]):
                print(f" [{i}] {f}")
            inp = input("\nEnter Index or Full Path: ").strip().strip('"')
            if os.path.exists(inp): file_path = inp
            elif inp.isdigit(): file_path = os.path.join(default_dir, files[int(inp)])

    if file_path and os.path.exists(file_path):
        interactive_vi_viewer(file_path, forced_house=args.house)

if __name__ == '__main__':
    main()

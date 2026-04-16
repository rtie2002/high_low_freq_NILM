import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Slider, Button
import soundfile as sf
from datetime import datetime, timedelta

# ==========================================
# VI WAVEFORM VISUALIZER (ENHANCED)
# ==========================================

def interactive_vi_viewer(file_path):
    """
    Interactive viewer for UK-DALE VI .flac files.
    - Channel 0: Voltage
    - Channel 1: Current
    - Real-time X-axis
    - Autofit and navigation buttons
    """
    print(f"\nLoading waveform: {file_path}")
    
    # Load metadata
    info = sf.info(file_path)
    sr = info.samplerate
    total_frames = info.frames
    channels = info.channels
    duration = total_frames / sr
    
    # Extract timestamp from filename
    try:
        base_ts = int(os.path.basename(file_path).split('-')[1].split('_')[0])
        start_time = datetime.fromtimestamp(base_ts)
        start_dt_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    except:
        start_time = datetime(2013, 1, 1) # Fallback
        start_dt_str = "Unknown"

    print(f"Sample Rate: {sr} Hz")
    print(f"Duration:    {duration:.2f} seconds")
    print(f"Start Time:  {start_dt_str}")

    # Load data
    print("Reading data into memory...")
    data, _ = sf.read(file_path)
    
    # Initial State
    state = {
        'start_idx': 0,
        'view_span': int(sr * 0.1), # Default 0.1s
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    plt.subplots_adjust(bottom=0.25, hspace=0.3, left=0.08, right=0.95)
    
    # Initial Calculation
    end_idx = min(state['start_idx'] + state['view_span'], total_frames)
    
    def get_time_axis(s_idx, e_idx):
        offsets = np.arange(s_idx, e_idx) / sr
        return [start_time + timedelta(seconds=o) for o in offsets]

    t_axis = get_time_axis(state['start_idx'], end_idx)
    
    line_v, = ax1.plot(t_axis, data[state['start_idx']:end_idx, 0], color='#d62728', label='Voltage', linewidth=1)
    line_i, = ax2.plot(t_axis, data[state['start_idx']:end_idx, 1], color='#1f77b4', label='Current', linewidth=1)
    
    # Format X-axis as exact time (Simplified Decimals)
    xfmt = mdates.DateFormatter('%H:%M:%S.%3f') 
    ax2.xaxis.set_major_formatter(xfmt)
    fig.autofmt_xdate()

    ax1.set_ylabel('Voltage (V)')
    ax2.set_ylabel('Current (A)')
    ax1.set_title(f"UK-DALE Waveform: {os.path.basename(file_path)} | Start: {start_dt_str}")
    
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # ==========================================
    # CONTROLS
    # ==========================================
    ax_pos = plt.axes([0.1, 0.1, 0.45, 0.03])
    ax_span = plt.axes([0.1, 0.05, 0.45, 0.03])
    
    pos_slider = Slider(ax_pos, 'Start Sec', 0, max(0, duration - 0.01), valinit=0)
    span_slider = Slider(ax_span, 'Span Sec', 0.001, min(duration, 5.0), valinit=0.1)

    # Navigation Buttons
    ax_prev = plt.axes([0.6, 0.07, 0.08, 0.04])
    ax_next = plt.axes([0.69, 0.07, 0.08, 0.04])
    ax_fit  = plt.axes([0.78, 0.07, 0.12, 0.04])
    
    btn_prev = Button(ax_prev, '◀ Back')
    btn_next = Button(ax_next, 'Forward ▶')
    btn_fit  = Button(ax_fit, 'Autofit Y')

    def update(val=None):
        start_sec = pos_slider.val
        span_sec = span_slider.val
        
        start_idx = int(start_sec * sr)
        end_idx = int((start_sec + span_sec) * sr)
        end_idx = min(end_idx, total_frames)
        
        if end_idx <= start_idx: return

        segment = data[start_idx:end_idx]
        t_seg = get_time_axis(start_idx, end_idx)
        
        line_v.set_xdata(t_seg)
        line_v.set_ydata(segment[:, 0])
        line_i.set_xdata(t_seg)
        line_i.set_ydata(segment[:, 1])
        
        ax1.set_xlim(t_seg[0], t_seg[-1])
        fig.canvas.draw_idle()

    def do_autofit(event):
        y_v = line_v.get_ydata()
        if len(y_v) > 0:
            margin = (np.max(y_v) - np.min(y_v)) * 0.1 or 1
            ax1.set_ylim(np.min(y_v) - margin, np.max(y_v) + margin)
        
        y_i = line_i.get_ydata()
        if len(y_i) > 0:
            margin = (np.max(y_i) - np.min(y_i)) * 0.1 or 0.1
            ax2.set_ylim(np.min(y_i) - margin, np.max(y_i) + margin)
        
        fig.canvas.draw_idle()

    def move_prev(event):
        new_val = max(pos_slider.val - span_slider.val, 0)
        pos_slider.set_val(new_val)

    def move_next(event):
        new_val = min(pos_slider.val + span_slider.val, duration - span_slider.val)
        pos_slider.set_val(new_val)

    pos_slider.on_changed(update)
    span_slider.on_changed(update)
    btn_prev.on_clicked(move_prev)
    btn_next.on_clicked(move_next)
    btn_fit.on_clicked(do_autofit)

    print("\nVisualizer ready. Use sliders and buttons to navigate.")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='UK-DALE VI Waveform Visualizer')
    parser.add_argument('--path', type=str, help='Path to .flac file')
    
    args = parser.parse_args()
    
    file_path = args.path
    if not file_path:
        default_dir = r"C:\Users\Raymond Tie\Desktop\PhD\Code\multi-domain NILM\high_low_freq_NILM\dataset_preprocess\UK_DALE_16khz"
        print(f"\nScanning: {default_dir}")
        
        files = []
        if os.path.exists(default_dir):
            files = sorted([f for f in os.listdir(default_dir) if f.endswith('.flac')])
        
        if not files:
            print("No .flac files found in the default directory.")
            file_path = input("Please enter the full path to a .flac file: ").strip().strip('"')
        else:
            for i, f in enumerate(files):
                print(f" [{i}] {f}")
            inp = input("\nEnter Index or Full Path: ").strip().strip('"')
            if os.path.exists(inp):
                file_path = inp
            elif inp.isdigit() and int(inp) < len(files):
                file_path = os.path.join(default_dir, files[int(inp)])
                
    if file_path and os.path.exists(file_path):
        interactive_vi_viewer(file_path)
    else:
        print(f"\nError: File not found or invalid input: {file_path}")

if __name__ == '__main__':
    main()

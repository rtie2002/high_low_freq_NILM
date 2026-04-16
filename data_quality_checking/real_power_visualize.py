import numpy as np
import pandas as pd
import os
import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import yaml

# ==========================================
# CONFIGURATION & PATH RESOLUTION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, 'config', 'preprocess', 'ukdale.yaml')

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_paths(config):
    paths = config['paths']
    for key in ['data_dir', 'save_path']:
        if not os.path.isabs(paths[key]):
            paths[key] = os.path.normpath(os.path.join(PROJECT_ROOT, paths[key]))
    return paths

# ==========================================
# DATA PROCESSING UTILITIES
# ==========================================
def denormalize_zscore(data, mean, std):
    return data * std + mean

def detect_appliance_from_path(file_path, config):
    file_lower = os.path.basename(file_path).lower()
    for appliance in config['appliances'].keys():
        if appliance.lower() in file_lower:
            return appliance
    return None

# ==========================================
# UNIVERSAL DATA VIEWER (FREE X-AXIS)
# ==========================================
def interactive_viewer(file_path, config, denormalize=True):
    """
    Timeline Viewer with: 
    - Smooth scrolling with start_idx
    - Adjustable window size (Zoom in/out on X-axis)
    - Full statistics and line toggling
    """
    print(f"\nLoading data: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    cols = list(df.columns)
    appliance_name = detect_appliance_from_path(file_path, config)
    global_params = config['global_params']
    
    # Process all numeric power columns into full sequences
    sequences = {}
    for i, col in enumerate(cols):
        if i == 0: continue # Skip time column
        
        data = df.iloc[:, i].values
        label = col
        
        if denormalize:
            if i == 1: # Aggregate
                data = denormalize_zscore(data, global_params['aggregate_mean'], global_params['aggregate_std'])
                label = f"{col} (Watts)"
            elif i == 2 and appliance_name: # Appliance
                mean, std = config['appliances'][appliance_name]['mean'], config['appliances'][appliance_name]['std']
                data = denormalize_zscore(data, mean, std)
                label = f"{col} (Watts)"
        
        sequences[label] = data

    total_points = len(df)
    print(f"Total points in sequence: {total_points:,}")

    # Initial State
    state = {
        'start_idx': 0,
        'view_span': 1024, # How many points to show at once
        'selection_rect': None,
        'sel_start': None
    }

    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.25, left=0.08, right=0.95, top=0.95)
    
    lines = {}
    label_to_legline = {}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Initial Plotting
    curr_end = min(state['start_idx'] + state['view_span'], total_points)
    x_range = np.arange(state['start_idx'], curr_end)
    
    for i, (label, data) in enumerate(sequences.items()):
        line, = ax.plot(x_range, data[state['start_idx']:curr_end], label=label, color=colors[i%len(colors)], alpha=0.8, picker=5)
        lines[label] = line

    leg = ax.legend(loc='upper right')
    for legline in leg.get_lines():
        legline.set_picker(5)
        label_to_legline[legline.get_label()] = legline

    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Power (Watts)' if denormalize else 'Value')
    ax.set_title(f"Visualizing: {appliance_name or 'Sequence'} | Span: {state['view_span']} pts")

    # ==========================================
    # CONTROLS
    # ==========================================
    # 1. Timeline Slider (Start Position)
    ax_pos = plt.axes([0.1, 0.12, 0.5, 0.03])
    pos_slider = Slider(ax_pos, 'Start Pos', 0, max(0, total_points - 100), valinit=0, valstep=1, valfmt='%d')
    
    # 2. Span Slider (X-Axis Zoom)
    ax_span = plt.axes([0.1, 0.07, 0.5, 0.03])
    span_slider = Slider(ax_span, 'View Span', 100, min(total_points, 50000), valinit=state['view_span'], valstep=100)

    # Buttons
    ax_prev = plt.axes([0.65, 0.09, 0.08, 0.04])
    ax_next = plt.axes([0.74, 0.09, 0.08, 0.04])
    ax_fit  = plt.axes([0.83, 0.09, 0.12, 0.04])
    btn_prev = Button(ax_prev, '◀ Back')
    btn_next = Button(ax_next, 'Forward ▶')
    btn_fit  = Button(ax_fit, 'Fit Waveform')

    def redraw_view(val=None):
        state['start_idx'] = int(pos_slider.val)
        state['view_span'] = int(span_slider.val)
        
        end_idx = min(state['start_idx'] + state['view_span'], total_points)
        x_new = np.arange(state['start_idx'], end_idx)
        
        for label, line in lines.items():
            line.set_xdata(x_new)
            line.set_ydata(sequences[label][state['start_idx']:end_idx])
            
        ax.set_xlim(state['start_idx'], end_idx)
        ax.set_title(f"Visualizing: {appliance_name} | {state['start_idx']:,} → {end_idx:,} (Span: {state['view_span']:,})")
        fig.canvas.draw_idle()

    def auto_fit(event):
        y_min, y_max = float('inf'), float('-inf')
        found = False
        for line in lines.values():
            if line.get_visible():
                y = line.get_ydata()
                if len(y) > 0:
                    y_min, y_max, found = min(y_min, np.min(y)), max(y_max, np.max(y)), True
        if found:
            span = (y_max - y_min) or 1
            ax.set_ylim(y_min - span*0.1, y_max + span*0.1)
            fig.canvas.draw_idle()

    # Interactivity Features
    def on_pick(event):
        target = event.artist
        line = target if target in lines.values() else lines.get(target.get_label())
        if line:
            vis = not line.get_visible()
            line.set_visible(vis)
            if line.get_label() in label_to_legline:
                label_to_legline[line.get_label()].set_alpha(1.0 if vis else 0.2)
            fig.canvas.draw_idle()

    def on_mouse_press(event):
        if event.inaxes != ax: return
        if event.button == 1: # Left click: Selection
            state['sel_start'] = (event.xdata, event.ydata)
            if state['selection_rect']: state['selection_rect'].remove()
            state['selection_rect'] = plt.Rectangle((event.xdata, event.ydata), 0, 0, fill=False, color='red', linestyle='--')
            ax.add_patch(state['selection_rect'])
        elif event.button == 3: # Right click: Clear
            if state['selection_rect']: state['selection_rect'].remove(); state['selection_rect'] = None
            fig.canvas.draw_idle()

    def on_mouse_move(event):
        if state['sel_start'] and event.inaxes == ax:
            w, h = event.xdata - state['sel_start'][0], event.ydata - state['sel_start'][1]
            state['selection_rect'].set_width(w); state['selection_rect'].set_height(h)
            fig.canvas.draw_idle()

    def on_mouse_release(event):
        if state['sel_start'] and event.button == 1:
            x0, x1 = sorted([state['sel_start'][0], event.xdata])
            state['sel_start'] = None
            print("\n" + "="*40 + "\nPERIOD STATISTICS (From Global Time)")
            for label, line in lines.items():
                if not line.get_visible(): continue
                x, y = line.get_xdata(), line.get_ydata()
                mask = (x >= x0) & (x <= x1)
                sub = y[mask]
                if len(sub) > 0:
                    print(f"{label:20} | Mean: {sub.mean():7.2f}W | Max: {sub.max():7.2f}W | Points: {len(sub)}")
            print("="*40)

    # Bindings
    pos_slider.on_changed(redraw_view)
    span_slider.on_changed(redraw_view)
    btn_prev.on_clicked(lambda e: pos_slider.set_val(max(0, state['start_idx'] - state['view_span']//2)))
    btn_next.on_clicked(lambda e: pos_slider.set_val(min(total_points, state['start_idx'] + state['view_span']//2)))
    btn_fit.on_clicked(auto_fit)
    
    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)

    auto_fit(None)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Timeline NILM Visualizer')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG)
    parser.add_argument('--path', type=str)
    
    args = parser.parse_args()
    config = load_config(args.config)
    paths = get_paths(config)
    
    file_path = args.path
    if not file_path:
        print(f"\nScanning: {paths['save_path']}")
        files = sorted([f for f in os.listdir(paths['save_path']) if f.endswith('.csv')]) if os.path.exists(paths['save_path']) else []
        for i, f in enumerate(files): print(f" [{i}] {f}")
        inp = input("\nEnter Index, Appliance, or Full Path: ").strip().strip('"')
        if os.path.exists(inp): file_path = inp
        elif inp.isdigit() and int(inp) < len(files): file_path = os.path.join(paths['save_path'], files[int(inp)])
        elif inp: 
            file_path = os.path.join(paths['save_path'], f"{inp}_training.csv")

    if file_path and os.path.exists(file_path):
        interactive_viewer(file_path, config)
    else:
        print(f"Error: File not found {file_path}")

if __name__ == '__main__':
    main()

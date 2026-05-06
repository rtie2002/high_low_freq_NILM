"""
Algorithm 1: Data Cleaning and Selection for Multivariate Appliance Power Data
Based on the paper: "A diffusion model-based framework to enhance the robustness 
of non-intrusive load disaggregation"

This script implements Algorithm 1 to select effective parts of appliance data
and prepare it for multivariate diffusion model training.

IMPORTANT: According to the paper (Section 4.1), Algorithm 1 is applied ONLY to 
TRAINING data: "When synthesizing data, we execute Algorithm 1 on the training data, 
send it to the diffusion model for synthetic data training..."

- Training data: Apply Algorithm 1 → Used for diffusion model training
- Validation/Test data: NOT processed by Algorithm 1 → Used for NILM model evaluation

Input: Multivariate CSV file (format: aggregate, appliance, minute_sin, minute_cos, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos)
Output: CSV with 9 columns (appliance + 8 sin/cos time features) - MinMax normalized appliance power, temporal features preserved

Usage:
    # Process training file
    python algorithm1_v2_multivariate.py --appliance_name fridge
    
Workflow:
    1. Read multivariate CSV (10 columns with headers)
    2. Extract appliance power column
    3. Apply Algorithm 1 (select effective parts based on threshold and window)
    4. Apply MinMaxScaler normalization to appliance power only
    5. Keep sin/cos temporal features unchanged
    6. Save to output CSV (appliance + 8 time features, no aggregate)
"""

import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import yaml

# Helper to load config with relative path
script_dir = os.path.dirname(os.path.abspath(__file__))
# New Path: Pointing to high_low_freq_NILM/config/preprocess/ukdale.yaml
CONFIG_PATH = os.path.normpath(os.path.join(script_dir, '..', 'config', 'preprocess', 'ukdale.yaml'))

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# Helper to map config appliance structure to script expectation
APPLIANCE_PARAMS = {}
for app_name, app_conf in CONFIG['appliances'].items():
    APPLIANCE_PARAMS[app_name] = {
        'on_power_threshold': app_conf['on_power_threshold'],
        'mean': app_conf['mean'],
        'std': app_conf['std'],
        'max_power': app_conf['max_power'],
        'max_power_clip': app_conf.get('max_power_clip'),
        'min_off_duration': app_conf.get('min_off_duration', 100), # Default 10 min
    }

def remove_isolated_spikes(power_sequence, window_size=5, spike_threshold=3.0, 
                          background_threshold=50):
    """
    Remove isolated spikes that suddenly appear when surrounding data is near zero.
    
    A spike is detected when:
    1. Current value is significantly higher than surrounding median
    2. Surrounding values are mostly near zero (below background_threshold)
    
    Args:
        power_sequence: 1D array of power values (in Watts)
        window_size: Size of window for median calculation (default: 5)
        spike_threshold: How many times larger than median to consider a spike (default: 3.0)
        background_threshold: Threshold to consider surrounding as "near zero" (default: 50W)
    
    Returns:
        power_sequence with isolated spikes removed (set to 0)
        num_spikes_removed: Number of spikes detected and removed
    """
    power_sequence = power_sequence.copy()
    n = len(power_sequence)
    num_spikes = 0
    
    # Pad array for edge handling
    half_window = window_size // 2
    padded = np.pad(power_sequence, half_window, mode='edge')
    
    for i in range(n):
        current_value = power_sequence[i]
        
        # Skip if current value is already low
        if current_value < background_threshold:
            continue
        
        # Get surrounding values (excluding center point)
        window_start = i
        window_end = i + window_size
        window = padded[window_start:window_end]
        
        # Calculate median of surrounding values (excluding center)
        surrounding = np.concatenate([window[:half_window], window[half_window+1:]])
        median_surrounding = np.median(surrounding)
        
        # Check if surrounding is mostly near zero
        low_values_count = np.sum(surrounding < background_threshold)
        is_background_low = low_values_count >= (len(surrounding) * 0.6)  # 60% threshold
        
        # Detect spike: current value is much higher than surrounding AND surrounding is low
        if is_background_low and current_value > spike_threshold * median_surrounding:
            # Additional check: current value should be significantly above background
            if current_value > background_threshold * 2:
                # Isolated spike detected - remove it
                power_sequence[i] = 0
                num_spikes += 1
    
    return power_sequence, num_spikes


def algorithm1_data_cleaning_multivariate(df, appliance_col, x_threshold, l_window=100, x_noise=0,
                                          remove_spikes=True, spike_window=5, spike_threshold=3.0,
                                          background_threshold=50, clip_max=None, max_power=None,
                                          min_off_duration=1, min_on_duration=1):
    """
    Algorithm 1: Data Cleaning and Selection for Multivariate Appliance Power Data
    
    Input:
        df: DataFrame with columns [aggregate, appliance, minute_sin, minute_cos, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos]
        appliance_col: name of the appliance power column
        x_threshold: appliance start threshold (Watts)
        l_window: window length (default: 100, from paper Table 2)
        x_noise: power noise threshold (default: 0)
        remove_spikes: whether to remove isolated spikes (default: True)
        spike_window: window size for spike detection (default: 5)
        spike_threshold: threshold multiplier for spike detection (default: 3.0)
        background_threshold: threshold to consider background as "low" (default: 50W)
        clip_max: optional maximum value to clip outliers (default: None)
    
    Output:
        filtered DataFrame with all columns preserved, power columns normalized
        selected indices
    """
    power_sequence = df[appliance_col].values.copy()
    
    # Step 0: Remove isolated spikes (optional preprocessing)
    if remove_spikes:
        power_sequence, num_spikes = remove_isolated_spikes(
            power_sequence, 
            window_size=spike_window,
            spike_threshold=spike_threshold,
            background_threshold=background_threshold
        )
        print(f"  Spike removal: {num_spikes} isolated spikes detected and removed")
    
    # Step 1: Initialize T_selected as an empty list
    T_selected = []
    
    # Step 2: x[x < x_noise] = 0
    power_sequence[power_sequence < x_noise] = 0
    
    # Step 3: Thresholding (Raw ON points)
    is_on = (power_sequence >= x_threshold).astype(int)
    
    # Step 4: Close Gaps (Bridge OFF periods shorter than min_off_duration)
    # This is equivalent to NILMTK's 'closing' logic
    if np.any(is_on):
        on_indices = np.where(is_on)[0]
        for i in range(len(on_indices) - 1):
            gap = on_indices[i+1] - on_indices[i]
            if gap > 1 and gap <= min_off_duration:
                # Close the gap
                is_on[on_indices[i]:on_indices[i+1]] = 1

    # Step 5: Filter Short Activations (Remove ON segments shorter than min_on_duration)
    if np.any(is_on):
        # Find start and end of each ON segment
        diff = np.diff(np.concatenate([[0], is_on, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for s, e in zip(starts, ends):
            duration = e - s
            if duration < min_on_duration:
                is_on[s:e] = 0

    # Step 6: Expand windows (l_window) around the final valid activations
    final_is_selected = np.zeros_like(is_on)
    if np.any(is_on):
        valid_on_indices = np.where(is_on)[0]
        for t in valid_on_indices:
            start_win = max(0, t - l_window)
            end_win = min(len(is_on), t + l_window + 1)
            final_is_selected[start_win:end_win] = 1
            
    T_selected = np.where(final_is_selected)[0].tolist()
    
    # Step 10: Select rows based on T_selected indices
    df_selected = df.iloc[T_selected].copy()
    
    # Step 10.5: Clip outliers if specified (only for appliance power)
    if clip_max is not None:
        num_clipped = np.sum(df_selected[appliance_col] > clip_max)
        df_selected[appliance_col] = np.clip(df_selected[appliance_col], 0, clip_max)
        if num_clipped > 0:
            print(f"  Clipped {num_clipped} values in '{appliance_col}' above {clip_max}W ({num_clipped/len(df_selected)*100:.2f}%)")
    
    # Step 11-12: Apply MinMax normalization using fixed max_power
    # Use fixed max power from params to ensure 1.0 = Max Power (e.g., 2000W)
    # This matches the denormalization logic in real_power_visualize.py
    
    if max_power is None:
        # Fallback to dynamic if not provided (though should be provided)
        print("  WARNING: max_power not provided, using dynamic max")
        max_power = df_selected[appliance_col].max()

    print(f"  Normalizing using fixed max_power: {max_power} W")
    
    # Clip values to max_power to ensure range [0, 1]
    df_selected[appliance_col] = df_selected[appliance_col].clip(upper=max_power)
    
    # Normalize: value / max_power
    df_selected[appliance_col] = df_selected[appliance_col] / max_power
    
    # Select output columns
    # If the input was simple (like ukdale_processing output), keep those columns
    # Otherwise, use the multivariate 9-column format
    original_cols = df.columns.tolist()
    multivariate_cols = [appliance_col] + ['minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 
                                          'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    
    # Check if we should use original format or multivariate
    is_multivariate = any(c in original_cols for c in multivariate_cols[1:])
    
    if not is_multivariate:
        # Keep original columns (e.g., time, aggregate, appliance)
        cols_to_keep = original_cols
    else:
        # Use multivariate columns
        cols_to_keep = [c for c in multivariate_cols if c in df_selected.columns]

    df_output = df_selected[cols_to_keep].copy()
    
    # Sin/Cos temporal features remain unchanged
    
    # Return None for scaler since we are using fixed max_power
    return df_output, T_selected, None

def plot_data_processing(power_data_original, x_cleaned, 
                        x_threshold, appliance_name, output_dir, max_samples=None):
    """
    Plot three informative graphs showing Algorithm 1's effect:
    1. Original data with threshold and selected regions highlighted
    2. Zoomed view of a startup event
    3. Final selected and normalized data
    
    Args:
        max_samples: Maximum number of samples to plot for overview. If None, plot all data.
    """
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Algorithm 1 Data Processing: {appliance_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    # Find startup events for highlighting
    startup_indices = np.where(power_data_original >= x_threshold)[0]
    
    # ============ Plot 1: Full original data with highlighted regions ============
    ax1 = fig.add_subplot(gs[0, :])
    
    # Determine sample size for overview
    if max_samples is None:
        sample_size = len(power_data_original)
    else:
        sample_size = min(max_samples, len(power_data_original))
    
    indices = np.arange(sample_size)
    
    # Plot original data
    ax1.plot(indices, power_data_original[:sample_size], 'b-', linewidth=0.5, alpha=0.6, label='Original data')
    
    # Highlight startup regions
    if len(startup_indices) > 0:
        startup_in_range = startup_indices[startup_indices < sample_size]
        if len(startup_in_range) > 0:
            ax1.scatter(startup_in_range, power_data_original[startup_in_range], 
                       c='red', s=1, alpha=0.5, label='Startup events (≥ threshold)')
    
    # Threshold line
    ax1.axhline(y=x_threshold, color='r', linestyle='--', linewidth=2, 
                label=f'Threshold: {x_threshold} W')
    
    ax1.set_title(f'Step 1: Original Data (Z-score denormalized to Watts)', 
                  fontweight='bold', fontsize=12)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Power (Watts)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Statistics box
    on_percentage = (len(startup_indices) / len(power_data_original)) * 100
    stats_text = f'Total samples: {len(power_data_original):,}\n'
    stats_text += f'Range: [{power_data_original.min():.0f}, {power_data_original.max():.0f}] W\n'
    stats_text += f'Startup events: {len(startup_indices):,} ({on_percentage:.2f}%)'
    ax1.text(0.02, 0.98, stats_text,
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # ============ Plot 2: Zoomed view of a startup event ============
    ax2 = fig.add_subplot(gs[1, 0])
    
    if len(startup_indices) > 0:
        # Find a good startup event to zoom into (middle of the data)
        mid_idx = len(startup_indices) // 2
        center = startup_indices[mid_idx]
        window = 500  # Show ±500 samples around startup
        
        start_idx = max(0, center - window)
        end_idx = min(len(power_data_original), center + window)
        
        zoom_indices = np.arange(start_idx, end_idx)
        zoom_data = power_data_original[start_idx:end_idx]
        
        ax2.plot(zoom_indices, zoom_data, 'b-', linewidth=1, label='Power')
        ax2.axhline(y=x_threshold, color='r', linestyle='--', linewidth=2, 
                   label=f'Threshold: {x_threshold} W')
        
        # Highlight the startup event
        startup_in_zoom = startup_indices[(startup_indices >= start_idx) & (startup_indices < end_idx)]
        if len(startup_in_zoom) > 0:
            ax2.scatter(startup_in_zoom, power_data_original[startup_in_zoom], 
                       c='red', s=20, alpha=0.7, label='Startup', zorder=5)
        
        ax2.set_title(f'Step 2: Zoomed View of Startup Event', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Power (Watts)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        zoom_text = f'Window: ±{window} samples\nCenter: index {center}'
        ax2.text(0.02, 0.98, zoom_text,
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    else:
        ax2.text(0.5, 0.5, 'No startup events found\n(all data below threshold)',
                ha='center', va='center', fontsize=12, color='red')
        ax2.set_title('Step 2: Zoomed View (No Events)', fontweight='bold', fontsize=12)
    
    # ============ Plot 3: Distribution comparison ============
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Create histograms
    ax3.hist(power_data_original, bins=50, alpha=0.5, label='Original', color='blue', density=True)
    
    # For cleaned data, we need to denormalize it back to see the distribution
    # But we don't have the scaler, so we'll just show it's focused on high power
    ax3.axvline(x=x_threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {x_threshold} W')
    
    ax3.set_title('Step 3: Power Distribution', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Power (Watts)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    dist_text = f'Original: mostly OFF state\nAlgorithm 1: keeps ON state'
    ax3.text(0.98, 0.98, dist_text,
            transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
    
    # ============ Plot 4: Final selected data (MinMax normalized) ============
    ax4 = fig.add_subplot(gs[2, :])
    
    # Show first N samples of cleaned data
    sample_size_cleaned = min(5000, len(x_cleaned))
    indices_cleaned = np.arange(sample_size_cleaned)
    
    ax4.plot(indices_cleaned, x_cleaned[:sample_size_cleaned], 
            'g-', linewidth=0.5, alpha=0.7)
    ax4.set_title(f'Step 4: Final Output - Selected & MinMax Normalized [0,1] (First {sample_size_cleaned:,} samples)', 
                  fontweight='bold', fontsize=12)
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Power (Normalized 0-1)')
    ax4.grid(True, alpha=0.3)
    
    retention_rate = len(x_cleaned) / len(power_data_original) * 100
    removed_samples = len(power_data_original) - len(x_cleaned)
    
    final_text = f'Selected samples: {len(x_cleaned):,}\n'
    final_text += f'Removed samples: {removed_samples:,}\n'
    final_text += f'Retention rate: {retention_rate:.2f}%\n'
    final_text += f'Range: [{x_cleaned.min():.4f}, {x_cleaned.max():.4f}]'
    
    ax4.text(0.02, 0.98, final_text,
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Show plot
    plt.show()

def main():
    """
    Apply Algorithm 1 to multivariate TRAINING data (as per paper requirement).
    
    According to the paper (Section 4.1): 
    "When synthesizing data, we execute Algorithm 1 on the training data, 
    send it to the diffusion model for synthetic data training..."
    """
    parser = argparse.ArgumentParser(
        description='Apply Algorithm 1 to multivariate TRAINING data for diffusion model')
    parser.add_argument('--appliance_name', type=str, required=True,
                        help='Appliance name: microwave, fridge, dishwasher, washingmachine, kettle')
    parser.add_argument('--input_file', type=str, 
                        default=None,
                        help='Input CSV file path (if not provided, uses training file from multivariate preprocessing)')
    parser.add_argument('--output_dir', type=str,
                        default='Data/datasets',
                        help='Output directory for processed CSV files')
    parser.add_argument('--window', type=int, default=CONFIG['algorithm1']['window_length'],
                        help=f"Window length (default: {CONFIG['algorithm1']['window_length']})")
    parser.add_argument('--remove_spikes', action='store_true', default=CONFIG['algorithm1']['remove_spikes'],
                        help=f"Remove isolated spikes (default: {CONFIG['algorithm1']['remove_spikes']})")
    parser.add_argument('--no_remove_spikes', action='store_false', dest='remove_spikes',
                        help='Disable spike removal')
    parser.add_argument('--spike_window', type=int, default=CONFIG['algorithm1']['spike_window'],
                        help=f"Window size for spike detection (default: {CONFIG['algorithm1']['spike_window']})")
    parser.add_argument('--spike_threshold', type=float, default=CONFIG['algorithm1']['spike_threshold'],
                        help=f"Spike threshold multiplier (default: {CONFIG['algorithm1']['spike_threshold']})")
    parser.add_argument('--background_threshold', type=float, default=CONFIG['algorithm1']['background_threshold'],
                        help=f"Background threshold in Watts (default: {CONFIG['algorithm1']['background_threshold']}W)")
    parser.add_argument('--clip_max', type=float, default=None,
                        help=f"Optional: Clip values above this maximum (default: {CONFIG['algorithm1']['clip_max']})")
    
    args = parser.parse_args()

    appliance_name = args.appliance_name.lower()
    
    if appliance_name not in APPLIANCE_PARAMS:
        raise ValueError(f"Unknown appliance: {appliance_name}. Must be one of: {list(APPLIANCE_PARAMS.keys())}")
    
    # Get appliance parameters
    params = APPLIANCE_PARAMS[appliance_name]

    # Determine clip_max: prioritize command line > appliance config > global config (args default)
    # Since args.clip_max defaults to global config, we need to check if we should prefer appliance specific.
    # Logic: If appliance specific clip exists, use it. But what if user explicitly passed --clip_max?
    # Ideally, specific appliance config should override default, but explicit input overrides everything.
    # Simplifying: If args.clip_max == CONFIG['algorithm1']['clip_max'] AND appliance has specific clip, use appliance logic.
    
    # Better logic: Use appliance specific clip if it exists, otherwise fallback to global.
    # Argparse default makes this tricky. We'll set default to None in argparse for clip_max 
    # and handle the defaulting logic here.
    
    clip_max = args.clip_max
    if clip_max is None:
        # Check appliance specific first
        clip_max = params.get('max_power_clip')
        
        # If still None, check global config
        if clip_max is None:
             clip_max = CONFIG['algorithm1']['clip_max']

    if clip_max is not None:
        print(f"  Configuration: using clip_max = {clip_max} W")

    x_threshold = params['on_power_threshold']
    min_off_duration = params.get('min_off_duration', 100)
    
    # Determine input file (multivariate CSV)
    if args.input_file is None:
        # Default: use training file from created_data directory
        input_file = f'created_data/UK_DALE/{appliance_name}_training_.csv'
    else:
        input_file = args.input_file
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Training file not found: {input_file}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'{appliance_name}_multivariate.csv')
    
    # Process training data
    print(f"\n{'='*60}")
    print(f"Applying Algorithm 1 to multivariate TRAINING data: {appliance_name}")
    print(f"{'='*60}")
    print(f"Reading: {input_file}")
    print(f"  Expected format: CSV with 10 columns (aggregate, appliance, 8 sin/cos time features)")
    
    # Read multivariate CSV with headers
    df = pd.read_csv(input_file)
    
    print(f"  CSV columns: {df.columns.tolist()}")
    print(f"  CSV shape: {df.shape}")
    print(f"  Original data length: {len(df):,}")
    
    # Verify expected columns
    expected_cols = ['aggregate', appliance_name, 'minute_sin', 'minute_cos',
                     'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']

    # Handle synthetic data case (no aggregate, maybe 'power' instead of appliance_name)
    if 'power' in df.columns and appliance_name not in df.columns:
        print(f"  Note: Renaming 'power' column to '{appliance_name}'")
        df.rename(columns={'power': appliance_name}, inplace=True)
        
    current_cols = list(df.columns)
    
    # Check if aggregate is missing (common in synthetic data)
    has_aggregate = 'aggregate' in df.columns
    
    if not has_aggregate:
        print("  Note: 'aggregate' column missing (Typical for synthetic data). Skipping aggregate checks.")
        # Expected cols without aggregate
        expected_cols = [c for c in expected_cols if c != 'aggregate']
    
    if current_cols != expected_cols:
        # Check if it's just a subset (e.g. extra index or missing optional)
        missing = [c for c in expected_cols if c not in current_cols]
        if missing:
            print(f"\n  WARNING: Missing expected columns: {missing}")
            print(f"  Got: {current_cols}")
    
    # Show data ranges
    print(f"\n  Data ranges:")
    if has_aggregate:
        print(f"    Aggregate (Z-score): [{df['aggregate'].min():.4f}, {df['aggregate'].max():.4f}]")
    print(f"    {appliance_name.capitalize()} (Z-score): [{df[appliance_name].min():.4f}, {df[appliance_name].max():.4f}]")
    print(f"    Time features found: {available_time_cols if 'available_time_cols' in locals() else 'None'}")
    
    # Denormalize appliance power for Algorithm 1
    mean = params['mean']
    std = params['std']
    print(f"\nDenormalizing appliance power (Z-score inverse):")
    print(f"  Mean: {mean} W, Std: {std} W")
    
    df_denorm = df.copy()
    df_denorm[appliance_name] = df[appliance_name] * std + mean
    # Note: aggregate column is not denormalized as it will be removed in output
    
    print(f"  Denormalized {appliance_name} range: [{df_denorm[appliance_name].min():.2f}, {df_denorm[appliance_name].max():.2f}] W")
    
    # Apply Algorithm 1
    print(f"\nApplying Algorithm 1:")
    print(f"  Threshold: {x_threshold} W")
    print(f"  Window length: {args.window}")
    print(f"  Spike removal: {'Enabled' if args.remove_spikes else 'Disabled'}")
    
    df_filtered, T_selected, scaler_app = algorithm1_data_cleaning_multivariate(
        df_denorm,
        appliance_col=appliance_name,
        x_threshold=x_threshold,
        l_window=args.window,
        remove_spikes=args.remove_spikes,
        spike_window=args.spike_window,
        spike_threshold=args.spike_threshold,
        background_threshold=args.background_threshold,
        clip_max=clip_max,
        max_power=params['max_power'],
        min_off_duration=params.get('min_off_duration', CONFIG['algorithm1'].get('min_off_duration_default', 10)),
        min_on_duration=params.get('min_on_duration', CONFIG['algorithm1'].get('min_on_duration_default', 1))
    )
    
    print(f"\n  Selected data length: {len(df_filtered):,}")
    print(f"  Reduction: {len(df) - len(df_filtered):,} samples removed")
    print(f"  Retention rate: {len(df_filtered)/len(df)*100:.2f}%")
    
    # Show filtered data ranges
    print(f"\n  Filtered data ranges:")
    print(f"    {appliance_name.capitalize()} (MinMax): [{df_filtered[appliance_name].min():.4f}, {df_filtered[appliance_name].max():.4f}]")
    
    # Only print ranges for time columns that actually exist
    for col in ['minute_sin', 'hour_sin', 'dow_sin', 'month_sin']:
        if col in df_filtered.columns:
            print(f"      {col}: [{df_filtered[col].min():.4f}, {df_filtered[col].max():.4f}]")
    
    # Calculate selected duration (assuming 6s sampling)
    sample_seconds = CONFIG['global_params'].get('sample_seconds', 6)
    total_seconds = len(df_filtered) * sample_seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    # Calculate number of independent "Real Cycles" 
    # Logic: Use the appliance-specific min_off_duration to bridge gaps
    natural_gap = params.get('min_off_duration', 100)
    num_cycles = 0
    
    # Identify indices where power is naturally above threshold
    raw_power = df_denorm[appliance_name].values
    raw_on_indices = np.where(raw_power >= x_threshold)[0]
    
    if len(raw_on_indices) > 0:
        num_cycles = 1
        for i in range(len(raw_on_indices) - 1):
            # If the gap between two actual ON points is > natural_gap
            if raw_on_indices[i+1] - raw_on_indices[i] > natural_gap:
                num_cycles += 1

    # Save CSV with appropriate headers
    df_filtered.to_csv(output_file, index=False, header=True)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS: Algorithm 1 processing complete!")
    print(f"{'='*60}")
    print(f"  Saved: {output_file}")
    print(f"  Rows: {len(df_filtered):,}")
    print(f"  Total Duration: {hours}h {minutes}m ({total_seconds:,} seconds)")
    print(f"  Detected Cycles: {num_cycles} (bridging gaps < {natural_gap} samples)")
    print(f"  Columns: {df_filtered.columns.tolist()}")
    print(f"  Appliance power: MinMax normalized [0,1]")
    print(f"  Time features: Sin/Cos encoded, preserved from input")
    print(f"\n  Note: Only TRAINING data processed (as per paper requirement)")

if __name__ == '__main__':
    main()

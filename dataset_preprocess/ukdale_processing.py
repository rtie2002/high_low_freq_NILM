import pandas as pd
import time
import argparse
import numpy as np
import os
import yaml

def get_arguments():
    # Calculate base path relative to this script (script is in dataset_preprocess/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, 'config', 'preprocess', 'ukdale.yaml')

    parser = argparse.ArgumentParser(description='sequence to point learning example for NILM')
    parser.add_argument('--config', type=str, default=default_config,
                          help='The path to the config file')
    parser.add_argument('--appliance_name', type=str, default='all',
                          help='Appliance to process. Use "all" to process all appliances defined in config.')
    return parser.parse_args()

def load_dataframe(directory, building, channel, col_names=['time', 'data'], nrows=None):
    file_path = os.path.join(directory, 'house_' + str(building), 'channel_' + str(channel) + '.dat')
    df = pd.read_table(file_path,
                       sep="\s+",
                       nrows=nrows,
                       usecols=[0, 1],
                       names=col_names,
                       dtype={'time': str},
                       )
    return df

def remove_isolated_spikes(power_sequence, window_size=5, spike_threshold=3.0, 
                          background_threshold=50):
    """Exact spike removal logic from algorithm1_v2_multivariate.py"""
    power_sequence = power_sequence.copy()
    n = len(power_sequence)
    num_spikes = 0
    half_window = window_size // 2
    padded = np.pad(power_sequence, half_window, mode='edge')
    
    for i in range(n):
        current_value = power_sequence[i]
        if current_value < background_threshold:
            continue
        window_start = i
        window_end = i + window_size
        window = padded[window_start:window_end]
        surrounding = np.concatenate([window[:half_window], window[half_window+1:]])
        median_surrounding = np.median(surrounding)
        low_values_count = np.sum(surrounding < background_threshold)
        is_background_low = low_values_count >= (len(surrounding) * 0.6)
        
        if is_background_low and current_value > spike_threshold * median_surrounding:
            if current_value > background_threshold * 2:
                power_sequence[i] = 0
                num_spikes += 1
    return power_sequence, num_spikes

def apply_algorithm1_labeling(power_sequence, x_threshold, l_window=100, x_noise=0,
                             remove_spikes=True, spike_window=5, spike_threshold=3.0,
                             background_threshold=50, min_off_duration=1, min_on_duration=1):
    """
    Exact Algorithm 1 steps from algorithm1_v2_multivariate.py,
    returning the final mask as the ON/OFF label.
    """
    sequence = power_sequence.copy()
    
    # Step 0: Spike Removal
    if remove_spikes:
        sequence, _ = remove_isolated_spikes(sequence, spike_window, spike_threshold, background_threshold)
    
    # Step 2: Noise Floor
    sequence[sequence < x_noise] = 0
    
    # Step 3: Thresholding
    is_on = (sequence >= x_threshold).astype(int)
    
    # Step 4: Close Gaps
    if np.any(is_on):
        on_indices = np.where(is_on)[0]
        for i in range(len(on_indices) - 1):
            gap = on_indices[i+1] - on_indices[i]
            if gap > 1 and gap <= min_off_duration:
                is_on[on_indices[i]:on_indices[i+1]] = 1

    # Step 5: Filter Short Activations
    if np.any(is_on):
        diff = np.diff(np.concatenate([[0], is_on, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) < min_on_duration:
                is_on[s:e] = 0

    # Step 6: Expand windows (Window Expansion)
    final_is_selected = np.zeros_like(is_on)
    if np.any(is_on):
        valid_on_indices = np.where(is_on)[0]
        for t in valid_on_indices:
            start_win = max(0, t - l_window)
            end_win = min(len(is_on), t + l_window + 1)
            final_is_selected[start_win:end_win] = 1
            
    return final_is_selected

def main():
    args = get_arguments()
    
    # Load configuration from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    paths = config['paths']
    global_params = config['global_params']
    params_appliance = config['appliances']
    
    # Determine Project Root for path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Resolution: Convert relative paths in YAML to absolute paths
    for key in ['data_dir', 'save_path']:
        if not os.path.isabs(paths[key]):
            paths[key] = os.path.normpath(os.path.join(project_root, paths[key]))

    appliance_name = args.appliance_name

    # Determine the list of appliances to process
    if appliance_name == 'all':
        appliances_to_run = global_params['appliances_to_process']
    else:
        if appliance_name not in params_appliance:
            raise ValueError(f"Unknown appliance: '{appliance_name}'. Valid options: {list(params_appliance.keys())}")
        appliances_to_run = [appliance_name]

    for appliance_name in appliances_to_run:
        print(f"\n{'='*40}")
        print(f"Processing appliance: {appliance_name}")
        print(f"{'='*40}")
        _process_appliance(appliance_name, paths, global_params, params_appliance, config)

def _process_appliance(appliance_name, paths, global_params, params_appliance, config):
    """Process and save CSV data for a single appliance."""
    # Ensure the entire directory tree exists
    os.makedirs(paths['save_path'], exist_ok=True)

    start_time = time.time()
    sample_seconds = global_params['sample_seconds']
    validation_percent = global_params['validation_percent']
    testing_percent = global_params['testing_percent']

    # BIG SPEEDUP 2: O(1) List appending avoids O(N^2) memory copying bottlenecks
    train_houses_list = []

    # Priority: Appliance-level 'houses' > Global-level 'houses'
    houses = params_appliance[appliance_name].get('houses', global_params.get('houses', [1]))
    channel_map = params_appliance[appliance_name].get('channel_map', {})

    for h in houses:
        channel_id = channel_map.get(h)
        
        if channel_id is None:
            print(f"  !! [SKIP] House {h} not found in channel_map for {appliance_name}")
            continue
            
        print(f"Loading House {h} (Channel {channel_id})...")
        
        # Use numeric filtering for speed (filter BEFORE datetime conversion)
        start_t = global_params.get('start_time')
        end_t = global_params.get('end_time')
        target_tz = global_params.get('timezone', 'UTC')
        
        # Convert config times to float Unix timestamps, respecting the timezone
        if start_t:
            start_ts = pd.to_datetime(start_t).tz_localize(target_tz).timestamp()
        else:
            start_ts = None
            
        if end_t:
            end_ts = pd.to_datetime(end_t).tz_localize(target_tz).timestamp()
        else:
            end_ts = None

        # 1. Load Mains (Memory Optimized Version)
        mains_path = os.path.join(paths['data_dir'], f"house_{h}", "mains.dat")
        
        # === Progress Tracker ===
        print(f"  -> [Step 1/4] Loading Mains datastream... (This file is massive, might take 10-60s. Please wait.)")
        t0 = time.time()
        
        mains_df = pd.read_csv(mains_path, sep='\s+', header=None, engine='c')
        
        # House 1 has 2 mains (cols 1,2), House 2 has 1 mains (col 1)
        if mains_df.shape[1] >= 3:
            mains_df['aggregate'] = mains_df[1] + mains_df[2]
        else:
            mains_df['aggregate'] = mains_df[1]
            
        mains_df = mains_df[[0, 'aggregate']]
        mains_df.columns = ['time', 'aggregate']
        
        # Ultra-fast Duduplication (Dropping duplicates as raw integers saves massive RAM)
        mains_df.drop_duplicates(subset=['time'], keep='first', inplace=True)
        
        # Numeric Filter
        if start_ts: mains_df = mains_df[mains_df['time'] >= start_ts]
        if end_ts:   mains_df = mains_df[mains_df['time'] <= end_ts]
        
        # Convert subset to datetime and localize
        mains_df['time'] = pd.to_datetime(mains_df['time'], unit='s', utc=True).dt.tz_convert(target_tz)
        mains_df.set_index('time', inplace=True)
        mains_df.sort_index(inplace=True)
        
        print(f"  -> [Step 1/4 Done] Loaded Mains: {mains_df.index.min()} to {mains_df.index.max()} ({len(mains_df)} lines)")

        # 2. Load Appliance (Memory Optimized)
        app_path = os.path.join(paths['data_dir'], f"house_{h}", f"channel_{channel_id}.dat")
        
        print(f"  -> [Step 2/4] Loading Appliance submeter: channel_{channel_id}.dat...")
        t1 = time.time()
        
        app_df = pd.read_csv(app_path, sep='\s+', header=None,
                             usecols=[0, 1],
                             dtype={0: np.float64, 1: np.float32},
                             engine='c')
        app_df.columns = ['time', appliance_name]
        
        app_df.drop_duplicates(subset=['time'], keep='first', inplace=True)
        
        # Numeric Filter
        if start_ts: app_df = app_df[app_df['time'] >= start_ts]
        if end_ts:   app_df = app_df[app_df['time'] <= end_ts]

        # Convert subset to datetime and localize
        app_df['time'] = pd.to_datetime(app_df['time'], unit='s', utc=True).dt.tz_convert(target_tz)
        app_df.set_index('time', inplace=True)
        app_df.sort_index(inplace=True)
        
        print(f"  -> [Step 2/4 Done] Loaded Appliance: {app_df.index.min()} to {app_df.index.max()} ({len(app_df)} lines)")

        # ==========================================
        # TIME FILTERING (ALREADY DONE NUMERICALLY ABOVE)
        # ==========================================




        if mains_df.empty or app_df.empty:
            print(f"Warning: No data found for {appliance_name} in house {h} for the selected timeframe.")
            continue
            
        print(f"  -> [Step 2/4 Done - {time.time() - t1:.1f}s] Successfully loaded Appliance submeter ({len(app_df)} lines).")

        # 3. Resample and Align (High-Speed Separated Strategy)
        print(f"  -> [Step 3/4] Aligning and Resampling timestamps... (This requires CPU computation)")
        t2 = time.time()
        sample_period = f"{sample_seconds}s" # Use lowercase 's' to avoid FutureWarning
        
        # Resample appliance to 6s grid first
        app_df_resampled = app_df.resample(sample_period).mean()
        
        # Outer join + bfill: Preserves continuous time series.
        # Mains and appliance channels have slightly different timestamps.
        # 'outer' keeps all timestamps; bfill(limit=1) fills minor gaps (1 step max).
        # This matches the original script behavior and avoids discontinuous output.
        df_align = mains_df.join(app_df_resampled, how='outer') \
            .resample(sample_period).mean() \
            .bfill(limit=1)
        df_align = df_align.dropna()
        
        if df_align.empty:
            print("  !! [ERROR] Zero overlap found between Mains and Appliance timelines!")
            print(f"     Mains Range: {mains_df.index.min()} -- {mains_df.index.max()}")
            print(f"     App   Range: {app_df.index.min()} -- {app_df.index.max()}")
        else:
            print(f"  -> [Step 3/4 Done] Successfully aligned grid. Found {len(df_align)} overlapping rows.")
        
        df_align.reset_index(inplace=True)
        
        # Physical constraint: appliance power must not exceed aggregate power
        df_align[appliance_name] = np.minimum(df_align[appliance_name], df_align['aggregate'])

        # ── Snapshot real-power BEFORE normalization ──────────────────────
        df_align_real = df_align.copy()

        # ── Algorithm 1 Labeling ──────────────────────────────────────────
        app_params = params_appliance[appliance_name]
        algo1_cfg = config.get('algorithm1', {})
        
        # Extract params for the labeler
        threshold = app_params.get('on_power_threshold', 50)
        min_on = app_params.get('min_on_duration', 1)
        min_off = app_params.get('min_off_duration', 1)
        window = algo1_cfg.get('window_length', 0)
        
        print(f"  -> [Algorithm 1] Labeling ON/OFF status (Thresh={threshold}W, Window={window})...")
        
        on_off_label = apply_algorithm1_labeling(
            df_align_real[appliance_name].values,
            x_threshold=threshold,
            l_window=window,
            x_noise=algo1_cfg.get('x_noise', 0),
            remove_spikes=algo1_cfg.get('remove_spikes', True),
            spike_window=algo1_cfg.get('spike_window', 5),
            spike_threshold=algo1_cfg.get('spike_threshold', 3.0),
            background_threshold=algo1_cfg.get('background_threshold', 50),
            min_off_duration=min_off,
            min_on_duration=min_on
        )
        
        df_align['on_off'] = on_off_label
        df_align_real['on_off'] = on_off_label

        # ── Z-score Normalization ──────────────────────────────────────────
        mean_app = params_appliance[appliance_name]['mean']
        std_app = params_appliance[appliance_name]['std']
        df_align['aggregate'] = (df_align['aggregate'] - global_params['aggregate_mean']) / global_params['aggregate_std']
        df_align[appliance_name] = (df_align[appliance_name] - mean_app) / std_app

        # Append to holding list (Costs exactly 0 milliseconds)
        train_houses_list.append((df_align, df_align_real))
        
        # Free memory forcibly before next house iteration
        del mains_df, app_df
        import gc; gc.collect()

    # === Post-Processing: Single Shot Concatenation ===
    output_mode = global_params.get('output_mode', 'zscore').lower()
    
    if train_houses_list:
        train       = pd.concat([x[0] for x in train_houses_list], ignore_index=True)  # z-score
        train_real  = pd.concat([x[1] for x in train_houses_list], ignore_index=True)  # real watts
    else:
        train      = pd.DataFrame()
        train_real = pd.DataFrame()
    del train_houses_list

    def _save(df, df_real, split_name):
        """Save zscore / real / both versions for one split."""
        key      = split_name           # 'test', 'val', 'train'
        key_real = split_name + '_real' # 'test_real', 'val_real', 'train_real'
        if output_mode in ('zscore', 'both'):
            fname = paths['naming'][key].format(appliance=appliance_name)
            df.to_csv(os.path.join(paths['save_path'], fname), index=False, header=True)
            print(f"  -> [Z-score] Saved {split_name}: {len(df)} rows  →  {fname}")
        if output_mode in ('real', 'both'):
            fname_r = paths['naming'][key_real].format(appliance=appliance_name)
            df_real.to_csv(os.path.join(paths['save_path'], fname_r), index=False, header=True)
            print(f"  -> [Real W ] Saved {split_name}: {len(df_real)} rows  →  {fname_r}")

    # Split dataset
    test_len = int((len(train)/100)*testing_percent)
    val_len  = int((len(train)/100)*validation_percent)

    # Testing Set
    test      = train.tail(test_len).reset_index(drop=True)
    test_real = train_real.tail(test_len).reset_index(drop=True)
    if test_len > 0:
        train.drop(train.index[-test_len:], inplace=True)
        train_real.drop(train_real.index[-test_len:], inplace=True)
        _save(test, test_real, 'test')

    # Validation Set
    val      = train.tail(val_len).reset_index(drop=True)
    val_real = train_real.tail(val_len).reset_index(drop=True)
    if val_len > 0:
        train.drop(train.index[-val_len:], inplace=True)
        train_real.drop(train_real.index[-val_len:], inplace=True)
        _save(val, val_real, 'val')

    # Training Set
    if len(train) > 0:
        _save(train, train_real, 'train')

    print(f"Data processing finished.")
    print(f"Size of training set:  {len(train) / 10**6:.4f} M rows.")
    print(f"Total elapsed time: {(time.time() - start_time) / 60:.2f} min.")

if __name__ == '__main__':
    main()
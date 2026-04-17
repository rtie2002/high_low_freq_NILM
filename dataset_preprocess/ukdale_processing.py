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
        _process_appliance(appliance_name, paths, global_params, params_appliance)

def _process_appliance(appliance_name, paths, global_params, params_appliance):
    """Process and save CSV data for a single appliance."""
    # Ensure the entire directory tree exists
    os.makedirs(paths['save_path'], exist_ok=True)

    start_time = time.time()
    sample_seconds = global_params['sample_seconds']
    validation_percent = global_params['validation_percent']
    testing_percent = global_params['testing_percent']

    # BIG SPEEDUP 2: O(1) List appending avoids O(N^2) memory copying bottlenecks
    train_houses_list = []

    for h in params_appliance[appliance_name]['houses']:
        print(f"Loading House {h}...")
        
        # Use numeric filtering for speed (filter BEFORE datetime conversion)
        start_t = global_params.get('start_time')
        end_t = global_params.get('end_time')
        
        # Convert config times to float Unix timestamps
        start_ts = pd.to_datetime(start_t).timestamp() if start_t else None
        end_ts = pd.to_datetime(end_t).timestamp() if end_t else None

        # 1. Load Mains (Memory Optimized Version)
        mains_path = os.path.join(paths['data_dir'], f"house_{h}", "mains.dat")
        
        # === Progress Tracker ===
        print(f"  -> [Step 1/4] Loading Mains datastream... (This file is massive, might take 10-60s. Please wait.)")
        t0 = time.time()
        
        mains_df = pd.read_csv(mains_path, sep='\s+', header=None, 
                               usecols=[0, 1, 2], # UK-DALE House 1 has 2 mains channels! We need both.
                               dtype={0: np.float64, 1: np.float32, 2: np.float32}, 
                               engine='c')
        # Sum both mains channels to get the real aggregate power
        mains_df['aggregate'] = mains_df[1] + mains_df[2]
        mains_df = mains_df[[0, 'aggregate']]
        mains_df.columns = ['time', 'aggregate']
        
        # Ultra-fast Duduplication (Dropping duplicates as raw integers saves massive RAM)
        mains_df.drop_duplicates(subset=['time'], keep='first', inplace=True)
        
        # Numeric Filter
        if start_ts: mains_df = mains_df[mains_df['time'] >= start_ts]
        if end_ts:   mains_df = mains_df[mains_df['time'] <= end_ts]
        
        # Convert subset to datetime
        mains_df['time'] = pd.to_datetime(mains_df['time'], unit='s')
        mains_df.set_index('time', inplace=True)
        mains_df.sort_index(inplace=True)
        
        print(f"  -> [Step 1/4 Done] Loaded Mains: {mains_df.index.min()} to {mains_df.index.max()} ({len(mains_df)} lines)")

        # 2. Load Appliance (Memory Optimized)
        channel_id = params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)]
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

        # Convert subset to datetime
        app_df['time'] = pd.to_datetime(app_df['time'], unit='s')
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
        
        # BIG SPEEDUP: Resample individually
        mains_df_resampled = mains_df.resample(sample_period).mean()
        app_df_resampled = app_df.resample(sample_period).mean()
        
        # Use inner join to force finding ONLY overlapping timestamps
        df_align = mains_df_resampled.join(app_df_resampled, how='inner')
        
        if df_align.empty:
            print("  !! [ERROR] Zero overlap found between Mains and Appliance timelines!")
            print(f"     Mains Range: {mains_df.index.min()} -- {mains_df.index.max()}")
            print(f"     App   Range: {app_df.index.min()} -- {app_df.index.max()}")
        else:
            print(f"  -> [Step 3/4 Done] Successfully aligned grid. Found {len(df_align)} overlapping rows.")
        
        df_align.reset_index(inplace=True)
        
        # Physical constraint: appliance power must not exceed aggregate power
        df_align[appliance_name] = np.minimum(df_align[appliance_name], df_align['aggregate'])

        # Normalization
        mean_app = params_appliance[appliance_name]['mean']
        std_app = params_appliance[appliance_name]['std']

        df_align['aggregate'] = (df_align['aggregate'] - global_params['aggregate_mean']) / global_params['aggregate_std']
        df_align[appliance_name] = (df_align[appliance_name] - mean_app) / std_app

        # Append to holding list (Costs exactly 0 milliseconds)
        train_houses_list.append(df_align)
        
        # Free memory forcibly before next house iteration
        del mains_df, app_df
        import gc; gc.collect()

    # === Post-Processing: Single Shot Concatenation ===
    # Stitching all 5 houses together ONCE outside the loop saves exponential copying overhead.
    train = pd.concat(train_houses_list, ignore_index=True) if train_houses_list else pd.DataFrame()
    del train_houses_list

    # Split dataset
    test_len = int((len(train)/100)*testing_percent)
    val_len = int((len(train)/100)*validation_percent)

    # Testing Set
    test = train.tail(test_len)
    test.reset_index(drop=True, inplace=True)
    if test_len > 0:
        train.drop(train.index[-test_len:], inplace=True)
    test_file = paths['naming']['test'].format(appliance=appliance_name)
    test.to_csv(os.path.join(paths['save_path'], test_file), index=False, header=True)

    # Validation Set
    val = train.tail(val_len)
    val.reset_index(drop=True, inplace=True)
    if val_len > 0:
        train.drop(train.index[-val_len:], inplace=True)
    val_file = paths['naming']['val'].format(appliance=appliance_name)
    val.to_csv(os.path.join(paths['save_path'], val_file), index=False, header=True)

    # Training Set
    train_file = paths['naming']['train'].format(appliance=appliance_name)
    train.to_csv(os.path.join(paths['save_path'], train_file), index=False, header=True)

    print(f"Data processing finished.")
    print(f"Size of training set: {len(train) / 10**6:.4f} M rows.")
    print(f"Total elapsed time: {(time.time() - start_time) / 60:.2f} min.")

if __name__ == '__main__':
    main()
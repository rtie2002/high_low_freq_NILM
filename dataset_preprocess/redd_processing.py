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
    default_config = os.path.join(project_root, 'config', 'preprocess', 'redd.yaml')

    parser = argparse.ArgumentParser(description='REDD dataset preprocessing for NILM')
    parser.add_argument('--config', type=str, default=default_config,
                        help='Path to the REDD config file')
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
    if not os.path.exists(paths['save_path']):
        os.makedirs(paths['save_path'])

    start_time = time.time()
    sample_seconds = global_params['sample_seconds']
    validation_percent = global_params['validation_percent']
    testing_percent = global_params['testing_percent']

    # Initialize empty DFs with correct column names
    train = pd.DataFrame(columns=['time', 'aggregate', appliance_name])
    test = pd.DataFrame(columns=['time', 'aggregate', appliance_name])

    for h in params_appliance[appliance_name]['houses']:
        print(f"Loading House {h}...")

        # 1. Load Mains (REDD House 1 has channel 1 and 2 as mains)
        mains_df = None
        for m_chan in [1, 2]:
            path = os.path.join(paths['data_dir'], f"house_{h}", f"channel_{m_chan}.dat")
            df = pd.read_csv(path, sep=' ', header=None, names=['time', f'mains_{m_chan}'])
            df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='s')
            df.set_index('time', inplace=True)
            if mains_df is None:
                mains_df = df
            else:
                mains_df = mains_df.join(df, how='outer')
        
        mains_df['aggregate'] = mains_df['mains_1'] + mains_df['mains_2']
        mains_df = mains_df[['aggregate']]

        # 2. Load Appliance
        channel_id = params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)]
        app_path = os.path.join(paths['data_dir'], f"house_{h}", f"channel_{channel_id}.dat")
        app_df = pd.read_csv(app_path, sep=' ', header=None, names=['time', appliance_name])
        app_df['time'] = pd.to_datetime(pd.to_numeric(app_df['time']), unit='s')
        app_df.set_index('time', inplace=True)

        # ==========================================
        # TIME FILTERING
        # ==========================================
        start_date = config['global_params'].get('start_date')
        end_date = config['global_params'].get('end_date')

        if start_date:
            mains_df = mains_df.loc[start_date:]
            app_df = app_df.loc[start_date:]
        if end_date:
            mains_df = mains_df.loc[:end_date]
            app_df = app_df.loc[:end_date]

        if mains_df.empty or app_df.empty:
            print(f"Warning: No data found for {appliance_name} in house {h} for the selected timeframe.")
            continue

        # 3. Resample and Align
        sample_period = f"{sample_seconds}s"
        df_align = mains_df.join(app_df, how='outer'). \
            resample(sample_period).mean().bfill(limit=1)
        
        df_align = df_align.dropna()
        df_align.reset_index(inplace=True)

        # Physical constraint: appliance power must not exceed aggregate power
        df_align[appliance_name] = np.minimum(df_align[appliance_name], df_align['aggregate'])

        # Normalization
        mean_app = params_appliance[appliance_name]['mean']
        std_app = params_appliance[appliance_name]['std']
        df_align['aggregate'] = (df_align['aggregate'] - global_params['aggregate_mean']) / global_params['aggregate_std']
        df_align[appliance_name] = (df_align[appliance_name] - mean_app) / std_app

        # Separate test house from training houses
        if h == params_appliance[appliance_name]['test_build']:
            test = pd.concat([test, df_align], ignore_index=True)
        else:
            train = pd.concat([train, df_align], ignore_index=True)

        del mains1_df, mains2_df, mains_df, app_df, df_align

    # Split training into train + validation
    val_len = int((len(train) / 100) * validation_percent)
    val = train.tail(val_len)
    val.reset_index(drop=True, inplace=True)
    train.drop(train.index[-val_len:], inplace=True)

    # Save all splits using config-defined filenames
    test_file = paths['naming']['test'].format(appliance=appliance_name)
    val_file = paths['naming']['val'].format(appliance=appliance_name)
    train_file = paths['naming']['train'].format(appliance=appliance_name)

    test.to_csv(os.path.join(paths['save_path'], test_file), index=False, header=True)
    val.to_csv(os.path.join(paths['save_path'], val_file), index=False, header=True)
    train.to_csv(os.path.join(paths['save_path'], train_file), index=False, header=True)

    print(f"Data processing finished.")
    print(f"Training set: {len(train) / 10**6:.4f} M rows")
    print(f"Validation set: {len(val) / 10**6:.4f} M rows")
    print(f"Test set: {len(test) / 10**6:.4f} M rows")
    print(f"Total elapsed time: {(time.time() - start_time) / 60:.2f} min.")


if __name__ == '__main__':
    main()

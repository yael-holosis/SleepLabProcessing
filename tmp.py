import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.io import read_raw_edf
from tqdm import tqdm
from datetime import datetime, timedelta
import glob

def process_signal(edf_raw, channel_name, folder_path, std_th=2, y_min=None, y_max=None):
    # Get the data for specified channel
    signal_data = edf_raw.get_data(channel_name)[0]

    # Create dataframe with original data and exact timestamps
    start_time = edf_raw.info['meas_date']
    df = pd.DataFrame({
        'time': [start_time + timedelta(seconds=t) for t in edf_raw.times],
        'original_signal': signal_data
    })

    # Calculate mean and standard deviation
    median = np.median(df['original_signal'])
    mean = np.mean(df['original_signal'])
    std = np.std(df['original_signal'])
    
    # Filter out values exceeding std_th standard deviations from median
    df['filtered_signal'] = df['original_signal'].where(
        abs(df['original_signal'] - mean) <= std_th * std, np.nan)

    # Plot original signal
    plt.figure(figsize=(12, 4))
    plt.plot(df['time'], df['original_signal'], alpha=0.5)
    plt.title(f'Original {channel_name} Signal (median: {median:.2f}, min: {df["original_signal"].min():.2f}, max: {df["original_signal"].max():.2f})')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # Plot filtered signal
    plt.figure(figsize=(12, 4))
    plt.plot(df['time'], df['filtered_signal'])
    plt.title(f'Filtered {channel_name} Signal (±{std_th}σ), min: {df["filtered_signal"].min():.2f}, max: {df["filtered_signal"].max():.2f}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.show()

    # Save original and filtered data to CSV
    df.to_csv(f'{folder_path}/{channel_name.lower().replace(" ","_")}_data.csv', index=False)

    print(f"Data saved to: {folder_path}/{channel_name.lower().replace(' ','_')}_data.csv")
    total_points = len(df['filtered_signal'])
    removed_points = df['filtered_signal'].isna().sum()
    percentage_removed = (removed_points / total_points) * 100
    print(f"Removed {removed_points} points as outliers ({percentage_removed:.2f}%)")

# Iterate over all folders in the data directory
data_dir = 'data'
for folder in tqdm(os.listdir(data_dir), desc="Processing folders"):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):        
        print('Processing folder:', folder_path)
        edf_file = glob.glob(os.path.join(folder_path, '*.edf'))
        #edf_file = edf_file[0]
        if len(edf_file) == 0:
            print(f"No edf file found in {folder_path}")
            continue
        
        edf_file = edf_file[0]
        patient = edf_file.split('/')[-2]
        if os.path.exists(edf_file):
            edf_raw = read_raw_edf(edf_file, preload=True)
            process_signal(edf_raw, 'Pulse', folder_path, std_th=3, y_min=0, y_max=110)
            process_signal(edf_raw, 'BR flow', folder_path, std_th=3, y_min=0, y_max=30)


def calculate_awake_accuracy(df, awake_window, percentage_threshold):
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    # Calculate the threshold based on percentage of maximum value
    # Apply the percentage threshold directly
    threshold = percentage_threshold
    
    # Apply threshold to determine if signal is above threshold (1) or not (0)
    df_copy['AboveThreshold'] = (df_copy['PercentageCount'] > threshold).astype(int)
    
    # Calculate awake index based on time window in minutes
    awake_indices = []
    
    for idx, row in df_copy.iterrows():
        current_time = row['StartTime']
        # Define time window (current time +/- awake_window/2 minutes)
        window_start = current_time - pd.Timedelta(minutes=awake_window/2)
        window_end = current_time + pd.Timedelta(minutes=awake_window/2)
        
        # Get signals that fall within the time window
        window_signals = df_copy[
            (df_copy['StartTime'] >= window_start) & 
            (df_copy['StartTime'] <= window_end)
        ]
        
        # Calculate median of AboveThreshold values within the time window
        if len(window_signals) > 0:
            awake_index = window_signals['AboveThreshold'].median()
            # If the median is 0.5, set it to 1
            if awake_index == 0.5:
                awake_index = 1
        else:
            awake_index = 0
            
        awake_indices.append(awake_index)
    
    df_copy['AwakeIndex'] = awake_indices
    
    # Calculate confusion matrix metrics using sklearn for a shorter implementation
    from sklearn.metrics import confusion_matrix, accuracy_score
    
    # Filter out rows with NaN values in awake_gt
    valid_data = df_copy.dropna(subset=['awake_gt'])
    
    if len(valid_data) > 0:
        y_true = valid_data['awake_gt'].values
        y_pred = valid_data['AwakeIndex'].values
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        
        stats = {'Accuracy': accuracy, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    else:
        # Return empty stats if no valid data
        stats = {'Accuracy': float('nan'), 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    return stats

def calculate_awake_accuracy_ver2(df, awake_window, percentage_threshold, establish_baseline = True):
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    # Calculate the threshold based on percentage of maximum value
    # Apply the percentage threshold directly
    threshold = percentage_threshold
    
    # Calculate awake index based on time window in minutes
    median_percentage = []
    
    for idx, row in df_copy.iterrows():
        current_time = row['StartTime']
        # Define time window (current time +/- awake_window/2 minutes)
        window_start = current_time - pd.Timedelta(minutes=awake_window/2)
        window_end = current_time + pd.Timedelta(minutes=awake_window/2)
        
        # Get signals that fall within the time window
        window_signals = df_copy[
            (df_copy['StartTime'] >= window_start) & 
            (df_copy['StartTime'] <= window_end)
        ]
        
        # Calculate median of AboveThreshold values within the time window
        if len(window_signals) > 0:
            awake_index = window_signals['PercentageCount'].median()
        else:
            awake_index = 0
            
        median_percentage.append(awake_index)
    
    
    df_copy['MedianPercentage'] = median_percentage
    
    if establish_baseline:
        # print(f"Median percentage: {median_percentage}")
        print(f"Establishing baseline for {percentage_threshold} threshold")
        percentage_threshold = min(median_percentage) + percentage_threshold
        print(f"New threshold: {percentage_threshold}")
        
    df_copy['AwakeIndex'] = df_copy['MedianPercentage']>percentage_threshold
    
    # Calculate confusion matrix metrics using sklearn for a shorter implementation
    from sklearn.metrics import confusion_matrix, accuracy_score
    
    # Filter out rows with NaN values in awake_gt
    valid_data = df_copy.dropna(subset=['awake_gt'])
    
    if len(valid_data) > 0:
        y_true = valid_data['awake_gt'].values
        y_pred = valid_data['AwakeIndex'].values
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        
        stats = {'Accuracy': accuracy, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    else:
        # Return empty stats if no valid data
        stats = {'Accuracy': float('nan'), 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    return stats

from tqdm import tqdm
import pandas as pd
import os

results_folder = 'awake_analysis'
df_motion_with_gt = pd.read_csv(os.path.join(results_folder, 'motion_with_gt.csv'))
df_motion_with_gt['StartTime'] = pd.to_datetime(df_motion_with_gt['StartTime'])
window_times = range(2, 40, 2) 
establish_baseline = True
motion_percentage_thresholds = list(range(4, 30, 2))  
num_configs = len(window_times) * len(motion_percentage_thresholds)
df_results_per_session = pd.DataFrame(columns=['window_time', 'motion_percentage_th', 'session_id', 'accuracy', 'TP', 'TN', 'FP', 'FN'])
df_results = pd.DataFrame(columns=['window_time', 'motion_percentage_th', 'accuracy', 'TP', 'TN', 'FP', 'FN'])

with tqdm(total=num_configs, desc="Processing configs") as pbar:
    for window_time in window_times:
        for motion_percentage_th in motion_percentage_thresholds:
            config = {'window_time': window_time, 'motion_percentage_th': motion_percentage_th}
            for session_id in df_motion_with_gt['SessionID'].unique():
                df_session = df_motion_with_gt[df_motion_with_gt['SessionID'] == session_id].copy()
                patient = df_session['PatientStudyName'].unique()[0]
                stats = calculate_awake_accuracy_ver2(df_session, window_time, motion_percentage_th, establish_baseline)
                results = {**config, 'session_id': session_id, 'patient': patient, **stats}
                df_results_per_session = pd.concat([df_results_per_session, pd.DataFrame([results])], ignore_index=True)
            pbar.update(1)
            df_results_per_session.to_csv(os.path.join(results_folder, 'awake_accuracy_per_session.csv'), index=False)

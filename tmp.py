import os
import pandas as pd
import numpy as np

# Parameters
window_size = 10  # Number of samples to look before and after each sample

# Prepare data for linear regression
X = []  # Motion percentages from window
y = []  # Ground truth labels
groups = [] # Group by patient 

results_folder = 'awake_analysis'
df_motion_per_sample = pd.read_csv(os.path.join(results_folder, 'motion_per_sample_10_seconds.csv'))

# Iterate through sessions
for session_id in df_motion_per_sample['SessionID'].unique():
    # Get motion data for this session
    session_motion = df_motion_per_sample[df_motion_per_sample['SessionID'] == session_id].copy()
    
    # Skip if session has no data
    if session_motion.empty:
        continue
        
    patient_name = session_motion['PatientStudyName'].unique()[0]
    session_date = session_motion['SessionDate'].unique()[0]
    
    session_folder = os.path.join('data', f"{patient_name}_{session_date}")
    
    # Skip if sleep stages file doesn't exist
    if not os.path.exists(os.path.join(session_folder, 'sleep_stages.csv')):
        continue
        
    sleep_stages_df = pd.read_csv(os.path.join(session_folder, 'sleep_stages.csv'))
    sleep_stages_df['StartDateTime'] = pd.to_datetime(sleep_stages_df['StartDateTime'])
    sleep_stages_df['EndDateTime'] = pd.to_datetime(sleep_stages_df['EndDateTime'])
    
    # For each sample in the session
    for i in range(len(session_motion)):
        # Get window indices
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(session_motion), i + window_size//2 + 1)
        
        # Get motion percentages for window
        window_percentages = session_motion['MotionPercentage'].iloc[start_idx:end_idx].values
            
        # Pad with zeros if window is smaller than expected
        if len(window_percentages) < 2*window_size + 1:
            # Calculate padding needed on each side
            left_pad = max(0, window_size//2 - i)  # Pad at start if near beginning
            right_pad = max(0, (i + window_size//2 + 1) - len(session_motion))  # Pad at end if near end
            
            # Pad the array on both sides as needed
            window_percentages = np.pad(window_percentages, (left_pad, right_pad), 'constant')
            
        # Get sample start and end time
        sample_start = pd.to_datetime(session_motion['StartTime'].iloc[i])
        sample_end = pd.to_datetime(session_motion['EndTime'].iloc[i])
        
        # Find the sleep stage that contains the middle point of the sample
        sample_midpoint = sample_start + (sample_end - sample_start) / 2
        stage = sleep_stages_df[
            (sleep_stages_df['StartDateTime'] <= sample_midpoint) & 
            (sleep_stages_df['EndDateTime'] > sample_midpoint)
        ]
        
        # Skip if no sleep stage found
        if stage.empty:
            gt = np.nan
        else:
            gt = 1 if stage['Stage'].iloc[0] == 'Awake' else 0
            
        
        X.append(window_percentages)
        y.append(gt)
        groups.append(patient_name)

df_motion_per_sample['GroundTruth'] = y
X = np.array(X)
y = np.array(y)

import re
import os
import pandas as pd
from datetime import datetime, timedelta

# Create a dictionary to store artifact times for each session
artifact_dict = {}
results_with_ref = pd.read_csv('data/results_with_ref.csv')
# Go through all folders in data directory
for folder in os.listdir('data'):
    if os.path.isdir(os.path.join('data', folder)) and '_' in folder:
        # Extract patient and date from folder name
        patient, date = folder.split('_')
        
        # Look for artifact file
        artifact_file = os.path.join('data', folder, date[-8:] + '-' + patient[-4:])
        if os.path.exists(artifact_file):
            with open(artifact_file, 'r') as file:
                lines = file.readlines()
                
            session_artifacts = []
            
            flow_found = False
            for line in lines:
                if 'Events Channel' in line and flow_found:
                    break
                    
                if 'Events Channel' in line and 'Pulse' in line:
                    flow_found = True
                    
                if 'Artifact' in line and flow_found:
                    # Extract start time and duration
                    start_match = re.search(r'Start: (\d{2}:\d{2}:\d{2})', line)
                    duration_match = re.search(r'Duration \[ms\]: (\d+(?:\.\d+)?)', line)
                    
                    if start_match and duration_match:
                        start_time = start_match.group(1)
                        duration = float(duration_match.group(1)) / 1000.0  # Convert ms to seconds
                        
                        # Convert to datetime objects
                        start_dt = datetime.strptime(f"{date} {start_time}", "%Y%m%d %H:%M:%S")
                        end_dt = start_dt + timedelta(seconds=duration)
                        session_artifacts.append((start_dt, end_dt))
            
            artifact_dict[folder] = session_artifacts

# Add HasArtifact column to results_with_ref
results_with_ref['HasArtifact'] = False

# Check each row if it falls within an artifact period
for idx, row in results_with_ref.iterrows():
    patient = row['PatientStudyName']
    signal_time = pd.to_datetime(row['RadarStartTime'])
    signal_end = signal_time + timedelta(seconds=60)  # Signal ends 60 seconds later
    session_date = signal_time.strftime('%Y%m%d')
    folder_name = f"{patient}_{session_date}"
    
    if folder_name in artifact_dict:
        for start_dt, end_dt in artifact_dict[folder_name]:
            # Check if the two time periods overlap
            if max(signal_time, start_dt) <= min(signal_end, end_dt):
                results_with_ref.loc[idx, 'HasArtifact'] = True
                break

# Save updated results
results_with_ref.to_csv('data/results_with_ref.csv', index=False)
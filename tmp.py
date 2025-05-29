from enum import IntEnum

class SleepState(IntEnum):
    SLEEP = 0
    WAKE = 1

    # # Plot original score distributions
    # awake_scores = session_scores[session_data['GroundTruth'] == 1]
    # sleep_scores = session_scores[session_data['GroundTruth'] == 0]
    
    # ax3.hist(awake_scores, bins=30, density=True, alpha=0.5, label='Ground Truth: Awake', color='red')
    # ax3.hist(sleep_scores, bins=30, density=True, alpha=0.5, label='Ground Truth: Sleep', color='blue')
    
    # # Apply GMM to find optimal threshold
    # from scipy.stats import boxcox
    # scores_reshaped = session_scores.reshape(-1, 1)
    # scores_bc, lambda_bc = boxcox(scores_reshaped.flatten() + 1e-10)  # Add small constant to handle zeros
    # scores_bc = scores_bc.reshape(-1, 1)
    
    # gmm = GaussianMixture(n_components=2, random_state=0).fit(scores_bc)
    # labels = gmm.predict(scores_bc)
    
    # # Find GMM threshold
    # sorted_scores = np.sort(session_scores)
    # sorted_scores_bc = boxcox(sorted_scores + 1e-10, lambda_bc)
    # probs = np.exp(gmm.score_samples(sorted_scores_bc.reshape(-1, 1)))
    # intersection_idx = np.argmin(np.abs(np.diff(probs)))
    # gmm_threshold_tramsformed = sorted_scores_bc[intersection_idx]
    # gmm_threshold = sorted_scores[intersection_idx]
    
    # # Plot GMM threshold
    # ax3.axvline(x=gmm_threshold, color='purple', linestyle='--', label=f'GMM Threshold: {gmm_threshold:.2f}')
    # ax3.axvline(x=logistic_regression_threshold, color='black', linestyle='--', label=f'PR Threshold: {logistic_regression_threshold:.2f}')
    
    # # Add density curve
    # hist_density = np.histogram(sorted_scores, bins=50, density=True)
    # bin_centers = (hist_density[1][:-1] + hist_density[1][1:]) / 2
    # ax3.plot(bin_centers, hist_density[0], color='black', label='All Scores', linewidth=2)
    # ax3.set_xlabel('Logistic Regression Score')
    # ax3.set_ylabel('Density')
    # ax3.set_title('Logistic Regression Score Distribution by Ground Truth')
    # ax3.legend()
    
    # # Plot transformed score distributions with GMM
    # ax4.hist(scores_bc[session_data['GroundTruth'] == 0], bins=30, density=True, alpha=0.5, label='Ground Truth: Sleep', color='blue')
    # ax4.hist(scores_bc[session_data['GroundTruth'] == 1], bins=30, density=True, alpha=0.5, label='Ground Truth: Awake', color='red')
    # ax4.axvline(x=gmm_threshold_tramsformed, color='purple', linestyle='--', label=f'GMM Threshold: {gmm_threshold_tramsformed:.2f}')

    # # Plot GMM components
    # x = np.linspace(scores_bc.min(), scores_bc.max(), 1000).reshape(-1, 1)
    # for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
    #     pdf = norm.pdf(x, mean, np.sqrt(covar))
    #     ax4.plot(x, pdf * gmm.weights_[i], label=f'GMM Component {i}')
    
    # ax4.set_xlabel('Box-Cox Transformed Score')
    # ax4.set_ylabel('Density')
    # ax4.set_title('Transformed Score Distribution with GMM Components')
    # ax4.legend()
    
    # plt.tight_layout()
    # plt.show()
    
def webster_rules(predictions, sample_times, end_times):
    """
    Apply Cole-Kripke sleep-wake scoring rules to model predictions using efficient change detection
    
    Args:
        predictions: Array of binary predictions (SleepState.SLEEP=0, SleepState.WAKE=1)
        sample_times: Array of start timestamps for each prediction
        end_times: Array of end timestamps for each prediction
        
    Returns:
        Tuple containing:
        - Array of corrected predictions after applying Webster rules
        - List of indices where corrections were made (sleep->wake only)
    """
    # Make copy of predictions to modify
    corrected = predictions.copy()
    corrected_indices = []
    
    # Convert timestamps to minutes since start
    start_times_minutes = [(t - sample_times[0]).total_seconds() / 60 for t in sample_times]
    end_times_minutes = [(t - sample_times[0]).total_seconds() / 60 for t in end_times]

    # Find state changes using numpy diff
    import numpy as np
    state_changes = np.diff(predictions)
    change_points = np.nonzero(state_changes)[0] + 1  # +1 since diff reduces length by 1
    
    # Group consecutive indices into segments
    segments = []
    current_state = predictions[0]
    segment_start = 0
    
    for change_idx in change_points:
        segments.append({
            'start_idx': segment_start,
            'end_idx': change_idx - 1,
            'state': current_state,
            'duration': end_times_minutes[change_idx-1] - start_times_minutes[segment_start]
        })
        current_state = predictions[change_idx]
        segment_start = change_idx
    
    # Add final segment
    segments.append({
        'start_idx': segment_start,
        'end_idx': len(predictions) - 1,
        'state': current_state,
        'duration': end_times_minutes[-1] - start_times_minutes[segment_start]
    })

    # Process segments for rules 1-3 and 4-5
    for i, segment in enumerate(segments[:-1]):  # Skip last segment since we need to look ahead
        next_segment = segments[i + 1]
        
        # Rules 1-3: Wake followed by Sleep
        if segment['state'] == SleepState.WAKE and next_segment['state'] == SleepState.SLEEP:
            wake_duration = segment['duration']
            sleep_start_idx = next_segment['start_idx']
            
            correction_minutes = 0
            if wake_duration >= 15:
                correction_minutes = 4  # Rule 3
                rule = f"Rule 3: Wake {wake_duration:.1f}min -> correcting up to 4min sleep"
            elif wake_duration >= 10:
                correction_minutes = 3  # Rule 2
                rule = f"Rule 2: Wake {wake_duration:.1f}min -> correcting up to 3min sleep"
            elif wake_duration >= 4:
                correction_minutes = 1  # Rule 1
                rule = f"Rule 1: Wake {wake_duration:.1f}min -> correcting up to 1min sleep"
            
            if correction_minutes > 0:
                correction_end_time = start_times_minutes[sleep_start_idx] + correction_minutes
                correction_end_idx = sleep_start_idx + np.searchsorted(
                    start_times_minutes[sleep_start_idx:next_segment['end_idx']+1],
                    correction_end_time,
                    side='right'
                )
                
                print(f"{rule}, correcting indices {sleep_start_idx}-{correction_end_idx-1}")
                corrected[sleep_start_idx:correction_end_idx] = [SleepState.WAKE] * (correction_end_idx - sleep_start_idx)
                corrected_indices.extend(range(sleep_start_idx, correction_end_idx))
        
        # Rules 4-5: Sleep surrounded by Wake
        if i > 0 and segment['state'] == SleepState.SLEEP:
            prev_segment = segments[i-1]
            wake_before = prev_segment['duration']
            wake_after = next_segment['duration']
            sleep_duration = segment['duration']
            
            should_correct = False
            if wake_before >= 20 and wake_after >= 20 and sleep_duration <= 10:
                should_correct = True
                rule = f"Rule 5: Sleep {sleep_duration:.1f}min surrounded by {wake_before:.1f}min + {wake_after:.1f}min wake"
            elif wake_before >= 10 and wake_after >= 10 and sleep_duration <= 6:
                should_correct = True
                rule = f"Rule 4: Sleep {sleep_duration:.1f}min surrounded by {wake_before:.1f}min + {wake_after:.1f}min wake"
            
            if should_correct:
                print(f"{rule}, correcting indices {segment['start_idx']}-{segment['end_idx']}")
                new_corrections = [idx for idx in range(segment['start_idx'], segment['end_idx']+1)
                                 if idx not in corrected_indices]
                segment_length = segment['end_idx'] - segment['start_idx'] + 1
                corrected[segment['start_idx']:segment['end_idx']+1] = [SleepState.WAKE] * segment_length
                corrected_indices.extend(new_corrections)
    
    return corrected, corrected_indices

import pandas as pd

df = pd.read_csv('awake_analysis/logistic_regression_predictions_390.0.csv')
logistic_regression_predictions = df['logistic_regression_predictions'].tolist()
start_times = pd.to_datetime(df['StartTime'])
end_times = pd.to_datetime(df['EndTime'])

logistic_regression_predictions, corrected_indices = webster_rules(
    logistic_regression_predictions,
    start_times.tolist(),
    end_times.tolist()
)
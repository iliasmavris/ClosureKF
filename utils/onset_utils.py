"""
Utilities for computing onset times from predicted trajectories.
"""

import numpy as np
import torch


def predict_onset_from_trajectory(x_current, x_future, threshold, min_consecutive=3, predict_deltas=False):
    """
    Predict onset time from trajectory.
    
    Args:
        x_current: Scalar current displacement
        x_future: [H] predicted future displacement (or delta if predict_deltas=True)
        threshold: Displacement threshold for onset
        min_consecutive: Minimum consecutive steps above threshold
        predict_deltas: If True, x_future is delta, need to reconstruct absolute
    Returns:
        k: Onset index (0-indexed within future horizon) or None if no onset
    """
    # Convert to numpy if torch tensor
    if isinstance(x_current, torch.Tensor):
        x_current = x_current.cpu().numpy()
    if isinstance(x_future, torch.Tensor):
        x_future = x_future.cpu().numpy()
    
    # Reconstruct absolute displacement if needed
    if predict_deltas:
        # x_abs[k] = x_current + x_future[k] (NO cumsum - absolute-to-now delta)
        x_abs = x_current + x_future
    else:
        # Already absolute
        x_abs = x_future
    
    # Find first k where x_abs[k] >= threshold
    above_threshold = x_abs >= threshold
    
    if not np.any(above_threshold):
        return None
    
    # Find first occurrence
    first_above = np.where(above_threshold)[0]
    if len(first_above) == 0:
        return None
    
    k_start = first_above[0]
    
    # Check if we have min_consecutive consecutive steps above threshold
    if k_start + min_consecutive > len(x_abs):
        return None
    
    # Check if all steps from k_start to k_start+min_consecutive are above threshold
    if np.all(above_threshold[k_start:k_start + min_consecutive]):
        return k_start
    
    # If not, look for next valid onset
    for k in range(k_start + 1, len(x_abs) - min_consecutive + 1):
        if np.all(above_threshold[k:k + min_consecutive]):
            return k
    
    return None


def compute_onset_timing_error(pred_onsets, true_onsets, dt_fut, return_stats=True):
    """
    Compute onset timing error in samples and seconds.
    
    Args:
        pred_onsets: List of predicted onset indices (or None)
        true_onsets: List of true onset indices (or None)
        dt_fut: List of [H] arrays of future time_delta values
        return_stats: If True, return statistics dict
    Returns:
        If return_stats:
            dict with 'error_samples', 'error_seconds', 'mean_error_samples', 'mean_error_seconds', etc.
        Else:
            (error_samples_list, error_seconds_list)
    """
    error_samples_list = []
    error_seconds_list = []
    
    for pred_k, true_k, dt in zip(pred_onsets, true_onsets, dt_fut):
        if pred_k is None or true_k is None:
            continue
        
        # Convert to numpy if needed
        if isinstance(dt, torch.Tensor):
            dt = dt.cpu().numpy()
        
        # Error in samples
        error_samples = pred_k - true_k
        error_samples_list.append(error_samples)
        
        # Error in seconds: sum(dt[1:k_pred]) - sum(dt[1:k_true])
        # Note: dt[0] corresponds to t+1, so dt[1] corresponds to t+2, etc.
        # For index k, we need dt[0:k] (k samples)
        t_pred = np.sum(dt[0:pred_k]) if pred_k > 0 else 0.0
        t_true = np.sum(dt[0:true_k]) if true_k > 0 else 0.0
        error_seconds = t_pred - t_true
        error_seconds_list.append(error_seconds)
    
    if not return_stats:
        return error_samples_list, error_seconds_list
    
    # Compute statistics
    if len(error_samples_list) == 0:
        return {
            'error_samples': [],
            'error_seconds': [],
            'mean_error_samples': np.nan,
            'mean_error_seconds': np.nan,
            'std_error_samples': np.nan,
            'std_error_seconds': np.nan,
            'count': 0,
        }
    
    error_samples_arr = np.array(error_samples_list)
    error_seconds_arr = np.array(error_seconds_list)
    
    return {
        'error_samples': error_samples_list,
        'error_seconds': error_seconds_list,
        'mean_error_samples': np.mean(error_samples_arr),
        'mean_error_seconds': np.mean(error_seconds_arr),
        'std_error_samples': np.std(error_samples_arr),
        'std_error_seconds': np.std(error_seconds_arr),
        'median_error_samples': np.median(error_samples_arr),
        'median_error_seconds': np.median(error_seconds_arr),
        'count': len(error_samples_list),
    }

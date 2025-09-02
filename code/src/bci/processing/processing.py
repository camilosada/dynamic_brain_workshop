import pandas as pd
import numpy as np


def filter_somas(dff_traces, roi_table, soma_prob: float = 0.005):
    """
    Filters somas 
    
    Parameters
    ----------
    dff_traces : np.ndarray
        DFF Traces for given session
    roi_table : pd.DataFrame
        ROI table
    soma_prob : float, optional
        Default is 0.005 #TrustUs
    
    Returns
    -------
    valid_rois : pd.DataFrame
        ROI Table excluding unlikely somas and including CN (regardless of soma prob)
    
    Raises
    ------
    ValueError
        If more than 1 CN during BCI epoch
        
    """
    # remove ROIs with NaN traces
    valid_trace_ids = [i for i in range(dff_traces.shape[1]) if np.isnan(dff_traces[0, i])==False]
    # limit ROI table to non-nan traces
    roi_table_filtered = roi_table.loc[valid_trace_ids]
    
    # find somatic ROIs
    valid_rois = roi_table_filtered[roi_table_filtered['soma_probability'] > soma_prob]
    
    # add CN if not included
    target_roi_index = bci_trials['closest_roi'].unique()
    print(f'CN: {target_roi_index}')
    
    # if more than one CN found
    if len(target_roi_index) > 1:
        raise ValueError("More than one CN during BCI epoch")
    
    target_roi_index = target_roi_index[0]
    
    # add to ROI table if not already there
    if not(target_roi_index in valid_rois.index):
        valid_rois = pd.concat((valid_rois, roi_table_filtered.loc[[target_roi_index], :]), axis=0)
        valid_rois = valid_rois.sort_index()
        
    return valid_rois

def smooth_dff(dff: np.array, window: int = 10):
    """
    Smooth DFF signal over given sliding window
    
    Parameters
    ----------
    dff : np.array
        dff traces
    window : int, optional
        Smoothing window in frames, default is 10
        
    Returns
    -------
    smooth_dff : np.ndarray
        Smoothed dff traces
    """
    smooth_dff = np.full(dff.shape, np.nan)
    kernel = np.ones(window) / window
    
    for idx, trial in enumerate(dff):
        smooth_dff[idx] = np.convolve(trial, kernel, mode='same')
        
    return smooth_dff
    
    
def get_threshold_changes(high_thresh: np.ndarray = None,
                          low_thresh: np.ndarray = None,
                          n_trials: int = None,
                          bci_trials: pd.DataFrame = None):
    """
    Get the trials at which the threshold changes 
    
    Parameters
    ----------
    high_thresh : np.ndarray, optional
    low_thresh : np.ndarray, optional
    n_trials : int, optional
    bci_trials : pd.DataFrame, optional
    
    Returns
    -------
    thresh_index_changes : np.ndarray
        Indices of threshold changes
        
    Raises
    ------
    ValueError
        If wrong parameters are passed
    """
    if high_thresh is None and low_thresh is None and n_trials is None and bci_trials is None:
        raise ValueError("high_thresh or low_thresh, and n_trials required, or bci_trials required")
    elif bci_trials is not None:
        high_thresh = bci_trials['high']
        n_trials = len(bci_trials)
    elif high_thresh is not None and n_trials is not None:
        high_thresh = high_thresh
        n_trials = n_trials
        
    index_change_thresh = np.insert(np.where(np.insert(np.diff(high_thresh), 0, 0,)), 0, 0)
    index_change_thresh = np.append(index_change_thresh, n_trials)
    return index_change_thresh

def get_epoch_start_stop_frames(epoch_table: pd.DataFrame, epoch: str) -> tuple:
    """
    Get start and stop frames for given epoch
    
    Parameters
    ----------
    epoch_table : pd.DataFrame
        Epoch table
    epoch : str
        Which epoch to get times for
        Must be in ['photostim', 'spont', 'spont_01', 'BCI', 'spont_post', 'photostim_post']
        
    Returns
    -------
    tuple
        start and stop frames
        
    Raises
    ------
    ValueError
        If `epoch` param not allowed
    """
    possible_epochs = ['photostim', 'spont', 'spont_01', 'BCI', 'spont_post', 'photostim_post']
    if epoch not in possible_epochs:
        raise ValueError(f'`epoch` must be in {possible_epochs}')
        
    epoch_of_interest = epoch_table[epoch_table['stimulus_name'].str.contains(epoch)]
    start = epoch_of_interest.loc[epoch_of_interest.index[0]]['start_frame']
    stop = epoch_of_interest.loc[epoch_of_interest.index[0]]['stop_frame']
    return (start, stop)
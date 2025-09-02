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


def get_dff_by_trial(dff_smooth: np.ndarray, data: dict = None, 
                     epoch_data: dict = None, bci_trials: pd.DataFrame = None, 
                     frame_rate: float = None, ):
    """
    Reshape dff array to separate by trial
    
    Parameters
    ----------
    dff_smooth : np.ndarray
        dff trace with shape (n_rois, frames)
    data : dict, optional
        Dictionary containing ['bci_trials'] and ['frame_rate'], default is None
    epoch_data : dict, optional
        Dictionary containing ['start_bci_trial'], ['stop_bci_trial'], default is None
    bci_trials : pd.DataFrame, optional
        BCI trials dataframe, default is None
    frame_rate : float, optional
        Imaging frame rate, default is None
    start_bci_trial : np.ndarray, optional
        Array of BCI trial start frames, default is None
    stop_bci_trial : np.ndarray, optional
        Array of BCI trial end frames, default is None
        
    Returns
    -------
    dff_by_trial : np.ndarray
        DFF by trial in shape (n_rois, n_trials, max_trial_frames).
        From go_cue to threshold_crossing_times
    """
    # make sure have all params
    if data is None:
        if bci_trials is None and frame_rate is None:
            raise ValueError('need `bci_trials` and `frame_rate` or `data` dict')
    else:  # use dict if we have it
        bci_trials = data['bci_trials']
        frame_rate = data['frame_rate']
    # bci trials vars
    go_cue = bci_trials['go_cue']
    threshold_crossing_times = bci_trials['threshold_crossing_times']

    # find epoch params
    if epoch_data is None:
        if start_bci_trial is None and stop_bci_trial is None:
            raise ValueError('need `stop_bci_trial` and `start_bci_trial` or `epoch_data` dictionary')
    else:  # use dict if we have it
        start_bci_trial = epoch_data['start_bci_trial']
        stop_bci_trial = epoch_data['stop_bci_trial']    
    
    # get dimensions
    n_rois = dff_smooth.shape[0]  # number of neurons
    n_trials = len(start_bci_trial)  # number of trials 
    max_trial_duration = np.max(stop_bci_trial - start_bci_trial)
    
    # initialize
    dff_by_trial = np.full((n_rois, n_trials, max_trial_duration*2), np.nan)
    
    # 
    for trial, (start_idx, stop_idx) in enumerate(zip((go_cue * frame_rate).astype(int),
                                                  (threshold_crossing_times * frame_rate).astype(int))):
        # add dff_smooth for given trial window
        dff_by_trial[:, trial, :int(stop_idx-start_idx)] = dff_smooth[:, start_idx:stop_idx]
        
    return dff_by_trial  # shape (n_rois, n_trials, frames)

def get_bci_epoch_data(data: dict = None,
                       epoch_table: pd.DataFrame = None, 
                       bci_trials: pd.DataFrame = None, 
                       dff_traces: np.ndarray = None, 
                       frame_rate: float = None):
    """
    Get BCI epoch data (start/stop frames), convert thresh/zaber steps from time to frame
    
    Parameters
    ----------
    epoch_table : pd.DataFrame
        Epoch table 
    bci_trials : pd.DataFrame
        BCI trials data
    dff_traces : np.ndarray
        DFF traces 
    frame_rate : float
        Frame rate
        
    Returns
    -------
    dict
        Dictionary with BCI epoch data
        - 'start_bci_epoch': start frame
        - 'stop_bci_epoch': stop frame
        - 'start_bci_trial': trial start frames relative to epoch start
        - 'stop_bci_trial': trial stop frames relative to epoch start
        - 'thresh_crossing_frame': threshold crossing frames
        - 'zaber_step_frames': zaber steps in frames
        - 'go_cue_frame': go cue in frames
        - 'reward_frame': reward time in frames
        - 'low_thresh': low threshold values
        - 'high_thresh': high threshold values
        - 'dff_bci': transposed DFF trace for BCI epoch (ROIs x frames)
    """
    if data is not None:
        # check for valid dictionary
        required_keys = ['epoch_table', 'bci_trials', 'dff_traces', 'frame_rate']
        if not all(key in data.keys() for key in required_keys):
            missing_keys = [key for key in required_keys if key not in data.keys()]
            raise ValueError(f"`data` dictionary missing keys: {missing_keys}")
        # assign vars if dictionary is complete
        epoch_table = data['epoch_table']
        bci_trials = data['bci_trials']
        dff_traces = data['dff_traces']
        frame_rate = data['frame_rate']
        
    # otherwise make sure all other vars are passed
    elif data is None:
        required_params = [('epoch_table', epoch_table),('bci_trials', bci_trials),
                           ('dff_traces', dff_traces),('frame_rate', frame_rate)]
        
        missing_params = [name for name, value in required_params if value is None]
        if missing_params:
            raise ValueError(f"if not passing `data` dictionary, all parameters are required.\nmissing: {missing_params}")
    
    # get epoch frames
    bci_epochs = epoch_table[epoch_table['stimulus_name'].str.contains('BCI')]
    start_bci_epoch = bci_epochs.loc[bci_epochs.index[0]]['start_frame']
    stop_bci_epoch = bci_epochs.loc[bci_epochs.index[0]]['stop_frame']
    
    # calculate trial frames relative to epoch start
    start_bci_trial = bci_trials['start_frame'] - start_bci_epoch
    stop_bci_trial = bci_trials['stop_frame'] - start_bci_epoch
    
    # convert to frames
    thresh_crossing_frame = np.round(bci_trials['threshold_crossing_times'] * frame_rate).astype(int)
    zaber_step_frames = np.round(np.array(bci_trials['zaber_step_times'].tolist()) * frame_rate)
    go_cue_frame = np.round(bci_trials['go_cue'] * frame_rate).astype(int)
    reward_frame = np.round(bci_trials['reward_time'] * frame_rate).astype(int)
    
    # get low/high threshold arrays
    low_thresh = bci_trials['low']
    high_thresh = bci_trials['high']
    
    # subset to BCI epoch and transpose (ROIs x frames)
    dff_bci = dff_traces[start_bci_epoch:stop_bci_epoch, :].T
    
    return {
        'start_bci_epoch': start_bci_epoch,
        'stop_bci_epoch': stop_bci_epoch,
        'start_bci_trial': start_bci_trial,
        'stop_bci_trial': stop_bci_trial,
        'thresh_crossing_frame': thresh_crossing_frame,
        'zaber_step_frames': zaber_step_frames,
        'go_cue_frame': go_cue_frame,
        'reward_frame': reward_frame,
        'low_thresh': low_thresh,
        'high_thresh': high_thresh,
        'dff_bci': dff_bci
    }
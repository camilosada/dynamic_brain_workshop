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
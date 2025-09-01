import os
import pandas as pd


def align_thresholds(bci_trials: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns thresholds to trials in BCI stimulus trials dataframe
    Added columns are "trial", "low", "high"
    
    Parameters
    ----------
    bci_trials : pd.DataFrame
        BCI stimulus trials dataframe
    thresholds : pd.DataFrame
        BCI CN thresholds dataframe
        
    Returns
    -------
    bci_trials : pd.DataFrame
        BCI stimulus trials dataframe including "low" and "high" thresholds per trials
        
    Raises
    ------
    AssertionError
        If BCI trials dataframe is shorter than thresholds dataframe
    AssertionError
        If BCI trials dataframe already contains "low" or "high" columns
    """
    assert len(bci_trials) > len(thresholds), "BCI trials must be longer than thresholds dataframe"
    # if these columns exist, likely already have run this function
    assert 'low' not in bci_trials.columns, "BCI trials alrady contains threshold information"
    assert 'high' not in bci_trials.columns, "BCI trials alrady contains threshold information"
    
    thresholds = thresholds.set_index('trial')  # index starts at 2
    bci_trials = bci_trials.merge(thresholds, left_on=bci_trials.index, right_on=thresh_index.index, how='outer')
    bci_trials = bci_trials.drop(columns='key_0')
    
    # check difference between thresh and trials
    nan_low = temp['low'].isna().sum()
    nan_high = temp['high'].isna().sum()
    
    if nan_low == nan_high:  # this should be likely
        all_lows = nan_low
        print(f'total difference in dataframes: {all_lows}')  # number of nans
    else:
        print(f'difference between dataframes\n\tlow: {nan_low}')
        print(f'difference between dataframes\n\thigh: {nan_high}')

    return bci_trials
    
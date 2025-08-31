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
    # because thresholds begin at trial 2
    assert len(bci_trials) > len(thresholds), "BCI trials must be longer than thresholds dataframe"
    # if these columns exist, likely already have run this function
    assert 'low' not in bci_trials.columns, "BCI trials alrady contains threshold information"
    assert 'high' not in bci_trials.columns, "BCI trials alrady contains threshold information"

    # set up thresh df to fit bci trials df
    aligned_thresh = pd.DataFrame(index=range(len(bci_trials)), columns=thresholds.columns)
    delta = len(bci_trials) - len(thresholds) # should always (?) be 2 i think
    if delta > 2:
        print(f'Difference between BCI trials dataframe and thresholds dataframe is: {delta}')
    aligned_thresh.iloc[delta:] = thresholds.values  # add thresholds values to df
    bci_trials = pd.concat([bci_trials, aligned_thresh], axis=1)  # join dfs
    return bci_trials
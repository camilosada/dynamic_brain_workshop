import os
from hdmf_zarr import NWBZarrIO
import pandas as pd
import json


def load_nwb_session_file(session_name, data_dir: str = '/data'):
    """
    Loads NWB file for given session name

    Parameters
    ----------
    session_name : str
        NWB session name 
    data_dir : str or Path, default is '/data'
        Path to base data directory

    Returns
    -------
    nwbfile : nwbfile
        NWB file for given session

    Raises
    ------
    AssertionError
        If no NWB file found in session directory
    """
    # Get the directory for this dataset and load it
    bci_data_dir = os.path.join(data_dir, 'brain-computer-interface')
    print(f'BCI data directory: {bci_data_dir}\n')
    # Get the data folder for this session
    session_dir = os.path.join(bci_data_dir, session_name)
    print(f'Session directory: {session_dir}\n')
    # Now find the NWB file and set the path to load it
    nwb_file = [file for file in os.listdir(session_dir) if 'nwb' in file][0]
    print(f'NWB file: {nwb_file}')
    assert len(nwb_file) > 0, f"No NWB file found in {session_dir}"
    nwb_path = os.path.join(session_dir, nwb_file)
    print(f'NWB path: {nwb_path}')
    # Load the data
    with NWBZarrIO(str(nwb_path), 'r') as io:
        nwbfile = io.read()
    return nwbfile

def load_session_thresh_file(session_name: str, data_dir: str = '/data') -> pd.DataFrame:
    """
    Finds and loads threshold file for given session name
    
    Parameters
    ----------
    session_name : str
        Name of experiment session
    data_dir : str or Path, default is '/data'
        Path to base data directory
    """
    # get name of threshold file
    threshold_file_name = get_session_thresh_file_name(session_name, data_dir)
    file_path = os.path.join(data_dir, 'bci-thresholds', threshold_file_name)
    print(f'Found threshold file at: {file_path}')
    thresholds = pd.read_csv(file_path)
    return thresholds

# helper function for load_session_thresh_file
def get_session_thresh_file_name(session_name, data_dir: str = '/data') -> str:
    """
    Returns BCI threshold file name for given session name
    
    Parameters
    ----------
    session_name : str
        Name of experiment session
    data_dir : str or Path, default is '/data'
        Path to base data directory
        
    Returns
    -------
    threshold_file_name : str
        Name of BCI threshold file
        
    Raises
    ------
    AssertionError
        If no files found for given session name
        If more than one file found for given session name
    """
    mouse_id = session_name.split('_')[1]
    mouse_thresh_files = get_mouse_thresh_file_list(mouse_id=mouse_id, data_dir=data_dir)
    print(f'All threshold files for mouse {mouse_id}: {mouse_thresh_files}\n')
    threshold_file_name = [f for f in mouse_thresh_files if f in (session_name)]
    assert len(threshold_file_name) > 0, f"No files found for session ID: {session_name}"
    assert len(threshold_file_name) < 2, f'Multiple files found for session ID: {session_name}\nFound: {threshold_file_name}'
    return threshold_file_name[0]

# helper function for load_session_thresh_file
def get_mouse_thresh_file_list(mouse_id, data_dir: str = '/data') -> list:
    """
    Finds and returns BCI threshold files for given mouse ID
    
    Parameters
    ----------
    mouse_id : str or int
        Subject ID for mouse of interest
    data_dir : str or Path, default is '/data/'
        Path to base data directory
        
    Returns
    -------
    this_mouse_thresh_files : list
        List of BCI threshold file names for given mouse ID
        
    Raises
    ------
    AssertionError
        If path to threshold directory doesn't exist
        If no files found for mouse ID
    """
    mouse_id = str(mouse_id)
    # create path to thresh files directory
    bci_thresh_path = os.path.join(data_dir, 'bci-thresholds')
    assert os.path.exists(bci_thresh_path), "Path to thresholds not found, check data_dir"
    # only look for threshold files
    bci_thresh_files = [f for f in os.listdir('/data/bci-thresholds/') if f.startswith('s')]
    # get list of files for mouse ID
    this_mouse_thresh_files = [f for f in bci_thresh_files if f.split('_')[1] == mouse_id]
    # check if there are any files 
    assert len(this_mouse_thresh_files) > 0, f"No files found for ID: {mouse_id}"
    return this_mouse_thresh_files

def load_metadata(data_dir: str = '/data') -> pd.DataFrame:
    """
    Loads bci_metadata.csv as DataFrame
    
    Parameters
    ----------
    data_dir : str or Path, default is '/data'
        Path to data directory
    
    Returns
    -------
    metadata : pd.DataFrame
        metadata csv
        
    Raises
    ------
    FileNotFoundError
        If metadata csv not found
    """
    path = os.path.join(data_dir, 'bci_task_metadata/bci_metadata.csv')
    try:
        metadata = pd.read_csv(path)
    except FileNotFoundError as e:
        print(f'File not found at {path}.\n{e}')
        
    return metadata
    
def get_session_names(mouse_id: str or int, data_dir: str = '/data') -> list:
    """
    Finds session names for valid mouse_id, returns ordered by date
    
    Parameters
    ----------
    mouse_id : str or int
        Subject ID for mouse
    data_dir : str or Path, default is '/data'
        Path to data directory
    
    Returns
    -------
    array of str
    
    Raises
    ------
    AssertionError
        If mouse_id is invalid
    """
    metadata = load_metadata()
    mouse_md = metadata[metadata['subject_id'] == mouse_id]
    mouse_md.sort_values(by='session_date')
    return mouse_md['name'].values
    
    
def get_bci_trials(nwb_file) -> pd.DataFrame:
    """
    Get BCI trials dataframe from NWB File
    
    Parameters
    ----------
    nwb_file : NWB File
        For given session
        
    Returns
    -------
    bci_trials : pd.DataFrame
        BCI Trials dataframe
    """
    return nwb_file.stimulus['Trials'].to_dataframe()
    
def load_bci_trials(session_name: str, data_dir: str = '/data') -> pd.DataFrame:
    """
    Loads BCI Trials dataframe from given session name
    
    Parameters
    ----------
    session_name : str
        Name of experiment session
    data_dir : str or Path, default is '/data'
    
    Returns
    -------
    bci_trials : pd.DataFrame
        Behavior table
    """
    nwb_file = load_nwb_session_file(session_name, data_dir)
    bci_trials = get_bci_trials(nwb_file)
    return bci_trials

def get_dff(nwb_file) -> np.ndarray:
    """
    Get dff traces for given NWB File
    
    Parameters
    ----------
    nwb_file : NWB File
        For given session
        
    Returns
    -------
    dff : np.ndarray
        dff traces from NWB file
    """
    return nwbfile.processing["processed"].data_interfaces["dff"].roi_response_series["dff"].data

def get_epoch_table(nwb_file) -> pd.DataFrame:
    """
    Get epoch table for given NWB File
    
    Parameters
    ----------
    nwb_file : NWB File
        For given session
        
    Returns
    -------
    epoch_table : pd.DataFrame
        Epoch table for given session
    """
    return nwbfile.intervals["epochs"].to_dataframe()

def load_filtered_metadata(data_dir: str = '/data'):
    """
    Load metadata.csv and filter to exclude sessions with no threshold data or duplicated sessions 
    
    Parameters
    ----------
    data_dir : str or Path, optional
        Path to data directory
        
    Returns
    -------
    metadata : pd.DataFrame
        Filtered metadata table
        
    Raises
    ------
    FileNotFoundError
        If metadata or valid sessions file not found
    """
    # read in overall metadata
    try:
        metadata = pd.read_csv(os.path.join(data_dir, 'bci_task_metadata', 'bci_metadata.csv'))
    except FileNotFoundError as e:
        print(e)
    
    # read in valid sessions
    try:
        with open('valid_sessions.json', 'r') as f:
            valid_sessions = json.load(f)
    except FileNotFoundError as e:
        print(e)
        
    # remove rows that have unreadable NWBs 
    invalid_sessions = valid_sessions['invalid']
    filtered = metadata[~metadata['name'].isin(invalid_sessions)]
    
    # get row indices for sessions with thresh files
    filtered_idx_list = []
    for i in valid_sessions['all']:  # these are sessions that have threshold files
        for filtered_idx, j in enumerate(filtered['name'].values):
            if i in j:  # session string is first half of 'name' string
                filtered_idx_list.append(filtered_idx)
    
    # remove rows that don't have thresh files
    remove_idx_list = []
    for i in filtered.index:
        if i not in filtered_idx_list:
            remove_idx_list.append(i)
    filtered = filtered.drop(remove_idx_list)
    
    
    # this one is duplicated, remove
    dupe = 'single-plane-ophys_731015_2025-01-28_18-56-35_processed_2025-08-03_21-58-28'
    filtered = filtered[filtered['name'] != dupe]

    return filtered
    
def get_raw_fluorescence(nwb_file) -> pd.DataFrame:
    """
    Get raw fluorescence trace for given NWB File
    
    Parameters
    ----------
    nwb_file : NWB File
        For given session
        
    Returns
    -------
    raw_trace : np.ndarray
        Raw fluorescnence trace for given NWB file
    """
    return nwb_file.processing['processed'].data_interfaces['raw'].roi_response_series['ROI_fluorescence'].data
import os
from hdmf_zarr import NWBZarrIO
import pandas as pd


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
    print(f'NWB path: nwb_path')
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

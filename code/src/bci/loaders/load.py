import os
from hdmf_zarr import NWBZarrIO


def load_nwb_session_file(data_dir, session_name):
    """
    Loads NWB file for given session name

    Parameters
    ----------
    data_dir : str or Path
        Path to base data directory
    session_name : str
        NWB session name 

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
    assert len(nwb_file) > 0, f"No NWB file found in {session_dir}"
    nwb_path = os.path.join(session_dir, nwb_file)
    print(f'NWB path: nwb_path')
    # Load the data
    with NWBZarrIO(str(nwb_path), 'r') as io:
        nwbfile = io.read()
    return nwbfile
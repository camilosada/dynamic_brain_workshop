import os
from hdmf_zarr import NWBZarrIO


def load_nwb_session_file(data_dir,session_name):

    # Get the directory for this dataset and load it
    bci_data_dir = os.path.join(data_dir, 'brain-computer-interface')
    print(bci_data_dir)
    # Get the data folder for this session
    session_dir = os.path.join(bci_data_dir, session_name)
    print(session_dir)
    # Now find the NWB file and set the path to load it
    nwb_file = [file for file in os.listdir(session_dir) if 'nwb' in file][0]
    nwb_path = os.path.join(session_dir, nwb_file)
    print(nwb_path)
    # Load the data
    with NWBZarrIO(str(nwb_path), 'r') as io:
        nwbfile = io.read()
    return nwbfile
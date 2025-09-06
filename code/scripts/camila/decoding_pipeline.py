# General imports 
import os 
import sys
import logging
import platform
from os.path import join as pjoin
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as stats 
from skimage import measure
import matplotlib.pyplot as plt
import json
import zarr

# Pynwb imports
from hdmf_zarr import NWBZarrIO
from nwbwidgets import nwb2widget

# set paths
sys.path.insert(0, '/code/src')
data_dir = '/data'

# package imports
from bci.loaders import load
from bci.thresholds.thresholds import align_thresholds
from bci.processing import processing
from bci.trials.align import indep_roll
from scipy import ndimage
from sklearn.model_selection import cross_validate

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from scipy.stats import wilcoxon
import results

# -------------------------------
def smooth_dff_by_trial(dff, window=10):
    return ndimage.uniform_filter1d(dff, size=window, axis=-1, mode='reflect')

def get_dff_by_trial(dff_smooth: np.ndarray, data: dict = None, 
                     epoch_data: dict = None, bci_trials: pd.DataFrame = None, 
                     frame_rate: float = None, start_bci_trial: np.ndarray = None,
                    stop_bci_trial: np.ndarray = None):
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
    for trial, (start_idx, stop_idx) in enumerate(zip((start_bci_trial).astype(int),
                                                  ((threshold_crossing_times * frame_rate)+start_bci_trial).astype(int))):
        # add dff_smooth for given trial window
        dff_by_trial[:, trial, :int(stop_idx-start_idx)] = dff_smooth[:, start_idx:stop_idx]
        
    return dff_by_trial  # shape (n_rois, n_trials, frames)

def get_zaber_event_matrix(dff_by_trial: np.array, data: dict = None, zaber_frames: np.array = None, thresh_crossing_frames: np.array = None):
    """
    
    Parameters
    ----------
    dff_by_trial : np.array
        Array containing dff traces per trial, shape (n_rois, n_trials, n_frames).
    data : dict, optional
        Data dictionary containing ['bci_trials'], default is None
    zaber_frames : np.array, optional
        Array of zaber steps in frames, shape (n_trials, max(n_steps)), default is None
    thresh_crossing_frames : np.array, optional
        Array of threshold crossing times in frames, shape (n_trials, 1) default is None
        
    Returns
    -------
    zaber_event_matrix : np.array
        Binary array of zaber step times, shape (n_trials, n_frames)
        
    Raises
    ------
    ValueError
        If wrong parameters passed
    """
    if data is not None:
        # get zaber frames in right format
        zaber_frames = data['bci_trials']['zaber_step_times'] * data['frame_rate'] - data['bci_trials']['go_cue']
        zaber_frames = np.array(zaber_frames.tolist())
        
        # get threshold crossing frames in right format
        thresh_crossing_frames = np.round(data['bci_trials']['threshold_crossing_times']*data['frame_rate']).astype(int)
        thresh_crossing_frames = thresh_crossing_frames.values.reshape(-1, 1)
        
    elif zaber_frames is not None and thresh_crossing_frames is not None:
            if len(zaber_frames.shape) != 2:
                raise ValueError("zaber_frames shape must be (n_trials, max(n_steps))")
            if len(thresh_crossing_frames.shape) != 2:
                raise ValueError("thresh_crossing_frames shape must be (n_trials, 1)")
    else:
        raise ValueError("must pass either data dictionary OR zaber_frames AND thresh_crossing_frames")
    
    # check dff matrix shape
    if len(dff_by_trial.shape) != 3:
        raise ValueError("dff_by_trial shape must be (n_rois, n_trials, n_frames")
        
    # get mask of zaber steps before threshold crossing time
    mask = np.astype(np.where(zaber_frames < thresh_crossing_frames, zaber_frames, 0), int)
    
    # get zaber step coordinates
    rows, _ = np.where(mask != 0) 
    cols = mask[np.where(mask != 0, True, False)]
    
    # initialize
    zaber_event_matrix = np.zeros_like(dff_by_trial[0, :, :])
    
    # apply mask
    zaber_event_matrix[rows, cols] = 1
    
    return zaber_event_matrix

# ---------------------------------
# set data path
platstring = platform.platform()
system = platform.system()
if system == "Darwin":
    # macOS
    data_dir = "/Volumes/Brain2025/"
elif system == "Windows":
    # Windows (replace with the drive letter of USB drive)
    data_dir = "E:/"
elif "amzn" in platstring:
    # then on CodeOcean
    data_dir = "/data/"
else:
    # then your own linux platform
    # EDIT location where you mounted hard drive
    data_dir = "/media/$USERNAME/Brain2025/"
    
print('data directory set to', data_dir)


savepath='/scratch'
save_format='jpg'

# Load metadata csv file
metadata = pd.read_csv(os.path.join(data_dir, 'bci_task_metadata', 'bci_metadata.csv'))
# Get all mice available
subject_ids = np.sort(metadata['subject_id'].unique())
# Select one mice
n_subjects = len(subject_ids)


for subject_id in [754303]:

    # Select one subject metadata, sorted by 'session_number'
    this_mouse_metadata = metadata[metadata['subject_id']==subject_id].sort_values(by='session_number')
    # Pick one session for this mouse
    session_names = this_mouse_metadata.name.values
    print('Selected subject is', subject_id)
    #print('Selected session is', session_name)

    for i_session_name in session_names:
        try:
            data = load.load_session_data(i_session_name)
            data['bci_trials'] = processing.correct_bci_trials(data['bci_trials'])
            data['bci_trials'] = processing.get_valid_bci_trials(data['bci_trials'])
            epoch_data = processing.get_bci_epoch_data(data)
        except:
            continue
        subdir = f'{subject_id}/{i_session_name}/svm'
        output = Path(f'{savepath}/{subdir}')
        
        # Time to frames
        thresh_crossing_frames = np.round(data['bci_trials']['threshold_crossing_times']*data['frame_rate']).astype(int)
        go_cue_frames = np.round(data['bci_trials']['go_cue']*data['frame_rate']).astype(int) #+ data['bci_trials']['start_frame'] - epoch_data['start_bci_epoch']

        # Select valid ROI
        valid_rois, cn_index = processing.filter_somas(data)
        dff_bci = processing.get_valid_bci_traces(epoch_data['dff_bci'], valid_rois)
        # Process dff traces
        ddf_smooth = smooth_dff_by_trial(dff_bci)  # window = 10
        dff_by_trial_smooth1 = get_dff_by_trial(ddf_smooth, data, epoch_data)
        
        # Get rois mapping from roi id to position in the dff trace matrix  
        new_cn_index = processing.convert_roi_index(cn_index, valid_rois)

        frame_rate = data['frame_rate']

        # Process lick spout
        zaber_bool = get_zaber_event_matrix(dff_by_trial_smooth1, data=data)
        zaber_smooth = smooth_dff_by_trial(zaber_bool)

        gc_frame=go_cue_frames.values[0]
        dff_gc= dff_by_trial_smooth1[:,:,gc_frame:]
        zaber_gc = zaber_smooth[:,gc_frame:]
        n_neurons = dff_by_trial_smooth1.shape[0]
        n3rds = 3
        n_iterations = 100
        score_train = np.zeros((n_iterations,n3rds))
        score_test = np.zeros((n_iterations,n3rds))
        cv_scores_test = np.zeros((n_iterations,n3rds,5))
        svc_coeff = np.zeros((n_iterations,n3rds,n_neurons))

        seed = 2025
        rng = np.random.default_rng(seed)

        itr_seed = rng.choice(np.arange(0, 5000), size=n_iterations, replace=False)

        n_tr = dff_gc.shape[1]
        one3rd = int(n_tr/n3rds)

        for it_3rd in range(n3rds):

            for it in range(n_iterations):       
                iseed = itr_seed[it] 


                if it_3rd==0:
                    dff_gc3rd = dff_gc[:,:one3rd]
                    zaber_flat3rd = zaber_gc[:one3rd]
                elif it_3rd==1:
                    dff_gc3rd = dff_gc[:,one3rd:2*one3rd]
                    zaber_flat3rd = zaber_gc[one3rd:2*one3rd]
                elif it_3rd==2:
                    dff_gc3rd = dff_gc[:,2*one3rd:]
                    zaber_flat3rd = zaber_gc[2*one3rd:]

                # Flattern arrays
                mask = ~np.isnan(dff_gc3rd[0, :, :])
                dff_flat = dff_gc3rd[:, mask]
                zaber_flat = zaber_flat3rd[mask]
                zaber_flat_ones = np.where(zaber_flat != 0, 1, 0)
                x = dff_flat
                y = zaber_flat_ones
                svc = LinearSVC(class_weight='balanced',C=1)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=iseed)
                
                scores = cross_validate(svc,x.T,y,cv=cv)
                cv_scores_test[it,it_3rd] = scores['test_score']

                x_train, x_test, y_train, y_test = train_test_split(x.T, y, stratify=y, test_size=0.2, random_state=iseed)
                svc.fit(x_train, y_train)

                score_train[it,it_3rd] = svc.score(x_train,y_train)
                score_test[it,it_3rd] = svc.score(x_test,y_test)
                svc_coeff[it,it_3rd] = svc.coef_[0]

        fig,ax = plt.subplots(figsize=(6,5))
        ax.hist(score_test[:,0])
        ax.hist(score_test[:,1])
        ax.hist(score_test[:,2])
        output.mkdir(parents=True, exist_ok=True)
        figname= f'test_score.{save_format}'
        fig.savefig(output/figname,format=save_format,bbox_inches="tight")

        max_all = []
        for i in range(3):
            mean_coeff = np.mean(svc_coeff,axis=0)[i]
            max_all.append(np.max(abs(mean_coeff))+0.05)

        lim_ar = np.max(max_all)
        fig,ax = plt.subplots(1,3,figsize=(20,5),sharex=True,sharey=True)
        _=ax[0].hist(np.mean(svc_coeff,axis=0)[0],bins=np.arange(-lim_ar,lim_ar,0.01))
        _=ax[0].hist(np.mean(svc_coeff,axis=0)[0][new_cn_index],bins=np.arange(-lim_ar,lim_ar,0.01))
        i_roimax = np.argmax(np.abs(np.mean(svc_coeff,axis=0))[0])
        _=ax[0].hist(np.mean(svc_coeff,axis=0)[0][i_roimax],bins=np.arange(-lim_ar,lim_ar,0.01),color='r')

        _=ax[1].hist(np.mean(svc_coeff,axis=0)[1],bins=np.arange(-lim_ar,lim_ar,0.01))
        _=ax[1].hist(np.mean(svc_coeff,axis=0)[1][new_cn_index],bins=np.arange(-lim_ar,lim_ar,0.01))
        i_roimax = np.argmax(np.abs(np.mean(svc_coeff,axis=0))[1])
        _=ax[1].hist(np.mean(svc_coeff,axis=0)[1][i_roimax],bins=np.arange(-lim_ar,lim_ar,0.01),color='r')

        _=ax[2].hist(np.mean(svc_coeff,axis=0)[2],bins=np.arange(-lim_ar,lim_ar,0.01))
        _=ax[2].hist(np.mean(svc_coeff,axis=0)[2][new_cn_index],bins=np.arange(-lim_ar,lim_ar,0.01))
        i_roimax = np.argmax(np.abs(np.mean(svc_coeff,axis=0))[2])
        _=ax[2].hist(np.mean(svc_coeff,axis=0)[2][i_roimax],bins=np.arange(-lim_ar,lim_ar,0.01),color='r')

        output.mkdir(parents=True, exist_ok=True)
        figname= f'w_dist.{save_format}'
        fig.savefig(output/figname,format=save_format,bbox_inches="tight")

        # plot max weight neurons
        dff_bci_smooth = smooth_dff_by_trial(dff_bci) 

        start_bci_trial = epoch_data['start_bci_trial']
        stop_bci_trial = epoch_data['stop_bci_trial']    

        # get dimensions
        n_rois = dff_bci_smooth.shape[0]  # number of neurons
        n_trials = len(start_bci_trial)  # number of trials 
        max_trial_duration = np.max(stop_bci_trial - start_bci_trial)

        # initialize
        dff_by_trial_st_end = np.full((n_rois, n_trials, max_trial_duration*2), np.nan)

        for trial, (start_idx, stop_idx) in enumerate(zip((start_bci_trial).astype(int),stop_bci_trial.astype(int))):
            # add dff_smooth for given trial window
            dff_by_trial_st_end[:, trial, :int(stop_idx-start_idx)] = dff_bci_smooth[:, start_idx:stop_idx]


        for i in range(3):
            frames_before = 600
            shifts = ((thresh_crossing_frames) -frames_before).values
            dff_bci_alignon_thr = indep_roll(dff_by_trial_st_end,-shifts,axis=-1)

            i_roi = np.argmax(np.abs(np.mean(svc_coeff,axis=0))[i])
            fig,ax=plt.subplots(figsize=(22,8))
            plt.pcolormesh(dff_bci_alignon_thr[i_roi,:,:1000],cmap='magma',vmin=0,vmax=5)
            plt.colorbar(label='dFF')
            ax.vlines([frames_before],[0],[n_trials],'b',linestyle='--')
            plt.xlabel('2P frames')
            plt.ylabel('Trials')
            original_roi_idx = valid_rois.index[i_roi]
            plt.title(f'BCI ROI nº{original_roi_idx}')

            ax.set_xticks([frames_before-115,frames_before])
            ax.set_xticklabels([np.round(-115/frame_rate,1),0])
            ax.set_xlabel('Aligned on threshold crossing time [s]')

            output.mkdir(parents=True, exist_ok=True)
            figname= f'ddf_aligned_on_threshold_ROI{original_roi_idx}_{i+1}rd.{save_format}'
            fig.savefig(output/figname,format=save_format,bbox_inches="tight")

        # plot condition neuron
        frames_before = 600
        shifts = (thresh_crossing_frames -frames_before).values
        dff_bci_alignon_thr = indep_roll(dff_by_trial_st_end,-shifts,axis=-1)

        i_roi = new_cn_index
        original_roi_idx = valid_rois.index[i_roi]
        fig,ax=plt.subplots(figsize=(22,8))
        plt.pcolormesh(dff_bci_alignon_thr[i_roi,:,:1000],cmap='magma',vmin=0,vmax=5)
        plt.colorbar(label='dFF')
        ax.vlines([frames_before],[0],[n_trials],'b',linestyle='--')

        plt.ylabel('Trials')
        plt.title(f'BCI ROI nº{original_roi_idx}')
        ax.set_xticks([frames_before-115,frames_before])
        ax.set_xticklabels([np.round(-115/frame_rate,1),0])
        ax.set_xlabel('Aligned on threshold crossing time [s]')

        output.mkdir(parents=True, exist_ok=True)
        figname= f'ddf_aligned_on_threshold_ROI{original_roi_idx}CN.{save_format}'
        fig.savefig(output/figname,format=save_format,bbox_inches="tight")

        plt.close('all')
        print('saving results')
        print(output)
        res = results.Results({'pipeline':'svm','score_train':score_train,'score_test':score_test,'svc_coeff':svc_coeff,
            'rois_idx':valid_rois.index.values,'dff_by_trial_st_end':dff_by_trial_st_end})
        res.to_python_hdf5(output/'svm_results.h5')


from typing import List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_lick_spout_steps_dff(dff_signal:np.ndarray,zaber_steps:np.ndarray,frames_before:int,idx_threshold_change:List,subject_id:str,session_name:str,i_roi:int,cn:bool,colors:List,save:bool=False, savepath:str='/scratch',save_format:str='jpg'):
    """
    Plot delta F/F (dF/F) signals and lick spout steps across trials.
       
    Parameters
    ----------
    dff_signal : numpy.ndarray
        2D array of delta F/F signals with shape (n_trials, n_timepoints).
        Each row represents one trial's dF/F trace.
    zaber_steps : numpy.ndarray
        Array of Zaber step positions for each trial.
        Should have same number of rows as dff_signal.
    frames_before : int
        Number of frames before the alignment event.
        Used to mark the reward time point and set x-axis labels.
    idx_threshold_change : array-like
        Indices marking transitions between different threshold periods.
    subject_id : str
        Identifier for the experimental subject.
    session_name : str
        Identifier for the experimental session.
    i_roi : int
        Region of interest (ROI) number being plotted.
    cn : bool
        Flag indicating if this is a conditioned neuron (CN). If True, 
        adds 'CN' suffix to ROI label.
    save : bool, optional
        Whether to save the figure. Default is False.
    savepath : str, optional
        Base directory path for saving figures. Default is '/scratch'.
    save_format : str, optional
        File format for saved figure ('jpg', 'png', 'pdf', etc.). 
        Default is 'jpg'.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    
    """
    ntrials = dff_signal.shape[0]
    max_act = np.nanmax(dff_signal)

    y_space = np.arange(0,ntrials*max_act,max_act)
    if len(y_space)>ntrials:
        y_space = y_space[:-1]
    dff_signal = dff_signal + y_space.reshape(-1,1)
    
    fig,ax = plt.subplots(1,2,figsize=(20,15),width_ratios=[4, 1],sharey=True)

    _=ax[0].plot(dff_signal[:,:frames_before+100].T,color='dimgrey')
    _=ax[0].scatter(zaber_steps,np.zeros(zaber_steps.shape)+y_space.reshape(-1,1),color='b',marker='|')

    ax[0].vlines(frames_before, 0,ntrials*max_act, color='grey', label = 'Reward')
    
    # Add colored bands for different threshold periods
    for iidx in range(len(idx_threshold_change)-1):
        ax[0].axhspan(ymin=idx_threshold_change[iidx]*max_act, 
                  ymax=idx_threshold_change[iidx+1]*max_act,
                  xmin=0, xmax=frames_before, 
                  color=colors[iidx], alpha=0.1, 
                  label=f'Threshold {iidx+1}')

    diff_steps=np.abs(np.diff(zaber_steps,axis=1))
    mean_steps = np.nanmean(diff_steps,axis=1)
    std_steps = np.nanstd(diff_steps,axis=1)

    ax[1].errorbar(mean_steps,np.arange(0,ntrials*max_act,max_act), xerr=std_steps, color='r',fmt='-o', capsize=4, label='Data with Std Dev')

    ax[0].set_xticks([frames_before-100,frames_before])
    ax[0].set_xticklabels(['-100','0'])
    ax[0].set_yticks(y_space)
    ax[0].set_yticklabels(np.arange(1,ntrials+1))
    ax[0].set_xlabel('Frames aligned on threshold crossing time')
    ax[0].set_ylabel('Trials')
    if cn:
        i_roi = str(i_roi)+'CN'
    ax[0].set_title(f'ROI nº{i_roi}')
    plt.subplots_adjust(wspace=0, hspace=0)
    if save:
        subdir = f'{subject_id}/{session_name}'
        output = Path(f'{savepath}/{subdir}')
        output.mkdir(parents=True, exist_ok=True)
        figname= f'lick_spout_steps_dff_ROI{i_roi}.{save_format}'
        fig.savefig(output/figname,format=save_format,bbox_inches="tight")

    return fig
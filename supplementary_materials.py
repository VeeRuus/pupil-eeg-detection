# Supplementary figures and analysis

"""
Imports and constants
"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from datamatrix import series as srs, operations as ops, io, DataMatrix
import time_series_test as tst
import scipy.stats
from scipy.stats import linregress
from statsmodels.formula.api import mixedlm
from analysis_utils import *
plt.style.use('default')

"""
Read in the final cleaned datamatrix
"""

dm = io.readtxt('data-clean-R2.csv')

"""
PSDs
"""
# Ensure consistent epoch timing for all subjects
tmin, tmax = -1, 0

all_epochs = []  # List to store all epochs
montage = mne.channels.make_standard_montage('biosemi64')

for subject_nr, sdm in ops.split(dm.subject_nr):
    raw, events, metadata = read_subject(subject_nr)
    raw.set_channel_types({'EXG3':'eog'})
    raw.set_montage(montage, match_case=False)

    if subject_nr == 7:
        raw.info["bads"].extend(['FC2','CP2']) 
        raw.interpolate_bads()

    # Load full datamatrix as it goes into the analysis
    # Get only the datamatrix subset for this subject
    # In the metadata, add column 'remove' which is by default 1
    metadata['remove'] = 1
    # Mark all rows in the metadata that exist in the datamatrix by setting remove to 0
    trials = set(sdm.trialid)
    metadata.loc[metadata['trial_number'].isin(trials), 'remove'] = 0

    # Create epochs with defined tmin, tmax, and baseline
    fix_epoch = mne.Epochs(raw, eet.epoch_trigger(events, STIM_TRIGGER), tmin=tmin, tmax=tmax, 
                           metadata=metadata, baseline=(None, 0))
    fix_epoch = fix_epoch["remove == 0"]
    
    # Append epochs to the list
    all_epochs.append(fix_epoch)
    
    # Compute PSD
    psd = fix_epoch.compute_psd(fmin=1, fmax=30)
    
    # Plot and save individual
    fig = psd.plot(show=False, picks=CHANNELS)
    fig.suptitle(f'Subject {subject_nr}', fontsize=12, x=0.2)
    plot_filename = f'psd_subject_{subject_nr}.png'
    fig.savefig(plot_filename)
    plt.close(fig)

# Concatenate all epochs from different subjects
combined_epochs = mne.concatenate_epochs(all_epochs)
psd_combined = combined_epochs.compute_psd(fmin=1, fmax=30)
fig = psd_combined.plot(picks=CHANNELS)
fig.suptitle('Group average', fontsize=12, x=0.2)
plot_filename = f'average_psd.png'
fig.savefig(plot_filename)
plt.close(fig)

"""
Topoplots
"""

# Ensure consistent epoch timing for all subjects
tmin, tmax = -1, 0

all_epochs = []  # Note: Initialize the list to collect all epochs
montage = mne.channels.make_standard_montage('biosemi64')

for subject_nr, sdm in ops.split(dm.subject_nr):
    raw, events, metadata = read_subject(subject_nr)
    raw.set_channel_types({'EXG3':'eog'})
    raw.set_montage(montage, match_case=False)

    if subject_nr == 7:
        raw.info["bads"].extend(['FC2','CP2']) 
        raw.interpolate_bads()

    metadata['remove'] = 1
    trials = set(sdm.trialid)
    metadata.loc[metadata['trial_number'].isin(trials), 'remove'] = 0

    # Create epochs with defined tmin and tmax
    fix_epoch = mne.Epochs(raw, eet.epoch_trigger(events, STIM_TRIGGER), tmin=tmin, tmax=tmax, metadata=metadata, baseline=(None, 0))
    fix_epoch = fix_epoch["remove == 0"]

    all_epochs.append(fix_epoch)
    
    psd = fix_epoch.compute_psd()
    
    fig = psd.plot_topomap(show=False)
    fig.suptitle(f'Subject {subject_nr}', fontsize=12)
    plot_filename = f'topomap_subject_{subject_nr}.png'
    fig.savefig(plot_filename)
    plt.close(fig)

# Concatenate all epochs from different subjects
combined_epochs = mne.concatenate_epochs(all_epochs)
psd_combined = combined_epochs.compute_psd()

fig = psd_combined.plot_topomap()
fig.suptitle('Group average', fontsize=12, x=0.2)
plot_filename = f'average_topomap.png'
fig.savefig(plot_filename)
plt.close(fig)

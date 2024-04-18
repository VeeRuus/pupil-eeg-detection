"""This module implements helper functions for the intial steps of data parsing.
"""
import mne
import numpy as np
import multiprocessing as mp
import mne
from mne.time_frequency import tfr_morlet
from datamatrix import convert as cnv, operations as ops, functional as fnc
import eeg_eyetracking_parser as eet
from data_adapter import read_subject, get_eyetracking_data


SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 19]
FULL_FREQS = np.arange(4, 30, 1)
CHANNELS = 'O1', 'O2', 'Oz', 'POz', 'Pz', 'P3', 'P4', 'P7', 'P8'

BINS = 5
STIM_TRIGGER = 1
FACTORS = ['bin_pupil','correct']

EPOCHS_KWARGS = dict(tmin=-1, tmax=.1, picks='eeg',
                     preload=True, reject_by_annotation=False,
                     baseline=None)


def subject_data(subject_nr):
    """Performs preprocessing for a single participant. This involves basic
    EEG preprocessing and subsequent epoching and extraction of TFR data. The
    result is a single DataMatrix that contains all information for final
    analysis.
    
    Parameters
    ----------
    subject_nr: int
    
    Returns
    -------
    DataMatrix
    """
    print(f'Processing subject {subject_nr}')
    raw, events, metadata = read_subject(subject_nr)
    # First get the correct epoch (-2.5 until .5 around the stimulus)
    fix_epoch = mne.Epochs(raw, eet.epoch_trigger(events, STIM_TRIGGER), tmin=-2.5, tmax=.5,
                           metadata=metadata, picks=CHANNELS)
    fix_tfr = tfr_morlet(fix_epoch, freqs=FULL_FREQS, n_cycles=4, n_jobs=-1,
                             return_itc=False, use_fft=True, average=False)
    fix_tfr.crop(-2, 0) # for plotting etc use -1,0
    dm = cnv.from_pandas(metadata)
    dm.tfr = cnv.from_mne_tfr(fix_tfr, ch_avg=True)
    # We need to use this special z-scoring function to z-score each frequency
    # band separately to take into account 1/F power differences
    dm.tfr = z_by_freq(dm.tfr)
    # Get pupil size
    pupil = eet.PupilEpochs(raw, eet.epoch_trigger(events, STIM_TRIGGER), tmin=-2.5,
                            tmax=.5, metadata=metadata, baseline=None)
    dm.pupil = cnv.from_mne_epochs(pupil, ch_avg=True)
    #dm.pupil_mm = (0.00714*dm.pupil)**0.5 #not sure if this is an appropriate conversion for the specific lab
    dm.pupil_z = ops.z(dm.pupil)
    return dm


@fnc.memoize(persistent=True, key='merged-data')
def get_merged_data():
    """Merges data for all participants into a single DataMatrix. Uses
    multiprocessing for performance.
    
    Returns
    -------
    DataMatrix
    """
    # First call this once to have it memoized
    get_eyetracking_data()
    return fnc.stack_multiprocess(subject_data, SUBJECTS, processes=20)


def z_by_freq(col):
    """Performs z-scoring across trials, channels, and time points but 
    separately for each frequency.
    
    Parameters
    ----------
    col: MultiDimensionalColumn
    
    Returns
    -------
    MultiDimensionalColumn
    """
    zcol = col[:]
    for i in range(zcol.shape[1]):
        zcol._seq[:, i] = (
            (zcol._seq[:, i] - np.nanmean(zcol._seq[:, i]))
            / np.nanstd(zcol._seq[:, i])
        )
    return zcol

# def decode_subject(subject_nr):
#     read_subject_kwargs = dict(subject_nr=subject_nr)
#     return bdu.decode_subject(read_subject_kwargs=read_subject_kwargs,
#          factors=FACTORS, epochs_kwargs=EPOCHS_KWARGS,
#          trigger=STIM_TRIGGER, window_stride=1, window_size=90,
#          n_fold=4, epochs=4, read_subject_func=read_subject,
#          patch_dta_func=add_bin_pupil) #add bin pupil adds a bin pupil csv that is created in an earlier stage of the data processing
         

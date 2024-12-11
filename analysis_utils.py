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

IAF_DICT = {1 : 9.56, 
            2 : 8.37, 
            3 : 10.36,
            4 : 11.15,
            5 : 12.75,
            6 : 9.96,
            7 : 8.37,
            8 : 9.96, 
            9 : 9.16, 
            10 : 8.76,
            12 : 9.56,
            13 : 9.56,
            15 : 9.96, 
            16 : 8.76,
            17 : 9.56,
            19 : 10.76}

montage = mne.channels.make_standard_montage('biosemi64')

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
    raw.set_channel_types({'EXG3':'eog'})
    raw.set_montage(montage, match_case=False)
    if subject_nr == 6:
        raw.set_eeg_reference(ref_channels = "average")
    # First get the correct epoch (-2.5 until stim onset (t=0))
    # use eet.autoreject_epochs() instead. You can use it the same way.
    fix_epoch = eet.autoreject_epochs(raw, eet.epoch_trigger(events, STIM_TRIGGER), tmin=-2.5, tmax=0,
                           metadata=metadata, picks=CHANNELS)
    # fix_epoch = mne.Epochs(raw, eet.epoch_trigger(events, STIM_TRIGGER), tmin=-2.5, tmax=0,
    #                        metadata=metadata, picks=CHANNELS)
    # tfr first over full frequencies and then the individual alpha frequency
    fix_tfr = tfr_morlet(fix_epoch, freqs=FULL_FREQS, n_cycles=4, n_jobs=-1,
                             return_itc=False, use_fft=True, average=False)
    iaf_value = IAF_DICT.get(subject_nr, None)
    iaf_range = [iaf_value - 1, iaf_value + 1]
    iaf_tfr = tfr_morlet(fix_epoch, freqs=iaf_range, n_cycles=4, n_jobs=-1,
                             return_itc=False, use_fft=True, average=False) 
    fix_tfr.crop(-2, -0.5) # to filter out edge artefacts
    iaf_tfr.crop(-2, -0.5) # to filter out edge artefacts
    dm = cnv.from_pandas(metadata)
    dm.tfr = cnv.from_mne_tfr(fix_tfr, ch_avg=True)
    dm.tfr_iaf = cnv.from_mne_tfr(iaf_tfr, ch_avg=True)
    # We need to use this special z-scoring function to z-score each frequency
    # band separately to take into account 1/F power differences
    dm.tfr = z_by_freq(dm.tfr)
    dm.tfr_iaf = z_by_freq(dm.tfr_iaf)
    # Get pupil size
    pupil = eet.PupilEpochs(raw, eet.epoch_trigger(events, STIM_TRIGGER), tmin=-2.5,
                            tmax=.5, metadata=metadata, baseline=None)
    dm.pupil = cnv.from_mne_epochs(pupil, ch_avg=True)
    dm.pupil_z = ops.z(dm.pupil)
    # Get the eyetracking data
    # raw.set_channel_types({'GazeX':'eeg'})
    gaze_x = mne.Epochs(raw, eet.epoch_trigger(events, STIM_TRIGGER), tmin=-2.5, tmax=0,
                        metadata=metadata, picks=['GazeX'])
    dm.gaze_x = cnv.from_mne_epochs(gaze_x, ch_avg=True)
    gaze_y = mne.Epochs(raw, eet.epoch_trigger(events, STIM_TRIGGER), tmin=-2.5, tmax=0,
                        metadata=metadata, picks=['GazeY'])
    dm.gaze_y = cnv.from_mne_epochs(gaze_y, ch_avg=True)
    

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
    return fnc.stack_multiprocess(subject_data, SUBJECTS, processes=10)


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
         

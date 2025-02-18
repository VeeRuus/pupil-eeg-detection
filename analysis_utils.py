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
from datamatrix import MultiDimensionalColumn


SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 19]
FULL_FREQS = np.arange(4, 30, 1)
CHANNELS = 'O1', 'O2', 'Oz', 'POz', 'Pz', 'P3', 'P4', 'P7', 'P8'

BINS = 5
STIM_TRIGGER = 1
FACTORS = ['bin_pupil','correct']

EPOCHS_KWARGS = dict(tmin=-1, tmax=.1, picks='eeg',
                     preload=True, reject_by_annotation=False,
                     baseline=None)

THETA_BAND = (4, 7)
BETA_BAND = (13, 30)

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
    EEG preprocessing and subsequent epoching and extraction of PSD data. The
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
        
    # First get the correct epoch (-1 until stim onset (t=0))
    fix_epoch_fft = eet.autoreject_epochs(raw, 
                                          eet.epoch_trigger(events, STIM_TRIGGER), 
                                          tmin=-1, tmax=0,
                                          metadata=metadata, 
                                          picks=CHANNELS)
    # compute psd in pre-stim epoch 
    
    # FFT on pre-stim epoch
    psd_output = fix_epoch_fft.compute_psd(fmin=4, fmax=30, picks=CHANNELS) 
    # Extract PSD data and frequencies
    psd = psd_output.get_data()  # PSD data for the given frequency band
    freqs = psd_output.freqs  # Frequency information

    # Individual alpha frequency specific to the subject
    iaf_value = IAF_DICT.get(subject_nr, None)
    iaf_range = [iaf_value - 1, iaf_value + 1]

    # get power in each band
    theta_power = compute_band_power(psd, freqs, THETA_BAND)
    iaf_power = compute_band_power(psd, freqs, iaf_range)
    beta_power = compute_band_power(psd, freqs, BETA_BAND)

    # Z-scoring the power bands using the new function
    theta_power_z = z_score_power_by_frequency(theta_power)
    iaf_power_z = z_score_power_by_frequency(iaf_power)
    beta_power_z = z_score_power_by_frequency(beta_power)

    dm = cnv.from_pandas(metadata)
    dm.theta_power = MultiDimensionalColumn(shape=(9,))  # 9 corresponds to each channel
    dm.iaf_power = MultiDimensionalColumn(shape=(9,))  # 9 corresponds to each channel
    dm.beta_power = MultiDimensionalColumn(shape=(9,))  # 9 corresponds to each channel

    dm.theta_power._seq[fix_epoch_fft.metadata.index] = theta_power_z
    dm.iaf_power._seq[fix_epoch_fft.metadata.index] = iaf_power_z
    dm.beta_power._seq[fix_epoch_fft.metadata.index] = beta_power_z
    
    # TFR ANALYSIS first over full frequencies and then the individual alpha frequency
    fix_epoch_tfr = eet.autoreject_epochs(raw, 
                                      eet.epoch_trigger(events, STIM_TRIGGER), 
                                      tmin=-2.5, tmax=0,metadata=metadata, 
                                      picks=CHANNELS)
    fix_tfr = tfr_morlet(fix_epoch_tfr, freqs=FULL_FREQS, n_cycles=4, n_jobs=-1,
                             return_itc=False, use_fft=True, average=False)
    iaf_tfr = tfr_morlet(fix_epoch_tfr, freqs=iaf_range, n_cycles=4, n_jobs=-1,
                             return_itc=False, use_fft=True, average=False) 
    fix_tfr.crop(-2, -0.5) 
    iaf_tfr.crop(-2, -0.5) 
    # dm = cnv.from_pandas(metadata)
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


@fnc.memoize(persistent=True) # removed key= 'merged_data'
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

def compute_band_power(psd, freqs, band):
        band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.mean(psd[:, :, band_idx], axis=-1)

def z_score_power_by_frequency(power_data):
    """Performs z-scoring across trials, channels, and time points separately for each frequency.

    Parameters
    ----------
    power_data: np.ndarray
        A multi-dimensional array where the last dimension represents frequencies.

    Returns
    -------
    np.ndarray
        The z-scored data with the same shape as the input power_data.
    """
    z_power_data = np.empty_like(power_data)
    for freq_idx in range(power_data.shape[-1]):
        mean = np.nanmean(power_data[..., freq_idx])
        std = np.nanstd(power_data[..., freq_idx])
        z_power_data[..., freq_idx] = (power_data[..., freq_idx] - mean) / std
    return z_power_data

# def decode_subject(subject_nr):
#     read_subject_kwargs = dict(subject_nr=subject_nr)
#     return bdu.decode_subject(read_subject_kwargs=read_subject_kwargs,
#          factors=FACTORS, epochs_kwargs=EPOCHS_KWARGS,
#          trigger=STIM_TRIGGER, window_stride=1, window_size=90,
#          n_fold=4, epochs=4, read_subject_func=read_subject,
#          patch_dta_func=add_bin_pupil) #add bin pupil adds a bin pupil csv that is created in an earlier stage of the data processing
         

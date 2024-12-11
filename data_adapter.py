"""This module implements a read_subject() that behaves as the 
eet.read_subject() but reads the data in the custom format of this project.
"""
import mne
import numpy as np
from eyelinkparser import parse, defaulttraceprocessor
import mne
from datamatrix import convert as cnv, SeriesColumn, DataMatrix, \
     operations as ops, functional as fnc
from datamatrix._datamatrix._multidimensionalcolumn import \
    _MultiDimensionalColumn


DEPTH = 200


@fnc.memoize(persistent=True)
def read_subject(subject_nr):
    """A wrapper function that behaves as eet.read_subject().
    
    The data should be organized as follows:
    
    data/eyetracking/*.edf
    data/eeg/*.set
    data/eeg/*.fdt
    """
    eye_dm = get_eyetracking_data()
    raw, events, dm = merge_eeg_data(eye_dm, subject_nr)
    # Only keep regular columns because the dataframe doesn't deal well with
    # multidimensional columns, and the gaze and pupil data are added as
    # channels to the raw anyway
    dm = dm[[column for name, column in dm.columns
             if not isinstance(column, _MultiDimensionalColumn)]]
    metadata = cnv.to_pandas(dm)
    return raw, events, metadata


def phasefilter(name):
    return name in ['Fix', 'Stim', 'NoStim']


@fnc.memoize(persistent=True)
def get_eyetracking_data():
    """Reads the eye tracking data and returns it as a single DataMatrix with
    all subjects.
    """
    dm = parse(
        folder='data/eyetracking',
        traceprocessor=defaulttraceprocessor(blinkreconstruct=True,
                                             downsample=10,
                                             mode='advanced'),
        maxtracelen=DEPTH,
        multiprocess=20,
        phasefilter=phasefilter)
    dm.ptrace_stim = SeriesColumn(depth=DEPTH)
    dm.xtrace_stim = SeriesColumn(depth=DEPTH)
    dm.ytrace_stim = SeriesColumn(depth=DEPTH)
    dm.ptrace_Stim.depth = DEPTH
    dm.xtrace_Stim.depth = DEPTH
    dm.ytrace_Stim.depth = DEPTH
    dm.ptrace_NoStim.depth = DEPTH
    dm.xtrace_NoStim.depth = DEPTH
    dm.ytrace_NoStim.depth = DEPTH
    for row in dm:
        row.ptrace_stim = (row.ptrace_Stim if row.target_present == 'yes'
                           else row.ptrace_NoStim)
        row.xtrace_stim = (row.xtrace_Stim if row.target_present == 'yes'
                           else row.xtrace_NoStim)
        row.ytrace_stim = (row.ytrace_Stim if row.target_present == 'yes'
                           else row.ytrace_NoStim)
        row.onset_stim = (row.t_onset_Stim if row.target_present == 'yes'
                          else row.t_onset_NoStim)
    del dm.ptrace_Stim
    del dm.ptrace_NoStim
    del dm.xtrace_Stim
    del dm.xtrace_NoStim
    del dm.ytrace_Stim
    del dm.ytrace_NoStim
    dm.fix_dur = (dm.onset_stim - dm.t_onset_Fix) // 10
    return dm[dm.subject_nr, dm.target_present, dm.condition, dm.onset_stim,
              dm.ptrace_stim, dm.xtrace_stim, dm.ytrace_stim, dm.xtrace_Fix,
              dm.ytrace_Fix, dm.ptrace_Fix, dm.fix_dur, dm.target_present,
              dm.response_time, dm.correct, dm.response, dm.target_xpos,
              dm.target_ypos, dm.difficulty, dm.t_onset_Fix]


def parse_triggers(label):
    if label == '100':  # fix dot
        return 128


def merge_eeg_data(dm, subject_nr):
    """Merges the eeg and eye tracking data for a single subject. The dm is
    the eye-tracking datamatrix as returned by get_eyetracking_data. The
    return value is a raw, events, dm tuple which is almost compatible with
    eet.read_subject(), but not quite, which is why this function is itself
    wrapped in read_subject() defined above.
    """
    print(f'Merging EEG and eye-tracking data for subject {subject_nr}')
    # First only select only the subset of the dm for this subject 
    sdm = dm.subject_nr == subject_nr
    # Now read the EEG data and downsample it to 100 Hz, which is the same
    # as the eye-tracking data
    raw = mne.io.read_raw_eeglab(
        f'data/eeg/ICA_removed_sub{subject_nr}.set')
    raw.resample(100)
    # Some subject-specific hacks to clean up recording artifacts
    if subject_nr == 3:
        raw.annotations.delete(303)
        raw.annotations.delete(13)
    elif subject_nr == 19:
        sdm.condition[sdm.condition != 'practice'] = 'experiment'
    elif subject_nr == 4:
        sdm = sdm[:186] << sdm[190:]
    # Extract events from the annotations, and only keep those that correspond
    # to experimental trials (i.e. non-practice trials). We use the dm for
    # that, because the condition coding is a bit messy in the EEG data.
    event_idx, event_labels = mne.events_from_annotations(raw, parse_triggers)
    if subject_nr == 4:
        assert event_idx.shape == (1036, 3)
    else:
        assert event_idx.shape == (1040, 3)
    event_idx = event_idx[sdm[sdm.condition == 'experiment']]
    sdm = sdm.condition == 'experiment'
    assert len(sdm) == event_idx.shape[0]
    # add trial numbers
    sdm.trial_number = 0
    trials = list(range(1, len(sdm) + 1))
    sdm.trial_number = trials
    # Add an event for the stim. We use the fixation duration and onset for
    # this, rather than the original stimulus trigger, which again was a bit
    # unreliable.
    event_idx = event_idx
    event_idx[:, 2] = 128 + np.arange(len(event_idx)) % 128
    stim_event_idx = np.zeros(event_idx.shape, dtype=int)
    # Make sure that fixation duration is numeric
    sdm.fix_dur[sdm.fix_dur == str] = 0
    stim_event_idx[:, 0] = event_idx[:, 0] + sdm.fix_dur
    stim_event_idx[:, 2] = 1  # stim trigger code
    event_idx = np.concatenate([event_idx, stim_event_idx])
    event_idx = event_idx[event_idx[:, 0].argsort()]
    # Now add the eye-tracking data as three new channels to the EEG
    data = np.empty((3, len(raw)), dtype=float)
    data[:] = np.nan
    for (t, _, code), row in zip(event_idx[event_idx[:, 2] >= 128], sdm):
        data[0, t: t + DEPTH] = row.xtrace_Fix
        data[1, t: t + DEPTH] = row.ytrace_Fix
        data[2, t: t + DEPTH] = row.ptrace_Fix
    ch_names = ['GazeX', 'GazeY', 'PupilSize']
    ch_types = 'misc'
    info = mne.create_info(ch_names=ch_names, sfreq=raw.info['sfreq'],
                           ch_types=ch_types)
    tmp = mne.io.RawArray(data, info)
    raw.add_channels([tmp])
    # Boundary annotations indicate epochs that were rejected based on manual
    # inspection of the data. We recode these into bad_manual annotations so
    # that MNE automatically rejects them.
    bads = mne.Annotations(onset=[], duration=[], description=[])
    for a in raw.annotations:
        if a['description'] == 'boundary':
            bads.append(onset=a['onset'], duration=a['duration'],
                        description='bad_manual')
    raw.set_annotations(raw.annotations + bads)
    return raw, (event_idx, event_labels), sdm

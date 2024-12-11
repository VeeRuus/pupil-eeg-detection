"""
Imports and constants
"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from datamatrix import series as srs, operations as ops, io, DataMatrix, SeriesColumn
import time_series_test as tst
import scipy.stats
from scipy.stats import linregress
from statsmodels.formula.api import mixedlm
from analysis_utils import *
plt.style.use('default')

"""
The heaviest of the parsing is done by get_merged_data. This returns a 
datamatrix where the two most relevant columns are pupil and tfr, where tfr
is a 4D (trial, channel, frequency, time) column.
"""
# get_merged_data.clear()  # Uncomment to reparse
dm = get_merged_data()

# add trial number here before any trials are deleted so the numbers match the metadata
dm.trialid = 0
for subject, sdm in ops.split(dm.subject_nr):
    trials = list(range(1, len(sdm) + 1))
    dm.trialid[sdm] = trials
# check that they match
assert(dm.trial_number == dm.trialid)

# remove trials with no responses or no data in prestim period
dm = dm.correct != ''
dm = dm.fix_dur != 0
# crop epochs for analysis and plotting
dm.crop_tfr = dm.tfr[:,:,-100:] # t-1.5 to -.5 
dm.crop_iaf = dm.tfr_iaf[:,:,-100:]
dm.crop_pupil_z = dm.pupil_z[:, 100:200] # t-1.5 to -.5

"""
Check gaze position and add a column in the dm marking bad gaze trials
"""

dm.crop_gazex = dm.gaze_x[:, 100:200] # t-1.5 to -.5
dm.crop_gazey = dm.gaze_y[:, 100:200] # t-1.5 to -.5
dm.crop_gazex = srs.baseline(dm.crop_gazex, dm.crop_gazex, 0, 50) # since no drift correction was done, we baseline the eye position
dm.crop_gazey = srs.baseline(dm.crop_gazey, dm.crop_gazey, 0, 50) # since no drift correction was done, we baseline the eye position

center_x = 512  # 1024/2
center_y = 384  # 768/2

# initalise col for errors
dm.gaze_error = SeriesColumn(depth=100)
# Iterate over the rows to compute deviations for each sample
for row in dm:
    gaze_x = row.crop_gazex  # Get the series for gaze_x
    gaze_y = row.crop_gazey  # Get the series for gaze_y

    # Calculate deviation for each sample using Pythagorean theorem
    deviation = np.sqrt((gaze_x) ** 2 + (gaze_y) ** 2)

    # Store the deviation series in the new column
    row.gaze_error = deviation

# Compute deviations in degrees of visual angle

# Monitor and experiment specifics
from math import atan2, degrees
h = 34.7        # Monitor height in cm
d = 100        # Distance between monitor and participant in cm
r = 1080        # Vertical resolution of the monitor
# Calculate the number of degrees that correspond to a single pixel
deg_per_px = degrees(atan2(.5 * h, d)) / (.5 * r)

dm.gaze_error_deg = SeriesColumn(depth=100)
for row in dm:
    error_px = row.gaze_error  # Get the deviation in pixels
    # Convert deviation from pixels to degrees
    error_deg = error_px * deg_per_px
    # Store the deviation in degrees series in the new column
    row.gaze_error_deg = error_deg
    
# find trials where more than 10 samples are more than 3.03 degrees away from centre
dm.gaze_bad = 0

def check_consecutive(samples, threshold, num_consecutive):
    """ Check if there are more than num_consecutive values in samples larger than the threshold. """
    consecutive_count = 0
    for sample in samples:
        if sample > threshold:
            consecutive_count += 1
            if consecutive_count > num_consecutive:
                return True
        else:
            consecutive_count = 0
    return False

threshold = 3
num_consecutive = 10

# Iterate over the rows to check the deviation in degrees
for row in dm:
    gaze_error_deg = row.gaze_error_deg  # Get the deviation in degrees series

    # Check if more than 10 consecutive samples exceed the threshold
    if check_consecutive(gaze_error_deg, threshold, num_consecutive):
        row.gaze_bad = 1
    else:
        row.gaze_bad = 0

# check how many bad gaze trials each participant has
for subject_nr, sdm in ops.split(dm.subject_nr):
    print(f'subject {subject_nr}: {sdm.gaze_bad.sum} trials')

    
"""
Visually inspect the trials with eye movements
"""
for row in dm.gaze_bad == 1:
    plt.plot(row.gaze_x, color='blue')
    plt.axhline(512, color='blue')
    plt.plot(row.gaze_y, color='green')
    plt.axhline(384, color='green')
    plt.show()

"""
Remove trials with eye movements
"""
dm = dm.gaze_bad == 0

"""
Look at number of trials per participant
"""

for subject_nr, sdm in ops.split(dm.subject_nr):
    print(f'subject {subject_nr}: {len(sdm)} trials')
    
"""
Finding individual alpha peaks in pre-stimulus epochs
"""

# plots to visually inspect channel power in the fix epoch
for subject in SUBJECTS:
    raw, events, metadata = read_subject(subject)
    fix_epoch = mne.Epochs(raw, eet.epoch_trigger(events, STIM_TRIGGER), tmin=-2.5, tmax=0,
                           metadata=metadata, picks=CHANNELS)
    fix_epoch.compute_psd(fmin=8,fmax=13).plot()

# Defining a function to retrieve channel with max power and the max freq in that channel
def find_iaf(subject_id, epochs, channels):
    """
    Find the individual alpha frequency (IAF) for a given subject.

    Parameters:
    - subject_id (int): The subject number/id.
    - epochs (mne.Epochs): The MNE epochs object containing the data.
    - channels (list): List of channel names.

    Returns:
    - max_power_channel_name (str): The name of the channel with the highest power.
    - max_power_frequency (float): The frequency with the highest power in the selected channel.
    """

    # Compute the PSD in the specified frequency range
    psd = epochs.compute_psd(fmin=8, fmax=13, picks=channels) 
    epoch_spectrum = psd.get_data()  # Get PSD in a numpy array of shape epochs x channels x frequencies

    # Step 1: Average over epochs
    epochs_avg = np.mean(epoch_spectrum, axis=0)  # Result is of shape (channels, frequencies)

    # Step 2: Find the channel with maximum overall power by summing over frequency dimension
    avg_power_ch = np.sum(epochs_avg, axis=1)
    max_power_channel = np.argmax(avg_power_ch)  # Index of the channel with highest power

    # Find which frequency in this channel has max power
    channel_of_interest = epochs_avg[max_power_channel]  # This is of shape (frequencies,)
    max_power_frequency_idx = np.argmax(channel_of_interest)  # Index of the frequency with highest power

    # Retrieve the name of the channel and the frequency value
    max_power_channel_name = channels[max_power_channel]
    max_power_frequency = psd.freqs[max_power_frequency_idx]

    return max_power_channel_name, max_power_frequency

# Loop over subjects and find IAF first in all trials (pre-stim) and then for verification in odd and even trials
subject_nr = []
iaf_ch = []
iaf_freq = []
odd_ch = []
odd_freq = []
even_ch = []
even_freq = []

for subject in SUBJECTS:
    raw, events, metadata = read_subject(subject) # get raw data
    if subject == 6:
        raw.set_eeg_reference(ref_channels = "average") # subject 6 has bad mastoids so using an average reference instead
    fix_epoch = mne.Epochs(raw, eet.epoch_trigger(events, STIM_TRIGGER), tmin=-2.5, tmax=0,metadata=metadata, picks=CHANNELS) 
    # splitting event data and metadata into odd and even trials
    even_events = np.array([events[0][i:i+2] for i in range(0, len(events[0]), 4)]).reshape(-1, 3)
    even_metadata = metadata[0::2]
    
    odd_events = np.array([events[0][i:i+2] for i in range(2, len(events[0]), 4)]).reshape(-1, 3)
    odd_metadata = metadata[1::2]
    # epoching split data
    odd_epoch = mne.Epochs(raw, eet.epoch_trigger(odd_events, STIM_TRIGGER), tmin=-2.5, tmax=0,metadata=odd_metadata, picks=CHANNELS) 
    even_epoch = mne.Epochs(raw, eet.epoch_trigger(even_events, STIM_TRIGGER), tmin=-2.5, tmax=0,metadata=even_metadata, picks=CHANNELS) 
    
    channel, frequency = find_iaf(subject, fix_epoch, CHANNELS)
    odd_channel, odd_frequency = find_iaf(subject, odd_epoch, CHANNELS)
    even_channel, even_frequency = find_iaf(subject, even_epoch, CHANNELS)
    
    subject_nr.append(subject)
    iaf_ch.append(channel)
    iaf_freq.append(frequency)
    odd_ch.append(odd_channel)
    odd_freq.append(odd_frequency)
    even_ch.append(even_channel)
    even_freq.append(even_frequency)

iaf_dm = DataMatrix(length=len(SUBJECTS))
iaf_dm.subject = subject_nr
iaf_dm.channel = iaf_ch
iaf_dm.frequency = iaf_freq
iaf_dm.odd_channel = odd_ch
iaf_dm.odd_frequency = odd_freq
iaf_dm.even_channel = even_ch
iaf_dm.even_frequency = even_freq

io.writetxt(iaf_dm, 'individual_alpha_frequency.csv')


"""
Visualize overall power
"""

plt.imshow(dm.crop_tfr[...], aspect='auto')
plt.yticks(range(dm.crop_tfr.shape[1]), FULL_FREQS)
plt.colorbar()
tick_positions = np.linspace(0, 99, 5)
tick_labels = np.arange(-1.5, -0.25, 0.25)
plt.xticks(ticks=tick_positions, labels=tick_labels)
plt.xlabel('Time relative to stim onset (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Power in Pre-Stimulus Interval')
plt.savefig('power-spectrum.png', dpi=300)


"""
Extract mean power in theta, alpha, and beta frequency bands by averaging over
selected frequencies from the full spectrum
"""
dm.theta = dm.crop_tfr[:, :4][:, ...]
dm.alpha = dm.crop_tfr[:, 4:8][:, ...]
dm.beta = dm.crop_tfr[:, 8:][:, ...]

"""
Extract mean power in individual alpha frequency
"""

# dm.iaf = 0

# for subject, sdm in ops.split(dm.subject_nr):
    
#     # find IAF and round to nearest int
#     for row in iaf_dm:
#         if row.subject == subject:
#             iaf = row.frequency
#     iaf = round(iaf)
    
#     # find IAF index in FULL_FREQS
#     index = np.where(FULL_FREQS == iaf)[0][0]
    
#     # average
#     dm.iaf[sdm] = sdm.crop_tfr[:, index]
    
"""
Create columns for mean power and pupil size things
"""

dm.mean_pupil_z = srs.reduce(dm.crop_pupil_z[:,:])
dm.mean_theta = srs.reduce(dm.theta[:,:])
dm.mean_alpha = srs.reduce(dm.alpha[:,:])
dm.mean_beta = srs.reduce(dm.beta[:,:])
dm.mean_iaf = srs.reduce(dm.crop_iaf[:,:])

# positive z-scores are considered large
# dm.large_pupils = -1
# dm.large_pupils[dm.mean_pupil_z > 0] = 1
# dm.large_alpha = -1
# dm.large_alpha[dm.mean_alpha > 0] = 1
# dm.large_beta = -1
# dm.large_beta[dm.mean_beta > 0] = 1
# dm.large_theta = -1
# dm.large_theta[dm.mean_beta > 0] = 1

# Create bins
# dm.bin_pupil = 0
# for i, bdm in enumerate(ops.bin_split(dm.mean_pupil_z, bins=BINS)):
#     dm.bin_pupil[bdm] = i
# dm.bin_alpha = 0
# for i, bdm in enumerate(ops.bin_split(dm.mean_alpha, bins=BINS)):
#     dm.bin_alpha[bdm] = i
# dm.bin_theta = 0
# for i, bdm in enumerate(ops.bin_split(dm.mean_theta, bins=BINS)):
#     dm.bin_theta[bdm] = i
# dm.bin_beta = 0
# for i, bdm in enumerate(ops.bin_split(dm.mean_beta, bins=BINS)):
#     dm.bin_beta[bdm] = i

valid_dm = ((dm.mean_pupil_z != np.nan) &
            (dm.mean_alpha != np.nan) &
            (dm.mean_beta != np.nan) &
            (dm.mean_theta != np.nan) &
            (dm.mean_iaf != np.nan))


for subject_nr, sdm in ops.split(valid_dm.subject_nr):
    print(f'subject {subject_nr}: {len(sdm)} trials')

"""
Write data to disk
"""
io.writetxt(valid_dm[valid_dm.correct, valid_dm.mean_alpha, valid_dm.mean_theta, valid_dm.mean_beta, valid_dm.subject_nr, valid_dm.mean_iaf, dm.response, dm.mean_pupil_z, dm.trialid], 'data-clean.csv', delimiter=',')

"""
Below some initial visualizations: these are not (necessarily) included in the manuscript 
"""

"""
Overall accuracy and pupil size
"""
dm.bin_mean = 0
x = []
y = []
x_err = []
y_err = []
y_min = []
y_max = []

for i, bdm in enumerate(ops.bin_split(dm.mean_pupil_z, bins=BINS)):
    dm.bin_mean[bdm]= round(bdm['mean_pupil_z'].mean, 2)
    x.append(bdm['mean_pupil_z'].mean)
    ymean = bdm['correct'].mean
    y.append(ymean)
    x_err.append(bdm['mean_pupil_z'].std / len(bdm) ** .5)
    yerr = bdm['correct'].std / len(bdm) ** .5
    y_err.append(yerr)
    y_min.append(ymean - yerr)
    y_max.append(ymean + yerr)

sns.set_style("darkgrid", {"axes.facecolor": ".9"})   
plt.title('Correct ~ Pupil')
plt.fill_between(x, y_min, y_max, alpha=.2, color='orchid')
plt.plot(x, y, 'o-', color='darkmagenta')
# plt.errorbar(x, y, xerr=x_err, yerr=y_err, linestyle='-', color='lightseagreen')
plt.xlabel('Pupil Size')
plt.ylabel('Correct')
plt.show()
plt.savefig(f'correct-pupil-purple.png', dpi=300)



"""
Permutation testing
"""
import itertools as it
from pathlib import Path

def permutation_test(dm, dv, iv):
    print(f'permutation test {dv} ~ {iv}')
    result = tst.lmer_permutation_test(
        dm, formula=f'{dv} ~ {iv}', re_formula=f'~ {iv}',
        groups='subject_nr', winlen=2, suppress_convergence_warnings=True,
        iterations=1000)
    Path(f'output2/tfr-{dv}-{iv}.txt').write_text(str(result))
    

args = []
for dv, iv in it.product(['theta', 'alpha', 'beta'], FACTORS):
    args.append((dm, dv, iv))
with mp.Pool() as pool:
    pool.starmap(permutation_test, args)

"""
"""

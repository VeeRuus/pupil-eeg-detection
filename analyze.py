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
The heaviest of the parsing is done by get_merged_data. This returns a 
datamatrix where the two most relevant columns are pupil and tfr, where tfr
is a 4D (trial, channel, frequency, time) column.
"""
#get_merged_data.clear()  # Uncomment to reparse
dm = get_merged_data()
dm = dm.correct != ''
dm = dm.fix_dur != 0
dm.crop_tfr = dm.tfr[:,:,100:] # last second before stimulus presentation
dm.crop_pupil_mm = dm.pupil_mm[:, 150:250]
dm.crop_pupil_z = dm.pupil_z[:, 150:250]
"""
Look at number of trials per participant
"""
for subject_nr, sdm in ops.split(dm.subject_nr):
    print(f'subject {subject_nr}: {len(sdm)} trials')

"""
Plot pupil over time
"""

dm.trialid = 0
for subject, sdm in ops.split(dm.subject_nr):
    trials = list(range(1, len(sdm) + 1))
    dm.trialid[sdm] = trials


"""
Visualize overall power
"""
plt.imshow(dm.crop_tfr[...], aspect='auto')
plt.yticks(range(dm.crop_tfr.shape[1]), FULL_FREQS)
plt.xticks(range(0, 100, 10), range(0, 1000, 100))
plt.show()


"""
Extract mean power in theta, alpha, and beta frequency bands by averaging over
selected frequencies from the full spectrum
"""
dm.theta = dm.crop_tfr[:, :4][:, ...]
dm.alpha = dm.crop_tfr[:, 4:8][:, ...]
dm.beta = dm.crop_tfr[:, 8:][:, ...]


"""
Create columns for mean power and pupil size things
"""

dm.mean_pupil_z = srs.reduce(dm.crop_pupil_z[:,:])
#dm.mean_pupil_mm = srs.reduce(dm.crop_pupil_mm[:,:])
dm.mean_theta = srs.reduce(dm.theta[:,:])
dm.mean_alpha = srs.reduce(dm.alpha[:,:])
dm.mean_beta = srs.reduce(dm.beta[:,:])

print(dm.mean_beta[:10])
print(dm.beta[:10])

# positive z-scores are considered large
dm.large_pupils = -1
dm.large_pupils[dm.mean_pupil_z > 0] = 1
dm.large_alpha = -1
dm.large_alpha[dm.mean_alpha > 0] = 1
dm.large_beta = -1
dm.large_beta[dm.mean_beta > 0] = 1
dm.large_theta = -1
dm.large_theta[dm.mean_beta > 0] = 1

# Create bins
dm.bin_pupil = 0
for i, bdm in enumerate(ops.bin_split(dm.mean_pupil_z, bins=BINS)):
    dm.bin_pupil[bdm] = i
dm.bin_alpha = 0
for i, bdm in enumerate(ops.bin_split(dm.mean_alpha, bins=BINS)):
    dm.bin_alpha[bdm] = i
dm.bin_theta = 0
for i, bdm in enumerate(ops.bin_split(dm.mean_theta, bins=BINS)):
    dm.bin_theta[bdm] = i
dm.bin_beta = 0
for i, bdm in enumerate(ops.bin_split(dm.mean_beta, bins=BINS)):
    dm.bin_beta[bdm] = i

valid_dm = ((dm.mean_pupil_z != np.nan) &
            (dm.mean_alpha != np.nan) &
            (dm.mean_beta != np.nan) &
            (dm.mean_theta != np.nan))


"""
Write data to disk
"""
io.writetxt(valid_dm[valid_dm.correct, valid_dm.mean_alpha, valid_dm.mean_theta, valid_dm.mean_beta, valid_dm.subject_nr, valid_dm.large_pupils, valid_dm.large_beta, valid_dm.large_alpha, valid_dm.large_theta, dm.response, dm.mean_pupil_z, dm.trialid], 'data.csv')

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

for i, bdm in enumerate(ops.bin_split(dm.mean_pupil, bins=BINS)):
    dm.bin_mean[bdm]= round(bdm['mean_pupil'].mean, 2)
    x.append(bdm['mean_pupil'].mean)
    ymean = bdm['correct'].mean
    y.append(ymean)
    x_err.append(bdm['mean_pupil'].std / len(bdm) ** .5)
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
plt.savefig(f'correct-pupil-purple.png', dpi=300)

# more plots
dm_target = dm[dm.target_present == 'yes'] # this doesn't work'
dm_target.bin_mean_alpha = 0
x_alpha = []
y_alpha = []
x_err_alpha = []
y_err_alpha = []
y_min_alpha = []
y_max_alpha = []

for i, bdm in enumerate(ops.bin_split(dm_target.mean_alpha, bins=BINS)):
    dm_target.bin_mean_alpha[bdm]= round(bdm['mean_alpha'].mean, 2)
    x_alpha.append(bdm['mean_alpha'].mean)
    ymean = bdm['correct'].mean
    y_alpha.append(ymean)
    x_err_alpha.append(bdm['mean_alpha'].std / len(bdm) ** .5)
    yerr = bdm['correct'].std / len(bdm) ** .5
    y_err_alpha.append(yerr)
    y_min_alpha.append(ymean - yerr)
    y_max_alpha.append(ymean + yerr)

sns.set_style("darkgrid", {"axes.facecolor": ".9"})   
plt.title('Hits ~ Alpha')
plt.fill_between(x_alpha, y_min_alpha, y_max_alpha, alpha=.2, color='lightpink')
plt.plot(x_alpha, y_alpha, 'o-', color='deeppink')
# plt.errorbar(x, y, xerr=x_err, yerr=y_err, linestyle='-', color='lightseagreen')
plt.xlabel('Alpha power')
plt.ylabel('Correct (on target-present)')
plt.savefig(f'correct-alpha.png', dpi=300)

"""
Overall heatmap contrasting large and small pupils
"""
tfr_large = (dm.large_pupils == 1).crop_tfr[...]
tfr_small = (dm.large_pupils == -1).crop_tfr[...]
plt.imshow(tfr_large - tfr_small, aspect='auto')
plt.colorbar()
plt.yticks(range(dm.crop_tfr.shape[1]), FULL_FREQS)
plt.xticks(range(0, 100, 10), range(0, 1000, 100))
plt.title('Power Difference Large Pupils - Small Pupils')
plt.xlabel('Time (Samples)')
plt.ylabel('Frequency (Hz)')
plt.savefig(f'large-small-power-heatmap.png',dpi=300)
plt.show()

"""
Heatmap of large pupils
"""
plt.imshow((dm.large_pupils == 1).crop_tfr[...], aspect='auto')
plt.yticks(range(dm.crop_tfr.shape[1]), FULL_FREQS)
plt.colorbar()
plt.xticks(range(0, 100, 10), range(0, 1000, 100))
plt.title('Large Pupils')
plt.xlabel('Time (Samples)')
plt.ylabel('Frequency (Hz)')
plt.show()

"""
Heatmap small pupils
"""
plt.imshow((dm.large_pupils == -1).crop_tfr[...], aspect='auto')
plt.yticks(range(dm.crop_tfr.shape[1]), FULL_FREQS)
plt.colorbar()
plt.xticks(range(0, 100, 10), range(0, 1000, 100))
plt.title('Small Pupils')
plt.xlabel('Time (Samples)')
plt.ylabel('Frequency (Hz)')
plt.show()
plt.show()

"""
Overall heatmap contrasting correct and incorrect trials
"""
tfr_correct = (dm.correct == 1).crop_tfr[...]
tfr_incorrect = (dm.correct == 0).crop_tfr[...]
plt.imshow(tfr_correct - tfr_incorrect, aspect='auto')
plt.colorbar()
plt.yticks(range(dm.crop_tfr.shape[1]), FULL_FREQS)
plt.xticks(range(0, 100, 10), range(0, 1000, 100))
plt.title('Power Difference Correct - Incorrect')
plt.xlabel('Time (Samples)')
plt.ylabel('Frequency (Hz)')
plt.savefig(f'accuracy-power-heatmap.png',dpi=300)
plt.show()

"""
Heatmap correct
"""
plt.imshow((dm.correct == 1).crop_tfr[...], aspect='auto')
plt.yticks(range(dm.crop_tfr.shape[1]), FULL_FREQS)
plt.xticks(range(0, 100, 10), range(0, 1000, 100))
plt.colorbar()
plt.title('Correct')
plt.xlabel('Time (Samples)')
plt.ylabel('Frequency (Hz)')
plt.show()

"""
Heatmap incorrect
"""
plt.imshow((dm.correct == 0).crop_tfr[...], aspect='auto')
plt.yticks(range(dm.crop_tfr.shape[1]), FULL_FREQS)
plt.xticks(range(0, 100, 10), range(0, 1000, 100))
plt.colorbar()
plt.title('Incorrect')
plt.xlabel('Time (Samples)')
plt.ylabel('Frequency (Hz)')
plt.show()

"""
Scatterplots and correlation testing
"""
N = 13251
colors = np.random.rand(N)

plt.style.use('seaborn-darkgrid')
plt.scatter('mean_pupil','mean_beta', data=valid_dm, c=colors, cmap='Blues', alpha=.7)
plt.xlabel('Pupil Size (z)')
plt.ylabel('Beta power (z)')
plt.title('Trial-wise beta band activity and pupil size')
plt.savefig(f'beta-pupil-scatterplot.png',dpi=300)
plt.show()

plt.scatter('mean_pupil','mean_alpha', data=valid_dm, c=colors, cmap='Reds')
plt.xlabel('Pupil Size (z)')
plt.ylabel('Alpha power (z)')
plt.title('Trial-wise alpha band activity and pupil size')
plt.savefig(f'alpha-pupil-scatterplot.png',dpi=300)
plt.show()

print(scipy.stats.pearsonr(valid_dm.mean_pupil, valid_dm.mean_beta)) #sig
print(scipy.stats.pearsonr(valid_dm.mean_pupil, valid_dm.mean_alpha)) #sig
print(scipy.stats.pearsonr(valid_dm.mean_pupil, valid_dm.mean_theta)) #not sig


"""
Mixed linear modelling 
"""

model_pupil = mixedlm(formula='mean_pupil ~ mean_alpha * mean_beta * mean_theta * correct',
                      re_formula='~ mean_alpha * mean_beta * mean_theta * correct',
                      data = valid_dm, groups='subject_nr').fit()
print(model_pupil.summary())

model_alpha = mixedlm(formula='mean_alpha ~ mean_pupil * correct',
                      re_formula='~ mean_pupil * correct',
                      data = valid_dm, groups='subject_nr').fit()
print(model_alpha.summary())

model_beta = mixedlm(formula='mean_beta ~ mean_pupil * correct',
                     re_formula='~ mean_pupil * correct',
                     data = valid_dm, groups='subject_nr').fit()
print(model_beta.summary())

model_theta = mixedlm(formula='mean_theta ~ mean_pupil * correct',
                     re_formula='~ mean_pupil * correct',
                     data = valid_dm, groups='subject_nr').fit()
print(model_theta.summary())


"""
Additional visualization (lineplots of relationships)
"""
plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title('Alpha power ~ pupil size')
tst.plot(valid_dm, dv='alpha', hue_factor='large_pupils', linestyle_factor='correct',
         hues=['blue', 'red'])
plt.xticks(np.arange(0, 101, 10), np.arange(0, 1001, 100))
plt.xlabel('Time (s)')
plt.ylabel('Alpha power (z)')
plt.ylim(-.15, .15)
plt.subplot(223)
plt.title('Pupil size ~ alpha')
tst.plot(valid_dm, dv='crop_pupil', hue_factor='large_alpha', linestyle_factor='correct',
         hues=['blue', 'red'])
plt.xticks(np.arange(0, 101, 10), np.arange(0, 1001, 100))
plt.xlabel('Time (s)')
plt.ylabel('Pupil size (z)')
plt.ylim(-.20, .20)
plt.savefig('alpha-pupil-relationship.png', dpi=300)

def two_way(dm, dv, f1, f2, ylim=None):
    a = np.zeros([BINS, BINS])
    for bin1, bin2, bdm in ops.split(
            dm[f'bin_{f1}'], dm[f'bin_{f2}']):
        a[bin1, bin2] = bdm[dv].mean
    plt.figure(figsize=(12, 4))
    plt.suptitle(f'{dv} as a function of {f1} and {f2}')
    plt.subplots_adjust(wspace=.35)
    plt.subplot(131)
    plt.title('a) Interaction')
    plt.imshow(a, aspect='auto',
               vmin=ylim[0] if ylim else None,
               vmax=ylim[1] if ylim else None)
    plt.ylabel(f'{f1} (bin)')
    plt.xlabel(f'{f2} (bin)')
    plt.subplot(132)
    plt.title(f'b) Effect of {f1}')
    x = []
    y = []
    for _, bdm in ops.split(dm[f'bin_{f1}']):
        x.append(bdm[f'mean_{f1}'].mean)
        y.append(bdm[dv].mean)
    plt.plot(x, y, 'o-')
    if ylim:
        plt.ylim(*ylim)
    plt.ylabel(dv)
    plt.xlabel(f'{f1} (z)')
    plt.subplot(133)
    plt.title(f'c) Effect of {f2}')
    x = []
    y = []
    for _, bdm in ops.split(dm[f'bin_{f2}']):
        x.append(bdm[f'mean_{f2}'].mean)
        y.append(bdm[dv].mean)
    plt.plot(x, y, 'o-')
    if ylim:
        plt.ylim(*ylim)
    plt.ylabel(dv)
    plt.xlabel(f'{f2} (z)')
    plt.show()


two_way(dm, dv='correct', f1='pupil', f2='alpha', ylim=(.65, .75))
two_way(dm, dv='correct', f1='pupil', f2='theta', ylim=(.65, .75))
two_way(dm, dv='correct', f1='pupil', f2='beta', ylim=(.65, .75))

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
    Path(f'output/tfr-{dv}-{iv}.txt').write_text(str(result))
    

args = []
for dv, iv in it.product(['theta', 'alpha', 'beta'], FACTORS):
    args.append((dm, dv, iv))
with mp.Pool() as pool:
    pool.starmap(permutation_test, args)

"""
"""

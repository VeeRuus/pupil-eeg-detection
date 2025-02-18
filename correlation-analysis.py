"""
Supplementary statistical analyses Ruuskanen et al., 2024
"""

"""
imports and read data
"""
import datamatrix
from datamatrix import io, operations as ops, FloatColumn, DataMatrix
import numpy as np
from scipy.stats import pearsonr, ttest_1samp, norm

dm = io.readtxt('data-clean-R2.csv')

"""
cross-correlation analysis of alpha, beta and pupil
"""

# lag pupil by -1 (i.e., shift backwards)
dm.lagged_pupil = FloatColumn

for subject, sdm in ops.split(dm.subject_nr):
  lagged_pupil = np.roll(sdm.mean_pupil_z, -1)
  lagged_pupil[-1] = np.nan
  dm.lagged_pupil[sdm] = lagged_pupil

dm = ((dm.mean_pupil_z != np.nan) &
      (dm.mean_iaf_power != np.nan) &
      (dm.mean_beta_power != np.nan) &
      (dm.mean_theta_power != np.nan) &
      (dm.lagged_pupil != np.nan))


# correlate alpha and pupil and beta and pupil within participants

def compute_participant_correlations(dm):
    alpha_pupil = []
    alpha_pupil_p = []
    beta_pupil = []
    beta_pupil_p = []
    subs = []

    for subject, sdm in ops.split(dm.subject_nr):
        subs.append(sdm.subject_nr[0])

        # Compute correlation and p-value for alpha
        alpha_corr, alpha_p = pearsonr(sdm.mean_iaf_power, sdm.lagged_pupil)
        alpha_pupil.append(round(alpha_corr, 5))
        alpha_pupil_p.append(round(alpha_p, 5))

        # Compute correlation and p-value for beta
        beta_corr, beta_p = pearsonr(sdm.mean_beta_power, sdm.lagged_pupil)
        beta_pupil.append(round(beta_corr, 5))
        beta_pupil_p.append(round(beta_p, 5))

    # Create and populate the DataMatrix
    dm_corr = DataMatrix(length=len(subs))
    dm_corr.subject_nr = subs
    dm_corr.alpha_pupil = alpha_pupil
    dm_corr.alpha_pupil_p = alpha_pupil_p
    dm_corr.beta_pupil = beta_pupil
    dm_corr.beta_pupil_p = beta_pupil_p

    return dm_corr

dm_corr = compute_participant_correlations(dm)

io.writetxt(dm_corr, 'cross-correlations.csv')

"""
t-test correlations
"""

# Sample correlation coefficients from your DataMatrix
correlations_beta = dm_corr.beta_pupil  # Example: a list of correlation coefficients


# Apply Fisher's z-transformation
fisher_z_values = np.arctanh(correlations_beta)

# Perform a one-sample t-test against zero
t_statistic, p_value = ttest_1samp(fisher_z_values, 0)

print(f"T-statistic: {t_statistic}, P-value: {p_value}")

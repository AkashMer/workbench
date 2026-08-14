# Initialize required libraries
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Download the data files from the github repo
# 1. Get the data path
repo_root = Path.cwd()
data_path = repo_root / "data" / "heading-estimation-multisensory-proc"
# 2. Make the directory if it does not exist already
data_path.mkdir(parents=True, exist_ok=True)
# 3. Download the data file to data_path if not already present
if not (data_path / "data.csv").exists():
    data_url = "https://raw.githubusercontent.com/sharootonian/CombinationAndCompetitionHeadingDirection/main/data/data.csv"
    response = requests.get(data_url)
    (data_path / "data.csv").write_bytes(response.content)
if not (data_path / "dataconf.csv").exists():
    dataconf_url = "https://raw.githubusercontent.com/sharootonian/CombinationAndCompetitionHeadingDirection/main/data/dataconf.csv"
    response = requests.get(dataconf_url)
    (data_path / "dataconf.csv").write_bytes(response.content)    

# Load the file and check the structure
bva_data = pd.read_csv(data_path / "data.csv")
print(bva_data.columns)
print(bva_data.shape)

# Number of unique participants
len(bva_data.subj.unique())
# 30 participants, confirmed all participant data present

# Get a general description of the dataframe
bva_data.describe(include='all')
# Max 399 trials per participant
# target: seems uniformly distribution from 0 - 360
# respond makes sense to be > 360 but then error is between -180 to 180: needs further investigation
# trial_duration: most likely represents the duration of whole trial, no way to get response time (needs confirmation)
# I wanna confirm if fb and fb_true match and when offset is zero
# fb_time: Most likely time from start to onset of feedback image

# Load and get the description of dataconf file as well
bva_dataconf = pd.read_csv(data_path / "dataconf.csv")
bva_dataconf.describe(include='all')
# Only two new columns
# conf: A post-hoc cone angle asked from the participant for the door position in their response view
# viewAmount: needs confirmation but the view angle traversed while feedback was visible
# ie. serves as a measure of how much attention was payed to the feedback

# How does high respond correspond to error values?
bva_data.query("respond > 500")
# error = respond - target
(bva_data.respond - bva_data.target - bva_data.error).describe()
# Confirmed => respond is the cumulative angle for the whole trial duration
# Hence, makes sense why > 360

# Confirm trial duration is the whole trial
sns.histplot(bva_data, x='trial_duration') # right-skewed in msec
# Is fb_time < trial_duration always?
(bva_data.fb_time < bva_data.trial_duration).mean() # not 1
# Check which fb_time > trial_duration
bva_data.query("fb_time > trial_duration") # zero rows
# Maybe equal?
bva_data.query("fb_time == trial_duration")
# trial duration is start of encoding period to response button press

# Do fb and fb_true match when offset is zero
(bva_data.eval("fb_diff = fb - fb_true")
    .query("condition == 'FB' & fb_offset == 0")
    .describe()
)
# fb_offset is never zero
# On further confirmation, fb_offset is drawn two ways:
# 1. Gaussian with 0 mean and SD: 30 degrees (70% trials/participant)
# 2. Uniformly from -180 to 180 (30% trials/participant)

# Confirm viewAmount is zero for noFB trials
(
    bva_dataconf.query("condition == 'noFB'")
        .viewAmount.describe()
)
# All zeros, confirmed

# EDA of both types of trials
# General counts per subject across both trials
sns.histplot(bva_dataconf, x='subj', hue='condition',
                stat='proportion', common_norm=False, multiple='dodge')
# Some participants have low number of trials, but they are equally low for both conditions
# Like particiant s4, s5, s9, s10, s21(lowest)

# Error distribuion across both conditions
g = sns.displot(bva_dataconf, x='error', col='condition', kde=True)
for ax, cond in zip(g.axes.flat, g.col_names):
    subset = bva_dataconf.query("condition == @cond")
    mean_val = subset.error.mean()
    median_val = subset.error.median()
    ax.axvline(mean_val, color='red', linestyle='--')
    ax.axvline(median_val, color='green', linestyle='--')
    ax.text(mean_val, ax.get_ylim()[1]*0.9, f'mean={mean_val:.1f}', color='red')
    ax.text(median_val, ax.get_ylim()[1]*0.8, f'median={median_val:.1f}', color='green')
# Both errors are right skewed with slightly shifted centers away from zero error
# But the FB trials are more right skewed
# This suggests that participants generally overestimated the target
# Could this be due to the validity split of the fb_offset
# Since validity labels are not present, I will use label valid for gaussian (0, 30) +/- 3SD
plot_data = bva_dataconf.assign(
    cue_validity=lambda d: np.where(d.fb_offset.abs() <= 90, 'valid', 'nonvalid')
)
g = sns.displot(plot_data, x='error', row='cue_validity', col='condition', kde=True)
for (row_val, col_val), ax in g.axes_dict.items():
    subset = plot_data.query("condition == @col_val & cue_validity == @row_val")
    mean_val = subset.error.mean()
    median_val = subset.error.median()
    ax.axvline(mean_val, color='red', linestyle='--')
    ax.axvline(median_val, color='green', linestyle='--')
    ax.text(mean_val, ax.get_ylim()[1]*0.9, f'mean={mean_val:.1f}', color='red')
    ax.text(median_val, ax.get_ylim()[1]*0.8, f'median={median_val:.1f}', color='green')
# Still slghtly right skewed but much less than FB trials with valid cues

# How does error relate to target value?
# Bin the target into 10 bins of 36 degrees each
plot_data = plot_data.assign(target_bin=pd.cut(plot_data.target, bins=10, precision=0))
# Plot across both condition and cue validity
g = sns.FacetGrid(plot_data, col='condition', row='cue_validity')
g.map(sns.boxplot, 'target_bin', 'error')
g.set(xticklabels=[])
g.set_axis_labels('target (binned, low → high)', 'error')
# Error's variance grows with target angle
# => the variance of error is a function of target angle
# ie. larger the self-motion movement, larger errors
# => less reliability of the proprioceptory/vestibular senses for larger movements
# Thus, a naive forced fusion model is not the right choice here

# Subject level distribution of error variable
subj_plot_data = plot_data.query("condition == 'noFB' | cue_validity == 'valid'")
g = sns.displot(subj_plot_data, x='error', col='subj', col_wrap=6,
                    hue='condition', kind='kde', fill=True, alpha=0.4)
# Check the corresponding means and medians along the subjects
fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
sns.pointplot(subj_plot_data, x='subj', y='error', hue='condition',
              estimator='mean', ax=ax[0])
sns.pointplot(subj_plot_data, x='subj', y='error', hue='condition',
              estimator='median', ax=ax[1])
# Confirms the right skewness and also that FB trials have greater errors across all subjects
# even when the cue validity is good

# Check if a similar pattern holds for noFB vs uniformly derived validity
subj_plot_nonvalid_data = plot_data.query("condition == 'noFB' | cue_validity == 'nonvalid'")
fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
sns.pointplot(subj_plot_nonvalid_data, x='subj', y='error', hue='condition',
              estimator='mean', ax=ax[0])
sns.pointplot(subj_plot_nonvalid_data, x='subj', y='error', hue='condition',
              estimator='median', ax=ax[1])
# Similar pattern holds except fro s9; most likely from low trial count
count_data = (
    plot_data
    .assign(group=lambda d: np.where(d.condition == 'noFB', 'noFB', d.cue_validity))
    .groupby(['subj', 'group']).size().reset_index(name='count')
)
subj_order = sorted(count_data.subj.unique(), key=lambda s: int(s[1:]))
sns.pointplot(count_data, x='subj', y='count', hue='group', order=subj_order)
# s5, s9 and s21 have low trials across all types of trials

# Confidence spread analysis
# Confirm all trials had non-zero conf values
plot_data.query('conf == 0') # 6, most in s14 for low error values
# Get the same relationship of mean of conf across subjects and trial types
fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
sns.pointplot(subj_plot_data, x='subj', y='conf', hue='condition',
              estimator='mean', ax=ax[0])
sns.pointplot(subj_plot_data, x='subj', y='conf', hue='condition',
              estimator='median', ax=ax[1])
# and median
fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
sns.pointplot(subj_plot_nonvalid_data, x='subj', y='conf', hue='condition',
              estimator='mean', ax=ax[0])
sns.pointplot(subj_plot_nonvalid_data, x='subj', y='conf', hue='condition',
              estimator='median', ax=ax[1])
# Not a lot of difference, both track similarly across all subjects
# Slightly lower confidence (higher conf) in case of nonvalid cues
# Need to compare valid vs nonvalid cues
fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
sns.pointplot(plot_data, x='subj', y='conf', hue='cue_validity',
              estimator='mean', ax=ax[0])
sns.pointplot(plot_data, x='subj', y='conf', hue='cue_validity',
              estimator='median', ax=ax[1])
# Mean conf are higher ie. lower confidence for non-valid vs valid
# But medians do not show a clear separation, possible outliers?
sns.displot(plot_data, x='conf', col='cue_validity',
                kind='kde', fill=True, alpha=0.4)
# The distribution of nonvalid trials for conf is much more spread out
# Suggesting an effect of outliers on means, which the medians are not sensitive to
# Also, conf shows an interesting bump in the right tail for valid cue trials
# Confirm against noFB trials
sns.displot(plot_data, x='conf', col='condition',
                kind='kde', fill=True, alpha=0.4)
# Even those shows a smaller bump in the right tail
# Needs more exploration regarding this phenomenon at subject level
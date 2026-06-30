# Import necessary libraries
from pathlib import Path
import zipfile
import io
import numpy as np
import pandas as pd

# Check the structure of the data file
# 1. Get the data root
repo_root = Path.cwd()
data_path = repo_root / "data" / "scene-areas-hierarchy-rsa"

# 2. Initialize the subject 23 zip file
path_to_zip = data_path / "MRI_Scanning_sub23.zip"
zip_file = zipfile.ZipFile(path_to_zip)

# 3. Check the structure of the zip file
zip_file.namelist()[:10]
# 4 runs, t1 anatomical images, each file as a stack of slices

# What is the structure of the folders inside the zip file
file_list = zip_file.namelist()
[name for name in file_list if name.endswith('/')]
# Organized by bold runs and t1

# Get the number of files under each run to compute time for each run
bold_runs = ['bold_run1', 'bold_run2', 'bold_run3', 'bold_run4']
for bold_run in bold_runs:
    print(bold_run, ': ',
    sum(1 for name in file_list if bold_run in name and not name.endswith('/'))*2, 's') # 2 sec for TR
# ~16 mins for each run
# How is each run divided for the 4 period: walking, facing, targeting, choice

# Load the behavior zip file
path_to_behavior_zip = data_path / "fMRI_behavior.zip"
behavior_zip_file = zipfile.ZipFile(path_to_behavior_zip)
behavior_zip_file.namelist()
# Files are under fMRI_behavior/sub_xx_formal_rawdata.txt and sub_xx_formal_Time_record_t.txt

# What is the separator used in the text files?
io.TextIOWrapper(behavior_zip_file.open('fMRI_behavior/sub_23_formal_rawdata.txt')).readline()
# Tab separator
io.TextIOWrapper(behavior_zip_file.open('fMRI_behavior/sub_23_formal_Time_record_t.txt')).readline()
# Same here

# Define the internal path to the trial conditions and timing files
formal_rawdata_filepath = 'fMRI_behavior/sub_23_formal_rawdata.txt'
formal_Time_record_filepath = 'fMRI_behavior/sub_23_formal_Time_record_t.txt'
# Load the tab separated data
trial_conditions = pd.read_table(behavior_zip_file.open(formal_rawdata_filepath), header=None)
timing_onsets = pd.read_table(behavior_zip_file.open(formal_Time_record_filepath), header=None)
# Check the shape and head of each tables
trial_conditions.shape
trial_conditions.head()
# Columns of interest: 1 map, 2 walking direction, 8 allocentric direction and 9 egocentric direction
# What are the unique number of maps used?
trial_conditions[1].value_counts() # 3 maps
# What are the unique number of walking directions?
trial_conditions[2].value_counts() # 4 directions
# Check out the trial timings and period onset timings
timing_onsets.shape
timing_onsets.head()
# How many trials per run?
timing_onsets[12].value_counts()
# 40 trials for each of the bold 1-4 runs
# How are the periods marked?
np.diff(timing_onsets.iloc[0])
# Columns 4 & 6 gives onset of walking and facing periods, the ones I am interested in

# After some reading of the SM task yesterday, I think it would be better if I exclude
# the facing period as well since the SM task starts with the facing period, so it would
# be better to only have the walking period to capture the scene perception aspect of the
# task. Also the data description explicitly states, the walking period were identical for both
# HND task and SM task

# The video examples of the walking period for either task cover the following combinations
# map 1 - directions 2, 4
# map 4 - directions 2, 3, 4
# map 5 - directions 1, 3, 4
# I will have to check if the map notations are locally specific to each subject since
# each subject was only showed 3 maps pseudorandomly selected
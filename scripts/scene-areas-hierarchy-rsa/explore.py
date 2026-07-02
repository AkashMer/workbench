# Import necessary libraries
from pathlib import Path
import zipfile
import io
import numpy as np
import pandas as pd
import nibabel as nib

# All the fMRI data downloaded using the url manifest from scidb using aria2c
# Command:
# aria2c -c -x 16 -s 16 -k 1M --content-disposition \
# -d /mnt/c/Users/akash/Documents/workbench/data/scene-areas-hierarchy-rsa/raw_fmri \
# -i /mnt/c/Users/akash/Documents/workbench/data/scene-areas-hierarchy-rsa/scidb_manifest.txt

# Check the structure of the data file
# 1. Get the data root
repo_root = Path.cwd()
raw_data_path = repo_root / "data" / "scene-areas-hierarchy-rsa" / "raw_fmri"

# 2. Initialize the subject 23 zip file
path_to_zip = raw_data_path / "MRI_Scanning_sub23.zip"
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
# Close the zip file connection
zip_file.close()

# Load the behavior zip file
path_to_behavior_zip = raw_data_path / "fMRI_behavior.zip"
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
trial_conditions[1].value_counts() # 3 maps - 1, 3, 2
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
formal_rawdata_filepath8 = 'fMRI_behavior/sub_8_formal_rawdata.txt'
# Load the tab separated data
trial_conditions8 = pd.read_table(behavior_zip_file.open(formal_rawdata_filepath), header=None)
# What are the unique number of maps used?
trial_conditions8[1].value_counts() # 3 maps 1, 3, 2
# Same as subject 23

# Confirming for all subjects under fMRI
# Get all the subject ids for the fMRI data
subject_list = [int(name.split('_')[2]) for name in behavior_zip_file.namelist() 
                if 'MACOSX' not in name and 'rawdata' in name]
# Convfirm the number of subjects avaliable
len(subject_list) # 19 subjects, not IDed as 1-19

# Collate all the behavior data into one large dataframe
behavior_data = pd.DataFrame()
for sub in subject_list:
    filepath = f'fMRI_behavior/sub_{sub}_formal_rawdata.txt'
    trial_cond = pd.read_table(behavior_zip_file.open(filepath),
                                names = ['map', 'walking_direction', 'trial_type'],
                                usecols = [1, 2, 4])
    # Col 0 is not participant id
    trial_cond['participant_id'] = sub
    behavior_data = pd.concat([behavior_data, trial_cond])
# What are the unique map values across all subjects?
behavior_data['map'].value_counts()
# All subjects have maps 1, 2 & 3; so no information which 3/6 maps were pseudo randomly selected is available
# Categorical grouping of stimuli, first by the 4 walking directions and then by the 3 unique maps
# is the only way forward

# What is the counts of each grouping for Subject 23?
behavior_data.query('participant_id == 23')[['walking_direction', 'map']].value_counts()
# 12 trials for most of the combinations, maximum of 20 trials in 2,1 combination

# Next Step: Build Stimulus Categorical RDM
directions = behavior_data.query('participant_id == 23')['walking_direction'].unique()
# Sort them in ascending order
directions = np.sort(directions)
# Build the RDM for walking directions
walking_direction_rdm = (directions[:, None] != directions[None, :]).astype(int)

# Do the same for map
maps = behavior_data.query('participant_id == 23')['map'].unique()
# Sort them in ascending order
maps = np.sort(maps)
# Build the RDM for walking directions
map_rdm = (maps[:, None] != maps[None, :]).astype(int)
# Close the behavior zip file connection
behavior_zip_file.close()

# Extract the fMRI data and load
# 1. Extract under subject_23 folder
# extract_path = data_path / "extracted" / "subject_23"
# zip_file.extractall(extract_path)

# DCM to Nifti conversion done in terminal
# dcm2niix -z y -f "sub23_bold_run1" 
# -o data/scene-areas-hierarchy-rsa/nifti/sub23 
# data\scene-areas-hierarchy-rsa\extracted\subject_23\MRI_Scanning_sub23\bold_run1

# Load the nifti bold_run1 file
# nifti_path = data_path / 'nifti' / 'sub23' / 'sub23_bold_run1.nii.gz'
# sub23_bold_run1 = nib.load(nifti_path)
# # Check headers
# print(sub23_bold_run1.header)
# 4d image: 112 x 112 x 62 voxels x 495 volumes

# All dcm files converted to nifti in terminal
# Rearranged in BIDS format under the data/scene-areas-hierarchy-rsa/bids

# fMRIPrep setup
# docker run --rm \
#   -v -path to bids-:/data:ro \
#   -v -path to output-:/out \
#   -v -path to free surfer license-:/opt/freesurfer/license.txt:ro \
#   nipreps/fmriprep:25.2.0 \
#   /data /out participant \
#   --participant-label 23 \
#   --fs-license-file /opt/freesurfer/license.txt \
#   --fs-no-reconall \
#   --output-spaces MNI152NLin2009cAsym \
#   --nprocs 8 \
#   --omp-nthreads 4 \
#   --mem-mb 10000
# fMRIPrep test successful
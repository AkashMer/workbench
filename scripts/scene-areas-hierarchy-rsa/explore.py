# Import necessary libraries
from pathlib import Path
import zipfile
import io
import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
from nibabel.nicom import dicomwrappers
from ants import from_numpy, from_nibabel_nifti, image_read, registration, apply_transforms, list_to_ndimage, make_image
from ants import read_transform, to_nibabel_nifti
from templateflow import api as tflow
from scipy import ndimage
from scipy.spatial.transform import Rotation
from scipy.stats import mode, norm
from nilearn.masking import compute_epi_mask, intersect_masks
from nilearn.glm import first_level
from nilearn.image import concat_imgs
import matplotlib.pyplot as plt
import seaborn as sns
import rsatoolbox
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
from scipy.spatial import procrustes
from scipy.spatial.distance import squareform

# All the fMRI data downloaded using the url manifest from scidb using aria2c
# Command:
# aria2c -c -x 16 -s 16 -k 1M --content-disposition \
# -d /mnt/c/Users/akash/Documents/workbench/data/scene-areas-hierarchy-rsa/raw_fmri \
# -i /mnt/c/Users/akash/Documents/workbench/data/scene-areas-hierarchy-rsa/scidb_manifest.txt

# Check the structure of the data file
# 1. Get the data root
repo_root = Path.cwd()
data_path = repo_root / "data" / "scene-areas-hierarchy-rsa"
raw_data_path = data_path / "raw_fmri"

# # 2. Initialize the subject 23 zip file
# path_to_zip = raw_data_path / "MRI_Scanning_sub23.zip"
# zip_file = zipfile.ZipFile(path_to_zip)

# # 3. Check the structure of the zip file
# zip_file.namelist()[:10]
# # 4 runs, t1 anatomical images, each file as a stack of slices

# # What is the structure of the folders inside the zip file
# file_list = zip_file.namelist()
# [name for name in file_list if name.endswith('/')]
# # Organized by bold runs and t1

# # Get the number of files under each run to compute time for each run
# bold_runs = ['bold_run1', 'bold_run2', 'bold_run3', 'bold_run4']
# for bold_run in bold_runs:
#     print(bold_run, ': ',
#     sum(1 for name in file_list if bold_run in name and not name.endswith('/'))*2, 's') # 2 sec for TR
# # ~16 mins for each run
# # How is each run divided for the 4 period: walking, facing, targeting, choice
# # Close the zip file connection
# zip_file.close()

# Load the behavior zip file
path_to_behavior_zip = raw_data_path / "fMRI_behavior.zip"
behavior_zip_file = zipfile.ZipFile(path_to_behavior_zip)
# behavior_zip_file.namelist()
# # Files are under fMRI_behavior/sub_xx_formal_rawdata.txt and sub_xx_formal_Time_record_t.txt

# # What is the separator used in the text files?
# io.TextIOWrapper(behavior_zip_file.open('fMRI_behavior/sub_23_formal_rawdata.txt')).readline()
# # Tab separator
# io.TextIOWrapper(behavior_zip_file.open('fMRI_behavior/sub_23_formal_Time_record_t.txt')).readline()
# # Same here

# # Define the internal path to the trial conditions and timing files
# formal_rawdata_filepath = 'fMRI_behavior/sub_23_formal_rawdata.txt'
# formal_Time_record_filepath = 'fMRI_behavior/sub_23_formal_Time_record_t.txt'
# # Load the tab separated data
# trial_conditions = pd.read_table(behavior_zip_file.open(formal_rawdata_filepath), header=None)
# timing_onsets = pd.read_table(behavior_zip_file.open(formal_Time_record_filepath), header=None)
# # Check the shape and head of each tables
# trial_conditions.shape
# trial_conditions.head()
# # Columns of interest: 1 map, 2 walking direction, 8 allocentric direction and 9 egocentric direction
# # What are the unique number of maps used?
# trial_conditions[1].value_counts() # 3 maps - 1, 3, 2
# # What are the unique number of walking directions?
# trial_conditions[2].value_counts() # 4 directions
# # Check out the trial timings and period onset timings
# timing_onsets.shape
# timing_onsets.head()
# # How many trials per run?
# timing_onsets[12].value_counts()
# # 40 trials for each of the bold 1-4 runs
# # How are the periods marked?
# np.diff(timing_onsets.iloc[0])
# # Columns 4 & 6 gives onset of walking and facing periods, the ones I am interested in

# After some reading of the SM task yesterday, I think it would be better if I exclude
# the facing period as well since the SM task starts with the facing period, so it would
# be better to only have the walking period to capture the scene perception aspect of the
# task. Also the data description explicitly states, the walking period were identical for both
# HND task and SM task

# # The video examples of the walking period for either task cover the following combinations
# # map 1 - directions 2, 4
# # map 4 - directions 2, 3, 4
# # map 5 - directions 1, 3, 4
# # I will have to check if the map notations are locally specific to each subject since
# # each subject was only showed 3 maps pseudorandomly selected
# formal_rawdata_filepath8 = 'fMRI_behavior/sub_8_formal_rawdata.txt'
# # Load the tab separated data
# trial_conditions8 = pd.read_table(behavior_zip_file.open(formal_rawdata_filepath), header=None)
# # What are the unique number of maps used?
# trial_conditions8[1].value_counts() # 3 maps 1, 3, 2
# # Same as subject 23

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

# What is the counts of each grouping for Subject 15?
behavior_data.query('participant_id == 15')[['walking_direction', 'map']].value_counts()
# 12 trials for most of the combinations, maximum of 20 trials in 2,1 combination

# --- Scene Area Parcels ----
# Explore the structure of scene area ROI masks
scene_parcels_zip = zipfile.ZipFile(data_path / "scene_parcels.zip")
scene_parcels_zip.namelist()
# B/L PPA, RSC and TOS (OPA)
# Explore the internal structure of each area map
hdr = scene_parcels_zip.read('scene_parcels/lPPA.hdr')
img = scene_parcels_zip.read('scene_parcels/lPPA.img')
lPPA_file = nib.AnalyzeImage.make_file_map()
lPPA_file['header'].fileobj = io.BytesIO(hdr)
lPPA_file['image'].fileobj = io.BytesIO(img)
lPPA_map = nib.AnalyzeImage.from_file_map(lPPA_file)
# Check out the span and voxel size
lPPA_map.affine # Voxel size 2mm & MNI152NLin6Asym space
(np.array(lPPA_map.shape) - 1)*2 # Span: 156mm x 188mm x 136mm
# Are the values binary or probabilistic?
np.unique(lPPA_map.get_fdata()) # 0 & 1 - Binary

# Convert the scene area maps to ANTs format
scene_parcels_zip = zipfile.ZipFile(data_path / "scene_parcels.zip")
scene_areas_pairs = {1: ('lPPA', 'rPPA'), 2: ('lRSC', 'rRSC'), 3:('lTOS', 'rTOS')}
scene_map = np.zeros(lPPA_map.shape, dtype=np.int16)
for label, (left, right) in scene_areas_pairs.items():
    # Get the left data
    hdr_l = scene_parcels_zip.read(f'scene_parcels/{left}.hdr')
    img_l = scene_parcels_zip.read(f'scene_parcels/{left}.img')
    left_file = nib.AnalyzeImage.make_file_map()
    left_file['header'].fileobj = io.BytesIO(hdr_l)
    left_file['image'].fileobj = io.BytesIO(img_l)
    left_map = nib.AnalyzeImage.from_file_map(left_file).get_fdata()

    # Get the right data
    hdr_r = scene_parcels_zip.read(f'scene_parcels/{right}.hdr')
    img_r = scene_parcels_zip.read(f'scene_parcels/{right}.img')
    right_file = nib.AnalyzeImage.make_file_map()
    right_file['header'].fileobj = io.BytesIO(hdr_r)
    right_file['image'].fileobj = io.BytesIO(img_r)
    right_map = nib.AnalyzeImage.from_file_map(right_file).get_fdata()

    # Change the value from zero to appropriate label for each area
    scene_map[(left_map == 1) | (right_map == 1)] = label
# Sanity check
np.unique(scene_map) # 0, 1, 2, 3
# Change to nifti format
scene_nifti = nib.Nifti1Image(scene_map, lPPA_map.affine)
# Convert to ANTs format
scene_ants = from_nibabel_nifti(scene_nifti)
# Close the scene zip file
scene_parcels_zip.close()

# --- Primary Visual Area parcels ---
# Explore the primary area ROI masks
vis_parcels_zip = zipfile.ZipFile(data_path / "visfAtlas.zip")
vis_parcels_zip.namelist()
# Extract the V1/V2 volume and surface maps
atlas_dir = data_path / "atlases"
vis_parcels_zip.extract('visfAtlas/nifti_volume/visfAtlas_MNI152_volume.nii.gz', atlas_dir)
vis_parcels_zip.extract('visfAtlas/nifti_volume/visfAtlas_FSL.xml', atlas_dir)
for hemi in ['lh', 'rh']:
    for region in ['v1d', 'v1v', 'v2d', 'v2v']:
        vis_parcels_zip.extract(f'visfAtlas/FreeSurfer/MPM_{hemi}_{region}.label', atlas_dir)
# Close the zip file
vis_parcels_zip.close()
# Load the file volume file
visf_path = atlas_dir / "visfAtlas" / "nifti_volume" / "visfAtlas_MNI152_volume.nii.gz"
visf_img = nib.load(visf_path)
# Check the voxel size and span
visf_img.affine # Voxel size 1 mm & MNI152NLin6Asym space
(np.array(visf_img.shape) - 1)*1 # 181mm x 217mm x 181mm
np.unique(visf_img.get_fdata()) # Areas labelled by number from XML file
# Check the XML number-area mappings
xml_path = atlas_dir / "visfAtlas" / "nifti_volume" / "visfAtlas_FSL.xml"
print(xml_path.read_text())
#  {ld, lv, rd, rv} : V1 = {12, 15, 28, 31} & V2 = {13, 16, 29, 32}
# Both Julian and visfAtlas ROI maps are in the MNI152NLin6Asym space

# Convert the primary visual areas map to ANTs format
visf_ants = image_read(str(visf_path))
# Exclude other areas besides V1 & V2
v1_labels = [12, 15, 28, 31]
v2_labels = [13, 16, 29, 32]
visf_new_array = np.zeros(visf_ants.numpy().shape, dtype=np.int16)
# Mark the v1 areas
visf_new_array[np.isin(visf_ants.numpy(), v1_labels)] = 1
# Mark the v2 areas
visf_new_array[np.isin(visf_ants.numpy(), v2_labels)] = 2
# Update the ants object with other areas excluded
visf_visual_ants = visf_ants.new_image_like(visf_new_array)

# # Candidate selection logic
# # 1. Excluded if mean Frame Displacement (FD) > 0.2
# # 2. Remaining ranked using a rank-sum approach based on:
# #    - Standard deviation of the frame-to-frame DVARS profile (penalizes sudden spikes)
# #    - HND trial performance (secondary task engagement)
# bold_runs = ['bold_run1', 'bold_run2', 'bold_run3', 'bold_run4']
# subject_rows = []
# # Loop over each subject
# for sub in subject_list:
#     path_to_sub_zip = raw_data_path / f"MRI_Scanning_sub{sub}.zip"
#     run_metrics = []
#     # Loop over each bold run for this subject
#     for bold_run in bold_runs:
#         # Collate all frames from this bold run
#         zip_file = zipfile.ZipFile(path_to_sub_zip)
#         run_files = sorted(name for name in zip_file.namelist() if bold_run in name and not name.endswith('/'))
#         running_sum = np.zeros((112, 112, 62))
#         bold_ants_list = []
#         # Get all frames in ANTs format as a list
#         for file in run_files:
#             # Read DICOM file
#             dcm = pydicom.dcmread(io.BytesIO(zip_file.read(file)))
#             # Wrap the data to read directly from the zipped format
#             dcm_wrap = dicomwrappers.wrapper_from_data(dcm)
#             # Extract the 3D data for each frame
#             dcm_3d = dcm_wrap.get_data()
#             # Add to the running sum to get the mean value for normalization
#             running_sum += dcm_3d
#             # Convert to ANTs format
#             ants_format = from_numpy(dcm_3d, spacing=(2, 2, 2.3))
#             # Append to list
#             bold_ants_list.append(ants_format)
#         # Close the zip file
#         zip_file.close()

#         # Compute the mean frame and convert to ANTs format
#         mean_frame_ants = from_numpy(running_sum/len(run_files), spacing = (2, 2, 2.3))

#         # Compute the motion correction metrics on the BOLD run
#         bold_4d = list_to_ndimage(make_image((*bold_ants_list[0].shape, len(bold_ants_list)), pixeltype='unsigned int',
#                                                 origin = (*bold_ants_list[0].origin, 0)),
#                                     bold_ants_list)

#         # Mean FD Computation - Only using evenly spaced 50 frames
#         frame_fd_running_sum = 0
#         for frame in np.linspace(1, len(bold_ants_list) - 1, 50, dtype = int):
#             # Register each frame against previous frame
#             frame_reg = registration(fixed = bold_ants_list[frame-1], moving = bold_ants_list[frame],
#                                         type_of_transform = 'Rigid')
#             # Load the transformed frame and extract the parameters
#             tx = read_transform(frame_reg['fwdtransforms'][0]).parameters
#             # Compute the eular angles from the rotation of the image
#             euler_angles = Rotation.from_matrix(tx[:9].reshape((3,3))).as_euler('xyz', degrees = False)
#             # Convert to mm using 50 mm general head size
#             displacement = euler_angles * 50
#             # Get the translations from the transformed parameters
#             translations = tx[9:]
#             # Compute the FD for this frame
#             frame_fd = np.sum(np.abs(np.append(displacement, translations)))
#             # Add the each frame_fd to a running sum
#             frame_fd_running_sum += frame_fd
#         # Compute the mean FD
#         run_mean_fd = frame_fd_running_sum/50

#         # DVARS computation
#         brain_mask = compute_epi_mask(to_nibabel_nifti(mean_frame_ants)).get_fdata().astype(bool)
#         bold_run_signal = bold_4d.numpy().astype(np.float64)[brain_mask]
#         # Rescale so the whole-brain modal intensity becomes 1000, per Power et al. (2012)
#         brain_mode = mode(bold_run_signal, axis = None).mode
#         mode1000_signal = bold_run_signal / brain_mode * 1000
#         # Compute the voxel differences between adjacent frames
#         frame_diff = np.diff(mode1000_signal, axis = -1)
#         # Compute the RMS across all brain voxels
#         rms = np.sqrt(np.mean(frame_diff**2, axis = 0))
#         # Compute the standard deviation of frame-to-frame DVARS across the whole run
#         run_sd_dvars = np.std(rms)

#         # Collect this run's metrics
#         run_metrics.append({'mean_fd': run_mean_fd, 'sd_dvars': run_sd_dvars})

#     # Average mean FD and max SD(DVARS) across this subject's 4 runs
#     run_metrics_df = pd.DataFrame(run_metrics)
#     subject_row = {'mean_fd': run_metrics_df['mean_fd'].mean(), 'sd_dvars': run_metrics_df['sd_dvars'].max()}
#     subject_row['participant_id'] = sub
#     subject_rows.append(subject_row)
# # Collate results into one DataFrame, one row per subject
# candidate_metrics = pd.DataFrame(subject_rows)
# # Rearrange the columns so participant_id is the first column
# participant_ids = candidate_metrics.pop('participant_id')
# candidate_metrics.insert(0, 'participant_id', participant_ids)

# # Compute HND task accuracy per subject from the behavioral rawdata files
# hnd_rows = []
# for sub in subject_list:
#     filepath = f'fMRI_behavior/sub_{sub}_formal_rawdata.txt'
#     hnd_col = pd.read_table(behavior_zip_file.open(filepath), header=None, encoding='latin1')[5].str.strip()
#     hnd_trials = hnd_col[hnd_col.isin(['Correct', 'Incorrect'])]
#     hnd_rows.append({'participant_id': sub, 'hnd_accuracy': (hnd_trials == 'Correct').mean()})
# hnd_performance = pd.DataFrame(hnd_rows)

# # Merge HND accuracy into the candidate metrics
# candidate_metrics = candidate_metrics.merge(hnd_performance, on='participant_id')

# # Save the candidate metrics
# candidate_selection_path = data_path / "derivatives" / "candidate_selection"
# candidate_selection_path.mkdir(parents=True, exist_ok=True)
# candidate_metrics.sort_values('participant_id').to_csv(candidate_selection_path / "candidate_metrics.tsv", sep='\t', index=False)

# Load and explore the candidate metrics
candidate_metrics = pd.read_table(candidate_selection_path / "candidate_metrics.tsv")
# 1. Exclude if mean FD > 0.2
# 2. Rank remaining using a rank-sum of SD(DVARS) (ascending - lower is more stable)
#    and HND accuracy (descending - higher is better task engagement)
candidate_metrics = (candidate_metrics.query('mean_fd <= 0.2')
        .assign(sd_dvars_rank = lambda x: x['sd_dvars'].rank(ascending=True),
                hnd_accuracy_rank = lambda x: x['hnd_accuracy'].rank(ascending=False))
        .assign(rank_sum = lambda x: x['sd_dvars_rank'] + x['hnd_accuracy_rank'])
        .sort_values('rank_sum'))

# # Extract the fMRI data and load Subject 15 data
# path_to_zip = raw_data_path / "MRI_Scanning_sub15.zip"
# zip_file = zipfile.ZipFile(path_to_zip)
# extract_path = data_path / "extracted" / "subject_15"
# zip_file.extractall(extract_path)
# zip_file.close()

# DCM to Nifti conversion done for all bold runs in a wsl2 terminal using the command below
# mkdir -p data/scene-areas-hierarchy-rsa/bids/sub-15/func
# for run in 1 2 3 4; do
#   dcm2niix -z y -f "sub-15_task-sm_run-${run}_bold" \
#     -o data/scene-areas-hierarchy-rsa/bids/sub-15/func \
#     data/scene-areas-hierarchy-rsa/extracted/subject_15/MRI_Scanning_sub15/bold_run${run}
# done

# Do the same for the T1w image
# mkdir -p data/scene-areas-hierarchy-rsa/bids/sub-15/anat
# dcm2niix -z y -f "sub-15_T1w" \
#   -o data/scene-areas-hierarchy-rsa/bids/sub-15/anat \
#   data/scene-areas-hierarchy-rsa/extracted/subject_15/MRI_Scanning_sub15/t1

# Run fMRIPrep on the extracted and converted BIDS data for Subject 15 to get the preprocessed data
# docker run --rm \
#   -v "-absolute path to bids-":/data:ro \
#   -v "-absolute path to output-":/out \
#   -v "-absolute path to free surfer license-":/opt/freesurfer/license.txt:ro \
#   nipreps/fmriprep:25.2.0 \
#   /data /out participant \
#   --participant-label 15 \
#   --fs-license-file /opt/freesurfer/license.txt \
#   --output-spaces MNI152NLin2009cAsym \
#   --nprocs 8 \
#   --omp-nthreads 4 \
#   --mem-mb 16000
# Ran for 2.5 hours with the above command

# Load the preprocessed fMRI data for Subject 15
preprocessed_path = data_path / "derivatives" / "sub-15"
# Confirm the FD and DVARS values for the preprocessed data
# 24 motion parameters: 6 raw + 6 derivatives + 6 squared + 6 squared-derivatives
motion_bases = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
motion_variants = ['', '_derivative1', '_power2', '_derivative1_power2']
motion_columns = [base + variant for base in motion_bases for variant in motion_variants]
# Build the confound regressors DataFrame (24 motion parameters + FD + DVARS) per run
confounds_dfs = []
for run in range(1, 5):
    timeseries_confounds_path = preprocessed_path / "func" / f"sub-15_task-sm_run-{run}_desc-confounds_timeseries.tsv"
    timeseries_confounds = pd.read_table(timeseries_confounds_path)
    confounds_df = timeseries_confounds[motion_columns + ['framewise_displacement', 'dvars']]
    # Fill the NaN values of the first frame with 0
    confounds_df = confounds_df.fillna(0)
    # Append to the list of confounds DataFrames
    confounds_dfs.append(confounds_df)
# Display the mean mean FD and mean median DVARS values
print("Mean FD:", pd.Series([df['framewise_displacement'].mean() for df in confounds_dfs]).mean()) # 0.076, 0.145
print("Mean of per-run median DVARS:", pd.Series([df['dvars'].median() for df in confounds_dfs]).mean()) # 45.80, 41.40
# This confirms that the preprocessed data for Subject 15 is of good quality and
# matches the estimated values from the raw data

# Load the preprocessed bold runs for Subject 15
preprocessed_bold_path = preprocessed_path / "func"
preprocessed_bold_runs = [nib.load(
                            preprocessed_bold_path / f"sub-15_task-sm_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz") 
                            for run in range(1, 5)]
# Compute the frame times for each run
run_frame_times = [np.arange(preprocessed_bold_runs[run].shape[-1]) * 2 for run in range(4)] # 2 sec TR

# Find the onset times for the walking period for each run
# Load the behavior zip file
behavior_zip_file = zipfile.ZipFile(path_to_behavior_zip)
# Define the internal path to the trial timing files
formal_Time_record_filepath = 'fMRI_behavior/sub_15_formal_Time_record_t.txt'
# Load the tab separated data
timing_onsets = pd.read_table(behavior_zip_file.open(formal_Time_record_filepath), header=None)
# Close the behavior zip file connection
behavior_zip_file.close()
# Get the onset times for the walking period for each run
trial_onsets = [timing_onsets[timing_onsets[12] == run][4].values for run in range(1, 5)]
# Get the onset times for the facing period for each run
trial_offsets = [timing_onsets[timing_onsets[12] == run][6].values for run in range(1, 5)]
# These will serve as the end point for the trials for our purposes
# Since bold activity is expected to be delayed

# Compute a combined brain mask across all 4 runs
subject_mask_paths = [preprocessed_path / "func" / f"sub-15_task-sm_run-{run}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
                       for run in range(1, 5)]
subject_mask = intersect_masks(subject_mask_paths, threshold=1.0)

# Compute the number of trials per run
n_trials_per_run = [len(trial_onsets[run]) for run in range(4)]

# Attach map/walking_direction conditions to each trial
sub15_conditions = behavior_data.query('participant_id == 15').reset_index(drop=True)
sub15_conditions['run'] = np.repeat(np.arange(1, 5), n_trials_per_run)
sub15_conditions['trial_index'] = np.concatenate([np.arange(1, n + 1) for n in n_trials_per_run])

# Condition-level GLM: one regressor per map x direction combo instead of per trial
sub15_conditions['map_direction'] = (sub15_conditions['map'].astype(str) + '_' +
                                      sub15_conditions['walking_direction'].astype(str))
sub15_combos = [sub15_conditions.query('run == @run')['map_direction'].values for run in range(1, 5)]

# Define the events DataFrame with trial_type shared across a condition's trials
events_dfs = []
for run in range(4):
    events_dfs.append(pd.DataFrame({
        'onset': trial_onsets[run] / 1000,
        'duration': (trial_offsets[run] - trial_onsets[run]) / 1000,
        'trial_type': [f'combo_{c}' for c in sub15_combos[run]]
    }))

# Build the design matrices for each run
design_matrics = [
    first_level.make_first_level_design_matrix(
        frame_times = run_frame_times[run],
        events = events_dfs[run],
        add_regs = confounds_dfs[run][motion_columns],
        add_reg_names = motion_columns
    ) for run in range(4)
]

# Build the GLM model
glm_model = first_level.FirstLevelModel(
    mask_img=intersect_masks(subject_mask_paths, threshold=1.0),
    minimize_memory=False
)

# Fit the model to the preprocessed bold runs and design matrices
glm_model.fit(
    run_imgs = preprocessed_bold_runs,
    design_matrices = design_matrics
)

# Define the unique conditions across all runs
conditions = sorted(sub15_conditions['map_direction'].unique())

# One beta per condition per run
betas = []
beta_meta = []
for run in range(4):
    run_columns = design_matrics[run].columns
    for c in conditions:
        combo_name = f'combo_{c}'
        # Only compute the contrast if the condition is present in this run's design matrix
        if combo_name not in run_columns:
            continue
        beta_map = glm_model.compute_contrast(
            [combo_name if r == run else np.zeros(design_matrics[r].shape[1]) for r in range(4)],
            output_type='effect_size'
        )
        betas.append(beta_map)
        beta_meta.append({'run': run + 1, 'map_direction': c})
betas_4d = concat_imgs(betas)
# Prepare a DataFrame with the run and map_direction
beta_meta = pd.DataFrame(beta_meta)
beta_meta[['map', 'walking_direction']] = beta_meta['map_direction'].str.split('_', expand=True)

# Get the per run residuals for crossnobis distance computation
run_residuals = glm_model.residuals

# ROI masking
# Use the same docker container to get the freesurfer surface masks from visfatlas
# mkdir -p data/scene-areas-hierarchy-rsa/derivatives/sub-15/label
# for hemi in lh rh; do
#   for region in v1d v1v v2d v2v; do
#     docker run --rm \
#       -v "$(pwd)/data/scene-areas-hierarchy-rsa:/data" \
#       --entrypoint bash \
#       nipreps/fmriprep:25.2.0 \
#       -c "export SUBJECTS_DIR=/data/derivatives/sourcedata/freesurfer && \
#           export FS_LICENSE=/data/license.txt && \
#           mri_label2label \
#             --srclabel /data/atlases/visfAtlas/FreeSurfer/MPM_${hemi}_${region}.label \
#             --srcsubject fsaverage \
#             --trgsubject sub-15 \
#             --trglabel /data/derivatives/sub-15/label/${hemi}.${region}.label \
#             --hemi ${hemi} \
#             --regmethod surface"
#   done
# done

# Do the same for a new silson atlas which has clearly named areas in the surface files
# for hemi in lh rh; do
#   for region in PPA OPA MPA; do
#     docker run --rm \
#       -v "$(pwd)/data/scene-areas-hierarchy-rsa:/data" \
#       --entrypoint bash \
#       nipreps/fmriprep:25.2.0 \
#       -c "export SUBJECTS_DIR=/data/derivatives/sourcedata/freesurfer && \
#           export FS_LICENSE=/data/license.txt && \
#           mri_surf2surf \
#             --srcsubject fsaverage \
#             --trgsubject sub-15 \
#             --hemi ${hemi} \
#             --sval /data/atlases/silson_atlas/fs_average.${region}.allloc.group_constrained_800.${hemi}.gii \
#             --tval /data/derivatives/sub-15/label/${hemi}.${region,,}.gii \
#             --sfmt gii \
#             --tfmt gii"
#   done
# done

# Select the top 800 vertices for all areas so there is near uniform number of voxels for each area
silson_regions = ['ppa', 'opa', 'mpa']
visf_regions = ['v1d', 'v1v', 'v2d', 'v2v']
vertex_selection = {}
for hemi in ['lh', 'rh']:
    for region in silson_regions:
        gii_path = preprocessed_path / "label" / f"{hemi}.{region}.gii"
        prob_data = nib.load(gii_path).darrays[0].data
        # Drop the zeros
        nonzero_indices = np.nonzero(prob_data)[0]
        nonzero_probs = prob_data[nonzero_indices]
        # Get the indices top 800
        top_n = min(800, nonzero_indices.size)
        top_order = np.argsort(nonzero_probs)[::-1][:top_n]
        # Slice to only include top 800
        vertex_selection[f'{hemi}_{region}'] = nonzero_indices[top_order]
    for region in visf_regions:
        label_path = preprocessed_path / "label" / f"{hemi}.{region}.label"
        indices, probs = nib.freesurfer.io.read_label(label_path, read_scalars=True)
        # Get the indices of top 800 
        top_n = min(800, indices.size)
        top_order = np.argsort(probs)[::-1][:top_n]
        # Slice to only include top 800
        vertex_selection[f'{hemi}_{region}'] = indices[top_order]

# Write a top-800 .label file per region, using XYZ from sub-15's own white surface
freesurfer_dir = data_path / "derivatives" / "sourcedata" / "freesurfer"
label_dir = preprocessed_path / "label"
for hemi in ['lh', 'rh']:
    surf_path = freesurfer_dir / "sub-15" / "surf" / f"{hemi}.white"
    # Get the XYZ coordinate frame of each hemisphere
    coords, _ = nib.freesurfer.io.read_geometry(surf_path)
    for region in silson_regions + visf_regions:
        # Get the indices of the vertices to write into the file
        indices = vertex_selection[f'{hemi}_{region}']
        # Define the file output path, same place as labels/gii for the overall atlas maps
        out_path = label_dir / f"{hemi}.{region}.top800.label"
        with open(out_path, 'w') as f:
            f.write(f"#!ascii label, top 800 vertices, subject sub-15\n{indices.size}\n")
            for idx in indices:
                x, y, z = coords[idx]
                # Write the 5th column for probability as all 1 since areas are already chosen
                f.write(f"{idx} {x:.3f} {y:.3f} {z:.3f} 1.0\n")

# Create the volumetric masks from the surface vertices in docker container
# This basically samples each vertex at 10% interval depths
# A threshold of 30% is chosen as a spatial density filter
# ie. how many samples are in each voxel and drops any voxel with < 0.3 samples in it (1 cmm voxels)
# for hemi in lh rh; do
#   for region in ppa opa mpa v1d v1v v2d v2v; do
#     docker run --rm \
#       -v "$(pwd)/data/scene-areas-hierarchy-rsa:/data" \
#       --entrypoint bash \
#       nipreps/fmriprep:25.2.0 \
#       -c "export SUBJECTS_DIR=/data/derivatives/sourcedata/freesurfer && \
#           export FS_LICENSE=/data/license.txt && \
#           mri_label2vol \
#             --label /data/derivatives/sub-15/label/${hemi}.${region}.top800.label \
#             --temp /data/derivatives/sourcedata/freesurfer/sub-15/mri/T1.mgz \
#             --regheader /data/derivatives/sourcedata/freesurfer/sub-15/mri/T1.mgz \
#             --proj frac 0 1 .1 \
#             --fillthresh 0.3 \
#             --subject sub-15 \
#             --hemi ${hemi} \
#             --o /data/derivatives/sub-15/label/${hemi}.${region}.top800.nii.gz"
#   done
# done

# Resample each native-space ROI mask onto the MNI152NLin2009cAsym grid
t1w_to_mni_xfm = preprocessed_path / "anat" / "sub-15_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"
brain_mask_data = subject_mask.get_fdata().astype(bool)
roi_masks = {}
for hemi in ['lh', 'rh']:
    for region in silson_regions + visf_regions:
        # Path the surface extended to volume files created in docker
        native_mask_path = label_dir / f"{hemi}.{region}.top800.nii.gz"
        native_mask_ants = image_read(str(native_mask_path))
        # Transform to MNI space
        transformed_ants = apply_transforms(fixed=from_nibabel_nifti(subject_mask), moving=native_mask_ants,
                                           transformlist=[str(t1w_to_mni_xfm)],
                                           interpolator='genericLabel')

        roi_mask = to_nibabel_nifti(transformed_ants).get_fdata().astype(bool)
        # Drop any ROI voxel that falls outside the GLM's actual brain mask
        roi_masks[f'{hemi}_{region}'] = roi_mask & brain_mask_data

# Pull out actual effect sizes per ROI, voxels x (condition x run) observations
betas_4d_data = betas_4d.get_fdata()
roi_betas_data = {}
for name, mask in roi_masks.items():
    roi_betas_data[name] = betas_4d_data[mask]
# n_roi_voxels x n (condition, run) observations, per ROI
# ROI-masked residuals per run, needed for the crossnobis noise precision matrix
roi_residual_data = {}
for name, mask in roi_masks.items():
    roi_residual_data[name] = [res.get_fdata()[mask] for res in run_residuals]

# Voxel count per ROI actually going into roi_betas_data
voxel_counts_pre_gm = [{'roi': name, 'n_voxels': roi_data.shape[0]}
                       for name, roi_data in roi_betas_data.items()]
print(pd.DataFrame(voxel_counts_pre_gm))
#        roi  n_voxels
# 0   lh_ppa       329
# 1   lh_opa       232
# 2   lh_mpa       201
# 3   lh_v1d       196
# 4   lh_v1v       174
# 5   lh_v2d       173
# 6   lh_v2v       245
# 7   rh_ppa       308
# 8   rh_opa       281
# 9   rh_mpa       193
# 10  rh_v1d       195
# 11  rh_v1v       139
# 12  rh_v2d       191
# 13  rh_v2v       152
# ~10% drop expected since functional voxel volume is 2x2x2.3 cmm

# MAD-based outlier voxel exclusion (Iglewicz & Hoaglin), modified z > 3.5
# Applied uniformly to every ROI to exclude any outliers driving statistical analysis
for name in roi_betas_data:
    voxel_std = roi_betas_data[name].std(axis=1)
    mad = np.median(np.abs(voxel_std - np.median(voxel_std)))
    modified_z = 0.6745 * (voxel_std - np.median(voxel_std)) / mad
    keep_voxels = np.abs(modified_z) <= 3.5
    roi_betas_data[name] = roi_betas_data[name][keep_voxels]
    roi_residual_data[name] = [res[keep_voxels] for res in roi_residual_data[name]]

# Voxel count per ROI after outlier exclusion
voxel_counts_post_outlier = [{'roi': name, 'n_voxels': roi_data.shape[0]}
                              for name, roi_data in roi_betas_data.items()]
print(pd.DataFrame(voxel_counts_post_outlier))
# n_roi_voxels x n_timepoints for each run, per ROI
#        roi  n_voxels
# 0   lh_ppa       326
# 1   lh_opa       230
# 2   lh_mpa       199
# 3   lh_v1d       182
# 4   lh_v1v       167
# 5   lh_v2d       163
# 6   lh_v2v       214
# 7   rh_ppa       303
# 8   rh_opa       278
# 9   rh_mpa       193
# 10  rh_v1d       168
# 11  rh_v1v       134
# 12  rh_v2d       173
# 13  rh_v2v       143

# Neural RSA via rsatoolbox, crossnobis distance
# Full 12x12 map x direction RSM kept alongside the marginals, needed as-is tomorrow to
# subset down to whichever combos match the available DNN comparison videos
condition_rdm_objects = {}
for name, roi_data in roi_betas_data.items():
    # Define the dataset for rsatoolbox
    dataset = rsatoolbox.data.Dataset(
        measurements=roi_data.T,
        obs_descriptors={'map': beta_meta['map'].values,
                          'walking_direction': beta_meta['walking_direction'].values,
                          'map_direction': beta_meta['map_direction'].values,
                          'run': beta_meta['run'].astype(str).values}
    )
    # Compute the noise precision matrix for crossnobis distance
    noise_precision = [rsatoolbox.data.prec_from_residuals(residuals.T)
                        for residuals in roi_residual_data[name]]
    # Store the raw RDMs object directly, no DataFrame conversion
    condition_rdm_objects[name] = rsatoolbox.rdm.calc_rdm_crossnobis(dataset, descriptor='map_direction',
                                                                       noise=noise_precision, cv_descriptor='run')

# Display the full 12x12 map x direction ROI RDMs, left ROIs on top, right below
# Walking direction forms the big blocks, map forms the small blocks within each
direction_major_order = [f'{map_id}_{direction_id}'
                          for direction_id in range(1, 5)
                          for map_id in range(1, 4)]
# Show the areas linearly in a hierarchy for each hemisphere
region_order = ['v1d', 'v1v', 'v2d', 'v2v', 'ppa', 'mpa', 'opa']
roi_labels = [f'lh_{region}' for region in region_order] + [f'rh_{region}' for region in region_order]
# One shared color scale per hemisphere based on off-diagonal values across all areas
hemi_scales = {}
for hemi in ['lh', 'rh']:
    pooled_off_diagonal = []
    for region in region_order:
        matrix = condition_rdm_objects[f'{hemi}_{region}'].get_matrices()[0]
        pooled_off_diagonal.append(matrix[~np.eye(matrix.shape[0], dtype=bool)])
    pooled_off_diagonal = np.concatenate(pooled_off_diagonal)
    # 98th percentile of the off-diagonal to avoid outlier influence on the color scale
    hemi_scales[hemi] = np.percentile(np.abs(pooled_off_diagonal), 98)
# Plot the heatmaps
fig, axes = plt.subplots(2, len(region_order), figsize=(4 * len(region_order) + 1, 9),
                          constrained_layout=True,
                          gridspec_kw={'wspace': 0.15, 'hspace': 0.15})
for col, region in enumerate(region_order):
    for row, hemi in enumerate(['lh', 'rh']):
        name = f'{hemi}_{region}'
        rdm_obj = condition_rdm_objects[name]
        rdm_df = pd.DataFrame(rdm_obj.get_matrices()[0],
                               index=rdm_obj.pattern_descriptors['map_direction'],
                               columns=rdm_obj.pattern_descriptors['map_direction'])
        reordered_rdm = rdm_df.loc[direction_major_order, direction_major_order]
        vmax = hemi_scales[hemi]
        # No per-subplot colorbar - one shared colorbar per hemisphere is added at the end instead
        sns.heatmap(reordered_rdm, ax=axes[row, col], cmap='RdBu', vmin=-vmax, vmax=vmax,
                    center=0, cbar=False, square=True)
        axes[row, col].set_title(name, fontsize=20)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        # Mark the boundaries between the 4 walking-direction blocks
        for boundary in range(3, 12, 3):
            axes[row, col].axhline(boundary, color='black', linewidth=1)
            axes[row, col].axvline(boundary, color='black', linewidth=1)
# One vertical colorbar per hemisphere row
for row, hemi in enumerate(['lh', 'rh']):
    vmax = hemi_scales[hemi]
    mappable = plt.cm.ScalarMappable(cmap='RdBu', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    fig.colorbar(mappable, ax=axes[row, :], orientation='vertical',
                 fraction=0.05, pad=0.02, ticks=[-vmax, 0, vmax])
plt.show()

# RDM permutation test
n_permutations = 200
rng = np.random.default_rng(0) # Ensure reproducibility of the permutation test
permutation_results = {}
for name, roi_data in roi_betas_data.items():
    noise_precision = [rsatoolbox.data.prec_from_residuals(residuals.T)
                        for residuals in roi_residual_data[name]]

    # Reuse the already-computed RDM instead of recomputing it
    computed_matrix = condition_rdm_objects[name].get_matrices()[0]
    # Test statistic: mean distance between conditions, off-diagonal only
    mean_distance = computed_matrix[~np.eye(computed_matrix.shape[0], dtype=bool)].mean()

    null_mean_distances = []
    for _ in range(n_permutations):
        # Under H0, condition labels carry no information, so shuffle them
        # Shuffle within each run to keep the cv_descriptor structure intact
        shuffled_labels = (beta_meta.groupby('run')['map_direction']
                            .transform(lambda s: rng.permutation(s.values)))
        perm_dataset = rsatoolbox.data.Dataset(
            measurements=roi_data.T,
            obs_descriptors={'map_direction': shuffled_labels.values,
                              'run': beta_meta['run'].astype(str).values}
        )
        perm_rdm = rsatoolbox.rdm.calc_rdm_crossnobis(perm_dataset, descriptor='map_direction',
                                                       noise=noise_precision, cv_descriptor='run')
        perm_matrix = perm_rdm.get_matrices()[0]
        perm_mean_distance = perm_matrix[~np.eye(perm_matrix.shape[0], dtype=bool)].mean()
        null_mean_distances.append(perm_mean_distance)

    null_mean_distances = np.array(null_mean_distances)
    permutation_results[name] = {
        'mean_distance': mean_distance,
        'null_mean': null_mean_distances.mean(),
        'null_std': null_mean_distances.std(),
        # 95% CI of the null distribution
        'null_ci_low': np.percentile(null_mean_distances, 2.5),
        'null_ci_high': np.percentile(null_mean_distances, 97.5),
        'null_mean_distances': null_mean_distances,
    }
# Package into a dataframe
permutation_df = pd.DataFrame(permutation_results).T

# Plot each ROI's null distribution of mean distance, against the computed mean distance
# Define the colors for the null distribution and the computed mean distance
null_color = '#8a8fa3'
real_color = '#2924bd'
# Arrange them as left and right pairs
paired_roi_labels = [f'{hemi}_{region}' for region in region_order for hemi in ('lh', 'rh')]
# Generate the violin plot
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
violin_data = [permutation_results[roi]['null_mean_distances'] for roi in paired_roi_labels]
parts = ax.violinplot(violin_data, positions=np.arange(len(paired_roi_labels)), showmedians=True)
# Set the violin plot colors to the custom color defined above
for body in parts['bodies']:
    body.set_facecolor(null_color)
    body.set_edgecolor(null_color)
    body.set_alpha(0.6) # Reduce opacity
for key in ('cmedians', 'cbars', 'cmins', 'cmaxes'):
    parts[key].set_color(null_color)
# Plot the computed mean distance for each ROI as a scatter point
real_values = [permutation_results[roi]['mean_distance'] for roi in paired_roi_labels]
ax.scatter(np.arange(len(paired_roi_labels)), real_values, color=real_color, s=80, zorder=3,
           label='Computed mean distance')
# Vertical separators between regions
for sep in np.arange(1.5, len(paired_roi_labels), 2):
    ax.axvline(sep, color='lightgray', linewidth=1, zorder=0)
# Label the axes and add a title
ax.set_xticks(np.arange(len(paired_roi_labels)))
ax.set_xticklabels([roi.upper() for roi in paired_roi_labels])
ax.set_ylabel('Mean off-diagonal crossnobis distance')
ax.set_title('RDM robustness check: real distance vs label-permuted null')
ax.legend(loc='upper right', frameon=False)
plt.show()
# Subject 12 has similar RDMs and Permutations results

# Compute z-scores for every ROI against its own null distribution
roi_z_scores = {}
for name, res in permutation_results.items():
    # Compute the z-scores
    z = (res['mean_distance'] - res['null_mean']) / res['null_std']
    roi_z_scores[name] = z
roi_z_table = pd.DataFrame({'z_score': roi_z_scores})
roi_z_table['percentile'] = norm.cdf(roi_z_table['z_score']) * 100
# Only print ROIs with z > 1 (ie. above 84th percentile)
print(roi_z_table[roi_z_table['z_score'] > 1].sort_values('z_score', ascending=False))
# Selected areas are with possible hypothetical connections based on anatomical proximity
# left V1d => left OPA; right V1d => right OPA; left pooled V1 => left pooled V2 => left MPA
# Check percentile of lh_v2d which did not pass the 84th percentile test
print(roi_z_table.loc['lh_v2d']['percentile']) # 69.20
# Not reliable; left pooled V1 => left V2v => left MPA

# Pool left V1d + V1v voxels together as the first stage in the MPA pathway
pooled_v1_beta = np.concatenate([roi_betas_data['lh_v1d'], roi_betas_data['lh_v1v']], axis=0)
# Compute the corresponing crossnobis ROI for this pooled area
dataset = rsatoolbox.data.Dataset(
    measurements=pooled_v1_beta.T,
    obs_descriptors={'map_direction': beta_meta['map_direction'].values,
                      'run': beta_meta['run'].astype(str).values}
)
noise_precision = [np.concatenate([res_v1d, res_v1v], axis=0)
                    for res_v1d, res_v1v in zip(roi_residual_data['lh_v1d'], roi_residual_data['lh_v1v'])]
noise_precision = [rsatoolbox.data.prec_from_residuals(res.T) for res in noise_precision]
pooled_v1_rdm = rsatoolbox.rdm.calc_rdm_crossnobis(
    dataset, descriptor='map_direction', noise=noise_precision, cv_descriptor='run')

# Since not all areas have good reliability, probably because this is one subject
# I need to confirm if the almost flat nature of scene RDMs has anything hidden
# once the participation of primary areas is removed using PCA
pr_rois = ['lh_v1d', 'lh_opa', 'rh_v1d', 'rh_opa', 'lh_v1', 'lh_v2v', 'lh_mpa']
participation_ratios = {}
for name in pr_rois:
    # Select the pooled V1 RDM object when the roi is V1
    rdm_obj = pooled_v1_rdm if name == 'lh_v1' else condition_rdm_objects[name]
    # Extract the 12x12 squared crossnobis distance matrix
    D = rdm_obj.get_matrices()[0]
    # Get the number of conditions
    n = len(conditions)
    # Build the centering matrix
    H = np.eye(n) - np.ones((n, n)) / n
    # Compute the similarity matrix
    S = -0.5 * H @ D @ H
    # Get the eigenvalues of the similarity matrix
    eigenvalues = np.linalg.eigvalsh(S)
    # Clip tiny negative eigenvalues (floating-point noise) to zero
    eigenvalues = np.maximum(eigenvalues, 0)
    # Compute the raw participation ratio
    pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    # Normalize by the max achievable rank (n - 1, centering removes one dimension)
    participation_ratios[name] = pr / (n - 1)
# Display the results grouped by hypothesized pathway
pathways = {
    'lh V1d -> OPA': ['lh_v1d', 'lh_opa'],
    'rh V1d -> OPA': ['rh_v1d', 'rh_opa'],
    'lh V1 -> V2v -> MPA': ['lh_v1', 'lh_v2v', 'lh_mpa'],
}
for pathway_name, rois in pathways.items():
    print(pathway_name)
    print(pd.Series({name: participation_ratios[name] for name in rois}))
    print()
# Equal participation along the pathways, with only left OPA showing good jump

# Compute MDS coordinates, stress, and Procrustes disparity for every ROI in every pathway
mds_rows = []
for pathway_name, rois in pathways.items():
    for stage, name in enumerate(rois):
        # Pooled V1 RDM object when the roi is V1
        rdm_obj = pooled_v1_rdm if name == 'lh_v1' else condition_rdm_objects[name]
        # Extract the matrix
        rdm_matrix = rdm_obj.get_matrices()[0]
        # Get the condition labels
        condition_labels = rdm_obj.pattern_descriptors['map_direction']
        # Scale the RDM down to 2 dimensions
        mds_estimator = MDS(n_components=2, dissimilarity='precomputed', random_state=23)
        mds_coords = mds_estimator.fit_transform(rdm_matrix)
        # Align every ROI MDS to the previous along each pathway
        disparity = None
        if stage == 0:
            pathway_reference = mds_coords
        else:
            pathway_reference, mds_coords, disparity = procrustes(pathway_reference, mds_coords)
        # Build a tidy row for each point
        for i, label in enumerate(condition_labels):
            mds_rows.append({
                'pathway': pathway_name,
                'stage': stage,
                'roi': name,
                'x': mds_coords[i, 0],
                'y': mds_coords[i, 1],
                'direction': label.split('_')[1],
                'map': label.split('_')[0],
                'stress': mds_estimator.stress_,
                'disparity': disparity,
            })
mds_df = pd.DataFrame(mds_rows)

# Plot the MDS coordinates
direction_palette = sns.color_palette('colorblind', n_colors=4)
g = sns.FacetGrid(mds_df, row='pathway', col='stage', row_order=pathways.keys(),
                   height=4, despine=True, sharex=False, sharey=False)
g.map_dataframe(sns.scatterplot, x='x', y='y', hue='direction', style='map',
                 palette=direction_palette, s=140);
# Grab legend handles/labels from a confirmed non-empty facet
legend_ax = next(ax for ax in g.axes.flat if ax.get_legend_handles_labels()[0])
handles, labels = legend_ax.get_legend_handles_labels()
# Loop over each to add subplot titles
legend_drawn = False
for (pathway_name, stage), ax in g.axes_dict.items():
    facet_data = mds_df[(mds_df['pathway'] == pathway_name) & (mds_df['stage'] == stage)]
    if facet_data.empty:
        # Draw the legend in one of the empty boxes of the first 2 pathways
        if not legend_drawn:
            ax.legend(handles, labels, loc='center', frameon=False)
            legend_drawn = True
        ax.set_title('')
        ax.axis('off')
        continue
    row = facet_data.iloc[0]
    # Add the ROI labels with MDS stress and Procrustes
    title = f'{row["roi"]}\nStress: {row["stress"]:.4f}'
    if pd.notna(row['disparity']):
        prev_roi = mds_df.loc[(mds_df['pathway'] == pathway_name) & (mds_df['stage'] == stage - 1),
                               'roi'].iloc[0]
        title += f'\nDisparity vs {prev_roi}: {row["disparity"]:.4f}'
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
# The space is clearly expanding across each pathway, the groupings are subtle
# Next step should be if the expanded space is not entirely built from low level features supplied by V1

# --- Partial RSA: is the higher-order structure explained by low level features from V1 alone? ---
# Fixed condition order for every model/neural RDM
condition_order = list(pooled_v1_rdm.pattern_descriptors['map_direction'])

# Function which returns the vectorized form of the upper triangle
def vectorize_rdm(rdm_obj, order):
    rdm_df = pd.DataFrame(rdm_obj.get_matrices()[0],
                           index=rdm_obj.pattern_descriptors['map_direction'],
                           columns=rdm_obj.pattern_descriptors['map_direction'])
    reordered = rdm_df.loc[order, order].values
    return squareform(reordered, checks=False)

# Build the 12x12 direction and map model RDMs with directions as primary and map as secondary
directions_12 = [label.split('_')[1] for label in condition_order]
maps_12 = [label.split('_')[0] for label in condition_order]
direction_rdm_12 = (np.array(directions_12)[:, None] != np.array(directions_12)[None, :]).astype(int)
map_rdm_12 = (np.array(maps_12)[:, None] != np.array(maps_12)[None, :]).astype(int)
direction_vec = direction_rdm_12[np.triu_indices(len(condition_order), k=1)]
map_vec = map_rdm_12[np.triu_indices(len(condition_order), k=1)]

# VIF for each predictor in {V1, direction, map} to look for collinearity of factors
vif_results = {}
for pathway_name, rois in pathways.items():
    v1_name = rois[0]
    v1_obj = pooled_v1_rdm if v1_name == 'lh_v1' else condition_rdm_objects[v1_name]
    v1_vec = vectorize_rdm(v1_obj, condition_order)
    predictors = pd.DataFrame({'v1': v1_vec, 'direction': direction_vec, 'map': map_vec})
    for col in predictors.columns:
        other_cols = [c for c in predictors.columns if c != col]
        r_squared = LinearRegression().fit(predictors[other_cols], predictors[col]).score(
            predictors[other_cols], predictors[col])
        vif_results[(pathway_name, col)] = 1 / (1 - r_squared)
vif_table = pd.Series(vif_results).rename('VIF')
print(vif_table)
# All are close to 1 for each left V1d, right V1d and left pooled V1
# No multicollinearity concerns

# higher_order_rdm ~ V1_rdm + direction_rdm + map_rdm, per pathway per stage
regression_results = {}
for pathway_name, rois in pathways.items():
    v1_name = rois[0]
    v1_obj = pooled_v1_rdm if v1_name == 'lh_v1' else condition_rdm_objects[v1_name]
    v1_vec = vectorize_rdm(v1_obj, condition_order)
    # Initialize the predictors for the model
    predictors = np.column_stack([v1_vec, direction_vec, map_vec])
    for stage, name in enumerate(rois):
        # Avoid baseline V1 stage
        if stage == 0:
            continue
        target_obj = condition_rdm_objects[name]
        target_vec = vectorize_rdm(target_obj, condition_order)
        # Initialize and fit the model
        fit = LinearRegression().fit(predictors, target_vec)
        # Extract the coefficients
        regression_results[(pathway_name, name)] = {
            'v1_coef': fit.coef_[0],
            'direction_coef': fit.coef_[1],
            'map_coef': fit.coef_[2],
        }
regression_df = pd.DataFrame(regression_results).T
print(regression_df)
#                              v1_coef  direction_coef  map_coef
# lh V1d -> OPA       lh_opa  0.078831        0.000674  0.000794
# rh V1d -> OPA       rh_opa -0.100562        0.000416 -0.000150
# lh V1 -> V2v -> MPA lh_v2v  0.134817        0.002051  0.000791
#                     lh_mpa -0.016874       -0.000049  0.000345
# Now to check if the coefficients can actually be interpreted

# Permutation null for the regression coefficients
coef_permutation_results = {}
for pathway_name, rois in pathways.items():
    v1_name = rois[0]
    # Reuse the already-computed noise precision
    v1_noise = noise_precision if v1_name == 'lh_v1' else \
        [rsatoolbox.data.prec_from_residuals(residuals.T) for residuals in roi_residual_data[v1_name]]
    v1_data = pooled_v1_beta if v1_name == 'lh_v1' else roi_betas_data[v1_name]
    for stage, name in enumerate(rois):
        # Avoid baseline V1 stage
        if stage == 0:
            continue
        target_noise = [rsatoolbox.data.prec_from_residuals(residuals.T)
                         for residuals in roi_residual_data[name]]
        target_data = roi_betas_data[name]

        null_direction_coefs = []
        null_map_coefs = []
        for _ in range(n_permutations):
            # Shuffle within each run to keep the cv_descriptor structure intact
            shuffled_labels = (beta_meta.groupby('run')['map_direction']
                                .transform(lambda s: rng.permutation(s.values)))
            # Rebuild V1's RDM from this shuffle
            v1_perm_dataset = rsatoolbox.data.Dataset(
                measurements=v1_data.T,
                obs_descriptors={'map_direction': shuffled_labels.values,
                                  'run': beta_meta['run'].astype(str).values}
            )
            v1_perm_rdm = rsatoolbox.rdm.calc_rdm_crossnobis(
                v1_perm_dataset, descriptor='map_direction', noise=v1_noise, cv_descriptor='run')
            # Rebuild the higher-order area's RDM from the same shuffle
            target_perm_dataset = rsatoolbox.data.Dataset(
                measurements=target_data.T,
                obs_descriptors={'map_direction': shuffled_labels.values,
                                  'run': beta_meta['run'].astype(str).values}
            )
            target_perm_rdm = rsatoolbox.rdm.calc_rdm_crossnobis(
                target_perm_dataset, descriptor='map_direction', noise=target_noise, cv_descriptor='run')

            # Vectorize both, fit the same regression, keep the direction/map coefficients
            v1_perm_vec = vectorize_rdm(v1_perm_rdm, condition_order)
            target_perm_vec = vectorize_rdm(target_perm_rdm, condition_order)
            perm_predictors = np.column_stack([v1_perm_vec, direction_vec, map_vec])
            perm_fit = LinearRegression().fit(perm_predictors, target_perm_vec)
            null_direction_coefs.append(perm_fit.coef_[1])
            null_map_coefs.append(perm_fit.coef_[2])

        null_direction_coefs = np.array(null_direction_coefs)
        null_map_coefs = np.array(null_map_coefs)
        coef_permutation_results[(pathway_name, name)] = {
            'direction_coef': regression_results[(pathway_name, name)]['direction_coef'],
            'null_direction_coefs': null_direction_coefs,
            'map_coef': regression_results[(pathway_name, name)]['map_coef'],
            'null_map_coefs': null_map_coefs,
        }

# Plot each target areas' null distribution of direction/map coefficients
# against the computed coefficient
fig, axes = plt.subplots(len(pathways), 1, figsize=(8, 5 * len(pathways)), constrained_layout=True)
for ax, (pathway_name, rois) in zip(axes, pathways.items()):
    # Target areas for this pathway only
    pathway_targets = [name for name in rois[1:]]
    # Two things per pathway direction, then map
    positions = np.arange(len(pathway_targets) * 2)
    null_data = []
    real_values = []
    tick_labels = []
    for name in pathway_targets:
        for coef_name in ['direction', 'map']:
            null_data.append(coef_permutation_results[(pathway_name, name)][f'null_{coef_name}_coefs'])
            real_values.append(coef_permutation_results[(pathway_name, name)][f'{coef_name}_coef'])
            tick_labels.append(f'{name}\n{coef_name}')
    parts = ax.violinplot(null_data, positions=positions, showmedians=True)
    # Set the violin plot colors
    for body in parts['bodies']:
        body.set_facecolor(null_color)
        body.set_edgecolor(null_color)
        body.set_alpha(0.6) # Reduce opacity
    for key in ('cmedians', 'cbars', 'cmins', 'cmaxes'):
        parts[key].set_color(null_color)
    # Plot the computed coefficient
    ax.scatter(positions, real_values, color=real_color, s=80, zorder=3,
               label='Computed coefficient')
    # Vertical separators between stages
    for sep in np.arange(1.5, len(positions), 2):
        ax.axvline(sep, color='lightgray', linewidth=1, zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel('Regression coefficient')
    ax.set_title(pathway_name)
    ax.legend(loc='upper right', frameon=False)
plt.show()
# None of the direction or map regressors lie outside the 95% confidence interval

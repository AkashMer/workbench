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
from templateflow import api as tflow
from scipy import ndimage

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

# 2. Initialize the subject 23 zip file
path_to_zip = raw_data_path / "MRI_Scanning_sub9.zip"
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
# Close the zip connection
scene_parcels_zip.close()

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

# Define the candidate selection logic
# 1. Exluded if ROI tSNR < 30
# 2. Remaining ranked by minimum mean FD
# 3. Ties broken by DVARS

# Collate all frames from bold_run1
zip_file = zipfile.ZipFile(path_to_zip)
run1_files = sorted(name for name in zip_file.namelist() if 'bold_run1' in name and not name.endswith('/'))
running_sum = np.zeros((112, 112, 62))
bold1_ants_list = []
# Get all frames in ANTs format as a list
for file in run1_files:
    # Read DICOM file
    dcm = pydicom.dcmread(io.BytesIO(zip_file.read(file)))
    # Wrap the data to read directly from the zipped format
    dcm_wrap = dicomwrappers.wrapper_from_data(dcm)
    # Extract the 3D data for each frame
    dcm_3d = dcm_wrap.get_data()
    # Add to the running sum to get the mean value for normalization
    running_sum += dcm_3d
    # Convert to ANTs format
    ants_format = from_numpy(dcm_3d, spacing=(2, 2, 2.3))
    # Append to list
    bold1_ants_list.append(ants_format)
# Close the zip file
zip_file.close()

# Compute the mean frame and convert to ANTs format
mean_frame_ants = from_numpy(running_sum/495, spacing = (2, 2, 2.3))
# Get the MNI152NLin6Asym from TemplateFlow
t1_path = tflow.get('MNI152NLin6Asym', resolution=1, suffix='T1w', desc = None)
# Convert to ANTs format
mni_template_ants = image_read(str(t1_path))
# Register the mean frame to MNI152NLin6Asym
ants_res = registration(fixed = mni_template_ants, moving = mean_frame_ants, type_of_transform = 'Affine')

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

# Transform both maps onto subject MNI space
visf_subject_ants = apply_transforms(fixed = mean_frame_ants,
                                        moving = visf_visual_ants,
                                        transformlist = ants_res['fwdtransforms'],
                                        whichtoinvert = [True],
                                        interpolator='nearestNeighbor')
scene_subject_ants = apply_transforms(fixed = mean_frame_ants,
                                        moving = scene_ants,
                                        transformlist = ants_res['fwdtransforms'],
                                        whichtoinvert = [True],
                                        interpolator='nearestNeighbor')

# Compute the motion correction metrics on the BOLD run
bold1_4d = list_to_ndimage(make_image((*bold1_ants_list[0].shape, len(bold1_ants_list)), pixeltype='unsigned int',
                                        origin = (*bold1_ants_list[0].origin, 0)),
                            bold1_ants_list)
# Compute mean and std across time axis
tSNR_3d = bold1_4d.numpy().mean(axis = 3) / bold1_4d.numpy().std(axis = 3)

# Compute the ROI tSNR for V1/V2
ndimage.mean(tSNR_3d, visf_subject_ants.numpy(), index = [1, 2])
# Compute the ROI tSNR for scene areas
ndimage.mean(tSNR_3d, scene_subject_ants.numpy(), index = [1, 2, 3])
# All are below 30, plan is to compute all 3 metrics for all subjects
# If all fail tSNR gate: shift the plan to normalize them on one scale
# use the weakest link logic like the last post
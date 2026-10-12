# Import necessary libraries
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from eyelinkio import read_edf
from nilearn.image import clean_img, load_img
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.signal import clean
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter



# Paths
data_path = Path.cwd() / "data"
func_dir = data_path / "derivatives" / "sub-26" / "func"
sub = "sub-26"
runs = [1, 2, 3, 4]

# Variance explained by each physiological noise component, per run
scree = []
for run in runs:
    # Component details from the confounds JSON file
    meta = json.loads((func_dir / f"{sub}_task-sm_run-{run}_desc-confounds_timeseries.json").read_text())
    ev = [meta[f"a_comp_cor_{i:02d}"]["VarianceExplained"] for i in range(50)]
    scree.append(pd.DataFrame({"run": run, "component": np.arange(50), "variance_explained": ev}))
# Scree plot, one panel per run
sns.relplot(data=pd.concat(scree), x="component", y="variance_explained", col="run", kind="line", marker="o")
plt.show()
# 10 seems like a good cutoff

# Temporal preprocessing per run: detrending, denoising, z-scoring
for run in runs:
    # File names for this run
    prefix = f"{func_dir / f'{sub}_task-sm_run-{run}'}_space-T1w"
    bold_path = f"{prefix}_desc-preproc_bold.nii.gz"
    bold_json = f"{prefix}_desc-preproc_bold.json"
    mask_path = f"{prefix}_desc-brain_mask.nii.gz"

    # Repetition time from the BOLD JSON file
    tr = json.loads(Path(bold_json).read_text())["RepetitionTime"]

    # Load the BOLD run and the brain mask
    bold_img = load_img(bold_path)
    mask_img = load_img(mask_path)

    # Step 1: detrending (regress out slow scanner drift)
    drift, _ = load_confounds(bold_path, strategy=("high_pass",))
    detrended = clean_img(bold_img, confounds=drift, t_r=tr, detrend=False, standardize=None, mask_img=mask_img)
    detrended.to_filename(f"{prefix}_desc-detrended_bold.nii.gz")

    # Step 2: denoising (regress out physiological noise estimated from white matter and CSF)
    noise, _ = load_confounds(bold_path, strategy=("high_pass", "compcor"), compcor="anat_combined", n_compcor=10)
    noise = noise.filter(like="comp_cor")
    denoised = clean_img(detrended, confounds=noise, t_r=tr, detrend=False, standardize=None, mask_img=mask_img)
    denoised.to_filename(f"{prefix}_desc-denoised_bold.nii.gz")

    # Step 3: z-score each voxel within this run
    zscored = clean_img(denoised, t_r=tr, detrend=False, standardize="zscore_sample", mask_img=mask_img)
    zscored.to_filename(f"{prefix}_desc-zscored_bold.nii.gz")

# Sanity check
for run in runs:
    prefix = f"{func_dir / f'{sub}_task-sm_run-{run}'}_space-T1w"
    zscored = load_img(f"{prefix}_desc-zscored_bold.nii.gz")
    mask = load_img(f"{prefix}_desc-brain_mask.nii.gz").get_fdata() > 0
    voxels = zscored.get_fdata()[mask]
    print(run, voxels.mean(), voxels.std())
# All in the range of e-06 and std 0.999

# Extract subject 26's eye-tracking file from the zip
eye_zip = data_path / "raw_fmri" / "fMRI_eyedate.zip"
with zipfile.ZipFile(eye_zip) as zip_file:
    zip_file.extract("fMRI_eyedate/el_sub26.edf", data_path / "extracted" / "subject_26")

# Load the eye-tracking file
sub26_eye_data = read_edf(data_path / "extracted" / "subject_26" / "fMRI_eyedate" / "el_sub26.edf")

# What does the file store?
print(sub26_eye_data)
# Only left eye data

# What are the attributes of this eyelink object?
[a for a in dir(sub26_eye_data) if not a.startswith("_")]
# dict
sub26_eye_data.keys() # ['info', 'discrete', 'times', 'samples']
sub26_eye_data["info"] # xpos x ypos x pupil size
(sub26_eye_data["samples"].shape, sub26_eye_data["times"].shape) # 4059.691 seconds of data
sub26_eye_data["discrete"].keys() # ['messages', 'buttons', 'fixations', 'saccades', 'blinks']
# What are messages?
sub26_eye_data["discrete"]["messages"].shape # 1831 messages
# calibration tags, validation tags, other eyelink tags - Not worth digging into further

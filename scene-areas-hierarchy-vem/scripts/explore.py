# Import necessary libraries
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from eyelinkio import read_edf
from nilearn.image import clean_img, load_img
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.signal import clean
from scipy.ndimage import convolve, find_objects, median_filter
from scipy.ndimage import label as label_regions
from scipy.signal import savgol_filter
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, skeletonize
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)

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

# Confirm time points when the run starts are marked
run_starts = {}
for time, msg in sub26_eye_data["discrete"]["messages"]:
    match = re.match(r"session: (\d+) trial: 0\s+screen: ITI$", msg.decode())  # Trial 0 ITI message
    if match:
        run_starts[int(match[1])] = time  # Run number -> start time in seconds
print(run_starts)
# {1: np.float64(0.038), 2: np.float64(1131.97), 3: np.float64(2134.146), 4: np.float64(3136.295)}

# Eyelink recorded at 1000 Hz; 1/2 slower than the fMRI sampling
tr = 2  # Seconds per volume
n_vols = {1: 483, 2: 463, 3: 468, 4: 464}  # Volumes per run
times = sub26_eye_data["times"]  # Sample times in seconds
samples = sub26_eye_data["samples"]  # xpos, ypos, pupil area
# Assign samples to each frame in each run
run_samples = {}
for run, start in run_starts.items():
    vol_index = np.floor((times - start) / tr).astype(int)  # Volume each sample falls in
    in_run = (vol_index >= 0) & (vol_index < n_vols[run])  # Samples inside this run
    run_samples[run] = (vol_index[in_run], samples[:, in_run])  # Volume index and samples for this run

# Per-volume eye regressors (pupil, gaze position, gaze speed, blinks, saccades)
blinks = sub26_eye_data["discrete"]["blinks"]
saccades = sub26_eye_data["discrete"]["saccades"]
x, y, pupil = samples

# Flag samples inside a blink
in_blink = np.zeros(len(times), dtype=bool)
for b_start, b_end in zip(blinks["stime"], blinks["etime"]):
    in_blink |= (times >= b_start) & (times <= b_end)

# Valid samples have gaze position, nonzero pupil, and no blink
valid = ~np.isnan(x) & ~np.isnan(y) & (pupil > 0) & ~in_blink

# Gaze speed in pixels per second between consecutive samples
speed = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)) * 1000

# Saccades that do not overlap a blink
clean_saccade = np.array([
    not np.any((blinks["stime"] <= s_end) & (blinks["etime"] >= s_start))
    for s_start, s_end in zip(saccades["stime"], saccades["etime"])
])

for run, start in run_starts.items():
    vols = pd.RangeIndex(n_vols[run])

    # Sample-level table for the samples inside this run
    vol_index = np.floor((times - start) / tr).astype(int)
    in_run = (vol_index >= 0) & (vol_index < n_vols[run])
    sample_table = pd.DataFrame({
        "vol": vol_index,
        "x": x,
        "y": y,
        "pupil": pupil,
        "speed": speed,
        "valid": valid,
        "blink": in_blink,
    })[in_run]
    good = sample_table[sample_table["valid"]].groupby("vol")
    all_samples = sample_table.groupby("vol")

    # Continuous features, one row per volume
    eye = pd.DataFrame(index=vols)
    eye["pupil_mean"] = good["pupil"].mean()
    eye["gaze_x_mean"] = good["x"].mean()
    eye["gaze_y_mean"] = good["y"].mean()
    eye["gaze_speed_mean"] = good["speed"].mean()
    eye["blink_fraction"] = all_samples["blink"].mean()

    # Event counts, by the volume each event starts in
    blink_vol = np.floor((blinks["stime"] - start) / tr).astype(int)
    saccade_vol = np.floor((saccades["stime"][clean_saccade] - start) / tr).astype(int)
    eye["blink_count"] = pd.Series(blink_vol).value_counts().reindex(vols, fill_value=0)
    eye["saccade_count"] = pd.Series(saccade_vol).value_counts().reindex(vols, fill_value=0)

    # Save next to the BOLD derivatives
    eye.to_csv(func_dir / f"{sub}_task-sm_run-{run}_desc-eye_timeseries.tsv", sep="	", index=False)
    print(run, eye.shape, eye.isna().any(axis=1).sum())
# Volumes with missing values in runs 1-4: 53, 38, 21 and 41
# Most likely MAR type missing data
# Z-score the eye regressors per run, fill missing volumes with the run mean and flag them
for run in runs:
    eye = pd.read_csv(func_dir / f"{sub}_task-sm_run-{run}_desc-eye_timeseries.tsv", sep="\t")
    present = eye["pupil_mean"].notna()  # Volumes with valid eye samples
    eye_z = (eye - eye.mean()) / eye.std()  # Mean and std from observed volumes only
    eye_z = eye_z.fillna(0)  # Run mean after z-scoring
    eye_z["eye_tracking_present"] = present.astype(int)  # 1 = valid eye data, 0 = filled
    eye_z.to_csv(func_dir / f"{sub}_task-sm_run-{run}_desc-eyezscored_timeseries.tsv", sep="	", index=False)

# Task example videos
period_frames = {
    "trial_examples_day_1_bhaa117.mp4": [
        (336, 696, "walking_direction_1"),
        (1135, 1495, "walking_direction_2"),
        (1967, 2329, "walking_direction_4"),
        (2832, 3194, "walking_direction_2"),
        (3681, 4042, "walking_direction_3"),
    ],
    "trial_examples_day_2_bhaa117.mp4": [
        (428, 792, "walking_direction_4"),
        (1233, 1594, "walking_direction_3"),
        (1714, 1834, "facing_direction"),
        (2673, 3034, "walking_direction_3"),
        (3553, 3921, "walking_direction_1"),
        (4041, 4164, "facing_direction"),
    ],
}

# Load both videos
videos = {
    video_name: cv2.VideoCapture(str(data_path / video_name))
    for video_name in period_frames
}

# One clip per walking direction and one facing clip, picked by checking the masks
selected_clips = {
    ("trial_examples_day_1_bhaa117.mp4", 336),
    ("trial_examples_day_1_bhaa117.mp4", 1135),
    ("trial_examples_day_2_bhaa117.mp4", 2673),
    ("trial_examples_day_2_bhaa117.mp4", 428),
    ("trial_examples_day_2_bhaa117.mp4", 1714),
}

# Load Mask R-CNN (COCO person) on the GPU
model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_score_thresh=0.3).eval().cuda()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # Dilation kernel
rng = np.random.default_rng(524)
clip_path = data_path / "derivatives" / "masked_clips"
clip_path.mkdir(exist_ok=True)

# Grab the frames of each selected period, mask the characters and fill them with white noise
clips = []
for video_name, periods in period_frames.items():
    video = videos[video_name]
    for start, end, label in periods:
        if (video_name, start) not in selected_clips:
            continue
        video.set(cv2.CAP_PROP_POS_FRAMES, start)  # Seek to the first frame
        frames = []
        for _ in range(end - start + 1):  # End frame is inclusive
            ok, frame = video.read()  # BGR array
            if not ok:
                break
            frames.append(frame)
        frames = np.stack(frames)
        masks = []
        for i in range(0, len(frames), 8):  # Batches of 8 frames
            batch = [torch.from_numpy(f[..., ::-1].copy()).permute(2, 0, 1).float().div(255).cuda() for f in frames[i:i + 8]]  # BGR to RGB tensors
            with torch.no_grad():
                results = model(batch)
            for result in results:
                person = (result["labels"] == 1) & (result["scores"] > 0.5)  # Confident person detections
                mask = (result["masks"][person, 0] > 0.5).any(0).cpu().numpy() if person.any() else np.zeros(frames.shape[1:3], bool)  # Union of person masks
                masks.append(cv2.dilate(mask.astype(np.uint8), kernel).astype(bool))  # Dilate the mask
        masks = np.stack(masks)
        frames = frames.astype(np.float32)
        mean = np.stack([f[~m].mean(0) for f, m in zip(frames, masks)])[:, None, None, :]  # Unmasked mean per frame and channel
        std = np.stack([f[~m].std(0) for f, m in zip(frames, masks)])[:, None, None, :]  # Unmasked std per frame and channel
        noise = np.clip(rng.normal(mean, std, frames.shape), 0, 255)  # Independent noise per channel
        frames = np.where(masks[..., None], noise, frames).astype(np.uint8)  # Masked pixels replaced by noise
        clips.append({"video": video_name, "label": label, "start": start, "frames": frames, "masks": masks})

        # Save the masked clip as lossless arrays for feature extraction
        np.save(clip_path / f"{label}_frames.npy", frames)
        np.save(clip_path / f"{label}_masks.npy", masks)

        # Save the masked clip as mp4 for viewing
        writer = cv2.VideoWriter(str(clip_path / f"{label}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 60, (frames.shape[2], frames.shape[1]))
        for frame in frames:
            writer.write(frame)
        writer.release()

# Release the video handles
for video in videos.values():
    video.release()

# Download PiDiNet code and weights
pidinet_commit = "d21aa881ed9c628571636fad39acfe1fad517ebd"
pidinet_path = data_path / "pidinet"
if not pidinet_path.exists():
    zip_path = data_path / "pidinet.zip"
    urllib.request.urlretrieve(f"https://github.com/hellozhuo/pidinet/archive/{pidinet_commit}.zip", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_path)
    # Rename the folder
    (data_path / f"pidinet-{pidinet_commit}").rename(pidinet_path)
    # Delete the zip
    zip_path.unlink()

# Import PiDiNet from the downloaded repo
sys.path.insert(0, str(pidinet_path))
from models.pidinet import pidinet

# Build the model
pidinet_model = pidinet(SimpleNamespace(config="carv4", dil=True, sa=True))
# Load the weights
checkpoint = torch.load(pidinet_path / "trained_models" / "table5_pidinet.pth", map_location="cpu", weights_only=False)
# Strip the module prefix
pidinet_model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()})
# Move to the GPU
pidinet_model = pidinet_model.cuda().eval()

# Junction types and 15 degree angle bins
junction_labels = ["T", "Y", "Arrow", "X"]
angle_bins = np.arange(0, 121, 15)
# Kernel that counts the 8 neighbours of a pixel
neighbours = np.ones((3, 3), int)
neighbours[1, 1] = 0
# Dilation kernel for the area around the masked characters
exclusion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
# ImageNet mean and std on the GPU
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()


# Contour map of a BGR frame (PiDiNet run on the frame enlarged 2x)
def contour_map(frame):
    height, width = frame.shape[:2]
    # Enlarge the frame 2x
    enlarged = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # BGR array to RGB tensor with values 0 to 1 on the GPU
    x = torch.from_numpy(enlarged[..., ::-1].copy()).permute(2, 0, 1)[None].float().cuda() / 255
    # Normalize the tensor, run PiDiNet and keep its final contour map
    with torch.no_grad():
        contours = pidinet_model((x - imagenet_mean) / imagenet_std)[-1][0, 0].cpu().numpy()
    # Shrink the contour map back to the frame size
    return cv2.resize(contours, (width, height), interpolation=cv2.INTER_AREA)


# Pixels of a one-pixel-wide line in order from one end to the other
def order_path(coords):
    pixels = {tuple(c) for c in coords}
    # Neighbours of a pixel that are on the line
    def pixel_neighbours(c):
        return [(c[0] + dy, c[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and (c[0] + dy, c[1] + dx) in pixels]
    # Pixels with one neighbour are the ends of the line
    ends = [c for c in pixels if len(pixel_neighbours(c)) == 1]
    if not ends:
        return None
    # Walk along the line from one end until there is no unseen neighbour
    path, current = [ends[0]], ends[0]
    seen = {current}
    while True:
        following = [n for n in pixel_neighbours(current) if n not in seen]
        if not following:
            return path
        current = following[0]
        seen.add(current)
        path.append(current)


# Junction type from the directions of the lines leaving a junction (Walther et al., 2023)
def junction_type(angles):
    angles = np.sort(angles)
    # Four lines are an X
    if len(angles) == 4:
        return "X"
    # Other numbers of lines get no type
    if len(angles) != 3:
        return None
    # Largest angle between neighbouring lines
    max_angle = np.diff(np.r_[angles, angles[0] + 360]).max()
    # Under 160 is Y, 160 to 200 is T, over 200 is Arrow
    if max_angle < 160:
        return "Y"
    return "T" if max_angle <= 200 else "Arrow"


# Junction types and minimum angles in a skeleton of one-pixel-wide lines
def find_junctions(skeleton):
    # Skeleton pixels with three or more neighbours
    branch = skeleton & (convolve(skeleton.astype(int), neighbours, mode="constant") >= 3)
    # Group branch pixels less than 2 px apart into one junction
    merged = cv2.dilate(branch.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    junction_ids, n_junctions = label_regions(merged, structure=np.ones((3, 3)))
    # Centre of each junction
    ys, xs = np.nonzero(branch)
    ids = junction_ids[ys, xs]
    n_pixels = np.maximum(np.bincount(ids, minlength=n_junctions + 1), 1)
    centre_y = np.bincount(ids, ys, n_junctions + 1) / n_pixels
    centre_x = np.bincount(ids, xs, n_junctions + 1) / n_pixels
    # Lines of the skeleton between the junctions
    line_ids, _ = label_regions(skeleton & ~merged, structure=np.ones((3, 3)))
    directions = {j: [] for j in range(1, n_junctions + 1)}
    # Direction of each line leaving each junction
    for line_id, box in enumerate(find_objects(line_ids), 1):
        # Pixels of the line
        coords = np.argwhere(line_ids[box] == line_id) + [box[0].start, box[1].start]
        # Skip lines under 6 px
        if len(coords) < 6:
            continue
        # Pixels of the line in order from one end to the other
        path = order_path(coords)
        if path is None:
            continue
        # Both ends of the line, each with the path starting at that end
        for end, from_end in ((path[0], path), (path[-1], path[::-1])):
            # Junction within 2 px of the end of the line
            near = junction_ids[max(end[0] - 2, 0):end[0] + 3, max(end[1] - 2, 0):end[1] + 3].max()
            if near == 0:
                continue
            # Pixel 8 steps along the line from the junction
            y, x = from_end[min(8, len(from_end) - 1)]
            # Direction from the junction centre to that pixel in degrees
            directions[near].append(np.degrees(np.arctan2(y - centre_y[near], x - centre_x[near])) % 360)
    # Type and smallest angle between neighbouring lines of each junction
    types, min_angles = [], []
    for angles in directions.values():
        junction = junction_type(angles)
        if junction:
            types.append(junction)
            sorted_angles = np.sort(angles)
            min_angles.append(np.diff(np.r_[sorted_angles, sorted_angles[0] + 360]).min())
    return types, min_angles


# Contour junction features for each volume of each masked clip
feature_names = junction_labels + [f"angle_{low}_{low + 15}" for low in angle_bins[:-1]]
feature_rows = []
for clip_label in ["walking_direction_1", "walking_direction_2", "walking_direction_3", "walking_direction_4", "facing_direction"]:
    # Load the clip without its last frame
    clip_frames = np.load(clip_path / f"{clip_label}_frames.npy")[:-1]
    clip_masks = np.load(clip_path / f"{clip_label}_masks.npy")[:-1]
    # Junction counts, angle histograms and unmasked areas of each frame
    counts = np.zeros((len(clip_frames), len(junction_labels)))
    histograms = np.zeros((len(clip_frames), len(angle_bins) - 1))
    areas = np.zeros(len(clip_frames))
    for i, (frame, mask) in enumerate(zip(clip_frames, clip_masks)):
        # Area around the masked characters
        excluded = cv2.dilate(mask.astype(np.uint8), exclusion_kernel).astype(bool)
        # Contour map thresholded into lines
        lines = apply_hysteresis_threshold(contour_map(frame), 0.2, 0.5)
        # Remove lines in the excluded area and along the image border
        lines[excluded] = False
        lines[:4], lines[-4:], lines[:, :4], lines[:, -4:] = False, False, False, False
        # Thin the lines to one pixel and remove lines under 20 px
        skeleton = remove_small_objects(skeletonize(lines), 20, connectivity=2)
        # Junction types and minimum angles of the frame
        types, min_angles = find_junctions(skeleton)
        # Store the counts, angle histogram and unmasked area of the frame
        counts[i] = [types.count(t) for t in junction_labels]
        histograms[i] = np.histogram(min_angles, bins=angle_bins)[0]
        areas[i] = (~excluded).sum()
    # Junctions per 1000 unmasked pixels in each window of 120 frames (one volume)
    for volume in range(len(clip_frames) // 120):
        window = slice(volume * 120, (volume + 1) * 120)
        totals = np.concatenate([counts[window].sum(0), histograms[window].sum(0)])
        feature_rows.append({"clip": clip_label, "volume": volume, **dict(zip(feature_names, 1000 * totals / areas[window].sum()))})
# Table of the features, saved as TSV
contour_junctions = pd.DataFrame(feature_rows)
contour_junctions.to_csv(data_path / "derivatives" / "contour_junction_features.tsv", sep="\t", index=False)

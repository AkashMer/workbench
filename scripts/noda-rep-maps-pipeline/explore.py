# Import necessary libraries
from pathlib import Path
from pymatreader import read_mat
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from sklearn.cluster import KMeans

# Load one calcium imaging file to confirm binning
# 1. Get the repo root
repo_root = Path.cwd()
data_path = repo_root / "data" / "noda-rep-maps-pipeline"

# 2. Define the path to the data file
data_file_path_cal = data_path / "calcium_excitatory" / "VISp" / "511509529.mat"

# 3. Load the file
data_cal = read_mat(data_file_path_cal)

# 4. Check out what is present in the file
for key, item in data_cal.items():
    if not key.startswith('__'):
        if isinstance(item, list):
            print(key, type(item), len(item))
        elif isinstance(item, np.ndarray):
            print(key, type(item), item.shape)
        else:
            print(key, type(item), item)
# Data stored in raw_pop_vector_info trials (neurons x time bins x trials)
binned_VISp_cal = data_cal['raw_pop_vector_info_trials']
# No mouse_id or similar information present

# Confirm if calcium inhibitory data is also has the same structure
# 1. Define the path to the data file
data_file_path_cal_inhib = data_path / "calcium_inhibitory" / "VISp" / "617395453.mat"

# 2. Load the data
data_cal_inhib = read_mat(data_file_path_cal_inhib)

# 3. Check out what is present in the file
for key, item in data_cal_inhib.items():
    if not key.startswith('__'):
        if isinstance(item, list):
            print(key, type(item), len(item))
        elif isinstance(item, np.ndarray):
            print(key, type(item), item.shape)
        else:
            print(key, type(item), item)
# Traces are in neurons x raw frames x days => unbinned
# I don't know enough about representational maps yet to understand how I can use
# interneuron data + data is unbinned. I am excluding calcium inhibitory data.

# Load one neuropixels file to confirm binning
# 1. Define the path to the data file
data_file_path = data_path / "neuropixels" / "session_787025148.mat"

# 2. Load the file
data = read_mat(data_file_path)

# 3. Check out what is present in the file
for key, item in data.items():
    if not key.startswith('__'):
        if isinstance(item, list):
            print(key, type(item), len(item))
        elif isinstance(item, np.ndarray):
            print(key, type(item), item.shape)
        else:
            print(key, type(item), item)
# Confirm
cell = data['informative_rater_mat'][0][0]
print(cell.shape)
# Seems like this is where the spiking information is stored 68 neurons x 27000 frames x 2 blocks
# No mouse_id or similar subject identifier stored in the processed data
# But what is stored in the 1 position?
cell_2 = data['informative_rater_mat'][1][0]
print(cell_2.shape)
# Matches the general shape but number of frames ie. repeats are lower
# visual_drift analysis file point to:
# 0 position is for Natural Movie 1 more repeats
# 1 position is for Shuffled Natural Movie 1
# Confirming this through repeats of movie in this one file
data['mean_pupil_movement_repeats'][0].shape
# 30 repeats, 2 blocks
data['mean_pupil_movement_repeats'][1].shape
# 10 repeats, 2 blocks
# Limiting myself to Natural Movie 1 for now
# The Shuffled part is most likely control for the experiement
# Define the area names as per visual_drift analysis
brain_areas = ['VISp','VISl','VISal','VISpm','VISrl','VISam','LGd','LP']
# Get data for all areas and movie 1
neuropixels_data = data['informative_rater_mat'][0]
# Confirm shape of each area
for i, area in enumerate(brain_areas):
    print(area, neuropixels_data[i].shape)
# Some are empty

# Compute the cohort size of neuropixels modality
manifest_path = os.path.join(os.path.expanduser('~'), 'allen_cache_ecephys', 'manifest.json')
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
# Get the number of unique mice in the Neuropixels data
neuropixels_session_table = cache.get_session_table()
neuropixels_session_table.columns
# The session_id most likely is the index id
neuropixels_session_table = neuropixels_session_table.reset_index()
neuropixels_session_table.columns
neuropixels_session_table.id
len(neuropixels_session_table.id)
# Matches the file names and the number of neuropixels files
# Compute the unique number of specimen_ids
neuropixels_session_table.specimen_id.nunique()
# Each session is equivalent to each mice. 58 total

# How many neurons recorded per mice per session?
neuropixels_area_wise = pd.DataFrame(columns=['session_id'] + brain_areas, index = range(58))
idx = 0
for f in data_path.glob('neuropixels/*.mat'):
    load_file = read_mat(f)
    neuropixels_area_wise.loc[idx, 'session_id'] = int(f.stem.replace('session_', ''))
    for num, area in enumerate(brain_areas):
        if load_file['informative_rater_mat'][0][num].size > 0:
            neuropixels_area_wise.loc[idx, area] = load_file['informative_rater_mat'][0][num].shape[0]
        else:
            neuropixels_area_wise.loc[idx, area] = 0
    idx +=1
# Combine both mouse id info and this area wise neurons into one table
neuropixels_area_wise = (
    neuropixels_area_wise.merge(neuropixels_session_table[['id', 'specimen_id']], how='left', left_on='session_id', right_on='id')
    .drop(columns='id')
    .set_index('session_id')
    # Sort by mice with the lowest minimum in across all areas in descending order
    .assign(min=lambda x: x[brain_areas].min(axis = 1))
    .sort_values('min', ascending=False)
    .drop(columns = 'min')
    .assign(total=lambda x: x[brain_areas].sum(axis = 1).astype(int))
)
# Order the columns and display
print(neuropixels_area_wise[['specimen_id'] + brain_areas + ['total']].head().to_string())
#             specimen_id VISp VISl VISal VISpm VISrl VISam LGd   LP  total
# session_id                                                               
# 755434585     730760270   75   39    42    62    49    94  44   27    432
# 756029989     734865738   51   30    51    90    24    72  60   27    405
# 750749662     726162197   52   20    46    64    41    64  82  142    511
# 719161530     703279284   52   40     9    18    10    37  71   28    265
# 791319847     769360779   93   56    43    17    58    49   8    9    333

# Compute the cohort size of calcium_excitatory modality
manifest_path_cal = os.path.join(os.path.expanduser('~'), 'allen_cache_ophys', 'manifest.json')
cache_cal = BrainObservatoryCache(manifest_file=manifest_path_cal)
calcium_session_info = cache_cal.get_experiment_containers()
calcium_session_table = pd.DataFrame(columns=["id", 'donor_name'], index=range(len(calcium_session_info)))
for i in range(len(calcium_session_info)):
    calcium_session_table.loc[i, 'id'] = calcium_session_info[i]['id']
    calcium_session_table.loc[i, 'donor_name'] = calcium_session_info[i]['donor_name']
# Get the id of the processed data
processed_data_id = [int(f.stem) for f in data_path.glob('calcium_excitatory/*/*.mat')] # 336
# Filter the session table
filtered_calcium_session_table = calcium_session_table[calcium_session_table['id'].isin(processed_data_id)]
# Compute the unique number of specimen_id = donor_name
filtered_calcium_session_table.donor_name.nunique()
# 193 mice, with multuple sessions

# How many neurons recorded per mice per session?
calcium_area_wise = pd.DataFrame(columns=['session_id', 'area', 'n_neurons'], index = range(336))
idx = 0
for f in data_path.glob('calcium_excitatory/**/*.mat'):
    calcium_area_wise.loc[idx, 'session_id'] = int(f.stem)
    calcium_area_wise.loc[idx, 'area'] = f.parent.name
    load_file = read_mat(f)
    calcium_area_wise.loc[idx, 'n_neurons'] = load_file['raw_pop_vector_info_trials'].shape[0]
    idx+=1
# Pivot to wide format and add mouse id info
calcium_area_wise = (
    calcium_area_wise
    .merge(filtered_calcium_session_table, how = 'left',
            left_on = 'session_id', right_on = 'id')
    .drop(columns = 'id')
    # Get the minimum number of neurons per area per mice
    .pivot_table(index = 'donor_name', columns = 'area', values = 'n_neurons', aggfunc='min')
    .fillna(0)
    .merge(filtered_calcium_session_table.groupby('donor_name')['id']
                                            .apply(list).reset_index(),
            how = 'left', on = 'donor_name')
)
# Convvert number of neurons to int type
calcium_area_wise[brain_areas[:6]] = calcium_area_wise[brain_areas[:6]].astype(int)
# Order by minimum number of neurons per area per mice
calcium_area_wise = (
    calcium_area_wise.assign(min=lambda x: x[brain_areas[:6]].min(axis = 1))
    .sort_values('min', ascending=False)
    .drop(columns = 'min')
    .assign(total=lambda x: x[brain_areas[:6]].sum(axis = 1).astype(int))
)
# Order the columns and display
print(calcium_area_wise[['donor_name'] + brain_areas[:6] + ['total']].head().to_string())
# donor_name  VISp  VISl  VISal  VISpm  VISrl  VISam  total
# 0       221470   237     0      0      0      0      0    237
# 97      307419     0     0      0      0     45      0     45
# 123     337438     0     0      7      0      0      6     13
# 124     337458   180     0      0      0      0      0    180
# 125     338502     0     0      0      0    369      0    369
# Do any mice have all areas recorded?
print(calcium_area_wise[(calcium_area_wise[brain_areas[:6]] > 0).all(axis=1)].to_string())
# Empty DataFrame
# Columns: [donor_name, VISal, VISam, VISl, VISp, VISpm, VISrl, id, total]
# Index: []
# No single mouse has all areas recorded, which goes against the ideal situation
# of one individual in the Noda paper

# Final verdict
# Picked session 755434585 / mouse 730760270: every area has >= 27 neurons,
# 432 neurons total, and behavior data (pupil + running speed summary stats)
# is fully clean (0% NaN)

# Bin the neuropixel data
binned_VISp = np.zeros((68, 30, 60))
# General logic
# Divide the session into 30 repeats x 30 bins x 30 frames x 2 blocks
reshaped_data = np.reshape(neuropixels_data[0], (68, 30, 30, 30, 2))
# Average frames across bins
binned_data = np.mean(reshaped_data, axis = 3)
# Transpose to change the structure to neurons x bins x repeats x blocks
binned_data = np.transpose(binned_data, (0, 2, 1, 3))
# Join data for 2 blocks
binned_VISp = np.reshape(binned_data, (68, 30, 60))
# Confirm
print(binned_VISp.shape)
# neurons x bins x trials

# Build RSM for each area
# Using trial to trial correlations instead of averaging across trials
rsms = {}
for i, area in enumerate(brain_areas):

    if(neuropixels_data[i].size > 0):
        n_neurons = neuropixels_data[i].shape[0]
        n_bins = 30
        n_trials = 60
        
        # Bin the data
        reshaped_data = np.reshape(neuropixels_data[i], (n_neurons, n_bins, n_bins, n_bins, 2))
        binned_data = np.mean(reshaped_data, axis = 3)
        binned_data = np.transpose(binned_data, (0, 2, 1, 3))
        binned_data = np.reshape(binned_data, (n_neurons, n_bins, n_trials))
        
        # Build the RSM
        res = np.zeros((n_bins, n_bins))
        cors = np.corrcoef(binned_data.reshape(n_neurons, n_bins*n_trials), rowvar=False)
        for j,k in itertools.product(range(n_bins), range(n_bins)):
            res[j,k] = np.mean(cors[j*n_trials:(j+1)*n_trials, k*n_trials:(k+1)*n_trials])
        rsms[area] = res

# Plot RSMs by area
fig, axes = plt.subplots(2, 3)
axes = axes.flatten()
plot_idx = 0
for item in rsms:
    axes[plot_idx].imshow(rsms[item], cmap = "RdBu")
    axes[plot_idx].set_title(item)
    plot_idx += 1

# Dimension reduction
pca = PCA(n_components=2)
fig, axes = plt.subplots(2, 3)
axes = axes.flatten()
plot_idx = 0
for item in rsms:
    dem_red = pca.fit_transform(rsms[item])
    axes[plot_idx].scatter(dem_red[:,0], dem_red[:,1])
    axes[plot_idx].set_title(item)
    plot_idx += 1

# Validate against the stimulus data
# Download Natural Movie 1
movie = cache.get_natural_movie_template(1)
print(movie.shape)
# Bin movie frames into 30 one-second bins
movie_binned = movie.reshape(30, 30, 304, 608).mean(axis=1)  # (30, 304, 608)
movie_flat = movie_binned.reshape(30, -1).astype(float)       # (30, 184832)
# Create a Pixel Similarity Matrix (30 x 30)
pixel_similarity = np.corrcoef(movie_flat)
print('Pixel similarity matrix shape:', pixel_similarity.shape)
# Plot
plt.figure(figsize=(6,6))
plt.imshow(pixel_similarity, cmap='RdBu')
plt.colorbar(label='Pixel correlation')
# PCA on this pixel similarity matrix to give groups
stimulus_dem_red = pca.fit_transform(pixel_similarity)
# Find the 2 clusters
stimulus_clusters = KMeans(n_clusters = 2).fit_predict(stimulus_dem_red)

# Use the pc1 dimension to provide groups to the reponse PCA
fig, axes = plt.subplots(2, 3)
axes = axes.flatten()
plot_idx = 0
for item in rsms:
    dem_red = pca.fit_transform(rsms[item])
    axes[plot_idx].scatter(dem_red[:,0], dem_red[:,1], c=stimulus_clusters, cmap = 'viridis')
    axes[plot_idx].set_title(item)
    plot_idx += 1

# What is the pixel similarity capturing?
fig, axes = plt.subplots(5, 6, figsize=(15, 10))
axes = axes.flatten()
for i in range(30):
    axes[i].imshow(movie_binned[i], cmap='gray')
    axes[i].set_title(f'Bin {i} | C{stimulus_clusters[i]}', 
                      color='yellow' if stimulus_clusters[i] == 0 else 'purple')
    axes[i].axis('off')
plt.tight_layout()
# The car appears around bins 9, 10, 11
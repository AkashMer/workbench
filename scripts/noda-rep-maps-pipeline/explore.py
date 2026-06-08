# Import necessary libraries
from pathlib import Path
from pymatreader import read_mat
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes
from sklearn.manifold import MDS
import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from sklearn.cluster import KMeans
import rsatoolbox
import imageio

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
# 432 neurons total

# Check behavioral data for the session_756029989
# 1. Define the path
data_file_path = data_path / "neuropixels" / "session_756029989.mat"

# 2. Load the file
data = read_mat(data_file_path)

# Get the mean pupil size, position and mean running speed for movie type 1
mean_pupil_size = data['mean_pupil_size_repeats'][0]
mean_pupil_size.shape
mean_running_speed = data['mean_running_speed_repeats'][0]
mean_running_speed.shape
# 10 repeats of 30 fps movie clip
repeats = range(10)

# Define a df to hold the behavioral data for plotting
behavioral_metrics = (pd.concat([
        pd.DataFrame(mean_pupil_size, columns=['Block 1', 'Block 2']).assign(metric='Mean Pupil Size'),
        pd.DataFrame(mean_running_speed,  columns=['Block 1', 'Block 2']).assign(metric='Mean Running Speed')
    ])
    .reset_index(names='repeat')
    .melt(id_vars = ['repeat', 'metric'], value_vars = ['Block 1', 'Block 2'],
            var_name = 'block', value_name = 'value')
)

# Plot in a Seaborn grid
g = sns.FacetGrid(behavioral_metrics, row = 'metric', sharey = False, height = 3, aspect = 2.5)
g.map_dataframe(sns.lineplot, x='repeat', y='value', hue='block', marker='o')
g.add_legend()
# There is quite a difference between 2 blocks
block1_mean = behavioral_metrics[behavioral_metrics['block'] == 'Block 1'].groupby('metric')['value'].mean()
block2_mean = behavioral_metrics[behavioral_metrics['block'] == 'Block 2'].groupby('metric')['value'].mean()
for ax, metric in zip(g.axes.flat, block1_mean.index):
    ax.axhline(block1_mean[metric], linestyle='--', alpha=0.5, color='blue')
    ax.axhline(block2_mean[metric], linestyle='--', alpha=0.5, color='orange')
# Both blocks are recorded under 2 different behavioral states; with block 1 showing stationary condition

# Get the neural activity data for block 1
neuropixels_data = [area[:, :, 0] for area in data['informative_rater_mat'][0][:8]]
# Define the number of bins
n_bins = 100
frames_per_bin = 9
# Define the number of trials
n_trials = 10
# Initialize a population reponse space variable
rsms = {}
# Bin the data in each area and compute the RDM
for idx, area in enumerate(brain_areas):
    # Get the number of neurons
    n_neurons = neuropixels_data[idx].shape[0]
    # Reshape into 10 repeats x 100 bins x 9 frames
    reshaped_data = np.reshape(neuropixels_data[idx], (n_neurons, n_trials, n_bins, frames_per_bin))
    # Average out the frame
    binned_data = np.mean(reshaped_data, axis = 3)
    # Transpose to change the structure to bins x repeats x neurons
    binned_data = np.transpose(binned_data, (2, 1, 0))
    # Flatten bins and repeats for cdist function
    binned_data = np.reshape(binned_data, (n_bins * n_trials, n_neurons))

    # Compute the disimilarity matrix using pearson correlation
    dist_matrix = cdist(binned_data, binned_data, metric='correlation')

    # Convert to a similarity matrix
    similarity_matrix = 1 - dist_matrix

    # Block the diagonal to preserve trial-to-trial reliability measure
    for i in range(n_bins):
        block = similarity_matrix[i*n_trials:(i+1)*n_trials, i*n_trials:(i+1)*n_trials]
        np.fill_diagonal(block, np.nan)

    # Fisher Z transforamtion to average correlation within bins
    z_matrix = np.arctanh(similarity_matrix)

    # Averge within bins
    res = np.zeros((n_bins, n_bins))
    for j, k in itertools.product(range(n_bins), range(n_bins)):
        block = z_matrix[j*n_trials:(j+1)*n_trials, k*n_trials:(k+1)*n_trials]
        if j == k:
            res[j, k] = np.nanmean(block)
        else:
            res[j, k] = np.mean(block)

    # Transform back to correlations
    rsm = np.tanh(res)

    # Save the result
    rsms[area] = pd.DataFrame(rsm, index = range(n_bins), columns=range(n_bins))

# Plot RSMs by area
fig, axes = plt.subplots(2, 4)
axes = axes.flatten()
plot_idx = 0
for item in rsms:
    sns.heatmap(rsms[item], ax=axes[plot_idx], cmap='RdBu', cbar=False, square=True,
                vmin = -1, vmax = 1)
    axes[plot_idx].set_title(item)
    axes[plot_idx].set_xticks([])
    axes[plot_idx].set_yticks([])
    plot_idx += 1

# Dimension reduction via MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=23, n_init=1)
neural_mds = {}
for area in rsms:
    rsm_for_mds = rsms[area].values.copy()
    np.fill_diagonal(rsm_for_mds, 1)
    neural_mds[area] = mds.fit_transform(1 - rsm_for_mds)

# Validate against the stimulus data
# Download Natural Movie 1
movie = cache.get_natural_movie_template(1)
# Bin movie frames into 100 9-frame bins (~ 300 ms)
movie_binned = movie.reshape(100, 9, 304, 608).mean(axis=1)
movie_flat = movie_binned.reshape(100, -1).astype(float)
# Create a Pixel Similarity Matrix (100 x 100)
pixel_similarity = 1 - cdist(movie_flat, movie_flat, metric='correlation')

# Apply MDS for visualization
stimulus_mds = MDS(n_components=2, dissimilarity='precomputed', random_state=23, n_init=1).fit_transform(1 - pixel_similarity)
stimulus_clusters = KMeans(n_clusters=2, random_state=23).fit_predict(stimulus_mds)

# Neural MDS colored by stimulus clusters — Procrustes-aligned to stimulus MDS
cluster_palette = sns.color_palette('Dark2', n_colors=2)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for plot_idx, area in enumerate(neural_mds):
    _, aligned, _ = procrustes(stimulus_mds, neural_mds[area])
    for c_idx in range(2):
        mask = stimulus_clusters == c_idx
        axes[plot_idx].scatter(aligned[mask, 0], aligned[mask, 1],
                               color=cluster_palette[c_idx], s=40)
    axes[plot_idx].set_title(area)
    axes[plot_idx].set_xticks([])
    axes[plot_idx].set_yticks([])

# What is the pixel similarity capturing?
# Check out the movie clip
imageio.mimwrite('scripts/noda-rep-maps-pipeline/attachments/natural_movie1.mp4', movie, format='ffmpeg', fps=30)
# Alley with couple till 4-5 seconds
# Wall with the 3rd individual's shadow on it from 6-10 seconds
# Car completely in view at around 12 seconds
# Couple enters around 15-16 second, walks to car and enter

# Compare both RSMs for each area
triu_idx = np.triu_indices(n_bins, k=1)
pixel_rdm = rsatoolbox.rdm.RDMs((1 - pixel_similarity)[triu_idx].reshape(1, -1))
rsa_r = {}
for area in brain_areas:
    rsm_vals = np.nan_to_num(rsms[area].values, nan=0.0)
    neural_rdm = rsatoolbox.rdm.RDMs((1 - rsm_vals)[triu_idx].reshape(1, -1))
    rsa_r[area] = float(rsatoolbox.rdm.compare(neural_rdm, pixel_rdm, method='spearman')[0, 0])

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(brain_areas, [rsa_r[a] for a in brain_areas],
              color=sns.color_palette('muted', len(brain_areas)))
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_ylabel('Spearman r  (neural RDM vs pixel RDM)')
ax.set_title('RSA: Session 756029989 Block 1 — neural RDMs vs pixel RDM')
for bar, area in zip(bars, brain_areas):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{rsa_r[area]:.3f}', ha='center', va='bottom', fontsize=9)

# undecided
Dr. Akash Mer
2026-06-06

## Stimulus Description

### Experimental Setup

The raw data came from the Allen Brain Observatory “Visual Coding”
project. Mice were head-fixed in front of a monitor and shown various
visual stimuli: drifting gratings, natural images and natural movies.
During this time, neural activity was recorded using two-photon calcium
imaging and Neuropixels probes in two different cohorts of mice. The
following analysis is limited to sessions when the stimulus was Natural
Movie 1. The stimulus are presented repeatedly and grouped in blocks per
session.

**Ethological significance of Natural Movie 1**: Natural movie 1 is a
~30 second black-and-white movie clip from the film *Touch of Evil*.
Thus the movie is more ethologically significant than gratings or
images. The scene contains a perspective from a human filming which
might limit its ideal ethological definition from a mouse’s perspective.

**Brain areas**: The brain areas recorded were VISp, VISl, VISal, VISpm,
VISrl, VISam, LGd, LP which match the visual perception circuitry in
mice.

**Context of the recorded activity**: All sessions were recorded under
similar conditions of passive viewing and the same recording paradigm.
Both datasets also include behavioral state measures: pupil metrics
(movement, position, size, width) and running speed. I plan to compare
the pupil size and movement and running speed to mean cross-trial
reliability of the population response to gauge how reliably each
block/session represents the stimulus.  
<!--
The mean cross-trial reliability will be computed from cross-trial correlations instead of averaging across trials. Session weights for the final representational map will be informed by both the number of neurons recorded in each area and each session's cross-trial reliability(if confirmed by behavioral co-variance).
-->

<details class="code-fold">
<summary>Imports</summary>

``` python
import os
from pathlib import Path
import numpy as np
import pandas as pd
from pymatreader import read_mat
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
```

</details>

<details class="code-fold">
<summary>Code</summary>

``` python
# Download Natural Movie 1
manifest_path = os.path.join(os.path.expanduser('~'), 'allen_cache_ecephys', 'manifest.json')
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
movie = cache.get_natural_movie_template(1)

# Bin movie frames into 30 one-second bins
movie_binned = movie.reshape(30, 30, 304, 608).mean(axis=1)  # (30, 304, 608)
movie_flat = movie_binned.reshape(30, -1).astype(float)       # (30, 184832)

# Create a Pixel Similarity Matrix (30 x 30)
pixel_similarity = np.corrcoef(movie_flat)

# PCA on the pixel similarity matrix to give groups
pca_random_state = 23
pca = PCA(n_components=2, random_state=pca_random_state)
stimulus_pca = pca.fit_transform(pixel_similarity)

# Check how the movie gets separated for 2 clusters
stimulus_clusters = KMeans(n_clusters=2, random_state=23).fit_predict(stimulus_pca)
# Deine color for 2 clusters
cluster_palette = sns.color_palette('Dark2', n_colors=2)

# Shared Styling
title_fontsize = 18
label_fontsize = 12
frame_aspect = movie_binned.shape[1] / movie_binned.shape[2]

# Plot with matrix and pca scatter in 1st column and binned movie frames in 2nd
fig = plt.figure(figsize=(16, 9))
subfigs = fig.subfigures(1, 2, width_ratios=[0.3, 1], wspace=0.02)
left_axes = subfigs[0].subplots(2, 1)
ax_heatmap, ax_scatter = left_axes

# Pixel similarity matrix
sns.heatmap(pixel_similarity, cmap='RdBu', square=True, xticklabels=False, yticklabels=False,
            cbar=False, ax=ax_heatmap)
ax_heatmap.set_title('Pixel similarity matrix', fontsize=title_fontsize, pad=8)
# Add the color legend in the space between 2 columns
cbar_ax = fig.add_axes([0.22, 0.56, 0.008, 0.28])
cbar = fig.colorbar(ax_heatmap.collections[0], cax=cbar_ax,
                    ticks=[pixel_similarity.min(), pixel_similarity.max()])
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
cbar.set_label('correlation', labelpad=-10)

# Stimulus structure
sns.scatterplot(x=stimulus_pca[:, 0], y=stimulus_pca[:, 1],
                hue=stimulus_clusters, palette=cluster_palette, legend=False, ax=ax_scatter)
ax_scatter.set_box_aspect(1)
ax_scatter.set_xlabel('PC1', fontsize=label_fontsize)
ax_scatter.set_ylabel('PC2', fontsize=label_fontsize)
ax_scatter.set_xticklabels([])
ax_scatter.set_yticklabels([])
ax_scatter.set_title('PCA of stimulus structure', fontsize=title_fontsize, pad=8)

# What is the pixel similarity capturing?
axes_frames = subfigs[1].subplots(5, 6, gridspec_kw={'left': 0.05, 'right': 0.98,
                                                     'wspace': 0.05, 'hspace': 0.02})
for i, ax in enumerate(axes_frames.flat):
    ax.imshow(movie_binned[i], cmap='gray')
    ax.set_box_aspect(frame_aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(cluster_palette[stimulus_clusters[i]])
        spine.set_linewidth(2)
subfigs[1].suptitle('Binned movie frames by cluster', fontsize=title_fontsize, y=0.90);
```

</details>

<div id="fig-stimulus-structure">

![](noda-rep-maps-pipeline_files/figure-commonmark/fig-stimulus-structure-output-1.png)

Figure 1: Stimulus Structure

</div>

The movie was binned using the same binning structure as the processed
data. I computed binwise pixel correlation which showed 2 distinct
regions around bins 9-12. This prompted me to apply KMeans
clustering(k=2) to the PCA of the correlation matrix to visualize the
structure. The movie clustered into two structurally distinct halves
which I can use for validation of my neural representational map. I
guessed two clusters before seeing the movie clip and the frame-by-frame
check of the bins gave the context for the 2 clusters. A car becomes
clearly visible around bins 9-12 preceded by a motion blur implying a
clear scene change.

## Cohort Size and Neural Population Statistics

### Cohort Size

| **Modality**                 | **Unique Mice** |
|------------------------------|-----------------|
| Calcium Imaging (Excitatory) | 193             |
| Neuropixels                  | 58              |

### Neurons recorded per area under each modality

The minimum and maximum neurons recorded across all sessions for each
modality and area:

<details class="code-fold">
<summary>Code</summary>

``` python
# Order of brain areas as per upstream data processing
brain_areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISrl', 'VISam', 'LGd', 'LP']

# Function to compute number of neurons in each area of the neuropixels data set
def neuropixels_area_counts(stimulus_areas):
    counts = {}
    for area_name, area in zip(brain_areas, stimulus_areas):
        if area.ndim == 3:
            counts[area_name] = area.shape[0]
        elif area.ndim == 2:
            counts[area_name] = 1
    return counts

# Get the current data-slug for the current folder and the data directory
current_dir = Path.cwd()
data_slug = current_dir.name
data_root = current_dir.parents[1] / "data" / data_slug

# Get neuron counts per area for the calcium excitatory method
neuron_counts = []
for file in data_root.glob('calcium_excitatory/*/*.mat'):
    mat = read_mat(str(file))
    neuron_counts.append({'modality': 'calcium_excitatory', 'area': file.parent.name,
                          'n_neurons': mat['raw_pop_vector_info_trials'].shape[0]})

# Get neuron counts per area for the Neuropixels method
for file in data_root.glob('neuropixels/*.mat'):
    mat = read_mat(str(file))
    for area_name, n in neuropixels_area_counts(mat['informative_rater_mat'][0]).items():
        neuron_counts.append({'modality': 'neuropixels', 'area': area_name, 'n_neurons': n})

# Save the result in a pandas DataFrame
neuron_stats = (pd.DataFrame(neuron_counts)
                .groupby(['modality', 'area'])['n_neurons']
                .agg(range=lambda x:f"{x.min()} - {x.max()}")
                .unstack('area')
                .droplevel(0, axis = 1)
                .reindex(columns=brain_areas)
                .rename_axis("Modality")
                .rename_axis("Brain Areas", axis=1)
                .rename({'calcium_excitatory':'Calcium Imaging (Excitatory)', 'neuropixels':'Neuropixels'})
                .fillna(0))
neuron_stats
```

</details>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

| Brain Areas | VISp | VISl | VISal | VISpm | VISrl | VISam | LGd | LP |
|----|----|----|----|----|----|----|----|----|
| Modality |  |  |  |  |  |  |  |  |
| Calcium Imaging (Excitatory) | 21 - 602 | 11 - 499 | 7 - 492 | 23 - 331 | 11 - 423 | 5 - 297 | 0 | 0 |
| Neuropixels | 14 - 126 | 14 - 100 | 9 - 185 | 13 - 115 | 10 - 111 | 17 - 135 | 1 - 90 | 2 - 170 |

</div>

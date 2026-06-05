# Import necessary libraries
from pathlib import Path
import scipy
import numpy as np

# Load one calcium imaging file to confirm binning
# 1. Get the repo root
repo_root = Path.cwd()

# 2. Define the path to the data file
data_file_path_cal = repo_root / "data" / "noda-rep-maps-pipeline" / "calcium_excitatory" / "VISp" / "511509529.mat"

# 3. Load the file
data_cal = scipy.io.loadmat(data_file_path_cal)

# 4. Check out what is present in the file
for key, val in data_cal.items():
    if not key.startswith('__'):
        print(key, np.array(val).shape)
# Data stored in raw_pop_vector_info trials (neurons x time bins x trials)

# Load one neuropixels file to confirm binning
# 1. Define the path to the data file
data_file_path = repo_root / "data" / "noda-rep-maps-pipeline" / "neuropixels" / "session_787025148.mat"

# 2. Load the file
data = scipy.io.loadmat(data_file_path)

# 3. Check out what is present in the file
for key, val in data.items():
    if not key.startswith('__'):
        print(key, np.array(val).shape)
# I can use this to create a mask over the neural activity data from Allen SDK?
# Confirm
cell = data['informative_rater_mat'][0, 0]
print(cell.shape)
# Seems like this is where the spiking information is stored
# Cross-check with valid units and units cutoff
valid_units = data['valid_units_drifting_gratings'][0, 0]
units_cutoff = data['units_cutoff_per_area'][0, 0]
print(valid_units.shape)
print(valid_units)
print(units_cutoff.shape)
print(units_cutoff)
# The processed data is in informative_rater_mat; no need to access AllenSDK for data

# Limiting myself to Natural Movie 1
# Checking repeats of movie in this one file
data['mean_pupil_movement_repeats'][0, 0].shape
# 10 repeats here for 1 area
# Define the area names
brain_areas = ['VISp','VISl','VISal','VISpm','VISrl','VISam','LGd','LP']
# Get data for all areas and movie 1
neuropixels_data = data['informative_rater_mat'][0, :8]
# Confirm shape of each area
for i, area in enumerate(brain_areas):
    print(area, neuropixels_data[i].shape)
# Some are empty

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
#
print(binned_VISp.shape)

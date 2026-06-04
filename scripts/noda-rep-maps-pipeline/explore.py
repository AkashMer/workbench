# Import necessary libraries
from pathlib import Path
import scipy
import numpy as np

# Load one neuropixels file to confirm binning
try:
    # 1. Find the folder slug for current file
    data_slug = Path(__file__).resolve().parent.name
    base_dir = Path(__file__).resolve().parent
    # 2. Step up to two levels
    repo_root = base_dir.parents[2]
except NameError:
    # Fallback to CWD
    repo_root = Path.cwd()

# Define the path to the data file
data_file_path = repo_root / "data" / "noda-rep-maps-pipeline" / "neuropixels" / "session_758798717.mat"

# Load the file
data = scipy.io.loadmat(data_file_path)

# Check out what is present in the file
for key, val in data.items():
    if not key.startswith('__'):
        print(key, np.array(val).shape)
# I can use this to create a mask over the neural activity data from Allen SDK
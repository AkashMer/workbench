# Undecided

## Data Source

- Following folders from [A dataset of visual- and memory-guided navigation combining fMRI/MEG recording and eye tracking for human cognition](https://doi.org/10.11922/sciencedb.01460) were used:
    - MRI_Scanning_sub23.zip
    - fMRI_behavior.zip
    - fMRI_eyedata.zip

### References

Zhang, B., & Naya, Y. (2022). A dataset of human fMRI/MEG experiments with eye tracking for spatial memory research using virtual reality. _Data in Brief_, _43_, 108380. [https://doi.org/10.1016/j.dib.2022.108380](https://doi.org/10.1016/j.dib.2022.108380)

## Setup Instructions

1. Clone this repo locally.
2. Recreate the environment using `mamba env create -f environment.yml` from this folder.
3. Download the following folders from https://doi.org/10.11922/sciencedb.01460,
    - MRI_Scanning_sub23.zip
    - fMRI_behavior.zip
    - fMRI_eyedata.zip
4. Paste those folders under `data/scene-areas-hierarchy-rsa` in this repo.
5. Run `scene-areas-hierarchy-rsa.qmd` to recreate the post.

*Note*: `explore.py` - Personal scratchpad for data exploration.
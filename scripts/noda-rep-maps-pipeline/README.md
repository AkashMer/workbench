# Estimation of Representational Maps in Mouse Visual Areas (Noda et al., 2024)

## Data Source

- Following folders under the data folder of [Repository containing scripts and processed data for Deitch et al., (2021)](https://github.com/zivlab/visual_drift.git) repo were used:
    - calcium_excitatory
    - neuropixels
- The data are a processed version of the publicly available neuronal data published in de Vries et al., 2020 and Siegle et al., 2021.

### References

1. Deitch et al., Representational drift in the mouse visual cortex, Current Biology (2021), https://doi.org/10.1016/j.cub.2021.07.062
2. de Vries et al., A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex, Nature neuroscience (2020), https://doi.org/10.1038/s41593-019-0550-9
3. Siegle et al., Survey of spiking in the mouse visual system reveals functional hierarchy, Nature (2021), https://doi.org/10.1038/s41586-020-03171-x

## Setup Instructions

1. Clone this repo locally.
2. Recreate the environment using `mamba env create -f environment.yml` from this folder.
3. Download the following folders from the repo https://github.com/zivlab/visual_drift.git,
    - data/calcium_excitatory
    - data/neuropixels
4. Paste those folders under `data/noda-rep-maps-pipeline` in this repo.
5. Run `noda-rep-maps-pipeline.qmd` to recreate the post.

*Note*: `explore.py` - Personal scratchpad for data exploration.

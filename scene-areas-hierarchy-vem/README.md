# Undecided

## Data Sources

- Preprocessed fMRI data (fMRIPrep/FreeSurfer outputs and ROI labels under `data/derivatives/`) borrowed from the output folders of [Multiple Regression RSA of Scene-Selective Areas in Human fMRI](https://github.com/AkashMer/workbench/tree/main/scene-areas-hierarchy-rsa).
- Behavioral data from [A dataset of human fMRI/MEG experiments with eye tracking for spatial memory research using virtual reality](https://doi.org/10.11922/sciencedb.01460)([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)):
    - `fMRI_behavior.zip`
- Following atlases were used to get the ROI labels:
    - Early visual cortex (V1/V2) atlas: [visfAtlas](https://download.brainvoyager.com/data/visfAtlas.zip)
    - Scene-selective area (PPA/OPA/MPA) parcels: [Place memory parcels and localizer](https://osf.io/xmhn7/)

### References

Zhang, B., & Naya, Y. (2022). A dataset of human fMRI/MEG experiments with eye tracking for spatial memory research using virtual reality. *Data in Brief*, *43*, 108380. https://doi.org/10.1016/j.dib.2022.108380

Rosenke, M., van Hoof, R., van den Hurk, J., Grill-Spector, K., & Goebel, R. (2021). A probabilistic functional atlas of human occipito-temporal visual cortex. *Cerebral Cortex*, *31*(1), 603–619. https://doi.org/10.1093/cercor/bhaa246

Steel, A. (2024). *Place memory parcels and localizer*. https://osf.io/xmhn7/

## Setup Instructions

### Notes

- `explore.py` — personal scratchpad for data exploration, not part of the reproducible pipeline.

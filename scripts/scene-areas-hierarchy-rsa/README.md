# Undecided

## Data Source

- Following folders from [A dataset of visual- and memory-guided navigation combining fMRI/MEG recording and eye tracking for human cognition](https://doi.org/10.11922/sciencedb.01460) were used:
    - MRI_Scanning_sub23.zip
    - fMRI_behavior.zip
    - fMRI_eyedata.zip
    - MRI_Scanning_sub8.zip
- Following video files from the supplementary data of [Medial Prefrontal Cortex Represents the Object-Based Cognitive Map When Remembering an Egocentric Target Location](https://doi.org/10.1093/cercor/bhaa117) were used:
    - trial_examples_Day_1_bhaa117.mp4
    - trial_examples_Day_2_bhaa117.mp4
- Following video files from the supplementary data of [Distinct networks coupled with parietal cortex for spatial representations inside and outside the visual field](https://doi.org/10.1016/j.neuroimage.2022.119041) were used:
    - 1-s2.0-S1053811922001707-mmc2.mp4
    - 1-s2.0-S1053811922001707-mmc3.mp4
    - 1-s2.0-S1053811922001707-mmc4.mp4
    - 1-s2.0-S1053811922001707-mmc5.mp4

### References

Zhang, B., & Naya, Y. (2022). A dataset of human fMRI/MEG experiments with eye tracking for spatial memory research using virtual reality. _Data in Brief_, _43_, 108380. [https://doi.org/10.1016/j.dib.2022.108380](https://doi.org/10.1016/j.dib.2022.108380)

Zhang, B., & Naya, Y. (2020). Medial prefrontal cortex represents the object-based cognitive map when remembering an egocentric target location. *Cerebral Cortex*, *30*(10), 5356–5371. https://doi.org/10.1093/cercor/bhaa117

Zhang, B., Wang, F., Zhang, Q., & Naya, Y. (2022). Distinct networks coupled with parietal cortex for spatial representations inside and outside the visual field. *NeuroImage*, *252*, 119041. https://doi.org/10.1016/j.neuroimage.2022.119041

## Setup Instructions

1. Clone this repo locally.
2. Recreate the environment using `mamba env create -f environment.yml` from this folder.
3. Download the following folders from https://doi.org/10.11922/sciencedb.01460,
    - MRI_Scanning_sub23.zip
    - fMRI_behavior.zip
    - fMRI_eyedata.zip
    - MRI_Scanning_sub8.zip
4. Download the following files from the supplementary section of https://doi.org/10.1093/cercor/bhaa117
    - trial_examples_Day_1_bhaa117.mp4
    - trial_examples_Day_2_bhaa117.mp4
5. Download the following files from the supplementary section of https://doi.org/10.1016/j.neuroimage.2022.119041
    - 1-s2.0-S1053811922001707-mmc2.mp4
    - 1-s2.0-S1053811922001707-mmc3.mp4
    - 1-s2.0-S1053811922001707-mmc4.mp4
    - 1-s2.0-S1053811922001707-mmc5.mp4
6. Paste those folders under `data/scene-areas-hierarchy-rsa` in this repo.
7. Run `scene-areas-hierarchy-rsa.qmd` to recreate the post.

*Note*: `explore.py` - Personal scratchpad for data exploration.
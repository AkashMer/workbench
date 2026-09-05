# Workbench

Storage space for the analysis scripts behind the posts in the Workbench section of my website (https://akashmer.github.io/Nole-Nexus), kept here for reproducibility.

## What's here

| Folder | Description |
|---|---|
| [`noda-rep-maps-pipeline`](noda-rep-maps-pipeline/) | [Estimation of Representational Maps in Mouse Visual Areas (Noda et al., 2024)](https://akashmer.github.io/Nole-Nexus/Workbench/noda-rep-maps-pipeline/noda-rep-maps-pipeline) |
| [`scene-areas-hierarchy-rsa`](scene-areas-hierarchy-rsa/) | [Multiple Regression RSA of Scene-Selective Areas in Human fMRI](https://akashmer.github.io/Nole-Nexus/Workbench/scene-areas-hierarchy-rsa/scene-areas-hierarchy-rsa) |
| [`heading-estimation-multisensory-proc`](heading-estimation-multisensory-proc/) | |

Each post folder contains:

| Subfolder | Description |
|---|---|
| **data** | Placeholder for the data used. Data itself is not stored here; see below. |
| **scripts** | The Quarto `.qmd` file and analysis code for that post, plus its own environment/setup files. |

## Licensing

- CODE in this repo is covered under the [MIT License](LICENSE).
- DATASETS under each post's `data/` folder are not included; they are linked per post and remain under their upstream licenses.
- POSTS rendered from these scripts are covered under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) on the site.

## Data

Data is NOT stored in this repo, as it is not mine to redistribute, and each post may draw on different sources. Every post's folder contains its own README documenting that post's data source, along with attribution and links. The matching website post carries the same attribution and links.

## Reproducing a post

Each post is self-contained under its own slug. Follow the instruction in the README file for each post to reproduce.
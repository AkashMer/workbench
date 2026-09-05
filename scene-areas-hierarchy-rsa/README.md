# Multiple Regression RSA of Scene-Selective Areas in Human fMRI

## Data Sources

- Raw fMRI/behavioral data from [A dataset of human fMRI/MEG experiments with eye tracking for spatial memory research using virtual reality](https://doi.org/10.11922/sciencedb.01460)([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)):
    - `MRI_Scanning_sub{8-26}.zip` (all 19 subjects; only participant 26 is preprocessed and used in the analysis)
    - `fMRI_behavior.zip`
- Early visual cortex (V1/V2) atlas: [visfAtlas](https://download.brainvoyager.com/data/visfAtlas.zip), downloaded automatically by the `download-atlases` cell.
- Scene-selective area (PPA/OPA/MPA) parcels: [Place memory parcels and localizer](https://osf.io/xmhn7/), downloaded automatically by the `download-atlases` cell via the OSF API.

### References

Zhang, B., & Naya, Y. (2022). A dataset of human fMRI/MEG experiments with eye tracking for spatial memory research using virtual reality. *Data in Brief*, *43*, 108380. https://doi.org/10.1016/j.dib.2022.108380

Rosenke, M., van Hoof, R., van den Hurk, J., Grill-Spector, K., & Goebel, R. (2021). A probabilistic functional atlas of human occipito-temporal visual cortex. *Cerebral Cortex*, *31*(1), 603–619. https://doi.org/10.1093/cercor/bhaa246

Steel, A. (2024). *Place memory parcels and localizer*. https://osf.io/xmhn7/

## Setup Instructions

### Option A: Docker (recommended, fully reproducible)

A [`Dockerfile`](https://github.com/AkashMer/workbench/blob/main/scene-areas-hierarchy-rsa/scripts/Dockerfile) is provided under `scripts/` which builds the same container used to render this post.

1. Clone this repo locally, then `cd` into `scene-areas-hierarchy-rsa/`.
2. Register for a free [FreeSurfer license](https://surfer.nmr.mgh.harvard.edu/registration.html) and place your `license.txt` under `data/`.
3. Download the raw fMRI data listed above into `data/raw_fmri/` using the provided [`scidb_manifest.txt`](https://github.com/AkashMer/workbench/blob/main/scene-areas-hierarchy-rsa/data/scidb_manifest.txt) and the `aria2c` command in the `download-data` code cell of the `.qmd`.
4. From `scripts/`, build the image:

```powershell
docker build -t scene-areas-hierarchy-rsa .
```

5. `cd` back out to the workbench repo root (the folder containing `scene-areas-hierarchy-rsa/`), then run the render:

```powershell
docker run --rm `
  -v "${PWD}\scene-areas-hierarchy-rsa:/workbench" `
  scene-areas-hierarchy-rsa
```

### Option B: Local environment

1. Clone this repo locally, then `cd` into `scene-areas-hierarchy-rsa/`.
2. Recreate the environment using `mamba env create -f environment.yml` from `scripts/`.
3. Follow steps 2-3 above for the FreeSurfer license and raw data.
4. Preprocessing (fMRIPrep, ROI mask construction) requires Docker regardless — see the `fmriprep-run` and `prepare-roi-masks` `eval: false` cells in the `.qmd` for the commands.
5. Register the `fmri` Jupyter kernel from the recreated environment, then uncomment the `jupyter: fmri` line in the `.qmd`'s YAML header:

```powershell
mamba activate scene-areas-hierarchy-rsa
python -m ipykernel install --user --name fmri --display-name "fmri"
```
6. From `scripts/`, run `quarto render scene-areas-hierarchy-rsa.qmd`.

### Notes

- `explore.py` — personal scratchpad for data exploration, not part of the reproducible pipeline.
- `select_top_vertices.py` — selects the top-N most probable, spatially contiguous vertices per ROI label.
- `scene-areas-hierarchy-rsa.bib` / `apa.csl` — bibliography and citation style

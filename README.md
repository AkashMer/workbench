# Workbench

Storage space for the analysis scripts behind the posts in the Workbench section of my website (https://akashmer.github.io/Nole-Nexus), kept here for reproducibility.

## What's here

| Folder | Description |
|---|---|
| **data** | Placeholder for the data used, organized by post. Data itself is not stored here; see below. |
| **scripts** | Jupyter Notebooks / R Markdown files containing the analysis code, organized by post. Mirrors the `data/` subfolder structure. |

## Licensing

- CODE in this repo is covered under the [MIT License](LICENSE).
- DATASETS under `data/` are not included; they are linked per post and remain under their upstream licenses.
- POSTS rendered from these scripts are covered under CC BY 4.0 on the site.

## Data

Data is NOT stored in this repo, as it is not mine to redistribute, and each post may draw on different sources. Every post's folder contains its own README documenting that post's data source, along with attribution and links. The matching website post carries the same attribution and links.

## Reproducing a post

Each post is self-contained under its own slug. To reproduce one:

1. Clone this repo locally.
2. Open that post's README at `scripts/<post-slug>/README.md`. It lists the data source(s), the exact files, and the setup for that post.
3. Install the dependencies from `scripts/<post-slug>/requirements.txt` (Python) or `renv.lock` (R).
4. Download the data into `data/<post-slug>/` as the README directs.
5. Run the notebook/script in `scripts/<post-slug>/`.
# Untitled

## Data Source

- Files in the data folder of [CombinationAndCompetitionHeadingDirection](https://github.com/sharootonian/CombinationAndCompetitionHeadingDirection) repo were used:
    - data.csv
    - dataconf.csv

### References

1. Harootonian, S. K., Ekstrom, A. D., & Wilson, R. C. (2022). Combination and competition between path integration and landmark navigation in the estimation of heading direction. *PLOS Computational Biology, 18(2)*, Article e1009222. https://doi.org/10.1371/journal.pcbi.1009222

## Setup Instructions

1. Clone this repo locally.
2. Make sure you have R version 4.4.3 installed.
3. From `heading-estimation-multisensory-proc/`, in an R console, run `renv::restore(project = "scripts")` to recreate the package library.
4. Run `quarto render scripts/heading-estimation-multisensory-proc.qmd` to recreate the post.

*Note*: `explore.R` - Personal scratchpad for data exploration.
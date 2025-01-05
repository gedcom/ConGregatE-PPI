# ConGregatE-PPI
## Context-specific aggregation of datasets for en masse protein-protein interaction prediction

Author: Tom Geddes

This repository contains an implementation of the ConGregatE-PPI deep learning method for PPI prediction. The repo contains the following items:
- ConGregate-PPI_code: Python implementation of the method
- comparison_code: R code for the generation of predictions and testing of alternative machine learning methods
- datasets: datasets used by the previous two items.
- ConGregatE-PPI.Rproj: R project for the repo (used exclusively by the comparison)
- dev.sh: Bash script for training and developing the project's artificial neural network models
- test.sh: Bash script for testing the project's artificial neural network models

To run any of the above, it is recommended to set the base directory as the working directory (or to open the RStudio project to run comparison code), otherwise the code may not be able to locate dataset files.

### Packages required

The following Python packages are required to run ConGregatE-PPI:
- `numpy`
- `keras`
- `os`
- `prettytable`
- `argparse`
- `csv`

`keras` was used in testing using a `tensorflow` backend.

The following R packages are required to perform the alternative method comparison scripts:
- `tidyverse`
- `caret`
- `MASS`
- `e1071`
- `class`
- `randomForest`
- `reshape2`
- `parallel`
- `impute`
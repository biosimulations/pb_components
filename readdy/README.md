# Multiscale-actin

A Vivarium wrapper for ReaDDy actin models.

## Installation

Install conda: https://docs.conda.io/en/latest/miniconda.html
(`conda` is currently required to install `readdy`.)

1. Create a virtual environment with conda-specific dependencies: `conda env create -f environment.yml`
2. Activate the environment: `conda activate multiscale-actin`
3. Install with pip dependencies: `pip install multiscale_actin` (or install in editable mode: `pip install -e .`)


### Installation for Mac's with ARM Chips

1. Create a virtual environment with conda-specific dependencies: `conda env create --platform osx-64 -f mac_env.yml`
2. Activate the environment: `conda activate multiscale-actin`
3. Run example: `python3 multiscale_actin/processes/create_readdy_pbif.py`
4. Visualize result with: https://simularium.allencell.org/viewer


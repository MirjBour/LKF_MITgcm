# LKF Detection with MITgcm

This repository contains the necessary scripts to extract Linear Kinematic Features (LKF) statistics from MITgcm outputs. These scripts are modifications of the LKF detection functions found in Nils Hutter's [lkf_tools repository](https://github.com/nhutter/lkf_tools).

## Overview

The primary script in this repository is `detect_dataset_MITgcm.py`, which utilizes several utility functions defined in the accompanying Python files. The scripts have been adapted to work specifically with MITgcm output data to detect and analyze LKFs.

## Repository Structure

- `detect_dataset_MITgcm.py`: Main script for detecting LKFs in MITgcm data.
- `dataset.py`: Contains dataset handling functions.
- `detection.py`: Core detection algorithms.
- `lkf_stats_tools.py`: Tools for calculating LKF statistics.
- `model_utils.py`: Utility functions for MITgcm data processing.
- `read_RGPS.py` and `rgps.py`: Scripts for reading RGPS data.
- `rw.py`: Read/write utilities.
- `tracking.py`: Functions for tracking LKFs over time.
- `_dir_filter.py`: Directory filtering utilities.

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/evalmr/LKF_MITgcm.git
    ```
2. Ensure you have the required dependencies installed.
3. Run the `detect_dataset_MITgcm.py` script with your MITgcm output data.

## Dependencies

- Python 3.x
- Required Python packages (e.g., NumPy, SciPy, etc.)

## Acknowledgments

This work is based on the LKF detection methods developed by Nils Hutter. Visit the [lkf_tools repository](https://github.com/nhutter/lkf_tools) and  [lkf_mitgcm2km repository](https://github.com/nhutter/lkf_mitgcm2km/tree/main) for more information.

The main MITgcm webpage can be found [here](https://mitgcm.readthedocs.io/en/latest/).
## Contact

For any questions or issues, please open an issue on this repository or contact [Eva Lemaire](eva.lemaire11@gmail.com).

# Simulation Data Directory

This directory contains the output data from our MITgcm simulations. Below is an overview of the structure and contents of this directory.

## Directory Structure

- `data/`: Contains all the raw output data from the simulations.
- `scripts/`: Includes scripts used for data processing and analysis.
- `results/`: Processed results and visualizations generated from the simulation data.

## Data Description

### Raw Data (`data/`)
- **Format**: NetCDF files
- **Naming Convention**: `output_YYYYMMDD.nc` where `YYYYMMDD` indicates the date of the simulation.

### Scripts (`scripts/`)
- **preprocess.py**: Script to preprocess the raw data.
- **analyze.py**: Script to perform various analyses on the preprocessed data.
- **visualize.py**: Script to generate plots and visualizations.

### Results (`results/`)
- **Statistics**: Contains statistical summaries and metrics derived from the simulations.
- **Plots**: Includes various plots and figures generated from the analysis scripts.

## Usage

1. **Preprocessing Data**
    ```bash
    python scripts/preprocess.py data/output_YYYYMMDD.nc
    ```

2. **Running Analysis**
    ```bash
    python scripts/analyze.py data/preprocessed_YYYYMMDD.nc
    ```

3. **Generating Visualizations**
    ```bash
    python scripts/visualize.py results/analysis_YYYYMMDD.json
    ```

## Dependencies

- Python 3.x
- Required Python packages: NumPy, SciPy, Matplotlib, NetCDF4

## Contact

For any questions or issues, please contact [Your Name] at [Your Email].



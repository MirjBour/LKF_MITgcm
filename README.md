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

## Contact

For any questions or issues, please open an issue on this repository or contact eva.lemaire11@gmail.com.


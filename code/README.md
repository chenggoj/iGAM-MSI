# Surface Analysis and iGAM Modeling Tools

This repository contains a set of Python scripts for analyzing surface properties of materials and building interpretable machine learning models for predicting metal-support interactions.

## Scripts

### 1. auto_surface_features.py

Automatically extracts various surface features from VASP output files.

Features:
- Surface energy calculation
- Work function estimation
- Surface roughness parameters
- Surface charge and dipole moment analysis
- Bond order calculations using DDEC6 method

Usage:

python3 auto_surface_features.py


### 2. surface_roughness.py

Calculates surface roughness parameters from a CONTCAR file.

Features:
- Identification of surface atoms
- Calculation of Ra, Rq, and Rmax roughness parameters
- Exports surface atom information to CSV

Usage:

python3 surface_roughness.py

## Requirements

- Python 3.7+
- ASE (Atomic Simulation Environment)
- NumPy
- Pandas
- Scikit-learn
- Interpret (for iGAM)
- Matplotlib

## Usage

1. Ensure your VASP output files are in the correct directory structure.
2. Run `auto_surface_features.py` to extract surface features.
3. Use `iGAM.py` to train and evaluate the iGAM model.
4. For specific surface roughness analysis, use `surface_roughness.py`.

# iGAM-MSI

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

iGAM-MSI: Shed light on Metal-Support Interactions through Interpretable Machine Learning


## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [NN-MD-database](#nn-md-database)
- [References](#references)
- [Citation](#citation)

## Introduction

In the realm of materials science, understanding is light. iGAM-MSI illuminates the complex world of Metal-Support Interactions (MSI) using Interpretable Generalized Additive Models (iGAM).

iGAM-MSI is an open-source project that leverages the power of iGAM to provide accurate and explainable predictions in materials science. With this package, you can train interpretable glassbox models and explain the intricacies of MSI systems. iGAM-MSI helps you understand your model's global behavior, or unravel the reasons behind individual predictions.

### Why Interpretability Matters in MSI Research

Interpretability in MSI research is essential for:

- **Model Debugging**: Understand why your model made specific predictions about metal-support interactions
- **Feature Engineering**: Identify ways to improve your model for better MSI phenomena capture
- **Material Design**: Leverage insights to design superior catalysts and supported metal systems
- **Scientific Discovery**: Uncover new insights about MSI through interpretable models

## Features

- Well-established iGAM models:
  - 12-features iGAM
  - 6-features iGAM
- Model training workflows
- Automated feature extraction workflow code


## Prerequisites

- Python 3.7+
- Linux
- Jupyter Notebook (for running IPython magic commands)

## Dependencies

This project requires the following main Python libraries:

- NumPy
- SciPy
- pandas
- ASE (Atomic Simulation Environment)
- scikit-learn
- scikit-optimize
- interpret-community
- matplotlib
- tqdm
- minepy
- statsmodels
- alive-progress

Note: Some libraries like `interpret-community` might have additional system dependencies. Please refer to their respective documentation for complete installation instructions.

## Usage



## NN-MD-database

The `NN-MD-database` directory contains all the original data and details from Neural Network Molecular Dynamics (NN-MD) simulations used to train iGAM-MSI models. This comprehensive database includes:

- Initial structures of metal nanoparticles on various supports
- Time-series data of important physical properties
- Visualization files for key interactions and phenomena
- Detailed simulation parameters and conditions

### Directory Structure

NN-MD-database/
├── {support_material}{miller_index}{OC22_trajectory_id}/
│   ├── initial_NNMD.pdb
│   ├── MD_Pt_contact_angle_adhesion_energy.csv
│   ├── MD_contact_angle_Normalized_MSI_descriptor.pdf
│   └── MD_Pt_Eadh_ChemicalPotential.pdf
├── Another_System/
│   └── ...
└── ...

Each subdirectory represents a specific metal-support system, named according to the convention: `{support_material}_{miller_index}_{OC22_trajectory_id}`. Each directory contains:

- `initial_NNMD.pdb`: Initial structure file of the Pt NP/support system for NN-MD simulations
- `MD_Pt_contact_angle_adhesion_energy.csv`: Time-series data including contact angle, adhesion energy, normalized MSI descriptor, and chemical potential of Pt
- `MD_contact_angle_Normalized_MSI_descriptor.pdf`: Visualization of contact angle and normalized MSI descriptor evolution
- `MD_Pt_Eadh_ChemicalPotential.pdf`: Visualization of Pt adhesion energy and chemical potential trends

This database serves as a valuable resource for researchers looking to:
- Validate computational models
- Explore trends in metal-support interactions
- Develop new descriptors for MSI phenomena

For more details on the database contents and usage, please refer to the [NN-MD-database README](NN-MD-database/README.md).


## References

For a detailed overview of iGAMs, please refer to the [original EBM repository](https://github.com/interpretml/interpret/).

## Citation

If you use this code, models, or the NN-MD-database in your research, please cite:

```bibtex


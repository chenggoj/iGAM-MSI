

# iGAM-MSI

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.16878887-blue)](https://doi.org/10.5281/zenodo.16878887)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41929--025--01417--3-blue)](https://doi.org/10.1038/s41929-025-01417-3)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

iGAM-MSI: Shed light on Metal-Support Interactions through Interpretable Machine Learning (Glass-box model)

<img src="./images/Overview.webp" width="600" alt="iGAM-MSI Overview">

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Dependencies](#dependencies)
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
- Automated feature extraction workflow code


## Prerequisites

- Python 3.7+
  
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

## References

For a detailed overview of iGAMs, please refer to the [original EBM repository](https://github.com/interpretml/interpret/).


## Citation

If you use this code, models, or the NN-MD-database in your research, please cite:

> Jiang C, Yan B, Goldsmith B, Linic S. *Predictive model for the discovery of sinter-resistant supports for metallic nanoparticle catalysts by interpretable machine learning*. Nat Catal (2025).

[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41929--025--01417--3-blue)](https://doi.org/10.1038/s41929-025-01417-3)

# iGAM-MSI
iGAM-MSI is a repository containing code and trained machine learning models for studying Metal-Support Interactions (MSI) using Interpretable Generalized Additive Models (iGAM). This project leverages the power of iGAM to provide accurate and explainable predictions in materials science. The publication DOI associated with the codes is: 

iGAM-MSI
Introduction
In the realm of materials science, understanding is light. iGAM-MSI illuminates the complex world of Metal-Support Interactions.
iGAM-MSI is an open-source code that leverages Interpretable Generalized Additive Models (iGAM) to study Metal-Support Interactions (MSI). With this package, you can train interpretable glassbox models and explain the intricacies of MSI systems. iGAM-MSI helps you understand your model's global behavior, or unravel the reasons behind individual predictions.
Interpretability in MSI research is essential for:

Model debugging - Why did my model make this prediction about a specific metal-support interaction?
Feature Engineering - How can I improve my model to better capture MSI phenomena?
Material Design - Can we use these insights to design better catalysts or supported metal systems?
Scientific Discovery - What new insights about MSI can we glean from these interpretable models?

For a detailed overview of iGAMs, please refer to the original EBM link (https://github.com/interpretml/interpret/)

Introducing iGAM for MSI
iGAM (Interpretable Generalized Additive Models) breathes new life into the study of Metal-Support Interactions. By combining modern machine learning techniques with traditional GAMs, we create models that are both highly accurate and fully interpretable.
Repository Contents
This repository includes:

Well-established iGAM models:

12-features iGAM
6-features iGAM


Model training workflows
Automated feature extraction workflow code

Dependencies
This project requires the following main Python libraries:

NumPy
SciPy
pandas
ASE (Atomic Simulation Environment)
scikit-learn
scikit-optimize
interpret-community
matplotlib
tqdm
minepy
statsmodels
alive-progress

To install all dependencies:
pip install -r requirements.txt

Python 3.7+ | Linux
Jupyter Notebook (for running IPython magic commands)
Note: Some libraries like interpret-community might have additional system dependencies. Please refer to their respective documentation for complete installation instructions.

Citation
If you use this code or models in your research, please cite:
[Include relevant citations]

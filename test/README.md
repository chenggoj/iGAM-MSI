# High-Throughput Screening for Metal-Support Interactions

This repository contains a Python script for high-throughput screening of metal-support interactions using a pre-trained interpretable machine learning model (iGAM).

## Overview

The script predicts the adhesion energy (E_adh) and contact angle for metal nanoparticles on various support materials based on 12 input features. It uses a pre-trained Explainable Boosting Machine (EBM) model for predictions.

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib

You can install the required packages using:

pip install pandas numpy scikit-learn joblib

## Usage

1. Prepare your input data:
   - Create a CSV file named `inputs.csv` with the following 12 features:
     E_surf, WF, Ra, NC_postive, NC_negative, Dipole_Z, rho_O, rho_M, M_SBO, O_SBO, Ef, Ehull

2. Ensure you have the pre-trained model file `ebm_model.joblib` in the same directory as the script.

3. Run the script:


4. The script will generate an `output.csv` file containing the original input data, predicted E_adh, and calculated contact angle.

## Input File Format

Your `inputs.csv` should have the following columns:
E_surf,WF,Ra,NC_postive,NC_negative,Dipole_Z,rho_O,rho_M,M_SBO,O_SBO,Ef,Ehull

Each row represents a different material or configuration.

## Output

The script will generate an `output.csv` file with the following additional columns:

- `Predicted_E_adh`: The predicted adhesion energy (J/m²)
- `Predicted_Contact_Angle`: The calculated contact angle based on the predicted E_adh

## Notes

- The contact angle calculation uses a default surface tension for platinum (γPt = 1.60218 J/m²). If you're using a different metal, you may need to modify this value in the `calculate_contact_angle` function.
- Ensure that your input data is properly normalized and scaled to match the training data used for the pre-trained model.

## Acknowledgements

This work is based on the iGAM-MSI project. For more details on the model and its development, please refer to the original paper.


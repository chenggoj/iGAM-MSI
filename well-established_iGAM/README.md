# Well-established iGAM Models for Metal-Support Interactions

This directory contains well-established Interpretable Generalized Additive Models (iGAM) for predicting metal-support interactions, along with scripts for high-throughput screening.

## Contents

1. `6-features_iGAM.joblib` (6.7MB)
   - A trained iGAM model using 6 key features for predicting metal-support interactions.

2. `12-features_iGAM.joblib` (66KB)
   - A more comprehensive iGAM model using 12 features for potentially more accurate predictions.

3. `predicted_from_6-features_iGAM.py` (2.2KB)
   - Python script for high-throughput screening using the 6-feature iGAM model.

4. `predicted_from_12-features_iGAM.py` (2.3KB)
   - Python script for high-throughput screening using the 12-feature iGAM model.

## Model Features

### 6-Feature Model
Features used: E_surf., Ra, rho_O, rho_M, Ef, Ehull

### 12-Feature Model
Features used: E_surf., WF, Ra, NC_postive, NC_negative, Dipole_Z, rho_O, rho_M, M_SBO, O_SBO, Ef, Ehull

## Usage

Both scripts follow a similar workflow:

1. Load and preprocess input data from a CSV file.
2. Load the respective trained iGAM model.
3. Predict adhesion energy (E_adh) using the model.
4. Calculate contact angle using the Young-Dupré equation.
5. Save results, including original data, predicted E_adh, and calculated contact angle, to a CSV file.

### Running the Scripts

For 6-feature model:
python3 predicted_from_6-features_iGAM.py

For 12-feature model:
python3 predicted_from_12-features_iGAM.py

Ensure that input CSV files (`inputs_6-features-iGAM.csv` or `inputs_12-features_iGAM.csv`) are present in the same directory before running the respective scripts.

## Output

The scripts will generate output CSV files (`output_6-features-iGAM.csv` or `outputs_12-features_iGAM.csv`) containing the original input data along with predicted adhesion energies and calculated contact angles.

## Note

The 12-feature model may provide more comprehensive predictions but requires more input data. The 6-feature model can be used when certain data is unavailable or for quicker screening processes.


import os
import pandas as pd
import numpy as np
from joblib import load
import math
def preprocess_data(filepath):
    """
    Load and preprocess the input dataset.
    """
    if os.path.exists(filepath):
        data = pd.read_csv(filepath)
        feature_names = ["E_surf.", "WF", "Ra", "NC_postive", "NC_negative", "Dipole_Z", "rho_O", "rho_M", "M_SBO", "O_SBO", "Ef",
"Ehull"] # all 12-feature
        X = data[feature_names]
    else:
        raise FileNotFoundError("Input dataset not found!")

    return X, data
def load_model(model_path):
    """
    Load the trained model from a file using joblib.
    """
    model = load(model_path)
    return model
def predict_eadh(model, X):
    """
    Predict E_adh using the loaded model.
    """
    return model.predict(X)
def calculate_contact_angle(eadh, gamma_pt=1.60218):
    """
    Calculate contact angle using Young-Dupré equation.
    E_adh = -γPt(cosθ + 1)
    """
    cos_theta = -eadh / gamma_pt - 1
    # Ensure cos_theta is within [-1, 1] to avoid math domain error
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta) * 180 / math.pi  # Convert to degrees
    return theta
def save_results_to_csv(input_data, eadh_pred, contact_angle, output_path):
    """
    Save the predicted E_adh, calculated contact angle, and original data to a CSV file.
    """
    output_data = input_data.copy()
    output_data['Predicted_E_adh'] = eadh_pred
    output_data['Predicted_Contact_Angle'] = contact_angle
    output_data.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
def main():
    input_data_path = "inputs.csv"
    model_path = "ebm_model.joblib"
    output_path = "output.csv"

    # Preprocess the input dataset
    X, input_data = preprocess_data(input_data_path)

    # Load the trained model
    model = load_model(model_path)

    # Predict E_adh
    eadh_pred = predict_eadh(model, X)

    # Calculate contact angle
    contact_angle = calculate_contact_angle(eadh_pred)

    # Save the results
    save_results_to_csv(input_data, eadh_pred, contact_angle, output_path)

    print("High-throughput screening completed successfully.")
if name == "main":
    main()

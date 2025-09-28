## Example

The `Example/` directory contains a complete demonstration of the 12-feature iGAM model for high-throughput predictions for Pt NP adhesion energy (J/m2) and Contact angle (°) on specific catalyst supports.

### Contents
Example/
- 12-features_iGAM.joblib           # Trained iGAM model file
- inputs_12-features_iGAM.csv       # Input features dataset
- outputs_12-features_iGAM.csv      # Prediction results
- predicted_from_12-features_iGAM.py # Execution script

### Case Study: BaO Surface Analysis

This example demonstrates the prediction of Pt NP adhesion performance on **6 different BaO surfaces**:
- BaO-mp-1342_(100)
- BaO-mp-1342_(110)
- *and other facets*

Surface inputs features

- **E_surf.** - Surface energy (J·m⁻²)
- **WF** - Work function (eV)
- **Ra** - Average roughness of the surface (Å)
- **Dipole_Z** - Surface average dipole moment in z direction (a.u.)

Atomic Densities

- **rho_O** - Surface oxygen density (atoms·Å⁻²)
- **rho_M** - Surface metal density (atoms·Å⁻²)

Bond Order Parameters

- **M_SBO** - Sum bond order of surface metal (dimensionless)
- **O_SBO** - Sum bond order of surface oxygen (dimensionless)

Electronic Properties

- **NC_postive** - Net atomic charge of surface metal (|e⁻|)
- **NC_negative** - Net atomic charge of surface oxygen (|e⁻|)

Bulk Stability Properties

- **Ef** - Formation energy per atom in bulk (eV·atom⁻¹)
- **Ehull** - Energy above convex hull in bulk (meV·atom⁻¹)

These 12 features comprehensively describe the surface geometric structure, electronic properties, chemical bonding characteristics, and thermodynamic stability of catalyst support materials, providing a complete set of material descriptors for predicting adhesion energy between Pt nanoparticles and supports.

### Output Predictions
The model generates:
- **Predicted_E_adh** (J/m²): Adhesion energy between metal nanoparticles and support
- **Predicted_Contact_Angle** (degrees): Calculated using Young-Dupré equation

### Running the Example
```bash
cd Example/
python3 predicted_from_12-features_iGAM.py

# NN-MD-database

This directory contains the Neural Network Molecular Dynamics (NN-MD) simulation data used for training and validating the iGAM-MSI models.

## Directory Structure

Each subdirectory represents a specific metal-support system, following the naming convention:

`{support_material}_{miller_index}_{OC22_trajectory_id}/`

For example: `TiO2_110_OC22-1234/`

## File Contents

Each system directory contains the following files:

1. `initial_NNMD.pdb`
   - Initial structure file of the Pt nanoparticle (NP) on the support surface
   - Format: PDB (Protein Data Bank)

2. `MD_Pt_contact_angle_adhesion_energy.csv`
   - Time-series data from the NN-MD simulation
   - Columns include:
     - Time step
     - Contact angle
     - Adhesion energy
     - Normalized MSI descriptor
     - Chemical potential of Pt

3. `MD_contact_angle_Normalized_MSI_descriptor.pdf`
   - Visualization of the evolution of contact angle and normalized MSI descriptor over time

4. `MD_Pt_Eadh_ChemicalPotential.pdf`
   - Visualization of Pt adhesion energy and chemical potential trends

## Usage

This database serves as a valuable resource for researchers in the field of metal-support interactions. It can be used to:

1. Validate computational models
2. Explore trends in metal-support interactions across different materials and surface orientations
3. Develop new descriptors for MSI phenomena
4. Train and test machine learning models for predicting MSI properties

## Data Processing

To process this data for use with iGAM-MSI models:

1. Extract relevant features from the CSV files
2. Normalize the data as required by your specific model
3. Split the data into training and testing sets

For detailed instructions on data processing and model training, please refer to the main README of the iGAM-MSI repository.

## Contributing

If you have additional NN-MD simulation data that you believe would be valuable to include in this database, please contact the repository maintainers or submit a pull request following the contribution guidelines in the main repository.

## Citation

If you use this data in your research, please cite:

```bibtex
[Include your citation here]

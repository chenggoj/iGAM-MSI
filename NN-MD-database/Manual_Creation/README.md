# Manual Cases: NN-MD Simulations of Metal-Support Interactions

## Overview

This directory contains comprehensive data and analysis from Neural Network Molecular Dynamics (NN-MD) simulations of Metal-Support Interaction (MSI) cases. The focus is on 3nm Pt nanoparticles (NPs) supported on various manually selected substrates.

## Directory Structure

The cases are organized in the following structure:

{support_material}/{miller_index}/LASP-NNMD

Where:
- support_material: Name and Materials Project ID of the support material (e.g., TiO2_mp-390)
- miller_index: Miller index of the support surface (e.g., 100)
- LASP-NNMD: Directory containing the NN-MD simulation results

## Contents

For each case, the following files are provided:

1. `initial_NNMD.pdb`: Initial structure of the Pt NP/support system for NN-MD simulations.

2. `MD_Pt_contact_angle_adhesion_energy.csv`: Time-series data including:
   - Contact angle
   - Adhesion energy
   - Normalized MSI descriptor
   - Chemical potential of Pt

3. `MD_contact_angle_Normalized_MSI_descriptor.pdf`: Visualization of contact angle and normalized MSI descriptor evolution.

4. `MD_Pt_Eadh_ChemicalPotential.pdf`: Visualization of Pt adhesion energy and chemical potential trends.

## Reproducibility

Due to file size limitations, the complete NN-MD trajectory files are not included. However, all results can be readily reproduced by:

1. Using the provided `initial_NNMD.pdb` as the starting structure.
2. Applying identical simulation conditions and software settings.

This approach ensures reproducibility while maintaining manageable file sizes.

## Significance

These manually selected cases provide valuable insights into the dynamics of metal-support interactions for specific systems of interest, crucial for understanding catalytic processes and materials design in nanoscale systems.

## Usage

Researchers can use this data to:
- Analyze metal-support interaction trends for specific materials
- Validate computational models
- Inform the design of supported metal catalysts for targeted applications

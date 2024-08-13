#! /home/cjiang1/miniconda3/envs/python3.10/bin/python3
"""
Author: Chenggong Jiang
Affiliation: University of Michigan
Email: chenggoj@umich.edu

This script calculates surface roughness parameters from a CONTCAR file.

Usage:
1. Make sure you have the 'ase' package installed (you can install it using 'pip install ase').
2. Place the CONTCAR file in the same directory as this script.
3. Run the script using 'python surface_roughness.py'.
4. The script will read the CONTCAR file, calculate the surface roughness parameters, and display the results.

Notes:
- The script assumes that the surface atoms are determined based on a height threshold (default is 0.9).
- You can adjust the 'surface_threshold' parameter in the 'main()' function if needed.
- The script calculates three surface roughness parameters: Ra, Rq, and Rmax.
- If the CONTCAR file is not found or an error occurs during processing, an error message will be displayed.
"""

import os
import numpy as np
from ase.io import read
import csv

def read_contcar_ase(file_path, surface_threshold=0.90):
    # Read the CONTCAR file using ASE
    atoms = read(file_path, format='vasp')
    
    # Get the lattice constant
    lattice_constant = atoms.cell.cellpar()[0]
    
    # Get the lattice vectors
    lattice_vectors = atoms.cell
    
    # Get the number and types of atoms
    atom_types = atoms.get_chemical_symbols()
    atom_numbers = atoms.get_atomic_numbers()
    
    # Get the atomic coordinates
    coordinates = atoms.get_positions()
    
    # Find the maximum z-coordinate
    max_z = np.max(coordinates[:, 2])
    
    # Determine the indices of surface atoms
    surface_indices = np.where(coordinates[:, 2] >= surface_threshold * max_z)[0]
    
    # Get the element types of surface atoms
    surface_elements = [atom_types[i] for i in surface_indices]
    
    print("Surface atom indices and elements:")
    for idx, elem in zip(surface_indices, surface_elements):
        print(f"{idx} {elem}")
    
    # Get the coordinates of surface atoms
    surface_coordinates = coordinates[surface_indices]
    
    return lattice_constant, lattice_vectors, atom_types, atom_numbers, surface_coordinates, surface_indices, surface_elements

def calculate_roughness(coordinates, surface_indices):
    # Calculate the average height
    avg_height = np.mean(coordinates[:, 2])
    
    # Calculate the absolute height deviations for each atom
    height_deviations = np.abs(coordinates[:, 2] - avg_height)
    
    # Calculate the Ra value
    ra = np.mean(height_deviations)
    
    # Calculate the Rq value
    rq = np.sqrt(np.mean(height_deviations**2))
    
    # Calculate the Rmax value (Maximum Roughness Depth)
    max_height_index = np.argmax(coordinates[:, 2])
    min_height_index = np.argmin(coordinates[:, 2])
    rmax = coordinates[max_height_index, 2] - coordinates[min_height_index, 2]
    
    # Get the corresponding atom indices for Rmax
    max_atom_index = surface_indices[max_height_index]
    min_atom_index = surface_indices[min_height_index]
    
    return ra, rq, rmax, max_atom_index, min_atom_index

def main():
    # Get the current working directory
    current_dir = os.getcwd()
    
    # Specify the CONTCAR file name
    contcar_file = 'CONTCAR'
    
    # Construct the full path to the CONTCAR file
    contcar_path = os.path.join(current_dir, contcar_file)
    
    # Check if the CONTCAR file exists
    if not os.path.exists(contcar_path):
        print(f"CONTCAR file not found in the current directory: {current_dir}")
        return
    
    # Specify the surface threshold (optional, default is 0.9)
    surface_threshold = 0.90
    
    try:
        # Read the CONTCAR file
        lattice_constant, lattice_vectors, atom_types, atom_numbers, surface_coordinates, surface_indices, surface_elements = read_contcar_ase(contcar_path, surface_threshold)
        
        # Save surface atom indices and elements to CSV file
        csv_file = 'surface_atoms_index.csv'
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"# Surface atoms identified using threshold: {surface_threshold}"])
            writer.writerow(["Index", "Element"])
            for idx, elem in zip(surface_indices, surface_elements):
                writer.writerow([idx, elem])
        
        print(f"Surface atom indices and elements saved to {csv_file}")

        # Calculate surface roughness parameters
        ra_value, rq_value, rmax_value, max_atom_index, min_atom_index = calculate_roughness(surface_coordinates, surface_indices)
        
        print(f"Arithmetic average roughness (Ra) = {ra_value:.3f} Å")
        print(f"Root mean square roughness (Rq) = {rq_value:.3f} Å")
        print(f"Maximal height difference (Rmax) = {rmax_value:.3f} Å between atoms {max_atom_index} and {min_atom_index}")
    except Exception as e:
        print(f"Error occurred while processing the CONTCAR file: {str(e)}")

if __name__ == '__main__':
    main()

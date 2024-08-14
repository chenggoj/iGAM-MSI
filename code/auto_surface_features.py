#! /home/cjiang1/miniconda3/envs/python3.10/bin/python3
import os
import subprocess
import pandas as pd
from ase.io import read
import numpy as np
from scipy.integrate import simps
import glob
from alive_progress import alive_bar

def extract_energy(directory):
    oszicar_path = os.path.join(directory, 'OSZICAR')
    with open(oszicar_path, 'r') as file:
        lines = file.readlines()
    energy_line = next((line for line in reversed(lines) if 'E0' in line), None)
    if energy_line:
        energy = float(energy_line.split()[4])
        return energy
    else:
        raise ValueError(f"No energy found in OSZICAR file: {oszicar_path}")


def calculate_surface_energy(E_perfect_slab, E_defected_slab, E_bulk, CONTCAR_slab, CONTCAR_bulk, E_defect, N_surface):
    """
    Calculate the surface energy of a slab with or without a defect.

    Parameters:
    E_perfect_slab: The total energy of the perfect slab (in eV).
    E_defected_slab: The total energy of the defected slab (in eV), default is None for perfect surface.
    E_bulk: The energy of an atom in the bulk (in eV).
    CONTCAR_slab: The path to the CONTCAR file of the slab.
    CONTCAR_bulk: The path to the CONTCAR file of the bulk.
    E_defect: The energy of the defect (in eV), default is 0 for perfect surface.

    Returns:
    E_surf: The surface energy (in eV/Å^2).
    E_surf_J_per_m2: The surface energy (in J/m^2).
    """
    # Check if any of the energies are not converged
    if isinstance(E_perfect_slab, str) or isinstance(E_bulk, str) or (
            E_defected_slab is not None and isinstance(E_defected_slab, str)):
        return 'Not converged'
    else:
        E_O2 = -9.854  # based on PBE+D3 functional

        # Read the CONTCAR file
        bulk = read(CONTCAR_bulk, format='vasp')
        slab = read(CONTCAR_slab, format='vasp')

        # Extract the lattice parameters
        cell_slab = slab.get_cell()
        print("<=====================================START TO CALCULATE SURFACE ENERGY===========================================>")	
        print(f"E_perfect_slab is {E_perfect_slab} eV")
        print(f"E_bulk is {E_bulk} eV")
        print(f"E_defected_slab is {E_defected_slab} eV")

        # Get types of atoms
        atom_types = set(bulk.get_chemical_symbols())
        print(f"atom_types is {atom_types}")
        metal_element = list(atom_types - {'O'})[0]

        # Calculate the number of each type of atom in the slab and the bulk
        bulk_counts = {atom: bulk.get_chemical_symbols().count(atom) for atom in atom_types}
        slab_counts = {atom: slab.get_chemical_symbols().count(atom) for atom in atom_types}

        # Calculate n value, assuming each type of atom has the same n value
        n_values = {atom: slab_counts[atom] / bulk_counts[atom] for atom in atom_types}
        N = n_values[metal_element]
        print(f"metal_element is {metal_element}")
        print(f"N_bulk is {N}")
        print(f"N_surface is {N_surface}")

        # Calculate the area of the slab surface
        # Here we assume the slab surface is along the xy plane
        A = abs(cell_slab[0][0] * cell_slab[1][1] - cell_slab[0][1] * cell_slab[1][0])
        print(f"Surface area is {A} Å^2")


        E_surf = (N_surface * E_perfect_slab - N * E_bulk) / (2 * A) + E_defect / A

        # 1 eV = 1.602176634e-19 J
        # 1 A = 1e-10 m
        E_surf_J_per_m2 = E_surf * (1.602176634e-19 / (1e-20))

        print(f"E_defect is {E_defect} eV")
        print(f"Surface energy is {E_surf} eV/¾E^2 or {E_surf_J_per_m2} J/m^2")
        print("<=====================================STOP TO CALCULATE SURFACE ENERGY===========================================>")
        return E_surf_J_per_m2, A

def calculate_work_function(directory):
    try:
        cmd = f"echo -e '426\\n3\\n' | vaspkit | grep 'Work Function'"
        result = subprocess.run(cmd, shell=True, check=True, cwd=directory, executable="/bin/bash", capture_output=True, text=True)
        output = result.stdout.strip()
        if output:
            work_function = float(output.split(':')[1].strip())
            return work_function
        else:
            return None
    except subprocess.CalledProcessError:
        return None

def calculate_surface_roughness(directory):
    os.chdir(directory)
    output = subprocess.check_output("surface_roughness.py", shell=True).decode('utf-8')
    os.chdir('../..')
    
    ra_line = next(line for line in output.split('\n') if line.startswith('Arithmetic average roughness'))
    rq_line = next(line for line in output.split('\n') if line.startswith('Root mean square roughness'))
    rmax_line = next(line for line in output.split('\n') if line.startswith('Maximal height difference'))
    
    ra = float(ra_line.split('=')[1].strip().split(' ')[0])
    rq = float(rq_line.split('=')[1].strip().split(' ')[0])
    rmax = float(rmax_line.split('=')[1].strip().split(' ')[0])
    
    return ra, rq, rmax


def calculate_surface_charge_dipole(directory):
    os.chdir(directory)
    output = subprocess.check_output("surface_charge_dipole.py", shell=True).decode('utf-8')
    os.chdir('../..')

    avg_positive_charge_line = next(
        (line for line in output.split('\n') if line.startswith('Average positive charge of surface atoms:')), None)
    avg_negative_charge_line = next(
        (line for line in output.split('\n') if line.startswith('Average negative charge of surface atoms:')), None)
    avg_surface_dipole_line = next(
        (line for line in output.split('\n') if line.startswith('Average surface dipole moment:')), None)

    avg_positive_charge = float(
        avg_positive_charge_line.split(':')[1].strip().split(' ')[0]) if avg_positive_charge_line else None
    avg_negative_charge = float(
        avg_negative_charge_line.split(':')[1].strip().split(' ')[0]) if avg_negative_charge_line else None

    if avg_surface_dipole_line:
        dipole_values = avg_surface_dipole_line.split('[')[1].split(']')[0].split()
        if len(dipole_values) == 3:
            avg_surface_dipole_x = float(dipole_values[0])
            avg_surface_dipole_y = float(dipole_values[1])
            avg_surface_dipole_z = float(dipole_values[2])
        else:
            print("No dipole_values")
            avg_surface_dipole_x = None
            avg_surface_dipole_y = None
            avg_surface_dipole_z = None
    else:
        print("No avg_surface_dipole_line")
        avg_surface_dipole_x = None
        avg_surface_dipole_y = None
        avg_surface_dipole_z = None

    return avg_positive_charge, avg_negative_charge, avg_surface_dipole_x, avg_surface_dipole_y, avg_surface_dipole_z


def get_bond_order(file_path, atom_number):
    """
    This function reads the bond order from a DDEC6 bond order analysis output file for a specified atom.

    Args:
        file_path (str): The path to the DDEC6 bond order analysis output file.
        atom_number (int): The ASE atom number for which to return the bond order. Indexing starts at 0.

    Returns:
        float: The bond order of the specified atom.
    """
    with open(file_path, 'r') as file:
        # Skip the first two lines which contain metadata
        next(file)
        next(file)

        # Loop over the lines in the file
        for i, line in enumerate(file):
            # If the line number matches the requested atom number (accounting for the two skipped lines),
            # parse the bond order from the line and return it
            if i == atom_number:
                elements = line.split()
                bond_order = float(elements[-1])
                print(f"#atom {atom_number} SBO in {file_path} is {bond_order}")
                return bond_order

    # If the atom number was not found in the file, raise an error
    raise ValueError(f"Atom number {atom_number} not found in file {file_path}.")


def calculate_surface_sbo(slab, surface_indices, dos_directory):
    surface_metal_sbo = []
    surface_oxygen_sbo = []

    for index in surface_indices:
        atom = slab[index]
        if atom.symbol == 'O':
            sbo = get_bond_order(os.path.join(dos_directory, 'DDEC6_even_tempered_bond_orders.xyz'), index)
            surface_oxygen_sbo.append(sbo)
        else:
            sbo = get_bond_order(os.path.join(dos_directory, 'DDEC6_even_tempered_bond_orders.xyz'), index)
            surface_metal_sbo.append(sbo)

    avg_surface_metal_sbo = np.mean(surface_metal_sbo) if surface_metal_sbo else 0
    avg_surface_oxygen_sbo = np.mean(surface_oxygen_sbo) if surface_oxygen_sbo else 0

    return avg_surface_metal_sbo, avg_surface_oxygen_sbo

def calculate_surface_densities(slab, surface_indices, surface_area):
    surface_atoms = slab[surface_indices]
    surface_oxygen_num = np.sum(surface_atoms.get_atomic_numbers() == 8)
    print(f"surface_oxygen_num is {surface_oxygen_num}")
    surface_metal_num = np.sum(surface_atoms.get_atomic_numbers() != 8)
    print(f"surface_metal_num is {surface_metal_num}")

    surface_oxygen_density = surface_oxygen_num / surface_area
    surface_metal_density = surface_metal_num / surface_area

    return surface_oxygen_density, surface_metal_density

def calculate_coordination_number(surface_indices, bonded_atoms_list):
    surface_coordination_numbers = [len(bonded_atoms_list[index]) for index in surface_indices]
    avg_surface_coordination_number = np.mean(surface_coordination_numbers)
    return avg_surface_coordination_number


def main():
    root_directory = os.getcwd()
    bulk_folders = [folder for folder in os.listdir(root_directory) if os.path.isdir(folder) and 'mp-' in folder]

    all_data = []

    with alive_bar(len(bulk_folders), title='Processing bulk folders') as bar:
        for bulk_folder in bulk_folders:
            bulk_directory = os.path.join(root_directory, bulk_folder)
            energy_bulk = extract_energy(bulk_directory)

            surface_folders = [folder for folder in os.listdir(bulk_directory) if
                               os.path.isdir(os.path.join(bulk_directory, folder))]

            for surface_folder in surface_folders:
                surface_directory = os.path.join(bulk_directory, surface_folder)
                dos_directory = os.path.join(surface_directory, 'DOS')

                print(f"processig on {dos_directory}")
                # Read CONTCAR files
                contcar_slab_path = os.path.join(surface_directory, 'CONTCAR')
                contcar_bulk_path = os.path.join(bulk_directory, 'CONTCAR')
                slab = read(contcar_slab_path)
                bulk = read(contcar_bulk_path)

                # Extract surface threshold
                surface_threshold = 0.9

                # Extract surface coordinates and indices
                surface_coordinates = slab.positions[
                    slab.positions[:, 2] >= surface_threshold * np.max(slab.positions[:, 2])]
                surface_indices = np.where(slab.positions[:, 2] >= surface_threshold * np.max(slab.positions[:, 2]))[0]

                # Check if the surface has oxygen vacancy
                if surface_folder.endswith('_Ov'):
                    # Surface with oxygen vacancy
                    energy_defected_slab = extract_energy(surface_directory)

                    # Find the perfect surface folder
                    perfect_surface_folder = surface_folder[:-3]  # Remove '_Ov' from the end
                    perfect_surface_directory = os.path.join(bulk_directory, perfect_surface_folder)

                    if not os.path.exists(perfect_surface_directory):
                        perfect_surface_folder += '_O'  # Try with '_O' suffix
                        perfect_surface_directory = os.path.join(bulk_directory, perfect_surface_folder)

                    if os.path.exists(perfect_surface_directory):
                        energy_slab = extract_energy(perfect_surface_directory)
                        E_O2 = -9.854  # based on PBE+D3 functional

                        contcar_perfect_slab_path = os.path.join(perfect_surface_directory, 'CONTCAR')
                        perfect_slab = read(contcar_perfect_slab_path)

                        # Get types of atoms
                        atom_types = set(bulk.get_chemical_symbols())
                        metal_element = list(atom_types - {'O'})[0]

                        # Calculate the number of each type of atom in the slab and the bulk
                        defected_slab_counts = {atom: slab.get_chemical_symbols().count(atom) for atom in atom_types}
                        perfect_slab_counts = {atom: perfect_slab.get_chemical_symbols().count(atom) for atom in atom_types}

                        # Calculate N_surface, N_surface_Ov
                        num = {atom: defected_slab_counts[atom] / perfect_slab_counts[atom] for atom in atom_types}
                        N_surface = num[metal_element]
                        N_surface_Ov = N_surface * perfect_slab_counts['O'] - defected_slab_counts['O']
                        print(f'The number of surface numbers on {surface_directory} is {N_surface}')
                        print (f'The number of surface oxygen vacancy on {surface_directory} is {N_surface_Ov}')
                        E_defect = energy_defected_slab - N_surface * energy_slab + N_surface_Ov * 0.5 * E_O2
                    else:
                        print(
                            f"Warning: Perfect surface folder not found for {surface_folder}. Skipping surface energy calculation.")
                        continue
                else:
                    # Perfect surface
                    energy_slab = extract_energy(surface_directory)
                    energy_defected_slab = None
                    N_surface = 1.0
                    E_defect = 0

                # Calculate surface energy
                surface_energy, surface_area = calculate_surface_energy(energy_slab, energy_defected_slab, energy_bulk,
                                                                        contcar_slab_path, contcar_bulk_path, E_defect, N_surface)

                # Calculate surface oxygen and metal densities
                surface_oxygen_density, surface_metal_density = calculate_surface_densities(slab, surface_indices,
                                                                                            surface_area)

                # Calculate surface metal and oxygen SBO
                avg_surface_metal_sbo, avg_surface_oxygen_sbo = calculate_surface_sbo(slab, surface_indices,
                                                                                      dos_directory)

                # Calculate work function
                work_function = calculate_work_function(dos_directory)

                # Calculate surface roughness parameters
                ra, rq, rmax = calculate_surface_roughness(dos_directory)

                # Calculate surface charge and dipole
                avg_positive_charge, avg_negative_charge, avg_surface_dipole_x, avg_surface_dipole_y, avg_surface_dipole_z = calculate_surface_charge_dipole(
                    dos_directory)

                # Append data to the list
                data = {
                    'Bulk Folder': bulk_folder,
                    'Surface Folder': surface_folder,
                    'DOS Folder': dos_directory,
                    'Surface Energy (J/m^2)': surface_energy,
                    'Work Function (eV)': work_function,
                    'Arithmetic Average Roughness (Ra)': ra,
                    'RMS Roughness (Rq)': rq,
                    'Maximal Height Difference (Rmax)': rmax,
                    'Average Positive Charge (e)': avg_positive_charge,
                    'Average Negative Charge (e)': avg_negative_charge,
                    'Average Surface Dipole Moment X (a.u.)': avg_surface_dipole_x,
                    'Average Surface Dipole Moment Y (a.u.)': avg_surface_dipole_y,
                    'Average Surface Dipole Moment Z (a.u.)': avg_surface_dipole_z,
                    'Surface Oxygen Density (atoms/Å^2)': surface_oxygen_density,
                    'Surface Metal Density (atoms/Å^2)': surface_metal_density,
                    'Average Surface Metal SBO': avg_surface_metal_sbo,
                    'Average Surface Oxygen SBO': avg_surface_oxygen_sbo
                }
                all_data.append(data)

            bar()  

    # Create a DataFrame with the extracted features
    df = pd.DataFrame(all_data)

    # Save the DataFrame to a CSV file
    output_file = os.path.join(root_directory, 'surface_features.csv')
    df.to_csv(output_file, index=False)
    print(f"Surface features saved to {output_file}")

if __name__ == '__main__':
    main()


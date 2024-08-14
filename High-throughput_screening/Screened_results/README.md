# High-Throughput Screening Results for Sintering-Resistant Oxide Supports

This repository contains the results of a high-throughput screening process for identifying promising sintering-resistant metal oxide supports from the OC22 database using 6-features iGAM or NN-MD predictions.

## Contents

### 1. All_results_summary.csv

This file summarizes the structural information of 10,689 oxide structures predicted by 6-features iGAM or NN-MD from the OC22 database.

### 2. Sintering-resistant_candidates_Stable_or_meta-stable.csv

This file contains a refined list of promising sintering-resistant metal oxide supports. The candidates meet the following criteria:
- Thermodynamically stable or meta-stable (0 ≤ ΔEhull ≤ 25 meV/atom)
- Melting points above 800°C
- Predicted contact angle at 800°C (tested with 3nm Pt NP) within 90° ± 10°
- Surface energy less than or equal to 3.0 J/m²

This file serves as a mapping of structural information for the most promising candidates.

### 3. Stable_or_meta-stable_promising_sintering-resistant_supports_from_OC22 (Directory)

This directory contains detailed information for the candidates listed in `Sintering-resistant_candidates_Stable_or_meta-stable.csv`. For each candidate, you will find:

- DFT optimization trajectory file (`{trajectory_id}.traj`): Can be read using ASE (Atomic Simulation Environment)
- DFT-relaxed structure file (POSCAR): Represents the final frame from the trajectory file

## Usage

1. `All_results_summary.csv` provides an overview of all screened structures.
2. `Sintering-resistant_candidates_Stable_or_meta-stable.csv` offers a focused list of the most promising candidates.
3. The `Stable_or_meta-stable_promising_sintering-resistant_supports_from_OC22` directory contains detailed structural information for further analysis and simulation.

## Notes

- The screening process combined thermodynamic stability, melting point, and predicted contact angle to identify the most promising sintering-resistant oxide supports.
- The provided trajectory files can be used for further computational studies or visualizations using ASE or other compatible software.
- POSCAR files represent the final, relaxed structures and can be used as input for additional calculations or analyses.

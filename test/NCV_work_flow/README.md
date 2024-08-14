# NCV_work_flow: Automated iGAM Model Training and Evaluation

## Overview

This directory contains an automated workflow for training, optimizing, and evaluating Interpretable Generalized Additive Models (iGAM) using both random cross-validation and nested cross-validation (NCV) strategies. The workflow is designed to process a large dataset, split it into various subsets, train models, and provide comprehensive performance analyses.

## Features

- Automated data splitting using random and stratified nested cross-validation
- iGAM model training and hyperparameter optimization
- Performance evaluation and comparison across different sampling methods
- Visualization of model performance
- Extraction and analysis of best hyperparameters
- Final model training on the entire dataset

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- interpret
- matplotlib
- tqdm

## File Structure

- `Total_set.csv`: Input data file
- `NCV_workflow.py`: Main script containing the entire workflow
- `iGAM.py`: Module containing the `run_iGAM()` function for model training and evaluation
- Generated directories:
  - `Random_CV/`: Contains results for random cross-validation
  - `Stratified_NCV_X/`: (X from 1 to 5) Contains results for each fold of nested cross-validation
  - `bagging_models/`: Stores the best models from each NCV fold
  - `final_model_dir/`: Contains the final model trained on the entire dataset

## Usage

1. Ensure all required packages are installed:

pip install pandas numpy scikit-learn interpret matplotlib tqdm

2. Place your `Total_set.csv` file in the same directory as the script.

3. Run the main script:

4. The script will automatically:
- Split the data
- Train and optimize models
- Collect and analyze results
- Generate performance visualizations

5. Review the output:
- Check the console for progress updates and summary statistics
- Examine the generated directories for detailed results
- View the performance graphs saved as PNG, SVG, and PDF files

## Output

- Model performance metrics (MAE, RMSE, R²) for each fold
- Visualizations comparing MAE across different sampling methods
- Best hyperparameters from the nested cross-validation process
- Final optimized model saved in `final_model_dir/`

## Customization

- Adjust the number of folds by modifying the `n_splits` parameter in `StratifiedKFold`
- Modify the `run_iGAM()` function in `iGAM.py` to change model training parameters or evaluation metrics

## Notes

- The workflow uses stratified sampling based on the 'Bulk' column in the dataset
- Ensure sufficient computational resources, as the process can be time-intensive for large datasets

## Troubleshooting

If you encounter any issues:
1. Check that all required packages are correctly installed
2. Ensure `Total_set.csv` is in the correct format and location
3. Verify that you have write permissions in the directory

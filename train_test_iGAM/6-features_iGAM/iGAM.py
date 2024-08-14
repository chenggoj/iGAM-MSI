#!/usr/bin/env python
# coding: utf-8

# In[ ]:

def run_iGAM():
    import os
    import sys
    import pandas as pd
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
    from interpret.glassbox import ExplainableBoostingRegressor
    from interpret import show
    from interpret.perf import RegressionPerf
    from tqdm import tqdm
    from shutil import copyfile
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    import psutil
    import time
    from interpret import set_visualize_provider
    from interpret.provider import InlineProvider
    from scipy.stats import pearsonr
    from minepy import MINE
    from matplotlib.lines import Line2D
    import matplotlib
    import matplotlib.patches
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    import logging
    set_visualize_provider(InlineProvider())
    get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
    # Setup logging
    import warnings
    logging.basicConfig(level=logging.INFO)

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        log = file if hasattr(file, 'write') else sys.stderr
        log.write(warnings.formatwarning(message, category, filename, lineno, line))
        logging.error(f"Warning caught: {message}, {category}, {filename}, {lineno}")

    warnings.showwarning = warn_with_traceback

    def select_param_space(level):
        """
        Returns the parameter space based on the desired level of granularity.
        """
        param_space = {
            "Debug": {
                'learning_rate': [0.01],
                'max_rounds': [50],
                'max_leaves': [2]
            },
            "Grid_fine": {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_rounds': [50, 150, 300],
                'max_leaves': [2, 4, 6],
                'min_samples_leaf': [1, 5, 10, 20],
                'max_bins': [32, 64, 128, 512],
                'early_stopping_rounds': [10, 30, 50],
                'early_stopping_tolerance': [0.0001, 0.01],
                'outer_bags': [1, 2, 3, 4, 5, 6, 7, 8],
                'inner_bags': [0, 1, 2]
            },
            "Grid_veryfine": {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_rounds': [50, 100, 150, 200, 250, 300],
                'max_leaves': [2, 3, 4, 5, 6],
                'min_samples_leaf': [1, 2, 5, 10, 20],
                'max_bins': [128, 256, 512],
                'early_stopping_rounds': [10, 20, 30, 40, 50],
                'early_stopping_tolerance': [0.0001, 0.001, 0.01],
                'outer_bags': [1, 2, 3, 4, 5, 6, 7, 8],
                'inner_bags': [0, 1, 2]
            },
            "Bayes_search": {
                'learning_rate': Real(0.01, 0.1),  # Reduced the upper limit to 0.1
                'max_rounds': Integer(50, 300),
                'max_leaves': Integer(2, 4),  # Reduced the upper limit to 4
                'min_samples_leaf': Integer(5, 10),  # Increased the lower limit to 5
                'max_bins': Integer(128, 512),
                'early_stopping_rounds': Integer(10, 20),  # Reduced the upper limit to 20
                'early_stopping_tolerance': Real(0.0001, 0.01),
                'outer_bags': Integer(1, 8),  # Added outer bags to the search space
                'inner_bags': Integer(0, 2)  # Added inner bags to the search space
            },
            "Random_search": {
                    'learning_rate': (0.001, 0.5),
                    'max_rounds': (10, 1000),
                    'max_leaves': (1, 20),
                    'min_samples_leaf': (1, 20),
                    'max_bins': (8, 1024),
                    'max_interaction_bins': (8, 1024),  
                    'early_stopping_rounds': (5, 100),
                    'early_stopping_tolerance': (0.0001, 0.1),
                    'outer_bags': (1,5),
                    'inner_bags': (0,4),
                    'smoothing_rounds':(0, 100)
                },

            "Random_then_bayes": {
                    'learning_rate': (0.01, 0.5),
                    'max_rounds': (10, 300),
                    'max_leaves': (2, 8),
                    'min_samples_leaf': (1, 20),
                    'max_bins': (32, 512),
                    'max_interaction_bins': (16, 256),  # Suggested range for max_interaction_bins
                    'early_stopping_rounds': (10, 100),
                    'early_stopping_tolerance': (0.0001, 0.01),
                    'outer_bags': (1,10),
                    'inner_bags': (0,2),
                    'smoothing_rounds':(0, 100)
            },
        }
        return param_space.get(level,
                               param_space["Debug"])  # Returns "debug" level by default if the level is not recognized

    def main(space_level="debug"):
        logging.info("Loading the data...")
        X, Y, X_train, X_test, y_train, y_test, y_train_with_remarks, y_test_with_remarks = prepare_data(
            "Total_set.csv")

        logging.info("Analyzing feature correlation...")
        analyze_feature_correlation(X, Y)

        # Use the function to get the appropriate param_space based on the desired level
        param_space = select_param_space(space_level)

        save_path = "results"

        logging.info("Running model analysis...")
        run_model_analysis(X_train, y_train, X_test, y_test, space_level, param_space, save_path=save_path)

        logging.info("Loading the best model...")
        with open(f"{save_path}/best_model_performance/best_model.pkl", "rb") as file:
            best_model = pickle.load(file)

        logging.info("Performing residual analysis...")
        residual_analysis(save_path, best_model, X_train, y_train_with_remarks, X_test, y_test_with_remarks)

        logging.info("Showing the model interpretability...")
        ebm_global, ebm_local, importances_avg_weight = show_model_interpretability(best_model, X, Y)

        return importances_avg_weight

    def prepare_data(filepath):
        """
        Load the data, process features and target variable, and return them.
        """
        # Check if both stratified_train_sets.csv and stratified_test_sets.csv exist
        if os.path.exists(filepath) and os.path.exists("stratified_train_sets.csv") and os.path.exists(
                "stratified_test_sets.csv"):
            # Load stratified datasets
            data = pd.read_csv(filepath)
            # select features
            #feature_names = ["E_surf.", "WF", "Ra", "NC_postive", "NC_negative", "Dipole_Z", "rho_O", "rho_M", "M_SBO", "O_SBO", "Ef",
#"Ehull"]
            feature_names = ["E_surf.", "Ra", "rho_O", "rho_M", "Ef", "Ehull"]
            X = data[feature_names]
            Y = data['E_adh']

            train_data = pd.read_csv("stratified_train_sets.csv")
            test_data = pd.read_csv("stratified_test_sets.csv")

            X_train = train_data[feature_names]
            y_train_with_remarks = train_data[['E_adh', 'File_pathway', 'Bulk', 'Surfaces']]
            y_train = y_train_with_remarks['E_adh']

            X_test = test_data[feature_names]
            y_test_with_remarks = test_data[['E_adh', 'File_pathway', 'Bulk', 'Surfaces']]
            y_test = y_test_with_remarks['E_adh']

        elif os.path.exists(filepath):
            # Random_sampling
            data = pd.read_csv(filepath)
            feature_names = ["E_surf.", "Ra", "rho_O", "rho_M", "Ef", "Ehull"]
            X = data[feature_names]
            Y = data['E_adh']

            y_with_remarks = data[['E_adh', 'File_pathway', 'Bulk', 'Surfaces']]
            X_train, X_test, y_train_with_remarks, y_test_with_remarks = train_test_split(X, y_with_remarks,
                                                                                          test_size=0.2,
                                                                                          random_state=42)
            y_train = y_train_with_remarks['E_adh']
            y_test = y_test_with_remarks['E_adh']

        else:
            raise FileNotFoundError("Required datasets not found!")

        return X, Y, X_train, X_test, y_train, y_test, y_train_with_remarks, y_test_with_remarks

    # Function to calculate Pearson correlation
    def pearson_corr(x, y):
        return pearsonr(x, y)[0]

    # Function to calculate Maximal Information Coefficient (MIC)
    def calc_mic(x, y):
        mine = MINE()
        mine.compute_score(x, y)
        return mine.mic()

    def analyze_feature_correlation(X, Y):
        """
        Calculate and visualize the correlation between features and the target.
        Includes both Pearson correlation and MIC.
        """
        mic_values = []
        r_values = []

        # Calculating MIC and Pearson's r for each feature
        for feature in X.columns:
            mic_values.append(calc_mic(X[feature], Y))
            r_values.append(pearson_corr(X[feature], Y))

        # Sorting data based on MIC values
        sorted_indices = np.argsort(mic_values)[::-1]
        feature_names_sorted = np.array(X.columns)[sorted_indices]
        mic_values_sorted = np.array(mic_values)[sorted_indices]
        r_values_sorted = np.array(r_values)[sorted_indices]

        # Plotting the data
        fig, ax = plt.subplots(figsize=(20, 15))
        ax.bar(feature_names_sorted, mic_values_sorted, color='blue', align='center')
        ax.bar(feature_names_sorted, -np.abs(r_values_sorted),
               color=np.where(r_values_sorted < 0, 'lightblue', 'lightcoral'), align='center')

        # Setting x and y labels
        ax.set_xlabel("Features", fontsize=18, weight='bold')
        y_ticks = np.linspace(-1, 1, 11)
        y_ticklabels = ["{:.1f}".format(abs(label)) if label != 0 else '0' for label in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticklabels, fontsize=16, weight='bold')

        # Annotating MIC and |Pearson's r| labels
        ax.text(-0.08, 0.8, 'MIC', transform=ax.transAxes, verticalalignment='center',
                horizontalalignment='right', fontsize=18, color='black', weight='bold', rotation=90)
        ax.text(-0.08, 0.2, "|Pearson's r|", transform=ax.transAxes, verticalalignment='center',
                horizontalalignment='right', fontsize=18, color='black', weight='bold', rotation=90)

        # Drawing the dashed line
        ax.axhline(0, linestyle='-', color='black', linewidth=2.5)

        # Setting x labels and rotating them
        ax.set_xticks(range(len(feature_names_sorted)))
        ax.set_xticklabels(feature_names_sorted, rotation=90, fontsize=16, weight='bold')

        # Adjusting space at the bottom for x labels
        plt.subplots_adjust(bottom=0.25)

        # Adding grid and legends
        ax.grid(axis='y')
        legend_elements = [Line2D([0], [0], color='blue', lw=4, label='MIC'),
                           Line2D([0], [0], color='lightcoral', lw=4, label="Pearson's r > 0"),
                           Line2D([0], [0], color='lightblue', lw=4, label="Pearson's r < 0")]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=16, frameon=True, handlelength=1.5,
                  edgecolor='black').get_frame().set_linewidth(1.5)

        # Saving plots in different formats
        plt.savefig('Total_feature_corelation_with_binding_energy.png', dpi=600)
        plt.savefig('Total_feature_corelation_with_binding_energy.svg', format='svg')
        plt.savefig('Total_feature_corelation_with_binding_energy.pdf', format='pdf')

        # Showing the plot
        plt.show()

    def run_model_analysis(X_train, y_train, X_test, y_test, space_level, param_space, save_path="results"):
        """
        Run the model analysis, including training, optimization, testing, and visualization.

        Parameters:
        - X_train, y_train: Training data and corresponding labels.
        - X_test, y_test: Testing data and corresponding labels.
        - param_space: Space of hyperparameters for search.
        - save_path: Directory path to save results, plots, and models.

        Returns:
        - None (results are saved to files).
        """

        # Ensure the save directory exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Ensure subdirectories exist
        cv_path = os.path.join(save_path, "cv10_results")
        if not os.path.exists(cv_path):
            os.makedirs(cv_path)

        best_model_path = os.path.join(save_path, "best_model_performance")
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)

        # Initialize the interpretable machine learning model
        features_num = int(X_train.shape[1])
        
        max_interactions = features_num * (features_num - 1) // 2
        ebm = ExplainableBoostingRegressor(interactions=max_interactions, random_state=42)

        # Set up the 10-fold cross-validation
        kf = KFold(n_splits=10)

        # Initialize a list to hold the performance of each model
        performance = []

        # Iterate through the 10 folds
        for idx, (train_index, valid_index) in tqdm(enumerate(kf.split(X_train), start=1), total=10,
                                                    desc='CV Iteration'):
            print(f"Iteration {idx}")

            # Split the data into training and validation sets for the current fold
            X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
            y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

            # Save the train and validation sets for the current fold
            current_iter_path = os.path.join(cv_path, f"iter_{idx:02d}")
            if not os.path.exists(current_iter_path):
                os.makedirs(current_iter_path)

            train_set = pd.concat([X_train_fold, y_train_fold], axis=1)
            valid_set = pd.concat([X_valid_fold, y_valid_fold], axis=1)
            train_set.to_csv(os.path.join(current_iter_path, "train_set.csv"), index=False)
            valid_set.to_csv(os.path.join(current_iter_path, "valid_set.csv"), index=False)

            # Fit the model with the optimal hyperparameters using the training set for the current fold
            if space_level == "Debug" or space_level == "Grid_fine" or space_level == "Grid_veryfine":
                print("-" * 80)
                print("Starting Gridsearch for the best hyperparameters")
                print(" " * 80)
                grid_search = GridSearchCV(
                    estimator=ebm,
                    param_grid=param_space,
                    scoring=make_scorer(mean_squared_error, greater_is_better=False),
                    cv=10,
                    n_jobs=-1,  # use all available CPU cores on the working node
                    verbose=1,
                    error_score=np.NINF  # optional: 'raise'
                )
                grid_search.fit(X_train_fold, y_train_fold)
                best_params = grid_search.best_params_
                print("Best hyperparameters: ", best_params)
                best_model = grid_search.best_estimator_
            elif space_level == "Random_search":
                print("-" * 80)
                print("Starting Randomized Search for hyperparameters")
                print(" " * 80)

                random_search = RandomizedSearchCV(
                    estimator=ebm,
                    param_distributions=param_space,
                    n_iter=2048,  # Adjust as needed
                    scoring=make_scorer(mean_squared_error, greater_is_better=False),
                    cv=10,
                    n_jobs=-1,  # use all available CPU cores on the working node
                    verbose=1,
                    error_score=np.NINF,  # optional: 'raise'
                    random_state=42  # Ensure reproducibility
                )
                random_search.fit(X_train_fold, y_train_fold)
                best_params = random_search.best_params_
                print("Best hyperparameters: ", best_params)
                best_model = random_search.best_estimator_

            elif space_level == "Random_then_bayes":
                print("-" * 80)
                print("Starting Random Search followed by BayesSearchCV for the best hyperparameters")
                print(" " * 80)
                
                print("-" * 80)
                print("Starting Random Search for roughly good hyperparameters")
                print(" " * 80)

                # Start with random search
                n_random_search_iterations = 2048  # Adjust as needed
                random_search = RandomizedSearchCV(
                    estimator=ebm,
                    param_distributions=param_space,
                    n_iter=n_random_search_iterations,
                    scoring=make_scorer(mean_squared_error, greater_is_better=False),
                    cv=10,
                    n_jobs=-1,  # use all available CPU cores on the working node
                    verbose=1,
                    error_score=np.NINF,  # optional: 'raise'
                    random_state=42  # Ensure reproducibility
                )
                random_search.fit(X_train_fold, y_train_fold)
                
                # Get the best parameters from random search
                best_random_params = random_search.best_params_

                # Create a refined parameter space around the best parameters
                refined_param_space = {
                    'learning_rate': Real(
                        max(best_random_params['learning_rate'] * 0.8, 0.01),
                        min(best_random_params['learning_rate'] * 1.2, 0.2)  # Adjusted to 0.2 based on your initial space
                    ),
                    'max_rounds': Integer(
                        max(int(best_random_params['max_rounds'] - 10), 50),
                        min(int(best_random_params['max_rounds'] + 10), 300)
                    ),
                    'max_leaves': Integer(
                        max(int(best_random_params['max_leaves'] - 1), 2),
                        min(int(best_random_params['max_leaves'] + 1), 6)  # Adjusted to 6 based on your initial space
                    ),
                    'min_samples_leaf': Integer(
                        max(int(best_random_params['min_samples_leaf'] - 2), 1),  # Adjusted lower bound to 1
                        min(int(best_random_params['min_samples_leaf'] + 2), 20)
                    ),
                    'max_bins': Integer(
                        max(int(best_random_params['max_bins'] - 50), 64),  # Adjusted lower bound to 64
                        min(int(best_random_params['max_bins'] + 50), 512)
                    ),
                    'early_stopping_rounds': Integer(
                        max(int(best_random_params['early_stopping_rounds'] - 5), 10),
                        min(int(best_random_params['early_stopping_rounds'] + 5), 50)  # Adjusted to 50 based on your initial space
                    ),
                    'early_stopping_tolerance': Real(
                        max(best_random_params['early_stopping_tolerance'] * 0.8, 0.0001),
                        min(best_random_params['early_stopping_tolerance'] * 1.2, 0.01)
                    ),
                }

                print("-" * 80)
                print("Starting BayesSearchCV for precisely the best hyperparameters around the sets from Random search")
                print(" " * 80)
                
                # Use BayesSearchCV with initial results from random search
                bayes_search = BayesSearchCV(
                    estimator=ebm,
                    search_spaces=refined_param_space,
                    scoring=make_scorer(mean_squared_error, greater_is_better=False),
                    cv=10,
                    n_jobs=-1,
                    verbose=1,
                    error_score=np.NINF,  # optional: 'raise'
                    n_iter=100,  # Adjust as needed
                    random_state=42,  # Ensure reproducibility
                )
                bayes_search.fit(X_train_fold, y_train_fold)

                best_params = bayes_search.best_params_
                print("Best hyperparameters: ", best_params)
                best_model = bayes_search.best_estimator_

            # Existing code for other search strategies
            elif space_level == "Bayes_search":
                # Use BayesSearchCV
                print("-" * 80)
                print("Starting BayesSearchCV for the best hyperparameters")
                print(" " * 80)
                bayes_search = BayesSearchCV(
                    estimator=ebm,
                    search_spaces=param_space,
                    scoring=make_scorer(mean_squared_error, greater_is_better=False),
                    cv=10,
                    n_jobs=-1,  # use all available CPU cores except 1 CPU core on the working node
                    verbose=1,
                    error_score=np.NINF,  # optional: 'raise'
                    random_state=42,
                    n_iter=1000  # number of iterations, you can adjust this
                )

                bayes_search.fit(X_train_fold, y_train_fold)
                best_params = bayes_search.best_params_
                print("Best hyperparameters: ", best_params)
                best_model = bayes_search.best_estimator_

            # Save the trained model for the current fold
            with open(f"{current_iter_path}/best_model.pkl", "wb") as f:
                pickle.dump(best_model, f)

            # Test the model for the current fold
            test_predictions = best_model.predict(X_test)
            train_predictions = best_model.predict(X_train)

            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))

            test_mae = mean_absolute_error(y_test, test_predictions)
            train_mae = mean_absolute_error(y_train, train_predictions)

            test_r2 = r2_score(y_test, test_predictions)
            train_r2 = r2_score(y_train, train_predictions)

            # Plot predicted vs actual for the current fold
            plt.figure(figsize=(10, 6))
            plt.scatter(train_predictions, y_train, alpha=0.7, color='b', label='train', s=160)
            plt.scatter(test_predictions, y_test, alpha=0.7, color='r', label='validation', s=160)
            plt.xlabel('Predicted $\mathit{\Delta E}_\mathrm{adh}$ (J/m2) from the best trained model', fontsize=15)
            plt.ylabel('Actual $\mathit{\Delta E}_\mathrm{adh}$ (J/m2) from NN-MD', fontsize=15)
            plt.title(f'Performance for the best trained iGAM (iteration {idx})', fontsize=15)
            plt.legend(fontsize=15)
            plt.grid(True)

            # Add R² scores and MAE to bottom-right corner
            x_range = np.max(y_test) - np.min(y_test)
            y_range = x_range
            x_loc = np.max(y_test) - x_range / 500
            y_loc = np.min(y_test) + y_range / 50
            plt.text(x_loc, y_loc,
                     'Train R²: {:.2f}\nValidation R²: {:.2f}\nTrain MAE: {:.2f}\nValidation MAE: {:.2f}'.format(
                         train_r2, test_r2, train_mae, test_mae), ha='right', va='bottom', fontsize=15)

            # Add Y=X line
            plt.plot([-3.5, 0.5], [-3.5, 0.5], 'k--')

            # Set axis limits
            plt.xlim([-3.5, 0.5])
            plt.ylim([-3.5, 0.5])

            plt.savefig(os.path.join(current_iter_path, "model_performance.png"), dpi=600)
            plt.savefig(os.path.join(current_iter_path, "model_performance.svg"), format='svg')
            plt.savefig(os.path.join(current_iter_path, "model_performance.pdf"), format='pdf')
            # plt.show()
            plt.close()

            # Append the performance of the current model to the list
            performance.append({
                'iteration': idx,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'best_params': best_params,
            })

        # Calculate the average performance
        average_performance = {
            'train_r2': np.mean([p['train_r2'] for p in performance]),
            'test_r2': np.mean([p['test_r2'] for p in performance]),
            'train_rmse': np.mean([p['train_rmse'] for p in performance]),
            'test_rmse': np.mean([p['test_rmse'] for p in performance]),
            'train_mae': np.mean([p['train_mae'] for p in performance]),
            'test_mae': np.mean([p['test_mae'] for p in performance]),
        }

        # Write the performance of each model and the average performance to a text file
        with open(os.path.join(cv_path, "CV10_total_performance.txt"), "w") as f:
            f.write(f"----------------Start of CV10----------------\n\n")
            for p in performance:
                f.write(f"Iteration {p['iteration']}:\n")
                f.write(f"Train R²: {p['train_r2']}\n")
                f.write(f"Test R²: {p['test_r2']}\n")
                f.write(f"Train RMSE: {p['train_rmse']}\n")
                f.write(f"Test RMSE: {p['test_rmse']}\n")
                f.write(f"Train MAE: {p['train_mae']}\n")
                f.write(f"Test MAE: {p['test_mae']}\n")
                f.write(f"Best parameters: {p['best_params']}\n\n")
            f.write(f"----------------Overall performance of CV10----------------\n\n")
            f.write("Average performance:\n")
            f.write(f"Train R²: {average_performance['train_r2']}\n")
            f.write(f"Test R²: {average_performance['test_r2']}\n")
            f.write(f"Train RMSE: {average_performance['train_rmse']}\n")
            f.write(f"Test RMSE: {average_performance['test_rmse']}\n")
            f.write(f"Train MAE: {average_performance['train_mae']}\n")
            f.write(f"Test MAE: {average_performance['test_mae']}\n")
            # Sort the performance list by test R² in descending order
            sorted_by_test_r2 = sorted(performance, key=lambda p: p['test_r2'], reverse=True)

            # Sort the performance list by test RMSE in ascending order
            sorted_by_test_rmse = sorted(performance, key=lambda p: p['test_rmse'])

            # Sort the performance list by test MAE in ascending order
            sorted_by_test_mae = sorted(performance, key=lambda p: p['test_mae'])

            # Write the best model information by test R² to the text file
            f.write("\nBest Model by Test R²\n")
            f.write(f"Iteration: {sorted_by_test_r2[0]['iteration']}\n")
            f.write(f"Train R²: {sorted_by_test_r2[0]['train_r2']}\n")
            f.write(f"Test R²: {sorted_by_test_r2[0]['test_r2']}\n")
            f.write(f"Train RMSE: {sorted_by_test_r2[0]['train_rmse']}\n")
            f.write(f"Test RMSE: {sorted_by_test_r2[0]['test_rmse']}\n")
            f.write(f"Train MAE: {sorted_by_test_r2[0]['train_mae']}\n")
            f.write(f"Test MAE: {sorted_by_test_r2[0]['test_mae']}\n\n")

            # Write the best model information by test RMSE to the text file
            f.write("Best Model by Test RMSE\n")
            f.write(f"Iteration: {sorted_by_test_rmse[0]['iteration']}\n")
            f.write(f"Train R²: {sorted_by_test_rmse[0]['train_r2']}\n")
            f.write(f"Test R²: {sorted_by_test_rmse[0]['test_r2']}\n")
            f.write(f"Train RMSE: {sorted_by_test_rmse[0]['train_rmse']}\n")
            f.write(f"Test RMSE: {sorted_by_test_rmse[0]['test_rmse']}\n")
            f.write(f"Train MAE: {sorted_by_test_rmse[0]['train_mae']}\n")
            f.write(f"Test MAE: {sorted_by_test_rmse[0]['test_mae']}\n\n")

            # Write the best model information by test MAE to the text file
            f.write("Best Model by Test MAE\n")
            f.write(f"Iteration: {sorted_by_test_mae[0]['iteration']}\n")
            f.write(f"Train R²: {sorted_by_test_mae[0]['train_r2']}\n")
            f.write(f"Test R²: {sorted_by_test_mae[0]['test_r2']}\n")
            f.write(f"Train RMSE: {sorted_by_test_mae[0]['train_rmse']}\n")
            f.write(f"Test RMSE: {sorted_by_test_mae[0]['test_rmse']}\n")
            f.write(f"Train MAE: {sorted_by_test_mae[0]['train_mae']}\n")
            f.write(f"Test MAE: {sorted_by_test_mae[0]['test_mae']}\n\n")
            f.write(f"----------------End of CV10----------------\n\n")
            f.write(
                f"I will use the best-trained hyperparameters by Iteration: {sorted_by_test_mae[0]['iteration']} to train and test the final model via the whole training set and test set\n\n")

        # Open and read the file
        with open(os.path.join(cv_path, "CV10_total_performance.txt"), "r") as f:
            content = f.read()

        start = content.find('----------------Start of CV10----------------')
        end = content.find('----------------Overall performance of CV10----------------')

        # Get the content of the iterations
        iteration_content = content[start:end]

        # Split the content into a list of strings for each iteration
        iterations = iteration_content.split('Iteration')

        # Initialize lists to store iteration numbers, training MAE, and testing MAE
        iteration_nums = []
        train_maes = []
        test_maes = []

        # Extract the iteration and MAE values from the file content
        for iteration in iterations[1:]:  # Skip the first split result because it's empty
            lines = iteration.split('\n')
            iteration_num = lines[0].strip().split(':')[0]  # Extract the iteration number
            iteration_nums.append(iteration_num)
            for line in lines[1:]:  # Skip the first line because it's the iteration number
                if 'Train MAE' in line:
                    train_maes.append(float(line.split(':')[-1].strip()))
                elif 'Test MAE' in line:
                    test_maes.append(float(line.split(':')[-1].strip()))

        # Create a list of tuples, each containing the iteration number, training MAE, and testing MAE
        performance = list(zip(iteration_nums, train_maes, test_maes))

        # Sort the performance list in ascending order of testing MAE
        performance.sort(key=lambda x: x[2])

        # Unzip the sorted list into iteration numbers, training MAEs, and testing MAEs
        iteration_nums, train_maes, test_maes = zip(*performance)

        # Plot CV10_MAE image
        # Plot the training and testing MAEs
        plt.figure(figsize=(10, 6))
        plt.plot(iteration_nums, train_maes, marker='o', linestyle='-', color='b', label='Train MAE')
        plt.plot(iteration_nums, test_maes, marker='o', linestyle='-', color='r', label='Validation MAE')
        plt.xlabel('Iteration Number', fontsize=15)
        plt.ylabel('MAE(eV)', fontsize=15)
        plt.title('Training MAE and Validation MAE over Iterations', fontsize=15)
        plt.legend(fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(True)
        plt.savefig(os.path.join(cv_path, "CV10_MAE.png"), dpi=600)
        plt.savefig(os.path.join(cv_path, "CV10_MAE.svg"), format='svg')
        plt.savefig(os.path.join(cv_path, "CV10_MAE.pdf"), format='pdf')
        plt.show()
        plt.close()

        # Load the best model by test MAE from iteration iter_num
        iter_num = sorted_by_test_mae[0]['iteration']
        with open(os.path.join(cv_path, f"iter_{iter_num:02d}/best_model.pkl"), "rb") as f:
            best_model = pickle.load(f)

        # Train the model on the full training set
        best_model.fit(X_train, y_train)

        # Make predictions on the full training and test sets
        train_predictions = best_model.predict(X_train)
        test_predictions = best_model.predict(X_test)

        # Save the full training set predictions and true values to a CSV file
        train_set = pd.DataFrame({'y_initial': y_train, 'y_predicted': train_predictions})
        train_set.to_csv(os.path.join(best_model_path, "train_set.csv"), index=False)

        # Save the test set predictions and true values to a CSV file
        test_set = pd.DataFrame({'y_initial': y_test, 'y_predicted': test_predictions})
        test_set.to_csv(f'{best_model_path}/test_set.csv', index=False)

        # Save the model to a file
        with open(f'{best_model_path}/best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)

        # Evaluate the performance of the model on the full training and test sets
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        train_mae = mean_absolute_error(y_train, train_predictions)
        train_r2 = r2_score(y_train, train_predictions)

        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_mae = mean_absolute_error(y_test, test_predictions)
        test_r2 = r2_score(y_test, test_predictions)

        # Plot predicted vs actual for the best_model_performance fold
        plt.figure(figsize=(10, 6))
        plt.scatter(train_predictions, y_train, alpha=0.7, color='b', label='train', s=160)
        plt.scatter(test_predictions, y_test, alpha=0.7, color='r', label='test', s=160)
        plt.xlabel(
            'Predicted $\mathit{\Delta E}_\mathrm{adh}$ (J/m2) from the best trained model on the whole training set',
            fontsize=15)
        plt.ylabel('Actual $\mathit{\Delta E}_\mathrm{adh}$ (J/m2) from NN-MD', fontsize=15)
        plt.title(f'Performance for the best trained iGAM on the whole dataset', fontsize=15)
        plt.legend(fontsize=15)
        plt.grid(True)

        # Add R² scores and MAE to bottom-right corner
        x_range = np.max(y_test) - np.min(y_test)
        y_range = x_range
        x_loc = np.max(y_test) - x_range / 500
        y_loc = np.min(y_test) + y_range / 50
        plt.text(x_loc, y_loc,
                 'Train R²: {:.2f}\nTest R²: {:.2f}\nTrain MAE: {:.2f}\nTest MAE: {:.2f}'.format(train_r2, test_r2,
                                                                                                 train_mae, test_mae),
                 ha='right', va='bottom', fontsize=15)

        # Add Y=X line
        plt.plot([-3.5, 0.5], [-3.5, 0.5], 'k--')

        # Set axis limits
        plt.xlim([-3.5, 0.5])
        plt.ylim([-3.5, 0.5])

        plt.savefig(os.path.join(best_model_path, "whole_model_performance.png"), dpi=600)
        plt.savefig(os.path.join(best_model_path, "whole_model_performance.svg"), format='svg')
        plt.savefig(os.path.join(best_model_path, "whole_model_performance.pdf"), format='pdf')
        plt.show()
        plt.close()

        # Write the model performance to a text file
        with open(os.path.join(best_model_path, "model_performance.txt"), "w") as f:
            f.write(f'Train R²: {train_r2:.4f}\n')
            f.write(f'Test R²: {test_r2:.4f}\n')
            f.write(f'Train RMSE: {train_rmse:.4f}\n')
            f.write(f'Test RMSE: {test_rmse:.4f}\n')
            f.write(f'Train MAE: {train_mae:.4f}\n')
            f.write(f'Test MAE: {test_mae:.4f}\n')
            f.write(f'Best hyperparameters: {best_params}\n')        
        return

    def residual_analysis(save_path, best_model, X_train, y_train_with_remarks, X_test, y_test_with_remarks,
                          color_train='lightblue', color_test='lightcoral'):
        """
        Perform residual analysis for the given model on training and test datasets.

        Args:
        - best_model: The trained model to be used for predictions.
        - X_train, y_train_with_remarks: Training data and corresponding labels with remarks.
        - X_test, y_test_with_remarks: Test data and corresponding labels with remarks.
        - color_train: Color for the training data boxplot. Default is 'lightblue'.
        - color_test: Color for the test data boxplot. Default is 'lightcoral'.

        Returns:
        - None (results are saved to files and plots are displayed).
        """

        def create_subdir(directory_name):
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)

        def to_subscript(label):
            subs = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
            return label.translate(subs)

        def plot_residuals(y_true_with_remarks, y_pred, model_name, dataset_name, grouping='Bulk', color='blue'):
            """
            Plots residuals using a boxplot grouped by a specified column label.

            Args:
            - y_true_with_remarks: The true labels dataframe with additional remarks.
            - y_pred: The predicted values.
            - model_name: The name of the model being evaluated.
            - dataset_name: The name of the dataset (e.g. "Training Set" or "Test Set").
            - grouping: The column name used for grouping in the boxplot. Default is 'Bulk'.
            - color: The color for the boxplot. Default is blue.
            """
            # Create subdirectory to save plots and CSVs
            subdir_name = f"{save_path}/residuals_and_plots"
            create_subdir(subdir_name)

            # Calculate residuals
            y_true = y_true_with_remarks['E_adh']
            residuals = y_true - y_pred

            # Extract labels from the dataframe based on the 'grouping' argument, convert to subscript, and process them
            y_true_with_remarks.loc[:, grouping] = y_true_with_remarks[grouping].str.split('_').str[0]
            labels = y_true_with_remarks[grouping].apply(lambda x: to_subscript(x)).tolist()
            # Create dataframe for residuals and labels
            residuals_df = pd.DataFrame({'Residuals': residuals, 'Label': labels})

            # Sort labels by their median residuals
            sorted_labels = residuals_df.groupby('Label').median().sort_values(by='Residuals').index.tolist()

            # Sort residuals_df based on the sorted_labels
            residuals_df['Label'] = pd.Categorical(residuals_df['Label'], categories=sorted_labels, ordered=True)
            residuals_df = residuals_df.sort_values('Label')

            # Plot boxplot of residuals grouped by labels with sorted order and add a bold dashed line at y=0
            plt.figure(figsize=(14, 8))
            boxplot = residuals_df.boxplot(column='Residuals', by='Label', vert=True, patch_artist=True, grid=True,
                                           fontsize=12)

            boxes = [patch for patch in boxplot.get_children() if isinstance(patch, matplotlib.patches.PathPatch)]
            for box in boxes:
                box.set_facecolor(color)
                # patch.set_facecolor(color)

            plt.axhline(0, color='black', linestyle='--', linewidth=2)  # Add dashed line at y=0
            plt.title(f'Boxplot of Residuals for {model_name} Grouped by {grouping} ({dataset_name})', fontsize=16)
            plt.ylabel('Residuals', fontsize=15)
            plt.xlabel(grouping, fontsize=15)
            plt.xticks(rotation=45)
            plt.suptitle('')  # Get rid of the automatic 'Boxplot grouped by' title
            plt.gca().spines['top'].set_visible(True)
            plt.gca().spines['right'].set_visible(True)
            plt.tight_layout()
            plt.savefig(os.path.join(subdir_name, f'Boxplot of Residuals ({model_name}_{dataset_name}_{grouping}).png'),
                        dpi=600)
            plt.savefig(os.path.join(subdir_name, f'Boxplot of Residuals ({model_name}_{dataset_name}_{grouping}).svg'),
                        format='svg')
            plt.savefig(os.path.join(subdir_name, f'Boxplot of Residuals ({model_name}_{dataset_name}_{grouping}).pdf'),
                        format='pdf')
            plt.show()

        def save_to_csv(y_true, y_pred, model_name, dataset_name):
            # Create subdirectory to save plots and CSVs
            subdir_name = f"{save_path}/residuals_and_plots"
            create_subdir(subdir_name)

            # Calculate residuals
            residuals = y_true['E_adh'] - y_pred

            # Create a new dataframe
            df = pd.DataFrame({
                'Original Data': y_true['E_adh'],
                'Predicted Data': y_pred,
                'Residuals': residuals
            })

            # Add the labels 'File_pathway', 'Bulk', 'Surfaces' from the original dataframe
            df = df.join(y_true[['File_pathway', 'Bulk', 'Surfaces']])
            # Save to CSV
            df.to_csv(os.path.join(subdir_name, f'Residuals_and_Predictions_{model_name}_{dataset_name}.csv'),
                      index=False)
            print(
                f"Saved residuals and predictions for {model_name} {dataset_name} to 'Residuals_and_Predictions_{model_name}_{dataset_name}.csv'")

        # Plot residuals and save residuals and predictions for both training and test data
        # Grouped by 'Bulk'
        plot_residuals(y_train_with_remarks, best_model.predict(X_train), "Best_Model", 'Training Set', grouping='Bulk',
                       color=color_train)
        plot_residuals(y_test_with_remarks, best_model.predict(X_test), "Best_Model", 'Test Set', grouping='Bulk',
                       color=color_test)
        save_to_csv(y_train_with_remarks, best_model.predict(X_train), "Best_Model", 'Training_Set')
        save_to_csv(y_test_with_remarks, best_model.predict(X_test), "Best_Model", 'Test_Set')

        # Grouped by "Surfaces"
        plot_residuals(y_train_with_remarks, best_model.predict(X_train), "Best_Model", 'Training Set',
                       grouping='Surfaces', color=color_train)
        plot_residuals(y_test_with_remarks, best_model.predict(X_test), "Best_Model", 'Test Set',
                       grouping='Surfaces', color=color_test)

    # Example usage:
    # residual_analysis(best_model, X_train, y_train_with_remarks, X_test, y_test_with_remarks)

    def show_model_interpretability(model, X, Y):
        """
        Displays the global and local interpretability of the model.

        Args:ExplainableBoostingRegressor(interactions=21)
        - model: The trained model.
        - X: The features from whole dataset including train and test.
        - Y: The actual_values from whole dataset including train and test.
        - instance_idx: The index of the instance for local interpretability.

        Returns:
        - ebm_global: Global interpretability object.
        - ebm_local: Local interpretability object for the specified instance.
        """
        # Global interpretability
        ebm_global = model.explain_global(name="Global analysis for features importance and role")
        show(ebm_global)

        # Local interpretability for all instances
        ebm_local = model.explain_local(X.iloc[1:-1], Y.iloc[1:-1],
                                        name="Local interpretability object for the specified instance")
        show(ebm_local)
        importances_avg_weight = model.term_importances(importance_type='avg_weight')
        print(f"Best EBM model feature average importance is {importances_avg_weight}")

        return ebm_global, ebm_local, importances_avg_weight
    
    # Space_level Options: "Debug" (Default), "Grid_fine" (Recommended), "Grid_veryfine", "Random_search", "Bayes_search", "Random_then_bayes" (Most efficient)
    main(space_level="Random_search")  

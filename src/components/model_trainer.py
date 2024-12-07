import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# Configuration class for model trainer
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")  # Path to save the trained model

# Class for training and evaluating models
class ModelTrainer:
    def __init__(self):
        # Initialize configuration for saving the trained model
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple machine learning models, evaluates their performance, 
        and selects the best model based on R² score.

        Parameters:
        - train_array (numpy array): Training data (features + target).
        - test_array (numpy array): Testing data (features + target).

        Returns:
        - r2_square (float): R² score of the best model on the test set.

        Raises:
        - CustomException: If no suitable model is found or an error occurs.
        """
        try:
            logging.info("Splitting training and test data into features and target.")
            # Split train and test arrays into features (X) and target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features for training
                train_array[:, -1],  # Target for training
                test_array[:, :-1],  # Features for testing
                test_array[:, -1],  # Target for testing
            )

            # Define the models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define hyperparameter grids for the models
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},  # No hyperparameters for Linear Regression
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
            }

            # Evaluate models and retrieve their performance
            logging.info("Evaluating models using the training and testing datasets.")
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Identify the best model based on the highest R² score
            best_model_score = max(model_report.values())  # Best R² score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Ensure the best model meets the minimum performance threshold
            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable performance.")

            logging.info(f"Best model: {best_model_name} with R² score: {best_model_score}")

            # Save the best model to a file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions on the test data with the best model
            predicted = best_model.predict(X_test)

            # Calculate the R² score for the best model on the test set
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            # Handle exceptions and provide detailed error messages
            raise CustomException(e, sys)

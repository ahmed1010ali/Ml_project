import os
import sys
import pickle
import dill 
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves a Python object to a specified file path using pickle.
    
    Parameters:
    - file_path (str): Path to save the serialized object.
    - obj: The Python object to be serialized and saved.

    Raises:
    - CustomException: If an error occurs during file saving.
    """
    try:
        # Extract the directory from the file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the file using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception for better debugging
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple models using GridSearchCV for hyperparameter tuning 
    and calculates their performance on training and testing datasets.

    Parameters:
    - X_train (array-like): Training features.
    - y_train (array-like): Training target values.
    - X_test (array-like): Testing features.
    - y_test (array-like): Testing target values.
    - models (dict): Dictionary of model names and their corresponding instances.
    - param (dict): Dictionary of hyperparameter grids for each model.

    Returns:
    - report (dict): Dictionary containing R² scores for each model on the test set.

    Raises:
    - CustomException: If an error occurs during model evaluation.
    """
    try:
        # Dictionary to store the test R² scores for each model
        report = {}

        # Iterate through each model in the dictionary
        for model_name, model_instance in models.items():
            # Retrieve the hyperparameter grid for the current model
            hyperparams = param[model_name]

            # Perform grid search to find the best hyperparameters
            gs = GridSearchCV(model_instance, hyperparams, cv=3)
            gs.fit(X_train, y_train)

            # Update the model instance with the best parameters
            model_instance.set_params(**gs.best_params_)

            # Train the model on the entire training set
            model_instance.fit(X_train, y_train)

            # Predict on the training set
            y_train_pred = model_instance.predict(X_train)

            # Predict on the testing set
            y_test_pred = model_instance.predict(X_test)

            # Calculate R² scores for training and testing datasets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test R² score in the report dictionary
            report[model_name] = test_model_score

        # Return the dictionary of test R² scores
        return report

    except Exception as e:
        # Raise a custom exception for better debugging
        raise CustomException(e, sys)

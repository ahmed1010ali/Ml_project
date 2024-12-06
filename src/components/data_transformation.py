import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

# Configuration class to define file paths for saving objects
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

# Main class for handling data transformation
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Method to create and return a preprocessing pipeline
    def get_data_transformer_object(self):
        """
        Creates preprocessing pipelines for numerical and categorical features.
        - Numerical pipeline: Imputes missing values, scales data.
        - Categorical pipeline: Imputes missing values, one-hot encodes, scales data.
        """
        try:
            # Define numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Numerical pipeline: handles missing values and scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Categorical pipeline: handles missing values, one-hot encoding, and scaling
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    # Method to apply transformations and save the preprocessing object
    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads data, applies transformations, and saves preprocessing object.
        - Returns transformed train/test arrays and preprocessor file path.
        """
        try:
            # Load train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Get the preprocessing pipeline
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Split input and target features for train and test data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing data")

            # Transform input features using the preprocessing pipeline
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed input features with target variables
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            # Save the preprocessing pipeline to a file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            # Return transformed data and file path
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)

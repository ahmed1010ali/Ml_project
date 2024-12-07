import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

# Configuration class for data ingestion file paths
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path to save training data
    test_data_path: str = os.path.join('artifacts', "test.csv")   # Path to save testing data
    raw_data_path: str = os.path.join('artifacts', "data.csv")    # Path to save raw data

# Class responsible for data ingestion
class DataIngestion:
    def __init__(self):
        # Initialize configuration for file paths
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Handles the ingestion of data:
        - Reads a dataset.
        - Splits it into training and testing datasets.
        - Saves raw, train, and test data as CSV files.
        """
        logging.info("Entered the data ingestion method/component.")
        try:
            # Read dataset from specified location
            df = pd.read_csv('notebook/data/stud.csv')  # Ensure correct dataset path
            logging.info('Dataset read into a DataFrame.')

            # Create necessary directories for saving files
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Initiating train-test split.")
            # Split data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed.")

            # Return paths for train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

# Main execution for data ingestion, transformation, and model training
if __name__ == "__main__":
    try:
        # Step 1: Data ingestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        # Step 2: Data transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        # Step 3: Model training
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))
    except Exception as e:
        logging.error(f"An error occurred: {e}")

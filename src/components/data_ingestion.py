# Import required modules from the standard library
import os                         # For file/directory handling
import sys                        # For accessing Python runtime environment and system-specific parameters

# Fix to include the root project directory in the module search path so `src` can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import custom exception and logging modules from the 'src' package
from src.exception import CustomException  # Handles exceptions in a customized way
from src.logger import logging             # Logs steps and issues during execution

# Import third-party library for data handling
import pandas as pd                        # For reading and handling CSV files

# Scikit-learn module for splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split

# Dataclass simplifies the creation of classes that just hold data
from dataclasses import dataclass

# Importing Data Transformation logic and its config class
from data_transformation import DataTransformation
from data_transformation import DataTransformationConfig

# Importing Model Trainer logic and its config class
from model_trainer import ModelTrainerConfig
from model_trainer import ModelTrainer

# Define configuration class to manage all file paths for data ingestion
@dataclass  # Automatically generates __init__ and other boilerplate methods
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path where training data will be saved
    test_data_path: str = os.path.join('artifacts', "test.csv")    # Path where test data will be saved
    raw_data_path: str = os.path.join('artifacts', "data.csv")     # Path to save raw original dataset

# Define main class responsible for ingesting the data
class DataIngestion:
    def __init__(self):
        # Instantiate the DataIngestionConfig to access file paths
        self.ingestion_config = DataIngestionConfig()

    # Method to perform the entire data ingestion workflow
    def initiate_data_ingestion(self):
        # Log that data ingestion process has started
        logging.info("Entered the data ingestion method/component")

        try:
            # Read the dataset into a pandas DataFrame from the specified file
            df = pd.read_csv(os.path.join('D:\DATA SCIENCE\ML-PROJECTS\student performance indicator\stud.csv'))

            # Log that reading was successful
            logging.info('Read the dataset as dataframe')

            # Ensure the directory for train/test/raw data exists; if not, create it
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw (unprocessed) data to the specified file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Log that train-test splitting is starting
            logging.info("Train-test split initiated")

            # Split the dataset into 80% training and 20% test data using a fixed random state for reproducibility
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training dataset to a CSV file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Save the test dataset to a CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            # Log that the ingestion process has completed
            logging.info("Ingestion of the data is completed")

            # Return paths to the train and test CSV files for downstream use
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        # Catch any exception that occurs and raise it as a custom exception for better traceback
        except Exception as e:
            raise CustomException(e, sys)

# This block will run only if this script is executed directly (not imported)
if __name__ == "__main__":
    # Step 1: Create an object of DataIngestion class
    obj = DataIngestion()

    # Step 2: Call the ingestion method to get paths to the train and test data
    train_data, test_data = obj.initiate_data_ingestion()

    # Step 3: Create an object of the DataTransformation class
    data_transformation = DataTransformation()

    # Step 4: Call the transformation function to process the train and test data
    # It returns numpy arrays typically used for ML model training
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Step 5: Model training (Uncomment when model training code is ready)
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))

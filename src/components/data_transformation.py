# Import system and dataclass utilities
import sys
from dataclasses import dataclass

# Add project root to the system path to resolve 'src' module imports
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import required libraries
import numpy as np                        # For numerical operations
import pandas as pd                       # For reading and handling datasets

# Scikit-learn modules for preprocessing
from sklearn.compose import ColumnTransformer             # For applying pipelines to column subsets
from sklearn.impute import SimpleImputer                  # For handling missing values
from sklearn.pipeline import Pipeline                     # For chaining preprocessing steps
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # For encoding and scaling

# Import custom components
from src.exception import CustomException                 # For custom error handling
from src.logger import logging                            # For logging execution steps
from src.utils import save_object                         # For saving objects like transformers

# Configuration class for storing transformation file path
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")  # Path to save preprocessing object

# Main class for data transformation
class DataTransformation:
    def __init__(self):
        # Load configuration settings for this component
        self.data_transformation_config = DataTransformationConfig()

    # Method to create and return the preprocessing object (a ColumnTransformer)
    def get_data_transformer_object(self):
        '''
        This function is responsible for setting up the transformation pipelines
        for both numerical and categorical columns.
        '''
        try:
            # Define the columns to apply transformations on
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical columns: fill missing values and scale
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),      # Fill missing with median
                    ("scaler", StandardScaler())                        # Standardize (mean=0, std=1)
                ]
            )

            # Pipeline for categorical columns: fill missing, one-hot encode, then scale
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),     # Fill missing with mode
                    ("one_hot_encoder", OneHotEncoder()),                     # Convert to one-hot
                    ("scaler", StandardScaler(with_mean=False))               # Scale without centering
                ]
            )

            # Log column information
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Create a ColumnTransformer to apply the above pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor  # Return the configured transformer

        except Exception as e:
            raise CustomException(e, sys)  # Raise a custom exception if anything fails

    # Method to apply the transformations on training and test datasets
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load the train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Get the preprocessor object (ColumnTransformer)
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Define the target/output column
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target for train set
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target for test set
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Fit and transform the training input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            # Only transform (not fit) the testing input features
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatenate input features and target values horizontally (column-wise)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Save the fitted preprocessing object to a file for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return transformed arrays and path to the saved object
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)  # Raise custom error if transformation fails

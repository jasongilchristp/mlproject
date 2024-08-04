import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTranformation, DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv") ## train data path
    test_data_path: str = os.path.join('artifacts', "test.csv") ## test data path
    raw_data_path: str = os.path.join('artifacts', "data.csv")  ## raw data path

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_injection(self):
        logging.info("Enetred the Data Ingestion Method")
        try:
            df = pd.read_csv("notebook\data\stud.csv") ## reading dataset as a dataframe
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) ## Saving raw data at raw data path location

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) ## spliting df into train and test datasets

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) ## saving train dataset at train data path location
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) ## saving test datset at test data path location

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path, ## returning path of train and test datasets
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_injection()

    data_transformation = DataTranformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
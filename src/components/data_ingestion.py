import os
import sys
import pandas as pd
# Manually adding the project root directory (D:\mlProject) to the system path
sys.path.append(r'D:\mlProject')

from src.exception import CustomException
from src.logger import logging
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # artifacts is the directory 
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Reading the CSV file
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read the dataset as Dataframe")

            # Creating the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Saving the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Train-Test Split
            logging.info("Train Test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Saving train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error occurred in data ingestion: {str(e)}")
            raise CustomException(e, sys)


if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()

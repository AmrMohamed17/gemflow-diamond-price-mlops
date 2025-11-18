import pandas as pd
import os
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    data_path: str = "artifacts/data.csv"

class DataIngestion:
    def __init__(self) -> None:
        self.data_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            print("Data Ingestion Started")
            data = pd.read_csv(self.data_config.data_path)
            data = data.drop(columns=['id'])
            train, test = train_test_split(data, test_size=0.25, random_state=42)
            
            
            return train, test
        except Exception as e:
            print(e)
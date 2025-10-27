import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder, StandardScaler

@dataclass
class DataTransformConfig():
  preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransform:
  def __init__(self) -> None:
    self.prep_path = DataTransformConfig()

  def initiate_data_transform(self, train_data, test_data):
    try:
      print("Data transform Started")
      cat_cols = ['cut', 'color', 'clarity']
      num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

      for col in cat_cols:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col])
        test_data[col] = le.transform(test_data[col])

      
      ss = StandardScaler()
      train_processed = pd.DataFrame(ss.fit_transform(train_data), columns = train_data.columns)
      test_processed = pd.DataFrame(ss.transform(test_data), columns = test_data.columns)
      
      print(train_processed.head())
      return train_processed, test_processed
    except Exception as e:
      print(e)

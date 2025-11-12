import os
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import  make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from src.utils.utils import save_object


@dataclass
class DataTransformConfig:
  file_path: str = os.path.join('artifacts', 'preprocessor.pkl') 



class DataTransform:

  def __init__(self) -> None:
    self.DataConfig = DataTransformConfig()

  def get_column_transformer(self):
    cat_cols = ['cut', 'color', 'clarity']
    num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

    cat_pipeline = make_pipeline(
      SimpleImputer(strategy='most_frequent'),
      OrdinalEncoder(categories=[
          ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],  # cut
          ['D', 'E', 'F', 'G', 'H', 'I', 'J'],               # color
          ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']  # clarity
        ],
        handle_unknown='use_encoded_value',
        unknown_value=-1
      ),
      StandardScaler()
    )


    num_pipeline = make_pipeline(
      SimpleImputer(strategy='median'),
      StandardScaler()
    )

    preprocessor = make_column_transformer(
      (num_pipeline, num_cols),
      (cat_pipeline, cat_cols)
    )

    return preprocessor


  def initiate_data_transform(self, train_data, test_data, test=False):
    try:

      cols = ['cut', 'color', 'clarity', 'carat', 'depth', 'table', 'x', 'y', 'z']
      print(train_data.columns)
      for col in cols:
        if col not in train_data.columns:
          raise Exception

      print("Data transform Started")
      prep_obj = self.get_column_transformer()

      target_col = 'price'
      y_train = train_data.pop(target_col)
      y_test = test_data.pop(target_col)


      X_train = pd.DataFrame(prep_obj.fit_transform(train_data), columns=prep_obj.get_feature_names_out())
      X_test = pd.DataFrame(prep_obj.transform(test_data), columns=prep_obj.get_feature_names_out())

      # train_processed = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
      # test_processed = pd.concat([X_test, y_test], axis=1)

      print(X_train.shape)
      print(X_train.head(1))
      print(X_test.head(1))
      print(y_train.head(1))
      print(y_test.head(1))
      

      if not test:
        save_object(self.DataConfig.file_path, prep_obj)

      return X_train, X_test, y_train, y_test

    except Exception as e:
      print(e)
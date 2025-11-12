import os
import pickle
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
      pickle.dump(obj, file_obj)

  except Exception as e:
    print(e)


def evaluate_models(X_train, X_test, y_train, y_test,models):
  try:
    report = {}

    for model_name, model in models.items():
      print(f"started {model_name}")
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      # print(f"predict for {model_name}: {y_pred[0:5]}")
      r2 = r2_score(y_test, y_pred)
      report[model_name] = r2
      print(f"{model_name} Ended Successfully")

    return report
  except Exception as e:
    print(e)


def evaluate_metrics(actual, predicted):
  try:
    r2 = r2_score(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return r2, mse, rmse
  except Exception as e:
    print(e)


def load_object(file_path):
  try:
    with open(file_path, "rb") as file_obj:
      return pickle.load(file_obj)
  except Exception as e:
    print(e)


def get_data_as_dataframe(carat, depth, table, x, y, z, cut, color, clarity):
  try:
    custom_data_input_dict = {
      'carat':[carat],
      'depth':[depth],
      'table':[table],
      'x':[x],
      'y':[y],
      'z':[z],
      'cut':[cut],
      'color':[color],
      'clarity':[clarity]
    }
    return pd.DataFrame(custom_data_input_dict)
  except Exception as e:
    print(e)
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_squared_error
from src.utils.utils import load_object, evaluate_metrics

@dataclass
class ModelEvaluationConfig:
  model_path: str = os.path.join('artifacts', 'model.pkl')
class ModelEvaluation:
  def __init__(self) -> None:
    self.config = ModelEvaluationConfig()

  def initiate_model_evaluate(self, test_data):
    try:
      X_test, y_test = test_data[:,:-1], test_data[:, -1]
      model = load_object(self.config.model_path)
      y_pred = model.predict(X_test)
      r2, mse, rmse = evaluate_metrics(y_test, y_pred)
      return r2, mse, rmse
    except Exception as e:
      print(e)
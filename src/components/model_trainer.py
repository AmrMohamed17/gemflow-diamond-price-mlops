import os
import pandas as pd
from dataclasses import dataclass
from src.utils.utils import evaluate_models, save_object
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.svm import SVC 


@dataclass
class ModelTrainerConfig:
  model_path: str = os.path.join('artifacts', 'model.pkl')
  test_path: str = os.path.join('artifacts', 'test.csv')

class ModelTrainer:
  def __init__(self) -> None:
    self.config = ModelTrainerConfig()

  def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
    print("Model Trainer Started")
    try:
      models = {
        'LinearRegression': LinearRegression(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        # 'RandomForestRegressor': RandomForestRegressor(),
        'XGBRegressor': XGBRegressor()
      }

      model_report = evaluate_models(X_train, X_test, y_train, y_test,models)
      print(model_report)

      best_model_score = max(model_report.values())
      best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

      print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

      save_object(self.config.model_path, models[best_model_name])
      test = pd.concat([X_test, y_test], axis=1)
      test.to_csv(self.config.test_path, index=False)


    except Exception as e:
      print(e)
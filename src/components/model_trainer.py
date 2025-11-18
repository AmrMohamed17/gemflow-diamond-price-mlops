import json
import os
import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import _name_estimators
from src.utils.utils import evaluate_models
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from src.utils.utils import save_object
import mlflow
import yaml


@dataclass
class ModelTrainerConfig:
  model_path: str = os.path.join('artifacts', 'model.pkl')
  # test_path: str = os.path.join('artifacts', 'test.csv')

class ModelTrainer:
  def __init__(self) -> None:
    self.config = ModelTrainerConfig()

  def initiate_model_trainer(self, X_train, X_test, y_train, y_test, test=False):
    print("Model Trainer Started")
    try:
      with open('params.yaml', 'r') as f:
          params = yaml.safe_load(f)

      model_params = params['model']    
      models = {
        # 'LinearRegression': LinearRegression(),
        # 'DecisionTreeRegressor': DecisionTreeRegressor(),
        'XGBRegressor': XGBRegressor(n_estimators= model_params['n_estimators'], 
                                    learning_rate= model_params['learning_rate'], 
                                    max_depth = model_params['max_depth'])
      }

      model_report = evaluate_models(X_train, X_test, y_train, y_test,models)
      print(model_report)

      best_model_score = max(model_report.values())
      best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

      print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

      if not test:
        # dagshub.init(repo_owner="AmrMohamed17", repo_name='gemflow-diamond-price-mlops', mlflow=True)
        # mlflow.set_tracking_uri("https://dagshub.com/AmrMohamed17/gemflow-diamond-price-mlops.mlflow")
        # mlflow.set_experiment("DiamondPricePred")
        # with mlflow.start_run():
        #   mlflow.log_params({
        #     "n_estimators": 300,
        #     "learning_rate": 0.05, 
        #     "max_depth" : 4
        #   })

        #   # best_model = models[best_model_name]
        #   # best_model.fit(X_train, y_train)

        #   mlflow.log_metric("r2_score", best_model_score)

        #   # mlflow.xgboost.log_model(models["XGBRegressor"], "model")

        metrics = {
            "r2_score": float(best_model_score),
            "best_model": best_model_name
        }

        os.makedirs('artifacts', exist_ok=True)
        with open('artifacts/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        save_object(
          file_path=self.config.model_path,
          obj=models[best_model_name]
        )

    except Exception as e:
      print(e)
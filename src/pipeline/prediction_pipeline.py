import os
from src.utils.utils import load_object
from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
  model_path: str = os.path.join('artifacts', 'model.pkl')
  preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class PredictionPipeline:
  def __init__(self) -> None:
    self.config = PredictionPipelineConfig()

  def predict(self, features):
    try:
      model = load_object(self.config.model_path)
      preprocessor = load_object(self.config.preprocessor_path)
      data = preprocessor.transform(features)
      preds = model.predict(data)
      return preds
    except Exception as e:
      print(e)
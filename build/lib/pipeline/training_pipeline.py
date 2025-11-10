from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransform
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
  def start_data_ingestion(self):
    try:
      data_ingestion = DataIngestion()
      train, test = data_ingestion.initiate_data_ingestion()
      return train, test
    except Exception as e:
      print(e)

  def start_data_transform(self, train_data, test_data):
    try:
      data_transform = DataTransform()
      return data_transform.initiate_data_transform(train_data, test_data)
    except Exception as e:
      print(e)

  def start_model_trainer(self, X_train, X_test, y_train, y_test):
    try:
      model_trainer = ModelTrainer()
      model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
    except Exception as e:
      print(e)

  def start_training(self):
    try:
      train, test = self.start_data_ingestion()
      X_train, X_test, y_train, y_test = self.start_data_transform(train, test)
      self.start_model_trainer(X_train, X_test, y_train, y_test)
    except Exception as e:
      print(e)

if __name__ == "__main__":
  obj = TrainingPipeline()
  obj.start_training()
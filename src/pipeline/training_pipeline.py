from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransform

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
      train_processed, test_processed = data_transform.initiate_data_transform(train_data, test_data)
      return train_processed, test_processed
    except Exception as e:
      print(e)


  def start_training(self):
    try:
      train, test = self.start_data_ingestion()
      train_arr, test_arr = self.start_data_transform(train, test)
    except Exception as e:
      print(e)

if __name__ == "__main__":
  obj = TrainingPipeline()
  obj.start_training()

  
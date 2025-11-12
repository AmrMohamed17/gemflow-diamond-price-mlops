import pytest
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransform
from src.components.model_trainer import ModelTrainer

@pytest.fixture(scope="module")
def sample_data():
    """Fixture to load and return the training and testing data."""
    data_ingestion = DataIngestion()
    train, test = data_ingestion.initiate_data_ingestion()
    return train, test

@pytest.fixture(scope="module")
def data_transform(sample_data):
    train, test = sample_data
    return DataTransform().initiate_data_transform(train, test, test=True)


# @pytest.fixture(scope="module")
# def model_trainer(data_transform):
#     X_train, X_test, y_train, y_test = data_transform
#     trainer = ModelTrainer()
#     trainer.initiate_model_trainer(X_train, X_test, y_train, y_test, test=True)
#     return trainer.config.model_path
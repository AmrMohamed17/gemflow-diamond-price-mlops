import pytest
from src.components.data_ingestion import DataIngestion

@pytest.fixture(scope="module")
def sample_data():
    """Fixture to load and return the training and testing data."""
    data_ingestion = DataIngestion()
    train, test = data_ingestion.initiate_data_ingestion()
    return train, test
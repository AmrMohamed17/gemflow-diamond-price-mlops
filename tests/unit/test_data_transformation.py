from src.components.data_transformation import DataTransform
import pandas as pd

def test_data_transformation(sample_data):
    train, test = sample_data
    X_train, X_test, y_train, y_test = DataTransform().initiate_data_transform(train, test)
    
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    assert not X_train.isnull().any().any()
    assert not X_test.isnull().any().any()

import os
import pytest
from src.components.model_trainer import ModelTrainer
from src.utils.utils import load_object


def test_model_trainer_creates_model_file(model_trainer):
    """Test that model trainer creates a model file"""
    model = model_trainer
    
    assert os.path.exists(model)
    


def test_model_can_be_loaded(data_transform):
    """Test that saved model can be loaded and used for prediction"""
    X_train, X_test, y_train, y_test = data_transform
    
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(X_train, X_test, y_train, y_test, test=True)
    
    # Load the model
    model = load_object(trainer.config.model_path)
    
    # Test prediction on a small sample
    predictions = model.predict(X_test.head(5))
    # print(X_train.shape)
    # print(X_test.shape)
    print(f"preds: {predictions}")
    
    assert predictions is not None
    assert len(predictions) == 5
    assert all(pred > 0 for pred in predictions)  # Prices should be positive


# def test_model_output_shape(data_transform):
#     """Test that model predictions have correct shape"""
#     X_train, X_test, y_train, y_test = data_transform
    
#     trainer = ModelTrainer()
#     trainer.initiate_model_trainer(X_train, X_test, y_train, y_test, test=True)
    
#     model = load_object(trainer.config.model_path)
#     predictions = model.predict(X_test)
    
#     assert predictions.shape[0] == X_test.shape[0]
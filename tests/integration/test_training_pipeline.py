import os
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.utils import load_object


def test_complete_training_pipeline():
    """Integration test: Run complete training pipeline end-to-end"""
    pipeline = TrainingPipeline()
    
    # Run the complete pipeline
    pipeline.start_training()
    
    # Check that all artifacts are created
    assert os.path.exists('artifacts/model.pkl')
    assert os.path.exists('artifacts/preprocessor.pkl')
    assert os.path.exists('artifacts/data.csv')


def test_pipeline_data_ingestion_step():
    """Test individual pipeline step: data ingestion"""
    pipeline = TrainingPipeline()
    train, test = pipeline.start_data_ingestion()
    
    # Check data is returned
    assert train is not None
    assert test is not None
    
    # Check train/test split ratio (approximately 80/20)
    total_size = len(train) + len(test)
    train_ratio = len(train) / total_size
    assert 0.7 < train_ratio < 0.85  # Allow some flexibility


def test_pipeline_data_transformation_step():
    """Test individual pipeline step: data transformation"""
    pipeline = TrainingPipeline()
    train, test = pipeline.start_data_ingestion()
    X_train, X_test, y_train, y_test = pipeline.start_data_transform(train, test)
    
    # Check that features and target are separated
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None
    
    # Check shapes match
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_pipeline_model_training_step():
    """Test individual pipeline step: model training"""
    pipeline = TrainingPipeline()
    train, test = pipeline.start_data_ingestion()
    X_train, X_test, y_train, y_test = pipeline.start_data_transform(train, test)
    pipeline.start_model_trainer(X_train, X_test, y_train, y_test)
    
    # Check model is saved
    assert os.path.exists('artifacts/model.pkl')
    
    # Check model can make predictions
    model = load_object('artifacts/model.pkl')
    predictions = model.predict(X_test.head(10))
    assert len(predictions) == 10


def test_trained_model_performance():
    """Integration test: Check that trained model has reasonable performance"""
    pipeline = TrainingPipeline()
    pipeline.start_training()
    
    # Load model and test data
    model = load_object('artifacts/model.pkl')
    preprocessor = load_object('artifacts/preprocessor.pkl')
    
    # Make predictions on test set
    import pandas as pd
    test_data = pd.read_csv('artifacts/data.csv')
    y_test = test_data['price']
    X_test = test_data.drop(columns=['price'])
    
    # Transform and predict
    X_test_transformed = preprocessor.transform(X_test)
    predictions = model.predict(X_test_transformed)
    
    # Calculate R2 score
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, predictions)
    
    # Model should have reasonable performance (R2 > 0.5)
    assert r2 > 0.5, f"Model R2 score is too low: {r2}"


def test_pipeline_artifacts_persistence():
    """Test that pipeline creates persistent artifacts"""
    pipeline = TrainingPipeline()
    pipeline.start_training()
    
    # Check that artifacts directory exists
    assert os.path.exists('artifacts')
    
    # Check that all expected files are created
    expected_files = ['model.pkl', 'preprocessor.pkl', 'data.csv']
    for file in expected_files:
        file_path = os.path.join('artifacts', file)
        assert os.path.exists(file_path), f"{file} not found"
        assert os.path.getsize(file_path) > 0, f"{file} is empty"
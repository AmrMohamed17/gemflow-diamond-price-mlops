import os
import tempfile
import pickle
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.utils.utils import (
    save_object, 
    load_object, 
    evaluate_models, 
    evaluate_metrics,
    get_data_as_dataframe
)


def test_save_and_load_object():
    """Test saving and loading objects with pickle"""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test_model.pkl")
        
        # Create a simple model
        model = LinearRegression()
        X_dummy = np.array([[1, 2], [3, 4], [5, 6]])
        y_dummy = np.array([1, 2, 3])
        model.fit(X_dummy, y_dummy)
        
        # Save and load
        save_object(file_path, model)
        loaded_model = load_object(file_path)
        
        # Test that loaded model works
        predictions = loaded_model.predict(X_dummy)
        assert predictions is not None
        assert len(predictions) == 3


def test_evaluate_metrics():
    """Test evaluation metrics calculation"""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 390, 510])
    
    r2, mse, rmse = evaluate_metrics(y_true, y_pred)
    
    # Check that metrics are calculated
    assert r2 is not None
    assert mse is not None
    assert rmse is not None
    
    # R2 should be between -inf and 1 (typically 0-1 for good models)
    assert r2 <= 1
    
    # MSE and RMSE should be positive
    assert mse > 0
    assert rmse > 0
    
    # RMSE should be square root of MSE
    assert abs(rmse - np.sqrt(mse)) < 0.001


def test_evaluate_metrics_perfect_prediction():
    """Test metrics with perfect predictions"""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([100, 200, 300, 400, 500])
    
    r2, mse, rmse = evaluate_metrics(y_true, y_pred)
    
    # Perfect prediction should have R2 = 1, MSE = 0, RMSE = 0
    assert abs(r2 - 1.0) < 0.001
    assert mse == 0
    assert rmse == 0


def test_get_data_as_dataframe():
    """Test converting input data to DataFrame"""
    df = get_data_as_dataframe(
        carat=0.5,
        depth=61.5,
        table=57.0,
        x=5.15,
        y=5.18,
        z=3.17,
        cut='Ideal',
        color='E',
        clarity='VS1'
    )
    
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    
    # Check all columns are present
    expected_columns = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color', 'clarity']
    assert all(col in df.columns for col in expected_columns)
    
    # Check values
    assert df['carat'].iloc[0] == 0.5
    assert df['cut'].iloc[0] == 'Ideal'
    assert df['color'].iloc[0] == 'E'


def test_evaluate_models_basic():
    """Test model evaluation returns proper report"""
    # Create simple dummy data
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    X_test = np.random.rand(20, 5)
    y_train = np.random.rand(100) * 100
    y_test = np.random.rand(20) * 100
    
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)
    
    models = {
        'LinearRegression': LinearRegression()
    }
    
    report = evaluate_models(X_train_df, X_test_df, y_train_series, y_test_series, models)
    
    # Check report structure
    assert isinstance(report, dict)
    assert 'LinearRegression' in report
    assert isinstance(report['LinearRegression'], float)
    assert report['LinearRegression'] <= 1  
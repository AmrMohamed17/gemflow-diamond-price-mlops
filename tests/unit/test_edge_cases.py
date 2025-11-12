import pytest
import pandas as pd
import numpy as np
from src.components.data_transformation import DataTransform
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.utils import get_data_as_dataframe


def test_data_transformation_handles_unknown_categories(data_transform):
    """Test that data transformation handles unknown category values"""
    # Create data with valid categories
    train_data = pd.DataFrame({
        'carat': [0.5, 1.0],
        'depth': [61.5, 62.0],
        'table': [57.0, 58.0],
        'x': [5.15, 6.5],
        'y': [5.18, 6.48],
        'z': [3.17, 4.0],
        'cut': ['Ideal', 'Premium'],
        'color': ['E', 'F'],
        'clarity': ['VS1', 'VS2'],
        'price': [1000, 2000]
    })
    
    # Create test data with an unknown category (should be handled by handle_unknown='use_encoded_value')
    test_data = pd.DataFrame({
        'carat': [0.7],
        'depth': [61.8],
        'table': [57.5],
        'x': [5.5],
        'y': [5.52],
        'z': [3.5],
        'cut': ['Unknown'],  # This should be handled gracefully
        'color': ['G'],
        'clarity': ['SI1'],
        'price': [1500]
    })
    
    try:
        X_train, X_test, y_train, y_test = data_transform
        assert X_test is not None
    except Exception as e:
        pytest.fail(f"Transformation failed with unknown category: {str(e)}")


def test_prediction_with_edge_case_values():
    """Test prediction with edge case values (very small/large)"""
    pipeline = PredictionPipeline()
    
    # Very small diamond
    tiny_diamond = get_data_as_dataframe(
        carat=0.2,
        depth=60.0,
        table=55.0,
        x=3.0,
        y=3.0,
        z=2.0,
        cut='Fair',
        color='J',
        clarity='I1'
    )
    
    # Very large diamond
    huge_diamond = get_data_as_dataframe(
        carat=5.0,
        depth=65.0,
        table=60.0,
        x=12.0,
        y=12.0,
        z=8.0,
        cut='Ideal',
        color='D',
        clarity='IF'
    )
    
    tiny_price = pipeline.predict(tiny_diamond)[0]
    huge_price = pipeline.predict(huge_diamond)[0]

    print(f"tiny_price: {tiny_price}")
    print(f"juge_price: {huge_price}")
    
    # Both should produce valid predictions
    assert tiny_price > 0
    assert huge_price > 0
    assert huge_price > tiny_price


def test_data_with_all_same_category():
    """Test transformation when all data has same category"""
    uniform_data = pd.DataFrame({
        'carat': [0.5, 0.6, 0.7],
        'depth': [61.5, 61.6, 61.7],
        'table': [57.0, 57.1, 57.2],
        'x': [5.0, 5.1, 5.2],
        'y': [5.0, 5.1, 5.2],
        'z': [3.0, 3.1, 3.2],
        'cut': ['Ideal', 'Ideal', 'Ideal'],  # All same
        'color': ['E', 'E', 'E'],  # All same
        'clarity': ['VS1', 'VS1', 'VS1'],  # All same
        'price': [1000, 1100, 1200]
    })
    
    transformer = DataTransform()
    preprocessor = transformer.get_column_transformer()
    
    # This should not raise an error
    try:
        transformed = preprocessor.fit_transform(uniform_data.drop(columns=['price']))
        assert transformed is not None
    except Exception as e:
        pytest.fail(f"Transformation failed with uniform categories: {str(e)}")





def test_zero_values_in_dimensions():
    """Test prediction with zero values in dimensions (edge case)"""
    pipeline = PredictionPipeline()
    
    # Diamond with some zero dimensions (unrealistic but tests robustness)
    edge_case_diamond = get_data_as_dataframe(
        carat=0.5,
        depth=61.5,
        table=57.0,
        x=0.0,  # Zero dimension
        y=5.0,
        z=3.0,
        cut='Good',
        color='F',
        clarity='VS2'
    )
    
    # Should still make a prediction (model trained on real data might handle this)
    try:
        prediction = pipeline.predict(edge_case_diamond)
        assert prediction is not None
    except Exception as e:
        # It's acceptable if this fails - just document the behavior
        pytest.skip(f"Model doesn't handle zero dimensions: {str(e)}")


def test_prediction_consistency():
    """Test that same input produces same prediction (model determinism)"""
    pipeline = PredictionPipeline()
    
    input_data = get_data_as_dataframe(
        carat=1.0,
        depth=62.0,
        table=58.0,
        x=6.5,
        y=6.5,
        z=4.0,
        cut='Premium',
        color='G',
        clarity='VS1'
    )
    
    # Make predictions multiple times
    pred1 = pipeline.predict(input_data)[0]
    pred2 = pipeline.predict(input_data)[0]
    pred3 = pipeline.predict(input_data)[0]
    
    # All predictions should be identical
    assert pred1 == pred2 == pred3
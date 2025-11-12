def test_data_transformation(data_transform):
    X_train, X_test, y_train, y_test = data_transform
    
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    assert not X_train.isnull().any().any()
    assert not X_test.isnull().any().any()
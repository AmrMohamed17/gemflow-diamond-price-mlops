def test_data_ingestion(sample_data):
    train, test = sample_data

    expected_columns = ["carat", "cut", "color", "clarity", "depth", "price"]
    for col in expected_columns:
        assert col in train.columns
        assert col in test.columns

    assert train.shape[1] == test.shape[1]

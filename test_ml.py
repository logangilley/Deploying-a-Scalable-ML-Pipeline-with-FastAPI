import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture
def data():
    df = pd.read_csv("data/census.csv")
    return df


def test_train_model(data):
    """Test that train_model returns a RandomForestClassifier."""
    X_train, y_train, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_inference(data):
    """Test that inference returns an array of the expected size."""
    X, y, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape[0] == X.shape[0]


def test_compute_model_metrics(data):
    """Test that compute_model_metrics returns precision, recall, and f1 between 0 and 1."""
    X, y, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

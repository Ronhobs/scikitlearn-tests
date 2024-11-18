import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.utils.validation import check_is_fitted
import pytest


def generate_data(samples=100, features=1, noise=0.1, random_state=42):
    """Generate random regression data."""
    return make_regression(
        n_samples=samples, n_features=features, noise=noise, random_state=random_state
    )


def fit_model(X, y):
    """Fit a simple linear regression model."""
    model = LinearRegression()
    model.fit(X, y)
    return model


def predict_and_evaluate(model, X, y):
    """Predict and evaluate using mean squared error."""
    y_pred = model.predict(X)
    return mean_squared_error(y, y_pred)


# Test cases
def test_generate_data():
    X, y = generate_data()
    assert X.shape[0] == 100
    assert len(y) == 100

    X, y = generate_data(samples=200, features=2)
    assert X.shape == (200, 2)
    assert len(y) == 200


def test_fit_model():
    X, y = generate_data()
    model = fit_model(X, y)

    # Check if the model is fitted
    check_is_fitted(model)

    # Coefficient check
    assert len(model.coef_) == X.shape[1]


def test_predict_and_evaluate():
    X, y = generate_data()
    model = fit_model(X, y)

    mse = predict_and_evaluate(model, X, y)
    assert mse < 1.0  # Should be a small error for simple regression


def test_edge_case_single_sample():
    X, y = generate_data(samples=1)
    model = fit_model(X, y)

    with pytest.raises(ValueError):
        predict_and_evaluate(model, X, y)


def test_fail_fit_model():
    with pytest.raises(ValueError):
        fit_model(None, None)


def test_fail_predict():
    X, y = generate_data()
    model = LinearRegression()  # Not fitted model

    with pytest.raises(ValueError):
        model.predict(X)

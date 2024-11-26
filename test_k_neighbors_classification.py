import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
import pytest


def generate_knn_data(samples=150, features=4, classes=3, random_state=42):
    """Generate random classification data for KNN."""
    return make_classification(
        n_samples=samples,
        n_features=features,
        n_classes=classes,
        n_informative=features - 1,
        n_redundant=1,
        random_state=random_state,
    )


def fit_knn(X, y, n_neighbors=3):
    """Fit a K-Neighbors classifier."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)
    return knn


def predict_and_evaluate_knn(knn, X, y):
    """Predict and evaluate using accuracy."""
    y_pred = knn.predict(X)
    return accuracy_score(y, y_pred)


# Test cases
def test_generate_knn_data():
    X, y = generate_knn_data()
    assert X.shape == (150, 4)
    assert len(y) == 150

    X, y = generate_knn_data(samples=200, features=6, classes=2)
    assert X.shape == (200, 6)
    assert len(np.unique(y)) == 2


def test_fit_knn():
    X, y = generate_knn_data()
    knn = fit_knn(X, y, n_neighbors=5)

    # Ensure model is fitted and neighbors parameter is correct
    assert knn.n_neighbors == 5
    assert hasattr(knn, "classes_")


def test_predict_and_evaluate_knn():
    X, y = generate_knn_data()
    knn = fit_knn(X, y, n_neighbors=3)

    acc = predict_and_evaluate_knn(knn, X, y)
    assert acc > 0.8  # Training data should have decent accuracy


def test_unfitted_knn():
    X, y = generate_knn_data()
    knn = KNeighborsClassifier()  # Unfitted classifier

    with pytest.raises(NotFittedError):
        knn.predict(X)


def test_invalid_neighbors():
    with pytest.raises(ValueError):
        fit_knn(np.random.rand(10, 3), np.random.randint(0, 2, size=10), n_neighbors=0)


def test_edge_case_single_neighbor():
    X, y = generate_knn_data()
    knn = fit_knn(X, y, n_neighbors=1)

    # Test with a single neighbor
    acc = predict_and_evaluate_knn(knn, X, y)
    assert acc == 1.0000  # Perfect accuracy with k=1 on training data to 4 decimal places


def test_high_dimensional_data():
    X, y = generate_knn_data(samples=100, features=100)
    knn = fit_knn(X, y, n_neighbors=3)

    # Ensure it can handle high-dimensional data
    acc = predict_and_evaluate_knn(knn, X, y)
    assert acc > 0.7


def test_edge_case_no_classes():
    X = np.random.rand(100, 5)
    y = np.array([])

    with pytest.raises(ValueError):
        fit_knn(X, y)

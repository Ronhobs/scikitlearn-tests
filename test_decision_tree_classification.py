import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.utils.validation import check_is_fitted
import pytest


def generate_classification_data(samples=100, features=5, classes=2, random_state=42):
    """Generate random classification data."""
    return make_classification(
        n_samples=samples,
        n_features=features,
        n_classes=classes,
        n_informative=features - 2,
        n_redundant=0,
        random_state=random_state,
    )


def fit_decision_tree(X, y, max_depth=None):
    """Fit a Decision Tree classifier."""
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    return clf


def predict_and_evaluate(clf, X, y):
    """Predict and evaluate using accuracy."""
    y_pred = clf.predict(X)
    return accuracy_score(y, y_pred)


# Test cases
def test_generate_classification_data():
    X, y = generate_classification_data()
    assert X.shape[0] == 100
    assert X.shape[1] == 5
    assert len(y) == 100

    X, y = generate_classification_data(samples=200, features=3, classes=3)
    assert X.shape == (200, 3)
    assert len(np.unique(y)) == 3


def test_fit_decision_tree():
    X, y = generate_classification_data()
    clf = fit_decision_tree(X, y)

    # Check if the model is fitted
    check_is_fitted(clf)

    # Test with max_depth
    clf = fit_decision_tree(X, y, max_depth=3)
    assert clf.get_depth() <= 3


def test_predict_and_evaluate():
    X, y = generate_classification_data()
    clf = fit_decision_tree(X, y)

    acc = predict_and_evaluate(clf, X, y)
    assert acc > 0.9  # Should have high accuracy on training data


def test_fail_fit_decision_tree():
    with pytest.raises(ValueError):
        fit_decision_tree(None, None)


def test_fail_predict_unfitted():
    X, y = generate_classification_data()
    clf = DecisionTreeClassifier()  # Unfitted classifier

    with pytest.raises(ValueError):
        clf.predict(X)


def test_edge_case_single_class():
    X, y = generate_classification_data(classes=1)
    with pytest.raises(ValueError):
        fit_decision_tree(X, y)


def test_edge_case_no_features():
    X = np.empty((100, 0))  # No features
    y = np.random.randint(0, 2, size=100)

    with pytest.raises(ValueError):
        fit_decision_tree(X, y)

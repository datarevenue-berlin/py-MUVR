import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from omigami.models.pls_classifier import PLSClassifier


@pytest.fixture
def pls_classifier():
    return PLSClassifier()


def test_fit(pls_classifier, mosquito):
    fitted_classifier = pls_classifier.fit(mosquito.X, mosquito.y)
    assert fitted_classifier is pls_classifier
    assert fitted_classifier.coef_ is not None
    assert fitted_classifier.feature_importances_ is not None
    assert len(fitted_classifier.feature_importances_) == mosquito.X.shape[1]


def test_prediction(pls_classifier, mosquito):
    fitted_classifier = pls_classifier.fit(mosquito.X, mosquito.y)
    y_pred = pls_classifier.predict(mosquito.X)
    assert y_pred is not None
    assert set(y_pred) == set(mosquito.y)
    assert y_pred.shape == mosquito.y.shape
    y_val = set(mosquito.y).pop()
    binary_y = (mosquito.y == y_val).astype(float)
    fitted_classifier = pls_classifier.fit(mosquito.X, binary_y)
    y_pred = pls_classifier.predict(mosquito.X)
    assert set(y_pred) == {0, 1}

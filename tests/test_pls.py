import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from py_muvr.models.pls import PLSClassifier, PLSRegressor
from sklearn.cross_decomposition import PLSRegression


@pytest.fixture
def pls_classifier():
    return PLSClassifier()


@pytest.fixture
def pls_regressor():
    return PLSRegressor()


def test_classifier_fit(pls_classifier, mosquito):
    fitted_classifier = pls_classifier.fit(mosquito.X, mosquito.y)
    assert fitted_classifier is pls_classifier
    assert fitted_classifier.coef_ is not None
    assert fitted_classifier.feature_importances_ is not None
    assert len(fitted_classifier.feature_importances_) == mosquito.X.shape[1]


def test_classifier_prediction(pls_classifier, mosquito):
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


def test_regressor_fit(pls_regressor):
    X = np.random.rand(10, 10)
    y = np.random.rand(10)
    sklearn_regressor = PLSRegression().fit(X, y)
    assert pls_regressor.fit(X, y)
    assert pls_regressor.fit(X[:, 0:1], y)
    with pytest.raises(ValueError):
        sklearn_regressor.fit(X[:, 0:1], y)


def test_regressor_predict(pls_regressor):
    X = np.random.rand(10, 10)
    y = np.random.rand(10)
    sklearn_regressor = PLSRegression().fit(X, y)
    pls_regressor.fit(X, y)
    y_pred = pls_regressor.predict(X)
    assert y_pred.shape == y.shape
    assert np.all(y_pred == sklearn_regressor.predict(X).ravel())

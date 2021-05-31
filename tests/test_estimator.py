import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.exceptions import NotFittedError
from py_muvr.models.estimator import Estimator
from py_muvr.models import make_estimator

models = [
    "RFC",
    RandomForestRegressor(10),
    SVR(kernel="linear"),
    "XGBC",
    "PLSC",
    "PLSR",
    SVC(kernel="linear", random_state=1),
    Pipeline([("norm", Normalizer()), ("model", SVC(kernel="linear", random_state=1))]),
    LinearRegression(),
    LogisticRegression(),
]


@pytest.mark.parametrize("est", models)
def test_make_estimator(est):
    estimator = make_estimator(est, 0)
    assert estimator
    assert isinstance(estimator, Estimator)


@pytest.mark.parametrize("est", ["yooo", SVC])
def test_make_estimator_errors(est):
    with pytest.raises(ValueError):
        estimator = make_estimator(est, 0)


@pytest.mark.parametrize("est", models)
def test_estimator_fit(est, dataset):
    estimator = make_estimator(est, 0)
    fit_estimator = estimator.fit(dataset.X, dataset.y)
    assert estimator is fit_estimator


@pytest.mark.parametrize(
    "est",
    models,
)
def test_estimator_predict(est, dataset):
    estimator = make_estimator(est, 0)
    with pytest.raises(NotFittedError):
        _ = estimator.predict(dataset.X)


@pytest.mark.parametrize(
    "est",
    models,
)
def test_estimator_predict(est, dataset):
    estimator = make_estimator(est, 0).fit(dataset.X, dataset.y)
    y = estimator.predict(dataset.X)
    assert y.size == dataset.y.size
    assert y.shape == dataset.y.shape


@pytest.mark.parametrize("est", models)
def test_estimator_clone(est, dataset):
    estimator = make_estimator(est, 0)
    estimator = estimator.fit(dataset.X, dataset.y)
    cloned = estimator.clone()
    assert cloned is not estimator
    with pytest.raises(NotFittedError):
        y_pred = cloned.predict(dataset.X)


@pytest.mark.parametrize("est", models)
def test_train_model(est, dataset):
    estimator = make_estimator(est, 0)
    trained_estimator = estimator.clone().fit(dataset.X, dataset.y)
    assert estimator is not trained_estimator


@pytest.mark.parametrize("estimator", models)
def test_get_feature_importancres(estimator, dataset):
    estimator = make_estimator(estimator, 0)
    estimator.fit(dataset.X, dataset.y)
    feature_importances = estimator.feature_importances
    assert any(feature_importances)
    assert all(feature_importances >= 0)
    assert len(feature_importances) == 12

import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.exceptions import NotFittedError
from omigami.model import Estimator, make_estimator
from omigami.data_models import InputData


@pytest.fixture
def dataset():
    X = np.random.rand(12, 12)
    y = np.random.choice([0, 1], 12)
    return InputData(X=X, y=y, groups=np.arange(12))


@pytest.mark.parametrize(
    "est", ["RFC", SVC(), Pipeline([("norm", Normalizer()), ("model", SVC())])]
)
def test_make_estimator(est):
    estimator = make_estimator(est, 0)
    assert estimator
    assert isinstance(estimator, Estimator)


@pytest.mark.parametrize("est", ["yooo", SVC])
def test_make_estimator_errors(est):
    with pytest.raises(ValueError):
        estimator = make_estimator(est, 0)


@pytest.mark.parametrize(
    "est", ["RFC", SVC(), Pipeline([("norm", Normalizer()), ("model", SVC())])]
)
def test_estimator_fit(est, dataset):
    estimator = make_estimator(est, 0)
    fit_estimator = estimator.fit(dataset.X, dataset.y)
    assert estimator is fit_estimator


@pytest.mark.parametrize(
    "est", ["RFC", SVC(), Pipeline([("norm", Normalizer()), ("model", SVC())])]
)
def test_estimator_predict(est, dataset):
    estimator = make_estimator(est, 0)
    with pytest.raises(NotFittedError):
        y_pred = estimator.predict(dataset.X)
    estimator = estimator.fit(dataset.X, dataset.y)
    y = estimator.predict(dataset.X)
    assert np.any(y)
    assert y.size == dataset.y.size
    assert y.shape == dataset.y.shape


@pytest.mark.parametrize(
    "est", ["RFC", SVC(), Pipeline([("norm", Normalizer()), ("model", SVC())])]
)
def test_estimator_clone(est, dataset):
    estimator = make_estimator(est, 0)
    estimator = estimator.fit(dataset.X, dataset.y)
    cloned = estimator.clone()
    assert cloned is not estimator
    with pytest.raises(NotFittedError):
        y_pred = cloned.predict(dataset.X)


@pytest.mark.parametrize(
    "est", ["RFC", SVC(), Pipeline([("norm", Normalizer()), ("model", SVC())])]
)
def test_train_model(est, dataset):
    estimator = make_estimator(est, 0)
    trained_estimator = estimator.train_model(dataset.X, dataset.y)
    assert estimator is not trained_estimator


@pytest.mark.parametrize(
    "estimator",
    [
        SVC(kernel="linear", random_state=0),
        RandomForestClassifier(random_state=0),
        Pipeline([("N", Normalizer()), ("C", SVC(kernel="linear", random_state=0))]),
    ],
)
def test_get_feature_importancres(estimator, dataset):
    estimator = make_estimator(estimator, 0)
    estimator.fit(dataset.X, dataset.y)
    feature_importances = estimator.feature_importances
    assert any(feature_importances)
    assert all(feature_importances > 0)
    assert len(feature_importances) == 12

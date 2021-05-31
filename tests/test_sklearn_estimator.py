import pytest
import numpy as np
from py_muvr.models import make_estimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression

models = [
    RandomForestRegressor(10),
    SVR(kernel="linear"),
    SVC(kernel="linear", random_state=1),
    Pipeline([("norm", Normalizer()), ("model", SVC(kernel="linear", random_state=1))]),
    Pipeline([("norm", Normalizer()), ("model", LinearRegression())]),
    LinearRegression(),
    LogisticRegression(),
]


@pytest.mark.parametrize("model", models)
def test_get_feature_importances(model, dataset):
    estimator = make_estimator(model, np.random.RandomState(0))
    estimator.fit(dataset.X, dataset.y)
    assert len(estimator.feature_importances) == dataset.X.shape[1]

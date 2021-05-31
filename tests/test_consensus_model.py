import pytest

import numpy as np

from py_muvr.exceptions import NotFitException
from py_muvr.consensus_model import ConsensusModel
from py_muvr.feature_selector import FeatureSelector
from py_muvr.models import ScikitLearnEstimator


@pytest.fixture
def fitted_selector(dataset):
    return FeatureSelector(
        n_outer=3,
        n_repetitions=2,
        metric="MISS",
        estimator="PLSC",
        random_state=0,
    ).fit(dataset.X, dataset.y)


def test_consensus_model_init(fitted_selector):
    fs = FeatureSelector(
        n_outer=3,
        n_repetitions=2,
        metric="MISS",
        estimator="PLSC",
        random_state=0,
    )
    with pytest.raises(NotFitException):
        _ = ConsensusModel(fs, "min")
    with pytest.raises(ValueError):
        _ = ConsensusModel(fitted_selector, "yo")
    assert ConsensusModel(fitted_selector, "min")
    assert ConsensusModel(fitted_selector, "max")
    assert ConsensusModel(fitted_selector, "mid")


def test_predict(fitted_selector, dataset):
    cm = ConsensusModel(fitted_selector, "min")
    y_pred = cm.predict(dataset.X)
    assert y_pred.shape == dataset.y.shape
    assert set(y_pred.tolist()) == set(dataset.y.tolist())


def test_extract_evaluators(fitted_selector):
    cm = ConsensusModel(fitted_selector, "min")
    for model in cm._models:
        assert isinstance(model, ScikitLearnEstimator)


def test_extract_feature_sets(fitted_selector):
    cm = ConsensusModel(fitted_selector, "min")
    assert (
        len(cm._feature_sets) == fitted_selector.n_repetitions * fitted_selector.n_outer
    )
    for features in cm._feature_sets:
        assert len(features) <= fitted_selector._n_features
        assert max(features) <= fitted_selector._n_features - 1
        assert min(features) >= 0


def test_predict_with_regressor(freelive):
    fs = FeatureSelector(
        n_outer=3,
        n_repetitions=2,
        metric="neg_mean_squared_error",
        estimator="PLSR",
        random_state=0,
    ).fit(freelive.X[:, :100], freelive.y)
    cm = ConsensusModel(fs, "mid")

    y_pred = cm.predict(freelive.X[:, :200])

    assert y_pred.shape == freelive.y.shape
    assert all(y_pred >= 0)
    assert np.floor(y_pred.mean()) == np.floor(freelive.y.mean())

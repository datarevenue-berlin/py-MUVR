import pytest
from omigami.exceptions import NotFitException
from omigami.consensus_model import ConsensusModel
from omigami.feature_selector import FeatureSelector
from omigami.models import ScikitLearnEstimator


@pytest.fixture
def fitted_selector(dataset):
    return FeatureSelector(
        n_outer=3, n_repetitions=2, metric="MISS", estimator="PLSC", random_state=0,
    ).fit(dataset.X, dataset.y)


def test_consensus_model(fitted_selector):
    fs = FeatureSelector(
        n_outer=3, n_repetitions=2, metric="MISS", estimator="PLSC", random_state=0,
    )
    with pytest.raises(NotFitException):
        cm = ConsensusModel(fs, "min", "classification")
    with pytest.raises(ValueError):
        cm = ConsensusModel(fitted_selector, "yo", "classification")
    with pytest.raises(ValueError):
        cm = ConsensusModel(fitted_selector, "min", "yo")
    assert ConsensusModel(fitted_selector, "min", "classification")
    assert ConsensusModel(fitted_selector, "max", "classification")
    assert ConsensusModel(fitted_selector, "mid", "classification")


def test_predict(fitted_selector, dataset):
    cm = ConsensusModel(fitted_selector, "min", "classification")
    y_pred = cm.predict(dataset.X)
    assert y_pred.shape == dataset.y.shape
    assert y_pred.size == dataset.y.size
    assert set(y_pred.tolist()) == set(dataset.y.tolist())


def test_extract_evaluators(fitted_selector):
    cm = ConsensusModel(fitted_selector, "min", "classification")
    models = cm._extract_evaluators(fitted_selector, "min")
    for model in models:
        assert isinstance(model, ScikitLearnEstimator)


def test_extract_feature_sets(fitted_selector):
    cm = ConsensusModel(fitted_selector, "min", "classification")
    feature_sets = cm._extract_feature_sets(fitted_selector, "min")
    assert len(feature_sets) == fitted_selector.n_repetitions * fitted_selector.n_outer
    for features in feature_sets:
        assert len(features) <= fitted_selector.n_features
        assert max(features) <= fitted_selector.n_features - 1
        assert min(features) >= 0

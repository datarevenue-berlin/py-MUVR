import pytest
import numpy as np
from sklearn.metrics import SCORERS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from omigami.data_models import FeatureRanks
from omigami.feature_evaluator import FeatureEvaluator, miss_score


@pytest.fixture
def feature_evaluator():
    fe = FeatureEvaluator(
        estimator="RFC",
        metric="MISS",
        random_state=0,
    )
    fe.n_initial_features = 12
    return fe


def test_feature_evaluator(feature_evaluator):
    assert feature_evaluator
    assert feature_evaluator._model_trainer
    assert hasattr(feature_evaluator._metric, "__call__")
    assert feature_evaluator.n_initial_features == 12


# TODO: this test is not passing because the pipeline has no attribute random_state
def test_evaluate_features(dataset):
    pipeline = Pipeline(
        [("normalizer", Normalizer()), ("model", SVC(kernel="linear", random_state=0))]
    )
    fe = FeatureEvaluator(
        estimator=pipeline,
        metric="MISS",
        random_state=0,
    )
    fe.n_initial_features = 12
    evaluation = fe.evaluate_features([0, 4, 6], 0, 0)
    assert evaluation
    assert evaluation.test_score
    assert evaluation.ranks
    assert isinstance(evaluation.ranks, FeatureRanks)
    assert evaluation.ranks.n_feats == 12
    assert evaluation.ranks[0]
    assert evaluation.ranks[1]
    with pytest.raises(ValueError):
        _ = evaluation.ranks[100]


@pytest.mark.parametrize(
    "estimator",
    [
        SVC(kernel="linear", random_state=0),
        RandomForestClassifier(random_state=0),
        Pipeline([("N", Normalizer()), ("C", SVC(kernel="linear", random_state=0))]),
    ],
)
def test_get_feature_importancres(estimator, dataset, feature_evaluator):
    estimator.fit(dataset.X, dataset.y)
    feature_importances = feature_evaluator._get_feature_importances(estimator)
    assert any(feature_importances)
    assert all(feature_importances > 0)
    assert len(feature_importances) == 12


class EstimatorWithFI:
    feature_importances_ = None


class EstimatorWithCoeffs:
    feature_coef_ = None


@pytest.mark.parametrize(
    ("estimator", "attribute", "values"),
    [
        (
            EstimatorWithFI,
            "feature_importances_",
            np.array([0.4, 0.05, 0.4, 0, 0, 0.15]),
        ),
        (EstimatorWithCoeffs, "coef_", np.array([[-0.4, -0.05, 0.4, 0, 0, 0.15]])),
    ],
)
def test_get_feature_rank(estimator, attribute, values, dataset, feature_evaluator):
    setattr(estimator, attribute, values)
    features = [1, 7, 8, 9, 10, 11]
    ranks = feature_evaluator._get_feature_ranks(estimator, features)

    assert ranks[1] == 1.5
    assert ranks[7] == 4
    assert ranks[8] == 1.5
    assert ranks[9] == 5.5
    assert ranks[10] == 5.5
    assert ranks[11] == 3


def test_make_metric(feature_evaluator):
    assert feature_evaluator._metric
    assert feature_evaluator._metric is miss_score


def test_miss_score():
    y_pred = np.array([1, 0, 0, 1])
    y_true = np.array([0, 1, 1, 0])
    assert miss_score(y_true, y_pred) == -4
    y_true = np.array([1, 0, 0, 1])
    assert miss_score(y_true, y_pred) == 0
    y_true = np.array([1, 0, 1, 0])
    assert miss_score(y_true, y_pred) == -2


def test_make_metric_from_string(feature_evaluator):
    for metric_id in SCORERS:
        assert feature_evaluator._make_metric_from_string(metric_id)
    assert feature_evaluator._make_metric_from_string("MISS") is miss_score
    with pytest.raises(ValueError):
        feature_evaluator._make_metric_from_string("yo")

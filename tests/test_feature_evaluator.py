import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from py_muvr.data_structures import FeatureRanks
from py_muvr.feature_evaluator import FeatureEvaluator
from py_muvr.models.sklearn_estimator import ScikitLearnEstimator
from py_muvr.data_splitter import DataSplitter


@pytest.fixture
def feature_evaluator():
    fe = FeatureEvaluator(
        estimator="RFC",
        metric="MISS",
        random_state=0,
    )
    fe.set_n_initial_features(12)
    return fe


def test_feature_evaluator(feature_evaluator):
    assert feature_evaluator
    assert feature_evaluator._estimator
    assert hasattr(feature_evaluator._metric, "__call__")
    assert feature_evaluator._n_initial_features == 12


def test_evaluate_features(dataset):
    pipeline = Pipeline(
        [("normalizer", Normalizer()), ("model", SVC(kernel="linear", random_state=0))]
    )
    fe = FeatureEvaluator(
        estimator=pipeline,
        metric="MISS",
        random_state=0,
    )
    fe.set_n_initial_features(12)
    ds = DataSplitter(n_outer=5, n_inner=4, random_state=0, input_data=dataset)
    split = ds.iter_outer_splits().__next__()
    evaluation_data = ds.split_data(dataset, split)
    evaluation = fe.evaluate_features(evaluation_data, [0, 4, 6])
    assert evaluation
    assert evaluation.test_score >= 0
    assert evaluation.ranks
    assert isinstance(evaluation.ranks, FeatureRanks)
    assert evaluation.ranks.n_feats == 12
    assert evaluation.ranks[0]
    assert evaluation.ranks[1]
    with pytest.raises(ValueError):
        _ = evaluation.ranks[100]


class EstimatorWithFI:
    feature_importances_ = None

    def set_params(*args, **kwargs):
        pass


class EstimatorWithCoeffs:
    feature_coef_ = None

    def set_params(*args, **kwargs):
        pass


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
    model = ScikitLearnEstimator(estimator, None)
    features = [1, 7, 8, 9, 10, 11]
    ranks = feature_evaluator._get_feature_ranks(model, features)

    assert ranks[1] == 1.5
    assert ranks[7] == 4
    assert ranks[8] == 1.5
    assert ranks[9] == 5.5
    assert ranks[10] == 5.5
    assert ranks[11] == 3

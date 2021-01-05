import pytest
import numpy as np
from omigami.data_models import InputData, FeatureRanks, FeatureEvaluationResults
from omigami.recursive_feature_eliminator import (
    RecursiveFeatureEliminator,
)


@pytest.fixture
def dataset():
    X = np.random.rand(12, 12)
    y = np.random.choice([0, 1], 12)
    return InputData(X=X, y=y, groups=np.arange(12))


@pytest.fixture
def inner_loop_results():
    return [
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[1, 2, 3], ranks=[3, 2, 1]), test_score=0.2,
        ),
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[1, 2, 3], ranks=[1.5, 1.5, 3]), test_score=0.2,
        ),
    ]



@pytest.fixture
def rfe():
    rfe = RecursiveFeatureEliminator(
        n_features=10, dropout_rate=0.3
    )
    return rfe


def test_feature_eliminator(rfe):
    assert rfe
    assert all(rfe.features == np.arange(10))


def test_remove_features_loop(rfe):
    rfe._remove_features = lambda feats, args: feats[:-1]

    for feat in rfe.iter_features():
        rfe.remove_features(feat)

    assert len(rfe.features) == 0


def test_stop_condition(rfe):
    assert rfe.stop_condition([10])
    assert not rfe.stop_condition([])


def test_iter_features(rfe):
    bools = [True, True, False, True, True]
    rfe.stop_condition = lambda args: bools.pop(0)

    for feat in rfe.iter_features():
        pass

    assert len(bools) == 2


@pytest.mark.parametrize("keep, selected", [(2, [2, 3]), (1, [2])])
def test_remove_features(rfe, inner_loop_results, keep, selected):
    selected_features = rfe._remove_features(inner_loop_results, keep)
    assert len(selected_features) == keep
    assert sorted(selected_features) == selected  # lowest avg ranks

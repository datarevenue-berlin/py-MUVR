import pytest
import numpy as np
from sklearn import datasets
from omigami.outer_looper import OuterLooper
from omigami.model_trainer import ModelTrainer


@pytest.fixture
def dataset():
    return datasets.make_classification(n_samples=20, n_features=5, random_state=42)


@pytest.fixture
def model_trainer(dataset):
    X, y = dataset
    return ModelTrainer(
        X=X,
        y=y,
        groups=np.arange(len(y)),
        n_inner=2,
        n_outer=2,
        estimator="RFC",
        metric="MISS",
    )


@pytest.fixture
def outer_looper(model_trainer):
    return OuterLooper(
        robust_minimum=0.05, features_dropout_rate=0.2, model_trainer=model_trainer,
    )


@pytest.fixture
def outer_train_results():
    return {
        (0, 1, 2, 3, 4): [
            {"score": 1, "feature_ranks": {0: 2.0, 1: 1.0, 2: 4.0, 3: 5.0, 4: 3.0}},
            {"score": 2, "feature_ranks": {0: 1.0, 1: 2.0, 2: 5.0, 3: 4.0, 4: 3.0}},
        ],
        (0, 1, 4, 2): [
            {"score": 1, "feature_ranks": {0: 1.0, 1: 2.0, 4: 3.0, 2: 4.0}},
            {"score": 2, "feature_ranks": {0: 1.0, 1: 2.0, 4: 3.0, 2: 4.0}},
        ],
        (0, 1, 4): [
            {"score": 3, "feature_ranks": {0: 2.0, 1: 1.0, 4: 3.0}},
            {"score": 2, "feature_ranks": {0: 1.0, 1: 2.0, 4: 3.0}},
        ],
        (0, 1): [
            {"score": 1, "feature_ranks": {0: 1.0, 1: 2.0}},
            {"score": 2, "feature_ranks": {0: 1.0, 1: 2.0}},
        ],
    }


def _select_best_features_and_score(outer_looper, outer_train_results):
    res = outer_looper._select_best_features_and_score(outer_train_results)
    for key in ("min", "max", "mid", "score"):
        assert key in res
    assert sorted(res["min"]) == (0, 1)
    assert len(res["score"]) == 4
    assert res["score"][2] == 3
    assert res["score"][5] == 3


def test_perform_outer_loop_cv(outer_looper):
    i = 0
    olcv = outer_looper._perform_outer_loop_cv(i).compute()
    assert olcv["test_results"]
    assert olcv["scores"] is not None
    assert olcv.test_results.MIN
    assert olcv.test_results.MAX
    assert olcv.test_results.MID
    assert olcv.test_results.MID.feature_ranks
    assert olcv.test_results.MID.feature_ranks
    assert olcv.test_results.MID.score is not None

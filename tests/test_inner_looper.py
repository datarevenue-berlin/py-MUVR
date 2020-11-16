import pytest
import numpy as np
from sklearn import datasets
from omigami.inner_looper import InnerLooper, InnerCVResult, InnerLoopResults
from omigami.model_trainer import ModelTrainer, TrainingTestingResult


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
def inner_looper(model_trainer):
    return InnerLooper(
        outer_index=1, features_dropout_rate=0.2, model_trainer=model_trainer,
    )


@pytest.fixture
def inner_results():
    return InnerCVResult(
        train_results=[
            TrainingTestingResult(
                feature_ranks={0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}, score=10,
            ),
            TrainingTestingResult(
                feature_ranks={0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}, score=9
            ),
            TrainingTestingResult(
                feature_ranks={0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}, score=4
            ),
        ],
        features=[0, 1, 2, 3, 4, 5],
    )


@pytest.fixture
def inner_loop_results(inner_results):
    features = [0, 1, 2, 3, 4, 5]
    results = InnerLoopResults()
    results[features] = inner_results
    return results


def test_run(inner_looper):
    res = inner_looper.run()
    assert len(res) == 4
    lengths = tuple(sorted(len(feats) for feats, _ in res))
    assert lengths == (2, 3, 4, 5)
    for _, inner_results in res:
        break
    assert isinstance(inner_results, InnerCVResult)
    assert len(inner_results) == len(inner_looper.splits)
    assert isinstance(inner_results[0], TrainingTestingResult)


def test_keep_best_features(inner_looper, inner_results):
    kept = inner_looper._keep_best_features(inner_results)
    kept = tuple(sorted(kept))
    assert kept == (0, 1, 2, 3, 4)


def test_get_closest_number_of_features(inner_loop_results):
    assert inner_loop_results.get_closest_number_of_features(100) == 6
    assert inner_loop_results.get_closest_number_of_features(2) == 6


def test_get_features_from_their_number(inner_loop_results):
    f = inner_loop_results.get_features_from_their_number(6)
    assert tuple(sorted(f)) == (0, 1, 2, 3, 4, 5)

import pytest
import numpy as np
from sklearn import datasets, metrics
from omigami.model_trainer import ModelTrainer, miss_score


@pytest.fixture
def dataset():
    return datasets.make_classification(n_samples=200, n_features=5, random_state=42)


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


def test_train_and_evaluate_on_segments(model_trainer):
    split = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 45, 36, 24, 36, 78], [3, 4, 5, 22, 33]
    features = [0, 4]
    res = model_trainer._train_and_evaluate_on_segments(split, features)
    assert res["score"] is not None
    assert res["feature_ranks"]
    assert res["feature_ranks"][0] < res["feature_ranks"][4]


def test_extract_feature_rank(model_trainer):
    class DummyEstimatorFI:
        feature_importances_ = None

    class DummyEstimatorCoeff:
        coef_ = None

    estimator = DummyEstimatorFI()
    feature_importances = np.array([0.05, 0.4, 0, 0, 0.15, 0.4])
    estimator.feature_importances_ = feature_importances
    features = [7, 8, 9, 10, 11, 12]
    ranks = model_trainer._extract_feature_rank(estimator, features)

    assert len(ranks) == len(estimator.feature_importances_)
    assert ranks[7] == 4
    assert ranks[8] == 1.5
    assert ranks[9] == 5.5
    assert ranks[10] == 5.5
    assert ranks[11] == 3
    assert ranks[12] == 1.5

    estimator = DummyEstimatorCoeff()
    coef = np.array([-0.05, 0.4, 0, 0, 0.15, -0.4])
    estimator.coef_ = [coef]

    ranks = model_trainer._extract_feature_rank(estimator, features)

    assert len(ranks) == len(coef)
    assert ranks[7] == 4
    assert ranks[8] == 1.5
    assert ranks[9] == 5.5
    assert ranks[10] == 5.5
    assert ranks[11] == 3
    assert ranks[12] == 1.5


def test_make_splits(model_trainer):
    splits = model_trainer._make_splits()
    n_outer_splits = model_trainer.n_outer
    n_inner_splits = model_trainer.n_outer * model_trainer.n_inner
    assert splits
    assert len(splits) == n_inner_splits + n_outer_splits

    # check test size and train size are as expected
    possible_test_size = {
        np.floor(model_trainer.X.shape[0] / model_trainer.n_outer),
        np.ceil(model_trainer.X.shape[0] / model_trainer.n_outer),
    }
    possible_train_size = {model_trainer.X.shape[0] - i for i in possible_test_size}
    for i in range(model_trainer.n_outer):
        train_idx, test_idx = splits[(i,)]
        assert len(train_idx) in possible_train_size
        assert len(test_idx) in possible_test_size

    # check there is no ntersection among the groups
    for i in range(model_trainer.n_outer):
        train_idx, test_idx = splits[(i,)]
        for j in range(model_trainer.n_inner):
            inner_train, valid_idx = splits[(i, j)]
            assert not set(inner_train).intersection(valid_idx)
            assert not set(test_idx).intersection(valid_idx)
            assert not set(inner_train).intersection(test_idx)


def test_make_metric(model_trainer):
    assert model_trainer.metric
    assert model_trainer.metric is miss_score


def test_miss_score():
    y_pred = np.array([1, 0, 0, 1])
    y_true = np.array([0, 1, 1, 0])
    assert miss_score(y_true, y_pred) == -4
    y_true = np.array([1, 0, 0, 1])
    assert miss_score(y_true, y_pred) == 0
    y_true = np.array([1, 0, 1, 0])
    assert miss_score(y_true, y_pred) == -2


def test_make_metric_from_string(model_trainer):
    for metric_id in metrics.SCORERS:
        assert model_trainer._make_metric_from_string(metric_id)
    model_trainer._make_metric_from_string("MISS") is miss_score
    with pytest.raises(ValueError):
        model_trainer._make_metric_from_string("yo")

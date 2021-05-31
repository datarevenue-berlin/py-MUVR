import pytest
import numpy as np
from py_muvr.models import sklearn_metrics


regression_true = np.array([1, 2, 3, 4, 5])
regression_pred = np.array([5, 4, 3, 2, 1])
classification_true = np.array([1, 0, 1, 2, 0, 0])
classification_pred = np.array([2, 0, 1, 1, 0, 1])
binary_classification_true = np.array([1, 0, 1, 1, 0, 0])
binary_classification_pred = np.array([1, 0, 1, 0, 1, 1])


@pytest.mark.parametrize(
    ("y_pred", "y_true", "res"),
    [
        (regression_pred, regression_true, -3),
        (regression_true, regression_true, 1),
    ],
)
def test_explained_variance(y_pred, y_true, res):
    metric = sklearn_metrics.SKLEARN_METRICS["explained_variance"]
    score = metric(y_true, y_pred)
    assert res == score


@pytest.mark.parametrize(
    ("y_pred", "y_true", "res"),
    [
        (regression_pred, regression_true, -3),
        (regression_true, regression_true, 1),
    ],
)
def test_r2(y_pred, y_true, res):
    metric = sklearn_metrics.SKLEARN_METRICS["r2"]
    score = metric(y_true, y_pred)
    assert res == score


@pytest.mark.parametrize(
    ("y_pred", "y_true", "res"),
    [
        (regression_pred, regression_true, -4),
        (regression_true, regression_true, 0),
    ],
)
def test_max_error(y_pred, y_true, res):
    metric = sklearn_metrics.SKLEARN_METRICS["max_error"]
    score = metric(y_true, y_pred)
    assert res == score


@pytest.mark.parametrize(
    ("y_pred", "y_true", "res"),
    [
        (regression_pred, regression_true, -2),
        (regression_true, regression_true, 0),
    ],
)
def test_neg_median_absolute_error(y_pred, y_true, res):
    metric = sklearn_metrics.SKLEARN_METRICS["neg_median_absolute_error"]
    score = metric(y_true, y_pred)
    assert res == score


@pytest.mark.parametrize(
    ("y_pred", "y_true", "res"),
    [
        (regression_pred, regression_true, -12 / 5),
        (regression_true, regression_true, 0),
    ],
)
def test_neg_mean_absolute_error(y_pred, y_true, res):
    metric = sklearn_metrics.SKLEARN_METRICS["neg_mean_absolute_error"]
    score = metric(y_true, y_pred)
    assert res == score


@pytest.mark.parametrize(
    ("y_pred", "y_true", "res"),
    [
        (regression_pred, regression_true, -8),
        (regression_true, regression_true, 0),
    ],
)
def test_neg_mean_squared_error(y_pred, y_true, res):
    metric = sklearn_metrics.SKLEARN_METRICS["neg_mean_squared_error"]
    score = metric(y_true, y_pred)
    assert res == score


@pytest.mark.parametrize(
    ("y_pred", "y_true", "res"),
    [
        (
            regression_pred,
            regression_true,
            -np.mean((np.log(1 + regression_true) - np.log(1 + regression_pred)) ** 2),
        ),
        (regression_true, regression_true, 0),
    ],
)
def test_neg_mean_squared_log_error(y_pred, y_true, res):
    metric = sklearn_metrics.SKLEARN_METRICS["neg_mean_squared_log_error"]
    score = metric(y_true, y_pred)
    assert res == score


@pytest.mark.parametrize(
    ("y_pred", "y_true", "res"),
    [
        (regression_pred, regression_true, -np.sqrt(8)),
        (regression_true, regression_true, 0),
    ],
)
def test_neg_root_mean_squared_error(y_pred, y_true, res):
    metric = sklearn_metrics.SKLEARN_METRICS["neg_root_mean_squared_error"]
    score = metric(y_true, y_pred)
    assert res == score


def test_neg_mean_poisson_deviance():
    metric = sklearn_metrics.SKLEARN_METRICS["neg_mean_poisson_deviance"]
    assert metric(regression_true, regression_pred) < 0
    assert metric(regression_true, regression_true) == 0


def test_neg_mean_gamma_deviance():
    metric = sklearn_metrics.SKLEARN_METRICS["neg_mean_gamma_deviance"]
    assert metric(regression_true, regression_pred) < 0
    assert metric(regression_true, regression_true) == 0


@pytest.mark.parametrize(
    ("y_pred", "y_true", "res"),
    [
        (classification_pred, classification_true, 0.5),
        (classification_true, classification_true, 1),
    ],
)
def test_accuracy(y_pred, y_true, res):
    metric = sklearn_metrics.SKLEARN_METRICS["accuracy"]
    score = metric(y_true, y_pred)
    assert res == score


@pytest.mark.parametrize(
    ("y_pred", "y_true", "res"),
    [
        (classification_pred, classification_true, 0.38888888888888884),
        (classification_true, classification_true, 1),
    ],
)
def test_balanced_accuracy(y_pred, y_true, res):
    metric = sklearn_metrics.SKLEARN_METRICS["balanced_accuracy"]
    score = metric(y_true, y_pred)
    assert res == score


@pytest.mark.parametrize(
    ("y_pred", "y_true", "res"),
    [
        (binary_classification_pred, binary_classification_true, 0.5),
        (binary_classification_true, binary_classification_true, 1),
    ],
)
def test_precision(y_pred, y_true, res):
    metric = sklearn_metrics.SKLEARN_METRICS["precision"]
    score = metric(y_true, y_pred)
    assert res == score


def test_precision_micro():
    metric = sklearn_metrics.SKLEARN_METRICS["precision_micro"]
    assert metric(classification_true, classification_pred) < 1
    assert metric(classification_true, classification_true) == 1


def test_precision_macro():
    metric = sklearn_metrics.SKLEARN_METRICS["precision_macro"]
    assert metric(classification_true, classification_pred) < 1
    assert metric(classification_true, classification_true) == 1


@pytest.mark.parametrize(
    ("y_pred", "y_true", "res"),
    [
        (binary_classification_pred, binary_classification_true, 2 / 3),
        (binary_classification_true, binary_classification_true, 1),
    ],
)
def test_recall(y_pred, y_true, res):
    metric = sklearn_metrics.SKLEARN_METRICS["recall"]
    score = metric(y_true, y_pred)
    assert res == score


def test_recall_macro():
    metric = sklearn_metrics.SKLEARN_METRICS["recall_macro"]
    assert metric(classification_true, classification_pred) < 1
    assert metric(classification_true, classification_true) == 1


def test_recall_micro():
    metric = sklearn_metrics.SKLEARN_METRICS["recall_micro"]
    assert metric(classification_true, classification_pred) < 1
    assert metric(classification_true, classification_true) == 1


@pytest.mark.parametrize(
    ("y_pred", "y_true", "res"),
    [
        (binary_classification_pred, binary_classification_true, 4 / 7),
        (binary_classification_true, binary_classification_true, 1),
    ],
)
def test_f1(y_pred, y_true, res):
    metric = sklearn_metrics.SKLEARN_METRICS["f1"]
    score = metric(y_true, y_pred)
    assert abs(res - score) < 1e-10


def test_f1_macro():
    metric = sklearn_metrics.SKLEARN_METRICS["f1_macro"]
    assert metric(classification_true, classification_pred) < 1
    assert metric(classification_true, classification_true) == 1


def test_f1_micro():
    metric = sklearn_metrics.SKLEARN_METRICS["f1_micro"]
    assert metric(classification_true, classification_pred) < 1
    assert metric(classification_true, classification_true) == 1


def test_jaccard():
    metric = sklearn_metrics.SKLEARN_METRICS["jaccard"]
    assert metric(binary_classification_true, binary_classification_pred) < 1
    assert metric(binary_classification_true, binary_classification_true) == 1


def test_jaccard_macro():
    metric = sklearn_metrics.SKLEARN_METRICS["jaccard_macro"]
    assert metric(classification_true, classification_pred) < 1
    assert metric(classification_true, classification_true) == 1


def test_jaccard_micro():
    metric = sklearn_metrics.SKLEARN_METRICS["jaccard_micro"]
    assert metric(classification_true, classification_pred) < 1
    assert metric(classification_true, classification_true) == 1

import pytest
import numpy as np
from py_muvr.models.sklearn_metrics import SKLEARN_METRICS
from py_muvr.models.metrics import make_metric, _make_metric_from_string, miss_score


def test_make_metric():
    metric = make_metric("MISS")
    assert metric
    assert metric is miss_score


def test_miss_score():
    y_pred = np.array([1, 0, 0, 1])
    y_true = np.array([0, 1, 1, 0])
    assert miss_score(y_true, y_pred) == -4
    y_true = np.array([1, 0, 0, 1])
    assert miss_score(y_true, y_pred) == 0
    y_true = np.array([1, 0, 1, 0])
    assert miss_score(y_true, y_pred) == -2


def test_make_metric_from_string():
    for metric_id in SKLEARN_METRICS:
        assert _make_metric_from_string(metric_id)
    assert _make_metric_from_string("MISS") is miss_score
    with pytest.raises(ValueError):
        _make_metric_from_string("yo")
